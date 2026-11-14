"""Tools compute the answers; budgets bound the conversation; every step is traced."""
from __future__ import annotations

import json

from conftest import tool_results
from msagent.config import Budgets
from msagent.gateway.budget import BudgetTracker
from msagent.kb import KnowledgeBase
from msagent.llm.base import ModelError
from msagent.llm.fake import FakeModel, call, say
from msagent.tenancy import TenantRegistry


def _coverage(session) -> dict:
    content = session.transcript[0]["content"]
    return json.loads(content.split("<member_coverage>\n")[1].split("\n</member_coverage>")[0])


def test_remaining_amounts_are_calculated_by_the_tool_for_the_current_plan_year(make_service):
    svc = make_service([say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "How much is left on the deductible?")
    acc = _coverage(session)["accumulators"]
    assert acc["plan_year"] == 2026
    assert acc["deductible"] == {"plan_amount": "$1,000.00", "met_so_far": "$154.48", "remaining": "$845.52"}
    assert acc["out_of_pocket_maximum"]["remaining"] == "$3,363.66"
    assert acc["physical_therapy_visits"] == {"plan_limit": 30, "used": 12, "remaining": 18}


def test_terminated_coverage_is_flagged_in_context(make_service):
    svc = make_service([say("ok")])
    session = svc.start("plan_a", "M100002")
    svc.send(session, "What is the urgent care copay?")
    cov = _coverage(session)
    assert cov["coverage_status"] == "TERMINATED" and "2026-08-31" in cov["coverage_note"]


def test_retrieval_uses_current_plan_year_unless_asked(make_service):
    svc = make_service([call("search_plan_documents", query="specialist visit", plan_year=2027),
                        call("search_plan_documents", query="specialist visit", plan_year=2031), say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "What does a specialist visit cost after renewal?")
    auto = session.transcript[0]["content"]
    assert "$40 copay" in auto and "$55 copay" not in auto          # auto-context: 2026 only
    next_year, bad_year = (json.loads(r["content"]) for r in tool_results(session))
    assert next_year["plan_year_searched"] == 2027 and "$55 copay" in json.dumps(next_year)
    assert next_year["is_current_plan_year"] is False
    assert bad_year["error"] == "tool_error"


def test_plan_rules_are_parsed_per_tenant(settings):
    kb = KnowledgeBase.load(settings.pack_dir, TenantRegistry.load(settings))
    assert kb.for_tenant("plan_a").policy().grievance_resolution_days == 30
    assert kb.for_tenant("plan_b").policy().grievance_resolution_days == 45
    design = kb.for_tenant("plan_a").plan_design("GOLD", 2027)
    assert (design.deductible_individual, design.oop_max_individual, design.pt_visit_limit) == (1500, 4500, 24)


def test_claims_tools_count_filter_and_compute_the_appeal_window(make_service, upstream):
    upstream.throttle_every = 3
    svc = make_service([call("find_claims", date_from="2026-01-01", date_to="2026-12-31", status="DENIED"),
                        call("find_claims", service_date="2026-08-05"),
                        call("get_claim", claim_id="C1000000001"), call("get_claim", claim_id="C1000000004"),
                        say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "Claims questions")
    denied, by_date, recent, old = (json.loads(r["content"]) for r in tool_results(session))
    assert denied["matching_count"] == 2 and denied["total_claims_on_file"] == 5
    assert by_date["claims"][0]["member_owes_total"] == "$40.00"
    assert recent["appeal"]["last_day_to_appeal"] == "2026-11-28"       # 2026-06-01 + 180 days
    assert recent["appeal"]["still_within_appeal_window"] is True
    assert old["appeal"]["last_day_to_appeal"] == "2026-09-16"
    assert old["appeal"]["still_within_appeal_window"] is False
    assert "member_id" not in json.dumps(denied)


def test_grievance_deadline_is_computed_from_the_tenants_own_rule(make_service):
    svc = make_service([call("get_grievances", grievance_id="G26-000001"), say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "Is grievance G26-000001 past the deadline?")
    res = json.loads(tool_results(session)[0]["content"])["grievances"][0]["resolution"]
    assert res["resolution_deadline"] == "2026-11-08" and res["past_resolution_deadline"] is True


def test_step_budget_stops_a_runaway_model(make_service):
    loop = FakeModel(lambda system, messages, tools: call("get_coverage"))
    svc = make_service(loop, Budgets(max_steps=8, max_model_calls_per_turn=50))
    session = svc.start("plan_a", "M100001")
    reply = svc.send(session, "loop forever")
    assert reply.outcome == "budget_exhausted" and "Nothing was changed" in reply.text
    assert session.budget.steps <= 8
    assert any(e.kind == "gateway_decision" and e.attrs.get("policy") == "budget"
               for e in session.tracer.events)


def test_model_call_budget_per_turn(make_service):
    loop = FakeModel(lambda system, messages, tools: call("get_coverage"))
    svc = make_service(loop, Budgets(max_steps=100, max_model_calls_per_turn=3))
    reply = svc.send(svc.start("plan_a", "M100001"), "loop")
    assert reply.outcome == "budget_exhausted" and len(loop.calls) == 3


def test_cost_and_latency_budgets():
    now = [0.0]
    tracker = BudgetTracker(Budgets(max_steps=10, max_cost_usd=0.01, turn_deadline_s=5), clock=lambda: now[0])
    assert tracker.check("model") is None
    tracker.charge_model(0.02)
    assert tracker.check("tool").limit == "cost"
    tracker = BudgetTracker(Budgets(turn_deadline_s=5), clock=lambda: now[0])
    now[0] = 6.0
    assert tracker.check("tool").limit == "latency"
    tracker.start_turn()
    assert tracker.check("tool") is None


def test_cost_is_metered_from_token_usage(make_service, settings):
    from msagent.llm.base import Usage, cost_usd

    assert cost_usd("claude-haiku-5-5", Usage(1_000_000, 1_000_000)) == 0.60
    assert cost_usd("local-model", Usage(1000, 1000)) == 0.0
    assert cost_usd("local-model", Usage(1_000_000, 0), price_in=2.0) == 2.0


def test_model_outage_degrades_safely(make_service, upstream):
    def boom(system, messages, tools):
        raise ModelError("provider down")

    svc = make_service(FakeModel(boom))
    reply = svc.send(svc.start("plan_a", "M100001"), "anything")
    assert reply.outcome == "unavailable" and upstream.posts == 0


def test_every_step_is_traced_with_decisions(make_service):
    svc = make_service([call("get_claim", claim_id="C1000000002"), call("nope"), say("PAID, member owes $40.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "Status of claim C1000000002?")
    kinds = [(e.kind, e.name, e.status) for e in session.tracer.events]
    assert ("turn", "rep_message", "info") in kinds
    assert ("gateway_decision", "get_claim", "ok") in kinds
    assert ("tool_call", "get_claim", "ok") in kinds
    assert ("gateway_decision", "nope", "denied") in kinds
    assert ("retrieval", "plan_documents", "ok") in kinds
    assert [k for k in kinds if k[0] == "model_call"] and kinds[-1][:2] == ("turn", "reply")
    model_event = next(e for e in session.tracer.events if e.kind == "model_call")
    assert {"input_tokens", "output_tokens", "cost_usd", "tool_calls"} <= set(model_event.attrs)
    assert [e.seq for e in session.tracer.events] == list(range(1, len(kinds) + 1))


def test_system_prompt_is_tenant_specific_and_stable(make_service):
    svc = make_service([say("a"), say("b"), say("c")])
    s1, s2 = svc.start("plan_a", "M100001"), svc.start("plan_a", "M777777")
    svc.send(s1, "q1"), svc.send(s2, "q2"), svc.send(svc.start("plan_b", "M200001"), "q3")
    sys_a1, sys_a2, sys_b = (c[0] for c in svc.model.calls)
    assert sys_a1 == sys_a2                               # cacheable prefix: no member data in it
    assert "Alpha Health" in sys_a1 and "Beta Care" not in sys_a1 and "Alpha Health" not in sys_b


def test_tools_state_the_yes_no_answer_so_the_model_does_not_derive_it(make_service):
    svc = make_service([call("get_claim", claim_id="C1000000001"), call("get_claim", claim_id="C1000000004"),
                        call("get_grievances"), say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "Can the member still appeal?")
    recent, old, grievances = (json.loads(r["content"]) for r in tool_results(session))
    assert recent["appeal"]["answer"].startswith("Yes") and "2026-11-28" in recent["appeal"]["answer"]
    assert old["appeal"]["answer"].startswith("No")
    assert grievances["grievances"][0]["resolution"]["answer"].startswith("Yes")


def test_an_empty_model_reply_is_retried_once(make_service):
    svc = make_service([say(""), say("The copay is $40.")])
    session = svc.start("plan_a", "M100001")
    assert svc.send(session, "Specialist copay?").text == "The copay is $40."
    svc = make_service([say(""), say("")])
    reply = svc.send(svc.start("plan_a", "M100001"), "Specialist copay?")
    assert "couldn't produce an answer" in reply.text and len(svc.model.calls) == 2
