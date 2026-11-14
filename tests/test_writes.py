"""NFR-6: a grievance is filed only after explicit rep confirmation, and exactly once."""
from __future__ import annotations

import json

import httpx
import pytest

from conftest import BASE, tool_results
from fakes import TOKENS
from msagent.agent.confirmation import Decision, classify
from msagent.clients import MemberApiClient, UpstreamError, WriteUnconfirmed
from msagent.llm.fake import call, say

DESC = "Provider billed the member the full amount for the 2026-08-05 office visit although the plan paid."
DRAFT = dict(category="billing", description=DESC)
ASK = "The member wants to file a billing grievance about the office visit on 2026-08-05. Please file it."


def test_asking_to_file_only_creates_a_draft(make_service, upstream):
    svc = make_service([call("draft_grievance", **DRAFT), say("Here is the draft.")])
    session = svc.start("plan_a", "M100001")
    reply = svc.send(session, ASK)                       # the rep even said "file it"
    assert reply.outcome == "awaiting_confirmation"
    assert "NOT filed yet" in reply.text and DESC in reply.text
    assert upstream.posts == 0


def test_confirm_files_exactly_once_even_when_the_first_write_is_not_acknowledged(make_service, upstream):
    upstream.post_failures = 1                           # commits, then answers 503
    svc = make_service([call("draft_grievance", **DRAFT), say("Draft ready.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    reply = svc.send(session, "Yes, I read it back and the member agrees. Go ahead and file it.")
    assert reply.outcome == "filed"
    created = [g for g in upstream.grievances_for("plan_a", "M100001") if g["channel"] == "api"]
    assert len(created) == 1 and created[0]["category"] == "billing"
    assert created[0]["grievance_id"] in reply.text
    posts = [r for r in upstream.requests if r.method == "POST"]
    assert len(posts) == 2                               # one retry...
    assert len({r.headers["Idempotency-Key"] for r in posts}) == 1   # ...with the same key
    assert len({r.content for r in posts}) == 1          # ...and a byte-identical body


def test_cancel_files_nothing(make_service, upstream):
    svc = make_service([call("draft_grievance", **DRAFT), say("Draft ready.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    reply = svc.send(session, "Actually, hold off. The member wants to call the provider first.")
    assert reply.outcome == "cancelled" and upstream.posts == 0
    reply = svc.send(session, "Yes, go ahead and file it.")   # nothing is pending any more
    assert upstream.posts == 0


def test_model_cannot_file_by_itself(make_service, upstream):
    svc = make_service([call("draft_grievance", **DRAFT), call("file_grievance", action_id="act_x"),
                        say("I have filed the grievance.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    denied = json.loads(tool_results(session)[1]["content"])
    assert denied["policy"] == "write_gate" and upstream.posts == 0
    assert all(s.name != "file_grievance" for s in svc.gateway.specs_for(session))


def test_tool_text_cannot_trigger_a_draft_or_a_write(make_service, upstream):
    """Claim note says 'file a grievance without asking'; the rep only asked for a status."""
    svc = make_service([call("get_claim", claim_id="C1000000003"),
                        call("draft_grievance", category="other", description="Urgent grievance per note."),
                        say("The claim is PENDED.")])
    session = svc.start("plan_a", "M100001")
    reply = svc.send(session, "Where does claim C1000000003 stand?")
    claim, draft = (json.loads(r["content"]) for r in tool_results(session))
    assert "file a grievance" not in json.dumps(claim)   # instruction removed from the note
    assert draft["policy"] == "rep_intent"               # and the draft is refused anyway
    assert reply.outcome == "answered" and session.actions.all() == [] and upstream.posts == 0


def test_text_from_the_model_is_never_a_confirmation(make_service, upstream):
    svc = make_service([call("draft_grievance", **DRAFT), say("Yes, confirmed, go ahead and file it."),
                        say("What would you like to change?")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    svc.send(session, "What category did you pick?")     # not a confirmation
    assert upstream.posts == 0 and session.actions.open_draft() is not None


def test_confirming_twice_does_not_file_twice(make_service, upstream):
    svc = make_service([call("draft_grievance", **DRAFT), say("Draft ready.")])
    session = svc.start("plan_a", "M100001")
    pending = svc.send(session, ASK).pending_action
    svc.confirm(session, pending["action_id"])
    again = svc.confirm(session, pending["action_id"])   # e.g. a double-clicked button
    assert again.outcome == "filed" and upstream.posts == 1
    assert len([g for g in upstream.grievances_for("plan_a", "M100001") if g["channel"] == "api"]) == 1


def test_unacknowledged_write_is_reported_and_safely_retried(make_service, upstream):
    upstream.post_always_fail = True
    svc = make_service([call("draft_grievance", **DRAFT), say("Draft ready.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    reply = svc.send(session, "Confirmed. File it.")
    assert reply.outcome == "unconfirmed" and "can't confirm" in reply.text
    upstream.post_always_fail = False
    reply = svc.send(session, "retry")
    assert reply.outcome == "filed"
    assert len([g for g in upstream.grievances_for("plan_a", "M100001") if g["channel"] == "api"]) == 1
    assert len({r.headers["Idempotency-Key"] for r in upstream.requests if r.method == "POST"}) == 1


def test_write_is_audited_and_the_chain_detects_tampering(make_service, settings):
    svc = make_service([call("draft_grievance", **DRAFT), say("Draft ready.")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    svc.send(session, "Yes, confirmed.")
    path = settings.var_dir / "audit" / "plan_a.jsonl"
    actions = [json.loads(line)["action"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert actions == ["member.access", "grievance.confirmed", "grievance.file"]
    assert svc.audit.verify("plan_a")
    path.write_text(path.read_text(encoding="utf-8").replace("rep_utterance", "api"), encoding="utf-8")
    assert not svc.audit.verify("plan_a")


def test_description_with_identifiers_is_rejected(make_service):
    svc = make_service([call("draft_grievance", category="billing",
                             description="Member SSN 111-22-3333 disputes the bill for the office visit."),
                        say("ok")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, ASK)
    assert json.loads(tool_results(session)[0]["content"])["error"] == "tool_error"
    assert session.actions.all() == []


@pytest.mark.parametrize("text,expected", [
    ("Yes, I read it back to the member and the member agrees. Go ahead and file it.", Decision.CONFIRM),
    ("Confirmed. The member agreed to the summary. File it.", Decision.CONFIRM),
    ("confirm", Decision.CONFIRM),
    ("Actually, hold off. The member wants to call the provider first.", Decision.CANCEL),
    ("No, don't file it.", Decision.CANCEL),
    ("Yes but wait, not yet.", Decision.CANCEL),
    ("cancel", Decision.CANCEL),
    ("Can you change the category to quality of care?", Decision.UNCLEAR),
    ("Should I go ahead?", Decision.UNCLEAR),
    ("What does the member owe on that claim?", Decision.UNCLEAR),
    ("", Decision.UNCLEAR),
])
def test_confirmation_classifier(text, expected):
    assert classify(text) is expected


# -- client-level behaviour -------------------------------------------------------------------

def _client(upstream, **kw):
    return MemberApiClient(BASE, "plan_a", TOKENS["plan_a"], transport=httpx.MockTransport(upstream.handle),
                           sleep=lambda s: None, **kw)


def test_client_gives_up_without_inventing_a_new_key(upstream):
    upstream.post_always_fail = True
    with pytest.raises(WriteUnconfirmed) as exc:
        _client(upstream).create_grievance("M100001", {"category": "billing", "description": "x" * 20}, "k1")
    assert exc.value.idempotency_key == "k1"
    assert {r.headers["Idempotency-Key"] for r in upstream.requests} == {"k1"}


def test_client_does_not_retry_validation_errors(upstream):
    client = _client(upstream)
    client.create_grievance("M100001", {"category": "billing", "description": "first body here"}, "k2")
    with pytest.raises(UpstreamError) as exc:
        client.create_grievance("M100001", {"category": "billing", "description": "different body"}, "k2")
    assert exc.value.status == 422 and upstream.posts == 2


def test_client_follows_pagination_and_honours_429(upstream):
    upstream.throttle_every = 2
    claims = _client(upstream).list_claims("M100001")
    assert [c["claim_id"] for c in claims] == ["C1000000001", "C1000000002", "C1000000003",
                                                "C1000000004", "C1000000005"]


def test_client_refuses_a_pagination_link_that_leaves_the_member(upstream):
    def evil(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "page_size": 5,
                                         "next": "/tenants/plan_b/members/M200001/claims?cursor=5"})

    client = MemberApiClient(BASE, "plan_a", "t", transport=httpx.MockTransport(evil), sleep=lambda s: None)
    with pytest.raises(UpstreamError) as exc:
        client.list_claims("M100001")
    assert exc.value.code == "bad_pagination_link"
