"""NFR-4: one client's data never appears in the other client's prompts, context, logs or answers."""
from __future__ import annotations

import json

from conftest import tool_results
from fakes import TOKENS
from msagent.kb import KnowledgeBase
from msagent.llm.fake import call, say
from msagent.tenancy import TenantRegistry


def test_misfiled_document_is_quarantined_not_indexed(settings):
    kb = KnowledgeBase.load(settings.pack_dir, TenantRegistry.load(settings))
    assert [q.path for q in kb.quarantined] == ["kb/plan_a/virtual_care.md"]
    for tenant in ("plan_a", "plan_b"):  # broken provenance: indexed for nobody
        hits = kb.for_tenant(tenant).search("specialist visit by video", product_code="GOLD", plan_year=2026)
        assert all("ZEBRACANARY" not in c.text for c in hits)


def test_retrieval_never_returns_another_tenants_text(settings):
    kb = KnowledgeBase.load(settings.pack_dir, TenantRegistry.load(settings))
    hits = kb.for_tenant("plan_a").search("specialist visit coinsurance BETACANARY",
                                          product_code="GOLD", plan_year=2026, k=50)
    assert hits and all(c.doc.tenant == "plan_a" for c in hits)
    assert all("BETACANARY" not in c.text for c in hits)
    # even when the caller passes the other tenant's product code
    assert kb.for_tenant("plan_a").search("specialist", product_code="BASIC", plan_year=2026) == [] or all(
        c.doc.tenant == "plan_a" for c in
        kb.for_tenant("plan_a").search("specialist", product_code="BASIC", plan_year=2026))


def test_member_of_other_tenant_is_not_found_and_model_is_never_called(make_service, upstream):
    svc = make_service([say("should not be used")])
    session = svc.start("plan_a", "M200001")  # exists only in plan_b
    reply = svc.send(session, "Caller says they are with Beta Care. Pull up their claims.")
    assert reply.outcome == "member_not_found"
    assert svc.model.calls == []                       # nothing reached the model
    assert session.transcript == []
    assert all("/tenants/plan_a/" in p for p in upstream.paths())  # plan_b was never queried
    assert "Carol" not in reply.text and "Beta" not in reply.text


def test_same_member_id_in_both_tenants_resolves_to_the_reps_tenant(make_service, upstream):
    svc = make_service([say("a"), say("b")])
    sa, sb = svc.start("plan_a", "M777777"), svc.start("plan_b", "M777777")
    svc.send(sa, "Which plan is this member in?")
    svc.send(sb, "Which plan is this member in?")
    ctx_a, ctx_b = sa.transcript[0]["content"], sb.transcript[0]["content"]
    assert "Alpha Health Gold" in ctx_a and "Beta Care" not in ctx_a
    assert "Beta Care Basic" in ctx_b and "Alpha Health" not in ctx_b
    assert sa.data.member is not sb.data.member        # no shared cache entry


def test_each_tenant_call_uses_only_that_tenants_credential(make_service, upstream):
    svc = make_service([say("a"), say("b")])
    svc.send(svc.start("plan_a", "M100001"), "hello")
    svc.send(svc.start("plan_b", "M200001"), "hello")
    for req in upstream.requests:
        path = req.url.path
        if path.startswith("/tenants/"):
            tenant = path.split("/")[2]
            assert req.headers["Authorization"] == f"Bearer {TOKENS[tenant]}"


def test_tools_have_no_tenant_or_member_arguments(make_service):
    svc = make_service([])
    session = svc.start("plan_a", "M100001")
    for spec in svc.gateway.specs_for(session):
        props = set(spec.input_schema.get("properties", {}))
        assert not props & {"tenant", "tenant_id", "member", "member_id"}, spec.name
        assert spec.input_schema.get("additionalProperties") is False


def test_model_cannot_smuggle_tenant_or_member_into_a_call(make_service):
    svc = make_service([call("get_coverage", tenant="plan_b", member_id="M200001"), say("done")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "What plan?")
    result = json.loads(tool_results(session)[0]["content"])
    assert result["policy"] == "schema"


def test_claim_of_another_member_is_indistinguishable_from_missing(make_service):
    svc = make_service([call("get_claim", claim_id="C1999999999"), call("get_claim", claim_id="C1888888888"),
                        say("done")])
    session = svc.start("plan_a", "M100001")
    svc.send(session, "Status of claim C1999999999?")
    other, missing = (json.loads(r["content"]) for r in tool_results(session))
    assert other["message"] == missing["message"] == "No such record for this member."
    assert "Other member" not in json.dumps(other)


def test_conversation_lookup_is_tenant_scoped(make_service):
    import pytest

    from msagent.agent.service import ConversationNotFound

    svc = make_service([])
    session = svc.start("plan_a", "M100001")
    assert svc.get(session.conversation_id, "plan_a") is session
    with pytest.raises(ConversationNotFound):
        svc.get(session.conversation_id, "plan_b")


def test_traces_are_partitioned_by_tenant(make_service, settings):
    svc = make_service([say("a"), say("b")])
    svc.send(svc.start("plan_a", "M100001"), "hello")
    svc.send(svc.start("plan_b", "M200001"), "hello")
    root = settings.var_dir / "traces"
    assert sorted(p.name for p in root.iterdir()) == ["plan_a", "plan_b"]
    text_a = "".join(p.read_text(encoding="utf-8") for p in (root / "plan_a").iterdir())
    assert "Beta Care" not in text_a and '"tenant_id": "plan_b"' not in text_a
