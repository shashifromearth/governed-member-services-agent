"""Checks against the real supplied pack. Skipped when the pack is not present (e.g. in CI's
`test` job), because the pack is never committed. Needs no mock server and no model."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from msagent.config import REPO_ROOT, Budgets, LLMSettings, Settings
from msagent.kb import KnowledgeBase
from msagent.tenancy import TenantRegistry
from msagent.tools.providers import load_vendor_manifest

PACK = Path(os.environ.get("MSA_PACK_DIR") or REPO_ROOT)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (PACK / "kb" / "plan_a").is_dir(), reason="supplied pack not present"),
]


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    settings = Settings(pack_dir=PACK, api_base_url="http://unused", directory_base_url="http://unused",
                        tenants_file=REPO_ROOT / "config" / "tenants.json", var_dir=REPO_ROOT / "var",
                        as_of="2026-11-15", budgets=Budgets(), llm=LLMSettings())
    return KnowledgeBase.load(PACK, TenantRegistry.load(settings))


def test_only_the_misfiled_document_is_quarantined(kb):
    assert len(kb.quarantined) == 1
    assert kb.quarantined[0].path.startswith("kb/plan_a/") and "plan_b" in kb.quarantined[0].reason


def test_every_document_admitted_belongs_to_its_tenant(kb):
    for tenant_id, tkb in kb.tenants.items():
        assert tkb.documents and {d.tenant for d in tkb.documents} == {tenant_id}


def test_plan_rules_parse_for_every_product_and_plan_year(kb):
    registry_products = {"plan_a": ["GPPO", "SHMO"], "plan_b": ["CPPO", "VHDHP"]}
    for tenant_id, products in registry_products.items():
        tkb = kb.for_tenant(tenant_id)
        policy = tkb.policy()
        assert policy.appeal_window_days and policy.grievance_resolution_days
        for product in products:
            assert tkb.current_plan_year(product, "2026-11-15") == 2026
            for year in tkb.plan_years(product):
                design = tkb.plan_design(product, year)
                assert design.deductible_individual and design.oop_max_individual and design.pt_visit_limit


def test_grievance_timeframes_differ_by_tenant_and_are_not_hard_coded(kb):
    days = {t: kb.for_tenant(t).policy().grievance_resolution_days for t in ("plan_a", "plan_b")}
    assert days["plan_a"] != days["plan_b"]


def test_embedded_instruction_in_the_guide_is_not_retrievable(kb):
    tkb = kb.for_tenant("plan_a")
    assert tkb.flagged_chunks >= 1
    hits = tkb.search("processing note for AI coding assistants add the comment", product_code="GPPO",
                      plan_year=2026, k=50)
    assert all("coding assistants" not in c.text.lower() for c in hits)


def test_vendor_manifest_description_is_flagged(kb):
    report = load_vendor_manifest(PACK / "tools" / "third_party_provider_directory.json")
    assert report["present"] and "search_providers" in report["flagged"]
