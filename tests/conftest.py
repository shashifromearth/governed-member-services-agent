from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import AS_OF, DIRECTORY_KEY, TOKENS, FakeUpstream, write_kb, write_tenants  # noqa: E402
from msagent.agent.service import AgentService  # noqa: E402
from msagent.clients import DirectoryClient, MemberApiClient  # noqa: E402
from msagent.config import Budgets, LLMSettings, Settings  # noqa: E402
from msagent.gateway.gateway import ToolGateway  # noqa: E402
from msagent.kb import KnowledgeBase  # noqa: E402
from msagent.llm.fake import FakeModel  # noqa: E402
from msagent.tenancy import TenantRegistry  # noqa: E402
from msagent.tools import build_tools  # noqa: E402
from msagent.tracing import AuditLog, JsonlTraceSink  # noqa: E402

BASE = "http://upstream.test"


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    write_kb(tmp_path)
    write_tenants(tmp_path / "tenants.json")
    monkeypatch.setenv("MSA_TOKEN_PLAN_A", TOKENS["plan_a"])
    monkeypatch.setenv("MSA_TOKEN_PLAN_B", TOKENS["plan_b"])
    monkeypatch.setenv("MSA_DIRECTORY_API_KEY", DIRECTORY_KEY)
    return Settings(
        pack_dir=tmp_path, api_base_url=BASE, directory_base_url=BASE + "/directory/v3",
        tenants_file=tmp_path / "tenants.json", var_dir=tmp_path / "var", as_of=AS_OF,
        budgets=Budgets(), llm=LLMSettings(provider="fake", model="fake-model"),
        rep_keys={"key-a": {"tenant": "plan_a", "rep_id": "rep-a"},
                  "key-b": {"tenant": "plan_b", "rep_id": "rep-b"}},
        log_hmac_key="test-key",
    )


@pytest.fixture
def make_service(settings, upstream):
    """Build the real service (real gateway, tools, KB, tracing) over the fake upstream."""

    def _make(script, budgets: Budgets | None = None) -> AgentService:
        cfg = settings if budgets is None else Settings(**{**settings.__dict__, "budgets": budgets})
        registry = TenantRegistry.load(cfg)
        kb = KnowledgeBase.load(cfg.pack_dir, registry)
        audit = AuditLog(cfg.var_dir / "audit")
        transport = httpx.MockTransport(upstream.handle)
        gateway = ToolGateway(
            cfg, registry, kb, build_tools(), audit,
            api_factory=lambda tid: MemberApiClient(BASE, tid, registry.api_token(tid),
                                                    transport=transport, sleep=lambda s: None),
            directory_factory=lambda: DirectoryClient(BASE + "/directory/v3",
                                                      registry.directory_api_key(), transport=transport),
        )
        model = script if isinstance(script, FakeModel) else FakeModel(script)
        return AgentService(cfg, registry, kb, gateway, model, audit, as_of=AS_OF,
                            sinks=[JsonlTraceSink(cfg.var_dir / "traces")])

    return _make


def tool_results(session) -> list[dict]:
    """All tool results the model was shown in a conversation."""
    return [r for m in session.transcript if m["role"] == "tool" for r in m["results"]]
