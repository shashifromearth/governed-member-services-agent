"""Provider adapters translate the neutral message format faithfully."""
from __future__ import annotations

import json

import httpx
import pytest

from msagent.config import LLMSettings, Settings
from msagent.llm import build_provider
from msagent.llm.anthropic_provider import AnthropicProvider
from msagent.llm.base import ModelError, ToolCall, ToolSpec
from msagent.llm.openai_compat import OpenAICompatProvider

TOOLS = [ToolSpec("get_claim", "One claim", {"type": "object", "properties": {"claim_id": {"type": "string"}},
                                           "required": ["claim_id"]})]
HISTORY = [
    {"role": "user", "content": "status of C1?"},
    {"role": "assistant", "text": "", "tool_calls": [ToolCall("c1", "get_claim", {"claim_id": "C1"})],
     "native": None},
    {"role": "tool", "results": [{"tool_call_id": "c1", "name": "get_claim", "content": '{"status":"PAID"}',
                                  "is_error": False}]},
]


def _openai(handler) -> OpenAICompatProvider:
    return OpenAICompatProvider(LLMSettings(provider="openai", model="m", base_url="http://llm.test/v1"),
                                transport=httpx.MockTransport(handler))


def test_openai_compat_round_trip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                {"id": "x1", "type": "function",
                 "function": {"name": "get_claim", "arguments": '{"claim_id": "C2"}'}}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7}})

    resp = _openai(handler).complete("SYSTEM", HISTORY, TOOLS)
    assert resp.tool_calls == [ToolCall("x1", "get_claim", {"claim_id": "C2"})]
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (50, 7)
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert seen["messages"][2]["tool_calls"][0]["function"]["name"] == "get_claim"
    assert seen["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"status":"PAID"}'}
    assert seen["tools"][0]["function"]["parameters"]["required"] == ["claim_id"]


def test_openai_compat_accepts_a_tool_call_printed_as_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '<tool_call>\n{"name": "get_claim", "arguments": {"claim_id": "C9"}}\n</tool_call>'}}]})

    resp = _openai(handler).complete("S", HISTORY[:1], TOOLS)
    assert resp.text == "" and resp.tool_calls[0].name == "get_claim"
    assert resp.tool_calls[0].arguments == {"claim_id": "C9"}


def test_openai_compat_leaves_ordinary_json_text_alone():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"name": "unknown_tool"}'}}]})

    resp = _openai(handler).complete("S", HISTORY[:1], TOOLS)
    assert resp.tool_calls == [] and resp.text == '{"name": "unknown_tool"}'


def test_openai_compat_errors_become_model_errors():
    with pytest.raises(ModelError):
        _openai(lambda r: httpx.Response(500, text="boom")).complete("S", HISTORY[:1], TOOLS)

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ModelError):
        _openai(down).complete("S", HISTORY[:1], TOOLS)


def test_anthropic_wire_format_groups_tool_results_and_replays_native_blocks():
    wire = AnthropicProvider._to_wire(HISTORY)
    assert wire[1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "c1", "name": "get_claim", "input": {"claim_id": "C1"}}]}
    assert wire[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": '{"status":"PAID"}', "is_error": False}]}
    native = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "hi"}]
    replay = AnthropicProvider._to_wire([{"role": "assistant", "text": "hi", "tool_calls": [],
                                          "native": native}])
    assert replay[0]["content"] is native                # passed back unchanged


def test_provider_selection_is_configuration(monkeypatch, tmp_path):
    for var in ("MSA_LLM_MODEL", "MSA_LLM_BASE_URL", "MSA_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MSA_LLM_PROVIDER", "ollama")
    settings = Settings.from_env(env_file=tmp_path / "none.env")
    assert (settings.llm.provider, settings.llm.base_url) == ("openai", "http://localhost:11434/v1")
    assert isinstance(build_provider(settings.llm), OpenAICompatProvider)
    monkeypatch.setenv("MSA_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("MSA_LLM_API_KEY", "test-key")
    settings = Settings.from_env(env_file=tmp_path / "none.env")
    assert settings.llm.model == "claude-haiku-5-5"
    assert isinstance(build_provider(settings.llm), AnthropicProvider)
    with pytest.raises(ValueError):
        build_provider(LLMSettings(provider="nope"))


@pytest.mark.parametrize("env,expected", [
    # no key anywhere: local model, no account needed
    ({}, ("openai", "qwen2.5:3b", "http://localhost:11434/v1", None, True)),
    ({"MSA_LOCAL_MODEL": "llama3.1:8b", "MSA_OLLAMA_URL": "http://ollama:11434/v1"},
     ("openai", "llama3.1:8b", "http://ollama:11434/v1", None, True)),
    # a hosted provider is used only when its key was supplied
    ({"ANTHROPIC_API_KEY": "k-a"}, ("anthropic", "claude-haiku-5-5", None, "k-a", False)),
    ({"OPENAI_API_KEY": "k-o"}, ("openai", "gpt-4o-mini", None, "k-o", False)),
    ({"ANTHROPIC_API_KEY": "k-a", "OPENAI_API_KEY": "k-o"}, ("anthropic", "claude-haiku-5-5", None, "k-a", False)),
    # an empty key is not a key
    ({"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}, ("openai", "qwen2.5:3b", "http://localhost:11434/v1", None, True)),
    # explicit choice wins over detection
    ({"MSA_LLM_PROVIDER": "ollama", "ANTHROPIC_API_KEY": "k-a"},
     ("openai", "qwen2.5:3b", "http://localhost:11434/v1", None, True)),
    ({"MSA_LLM_PROVIDER": "auto", "MSA_LLM_MODEL": "custom"}, ("openai", "custom", "http://localhost:11434/v1", None, True)),
])
def test_local_model_is_the_default_and_hosted_needs_a_key(env, expected):
    from msagent.config import resolve_llm

    assert resolve_llm(env) == expected


def test_unknown_provider_is_rejected():
    from msagent.config import resolve_llm

    with pytest.raises(ValueError):
        resolve_llm({"MSA_LLM_PROVIDER": "nope"})


def test_local_profile_gets_generous_time_limits(monkeypatch, tmp_path):
    for var in ("MSA_LLM_PROVIDER", "MSA_LLM_MODEL", "MSA_LLM_BASE_URL", "MSA_LLM_API_KEY", "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY", "MSA_LLM_TIMEOUT_S", "MSA_BUDGET_TURN_DEADLINE_S", "MSA_LOCAL_MODEL"):
        monkeypatch.delenv(var, raising=False)
    local = Settings.from_env(env_file=tmp_path / "none.env")
    assert (local.llm.timeout_s, local.budgets.turn_deadline_s) == (1200.0, 3600.0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    hosted = Settings.from_env(env_file=tmp_path / "none.env")
    assert (hosted.llm.provider, hosted.llm.timeout_s, hosted.budgets.turn_deadline_s) == ("anthropic", 120.0, 30.0)
