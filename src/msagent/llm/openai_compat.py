"""Adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

Covers OpenAI, Azure OpenAI, Ollama (``http://localhost:11434/v1``), vLLM, LM Studio and
similar gateways, so an open-weights model can be used with no code change.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx

from ..config import LLMSettings
from .base import Message, ModelError, ModelResponse, ToolCall, ToolSpec, Usage

_TOOL_TAG = re.compile(r"^\s*(?:<tool_call>|```(?:json)?)?\s*(\{.*\})\s*(?:</tool_call>|```)?\s*$", re.S)


def _inline_tool_call(text: str, known: set[str]) -> ToolCall | None:
    """Some open-weights models print a tool call as JSON text instead of using the tool-call
    channel. If the whole reply is exactly such an object naming a known tool, treat it as one."""
    match = _TOOL_TAG.match(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("name") not in known:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=obj["name"], arguments=args or {})


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, settings: LLMSettings, transport: httpx.BaseTransport | None = None):
        self.model = settings.model
        self._max_tokens = settings.max_output_tokens
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._http = httpx.Client(
            base_url=(settings.base_url or "https://api.openai.com/v1").rstrip("/"),
            headers=headers, timeout=settings.timeout_s, transport=transport,
        )

    @staticmethod
    def _to_wire(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            if role == "user":
                wire.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                out: dict[str, Any] = {"role": "assistant", "content": msg.get("text") or ""}
                if msg.get("tool_calls"):
                    out["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in msg["tool_calls"]
                    ]
                wire.append(out)
            elif role == "tool":
                for r in msg["results"]:
                    wire.append({"role": "tool", "tool_call_id": r["tool_call_id"],
                                 "content": r["content"]})
        return wire

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        if tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ModelError(f"model endpoint unreachable: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise ModelError(f"model endpoint returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else dict(args)
            except ValueError:
                parsed = {"_unparseable_arguments": str(args)[:200]}
            calls.append(ToolCall(id=raw.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                  name=fn.get("name", ""), arguments=parsed))
        text = (message.get("content") or "").strip()
        if not calls and tools:
            inline = _inline_tool_call(text, {t.name for t in tools})
            if inline is not None:
                calls, text = [inline], ""
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0)),
            stop_reason=choice.get("finish_reason") or "stop",
        )
