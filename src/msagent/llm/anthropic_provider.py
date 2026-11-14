"""Anthropic Messages API adapter (official SDK)."""
from __future__ import annotations

import json
from typing import Any

import anthropic

from ..config import LLMSettings
from .base import Message, ModelError, ModelResponse, ToolCall, ToolSpec, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: LLMSettings):
        self.model = settings.model
        self._max_tokens = settings.max_output_tokens
        kwargs: dict[str, Any] = {"timeout": settings.timeout_s, "max_retries": 2}
        if settings.api_key:
            kwargs["api_key"] = settings.api_key
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = anthropic.Anthropic(**kwargs)

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                wire.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                if msg.get("native") is not None:
                    # Replay the provider's own blocks unchanged (keeps thinking blocks valid).
                    wire.append({"role": "assistant", "content": msg["native"]})
                else:
                    blocks: list[dict[str, Any]] = []
                    if msg.get("text"):
                        blocks.append({"type": "text", "text": msg["text"]})
                    for call in msg.get("tool_calls", []):
                        blocks.append({"type": "tool_use", "id": call.id, "name": call.name,
                                       "input": call.arguments})
                    wire.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                # All results of one assistant turn go back in a single user message.
                wire.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["tool_call_id"], "content": r["content"],
                     "is_error": bool(r.get("is_error"))}
                    for r in msg["results"]
                ]})
        return wire

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                # Static per-tenant prefix (tools + system) is cacheable across conversations.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                       for t in tools],
                messages=self._to_wire(messages),
            )
        except anthropic.RateLimitError as exc:
            raise ModelError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ModelError(f"anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ModelError(f"anthropic connection error: {exc}") from exc

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input if isinstance(b.input, dict)
                     else json.loads(b.input or "{}"))
            for b in resp.content if b.type == "tool_use"
        ]
        usage = resp.usage
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return ModelResponse(
            text=text.strip(),
            tool_calls=calls,
            usage=Usage(input_tokens=usage.input_tokens + cached + created,
                        output_tokens=usage.output_tokens, cached_input_tokens=cached),
            stop_reason=resp.stop_reason or "end_turn",
            native=resp.content,
        )
