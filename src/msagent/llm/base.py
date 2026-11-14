"""Provider-neutral model interface.

The agent loop only knows these types. A provider adapter translates them to its wire format
and back, so swapping Anthropic for any OpenAI-compatible endpoint (OpenAI, Azure OpenAI,
Ollama, vLLM) is configuration, not code.

Conversation messages are plain dicts:

* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "text": str, "tool_calls": [ToolCall], "native": Any}``
  (``native`` is the provider's own representation, replayed verbatim to the same provider -
  needed e.g. for Anthropic thinking blocks)
* ``{"role": "tool", "results": [{"tool_call_id", "name", "content": str, "is_error": bool}]}``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    native: Any = None

    def as_message(self) -> Message:
        return {"role": "assistant", "text": self.text, "tool_calls": list(self.tool_calls),
                "native": self.native}


class ModelError(Exception):
    """The provider could not produce a response (after its own retries)."""


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse: ...


# USD per million tokens (input, output). Unknown / self-hosted models cost 0 unless overridden
# with MSA_LLM_PRICE_IN_PER_MTOK / MSA_LLM_PRICE_OUT_PER_MTOK.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
}


def cost_usd(model: str, usage: Usage, price_in: float | None = None,
             price_out: float | None = None) -> float:
    default_in, default_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    p_in = default_in if price_in is None else price_in
    p_out = default_out if price_out is None else price_out
    return (usage.input_tokens * p_in + usage.output_tokens * p_out) / 1_000_000
