"""Scripted model for tests: plays back a fixed sequence of responses (or a callable).

It is a test double only. It lets the gateway, tools, confirmation flow and budgets be tested
deterministically, including with adversarial "model" behaviour (e.g. a model that tries to
write without confirmation). It is never used to produce evaluation scores.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import Message, ModelResponse, ToolCall, ToolSpec, Usage

Script = list[ModelResponse] | Callable[[str, list[Message], list[ToolSpec]], ModelResponse]


def say(text: str) -> ModelResponse:
    return ModelResponse(text=text, usage=Usage(100, 20))


def call(name: str, _id: str | None = None, **arguments) -> ModelResponse:
    return ModelResponse(text="", tool_calls=[ToolCall(_id or f"call_{name}", name, arguments)],
                         usage=Usage(100, 20), stop_reason="tool_use")


class FakeModel:
    name = "fake"
    model = "fake-model"

    def __init__(self, script: Script):
        self._script = script
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        self.calls.append((system, [dict(m) for m in messages], list(tools)))
        if callable(self._script):
            return self._script(system, messages, tools)
        if not self._script:
            return say("(script exhausted)")
        return self._script.pop(0)
