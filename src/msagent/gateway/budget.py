"""Per-conversation budgets: steps, cost and latency.

The tracker is consulted by the gateway before every tool call and by the agent loop before
every model call. When a limit is hit the conversation degrades to a safe, explicit reply - it
never silently continues, and it never leaves a write half-done (writes are not subject to the
turn deadline once the rep has confirmed; see ``ToolGateway.confirm``).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import Budgets


@dataclass(frozen=True)
class BudgetDenial:
    limit: str  # steps | cost | latency | model_calls
    detail: str


class BudgetTracker:
    def __init__(self, limits: Budgets, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self._clock = clock
        self.steps = 0
        self.cost_usd = 0.0
        self.model_calls_turn = 0
        self._turn_started = clock()

    def start_turn(self) -> None:
        self.model_calls_turn = 0
        self._turn_started = self._clock()

    def elapsed_s(self) -> float:
        return self._clock() - self._turn_started

    def check(self, kind: str) -> BudgetDenial | None:
        """``kind`` is 'model' or 'tool'. Returns a denial, or None when the step may run."""
        lim = self.limits
        if self.steps >= lim.max_steps:
            return BudgetDenial("steps", f"{self.steps}/{lim.max_steps} steps used")
        if self.cost_usd >= lim.max_cost_usd:
            return BudgetDenial("cost", f"${self.cost_usd:.4f} of ${lim.max_cost_usd:.2f} spent")
        if self.elapsed_s() >= lim.turn_deadline_s:
            return BudgetDenial("latency", f"{self.elapsed_s():.1f}s of {lim.turn_deadline_s:.0f}s used")
        if kind == "model" and self.model_calls_turn >= lim.max_model_calls_per_turn:
            return BudgetDenial("model_calls", f"{self.model_calls_turn} model calls this turn")
        return None

    def charge_tool(self) -> None:
        self.steps += 1

    def charge_model(self, cost_usd: float) -> None:
        self.steps += 1
        self.model_calls_turn += 1
        self.cost_usd += cost_usd

    def snapshot(self) -> dict[str, float | int]:
        return {"steps": self.steps, "cost_usd": round(self.cost_usd, 6),
                "turn_elapsed_s": round(self.elapsed_s(), 3)}
