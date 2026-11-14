"""Regression gate: turns an evaluation summary into pass/fail checks for CI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Check:
    name: str
    value: float | int | None
    threshold: str
    passed: bool
    enforced: bool = True


def evaluate_gate(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[Check]:
    t = thresholds
    checks: list[Check] = []

    def at_least(name: str, value: float | None, minimum: float) -> None:
        checks.append(Check(name, value, f">= {minimum:.2f}", value is not None and value >= minimum))

    def at_most(name: str, value: float | None, maximum: float, enforced: bool = True) -> None:
        checks.append(Check(name, value, f"<= {maximum}", value is not None and value <= maximum,
                            enforced))

    at_most("safety violations (all cases)", summary["safety_violations"], t["safety_violations_max"])
    at_most("harness errors", summary["errors"], t["errors_max"])
    for behaviour in t["must_pass_behaviours"]:
        stats = summary["by_behaviour"].get(behaviour)
        if stats:
            at_least(f"pass rate: {behaviour}", stats["pass_rate"], 1.0)
    at_least("overall pass rate (adjudicated)", summary["pass_rate"], t["overall_pass_rate_min"])
    for tenant, stats in summary["by_tenant"].items():
        at_least(f"pass rate: tenant {tenant}", stats["pass_rate"], t["per_tenant_pass_rate_min"])
    at_most("average cost per conversation (USD)", summary["avg_cost_usd"], t["avg_cost_usd_max"])
    at_most("p95 platform latency per turn (s), excluding model time",
            summary["latency"]["platform_p95_s"], t["p95_platform_latency_s_max"])
    at_most("p95 end-to-end latency per turn (s)", summary["latency"]["turn_p95_s"],
            t["p95_turn_latency_s_max"], enforced=bool(t.get("enforce_turn_latency", False)))
    return checks


def gate_passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks if c.enforced)
