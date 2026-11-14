"""Evaluation harness.

    msagent eval                      # all cases, write reports/, apply the gate (exit 1 on failure)
    msagent eval --only EV-16,EV-32   # a subset (the gate is reported but not applied)
    msagent eval --no-gate

Runs each case in ``<pack>/eval/conversations.json`` through the real agent (real gateway, real
model provider, the mock upstream), observes state *outside* the agent - grievances actually on
file before and after, what was sent to the model, what was written to the trace - and scores it
deterministically.

The committed report (``reports/eval_report.md`` / ``eval_summary.json``) contains case ids,
verdicts and aggregate metrics only: no answers, no member data, no expected values. A detailed
per-case log for debugging is written to ``reports/local/``, which is git-ignored.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess  # noqa: S404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent import prompts
from ..agent.service import AgentService
from ..clients import MemberApiClient, NotFound
from ..config import REPO_ROOT, Settings
from ..gateway.gateway import member_identifiers
from ..session import Session
from .gate import evaluate_gate, gate_passed
from .report import render_markdown
from .scoring import CaseObservation, score_case

_REP_BLOCK = re.compile(r"<rep_message>.*?</rep_message>", re.S)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[k], 3)


def _model_context(svc: AgentService, session: Session) -> str:
    """Everything the model was shown, without the rep's own words (which may legitimately name
    another plan, as in the 'caller says they are with the other plan' cases)."""
    parts = [prompts.SYSTEM_TEMPLATE.format(tenant_name=session.ctx.tenant.name, as_of=session.as_of)]
    for msg in session.transcript:
        if msg["role"] == "user":
            parts.append(_REP_BLOCK.sub("", msg["content"]))
        elif msg["role"] == "assistant":
            parts.append(msg.get("text") or "")
        else:
            parts.extend(r["content"] for r in msg["results"])
    return "\n".join(parts)


def _derive_fact(kind: str, svc: AgentService, tenant: str, benefits: dict[str, Any]) -> str:
    """Expected values that depend on member data are derived at run time, never stored."""
    if kind == "deductible_remaining_current_plan_year":
        tkb = svc.kb.for_tenant(tenant)
        year = tkb.current_plan_year(benefits["product_code"], svc.as_of)
        design = tkb.plan_design(benefits["product_code"], year)
        met = benefits["accumulators"]["deductible_met_individual"]
        return f"${design.deductible_individual - met:,.2f}"
    raise ValueError(f"unknown derived fact '{kind}'")


def run_case(svc: AgentService, case: dict[str, Any], truth: MemberApiClient,
             adjudication: dict[str, Any] | None) -> dict[str, Any]:
    tenant, member_id = case["tenant"], case["member_id"]
    behaviour, facts = case["expected"]["behaviour"], list(case["expected"].get("facts", []))

    # Ground truth, read by the harness with the tenant's own credential (never via the agent).
    try:
        member = truth.get_member(member_id)
        benefits = truth.get_benefits(member_id)
        before = {g["grievance_id"] for g in truth.list_grievances(member_id)}
    except NotFound:
        member, benefits, before = None, None, set()

    def new_grievances() -> list[dict[str, Any]]:
        if member is None:
            return []
        return [g for g in truth.list_grievances(member_id) if g["grievance_id"] not in before]

    session = svc.start(tenant, member_id, "eval-rep")
    replies: list[dict[str, Any]] = []
    after_first, error = 0, None
    try:
        for i, turn in enumerate(case["turns"]):
            reply = svc.send(session, turn["text"])
            replies.append({"text": reply.text, "outcome": reply.outcome,
                            "pending_action": reply.pending_action, "usage": reply.usage})
            if i == 0:
                after_first = len(new_grievances())
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a failed run
        error = f"{type(exc).__name__}: {exc}"

    events = session.tracer.export()
    others = svc.registry.others(tenant)
    obs = CaseObservation(
        replies=replies,
        new_grievances_after_first_turn=after_first,
        new_grievances=new_grievances(),
        model_context=_model_context(svc, session),
        trace_text=json.dumps(events, ensure_ascii=False, default=str),
        tool_calls_ok=[e["name"] for e in events if e["kind"] == "tool_call" and e["status"] == "ok"],
        member_identifiers=member_identifiers(member) if member else [],
        other_tenant_markers=[m for t in others for m in (t.name, *t.network_ids)],
        error=error,
    )
    supplied = score_case(behaviour, facts, obs)
    adjudicated = supplied
    if adjudication:
        adj_facts = [(_derive_fact(f["derive"], svc, tenant, benefits) if isinstance(f, dict) else f)
                     for f in adjudication["facts"]]
        adjudicated = score_case(behaviour, adj_facts, obs)

    model_events = [e for e in events if e["kind"] == "model_call"]
    turn_latencies = [r["usage"].get("turn_latency_s", 0.0) for r in replies]
    model_s_by_turn: dict[int, float] = {}
    for e in model_events:
        model_s_by_turn[e["turn"]] = model_s_by_turn.get(e["turn"], 0.0) + (e["duration_ms"] or 0) / 1000
    platform = [round(max(lat - model_s_by_turn.get(i + 1, 0.0), 0.0), 3)
                for i, lat in enumerate(turn_latencies)]
    return {
        "id": case["id"], "tenant": tenant, "behaviour": behaviour, "turns": len(case["turns"]),
        "passed": adjudicated.passed, "passed_as_supplied": supplied.passed,
        "adjudicated": bool(adjudication),
        "failures": adjudicated.failures, "failures_as_supplied": supplied.failures,
        "safety_violations": adjudicated.safety_violations,
        "error": error,
        "outcomes": [r["outcome"] for r in replies],
        "tools": sorted({e["name"] for e in events if e["kind"] == "tool_call"}),
        "gateway_denials": sorted({f"{e['name']}:{e['attrs'].get('policy')}" for e in events
                                   if e["kind"] == "gateway_decision" and e["status"] == "denied"}),
        "guardrails": sorted({e["name"] for e in events if e["kind"] == "guardrail"}),
        "model_calls": len(model_events),
        "input_tokens": sum(e["attrs"].get("input_tokens", 0) for e in model_events),
        "output_tokens": sum(e["attrs"].get("output_tokens", 0) for e in model_events),
        "cost_usd": round(session.budget.cost_usd, 6),
        "turn_latency_s": turn_latencies, "platform_latency_s": platform,
        "_local": {"replies": [r["text"] for r in replies], "trace": events},  # stripped before commit
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(rows: list[dict[str, Any]], key: str = "passed") -> dict[str, Any]:
        n = len(rows)
        p = sum(1 for r in rows if r[key])
        return {"cases": n, "passed": p, "pass_rate": round(p / n, 4) if n else None}

    def group(field: str) -> dict[str, Any]:
        keys = sorted({r[field] for r in results})
        return {k: rate([r for r in results if r[field] == k]) for k in keys}

    turn = [x for r in results for x in r["turn_latency_s"]]
    platform = [x for r in results for x in r["platform_latency_s"]]
    costs = [r["cost_usd"] for r in results]
    return {
        **rate(results),
        "as_supplied": rate(results, "passed_as_supplied"),
        "safety_violations": sum(len(r["safety_violations"]) for r in results),
        "errors": sum(1 for r in results if r["error"]),
        "by_behaviour": group("behaviour"),
        "by_tenant": group("tenant"),
        "avg_cost_usd": round(statistics.fmean(costs), 6) if costs else None,
        "max_cost_usd": round(max(costs), 6) if costs else None,
        "avg_model_calls": round(statistics.fmean(r["model_calls"] for r in results), 2) if results else None,
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "latency": {"turn_p50_s": _percentile(turn, 50), "turn_p95_s": _percentile(turn, 95),
                    "platform_p50_s": _percentile(platform, 50),
                    "platform_p95_s": _percentile(platform, 95)},
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, check=False,  # noqa: S603,S607
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="msagent eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="path to conversations.json (default: <pack>/eval/conversations.json)")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports"))
    ap.add_argument("--thresholds", default=str(REPO_ROOT / "evaluation" / "thresholds.json"))
    ap.add_argument("--adjudications", default=str(REPO_ROOT / "evaluation" / "adjudications.json"))
    ap.add_argument("--no-gate", action="store_true", help="report only; always exit 0")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    svc = AgentService.build(settings)
    cases_path = Path(args.cases) if args.cases else settings.pack_dir / "eval" / "conversations.json"
    pack = json.loads(cases_path.read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    adjudications = json.loads(Path(args.adjudications).read_text(encoding="utf-8"))["cases"]
    cases = pack["cases"]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if pack.get("as_of") and pack["as_of"] != svc.as_of:
        print(f"WARNING: eval as_of {pack['as_of']} differs from environment clock {svc.as_of}",
              file=sys.stderr)

    truth = {tid: MemberApiClient(settings.api_base_url, tid, svc.registry.api_token(tid))
             for tid in svc.registry.ids()}
    started = time.time()
    results: list[dict[str, Any]] = []
    for case in cases:
        res = run_case(svc, case, truth[case["tenant"]], adjudications.get(case["id"]))
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        why = "; ".join(res["safety_violations"] + res["failures"])
        print(f"{res['id']}  {mark}  {res['behaviour']:<19} {sum(res['turn_latency_s']):6.1f}s  {why}",
              flush=True)

    summary = summarize(results)
    checks = evaluate_gate(summary, thresholds)
    full_run = not args.only
    passed = gate_passed(checks)
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "provider": svc.model.name, "model": svc.model.model,
        "as_of": svc.as_of, "eval_file_version": pack.get("version"),
        "cases_run": len(results), "full_run": full_run,
        "duration_s": round(time.time() - started, 1),
        "kb_quarantined": [q.__dict__ for q in svc.kb.quarantined],
    }
    out = Path(args.out)
    (out / "local").mkdir(parents=True, exist_ok=True)
    (out / "local" / "eval_details.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=1, default=str), encoding="utf-8")
    public = [{k: v for k, v in r.items() if k != "_local"} for r in results]
    document = {"meta": meta, "summary": summary, "gate": {"passed": passed, "applied": full_run,
                "checks": [c.__dict__ for c in checks]}, "thresholds": thresholds, "cases": public}
    suffix = "" if full_run else "_partial"
    (out / f"eval_summary{suffix}.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    (out / f"eval_report{suffix}.md").write_text(render_markdown(document, adjudications), encoding="utf-8")

    print(f"\n{summary['passed']}/{summary['cases']} passed (as supplied: "
          f"{summary['as_supplied']['passed']}/{summary['cases']}), "
          f"{summary['safety_violations']} safety violation(s)")
    for c in checks:
        flag = "ok  " if c.passed else ("FAIL" if c.enforced else "warn")
        print(f"  [{flag}] {c.name}: {c.value} (required {c.threshold})")
    print(f"GATE: {'PASSED' if passed else 'FAILED'}" + ("" if full_run else " (subset run - not applied)"))
    print(f"report: {out / f'eval_report{suffix}.md'}")
    if args.no_gate or not full_run:
        return 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
