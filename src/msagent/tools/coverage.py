"""Coverage, accumulators, identity check and plan-document search."""
from __future__ import annotations

import hmac
from typing import Any

from pydantic import Field

from ..gateway.gateway import Tool, ToolError, ToolKind, ToolRuntime
from .common import Args, NoArgs, usd

MAX_SSN_CHECKS = 3


def coverage_view(rt: ToolRuntime) -> dict[str, Any]:
    """Minimum-necessary coverage summary: no name, DOB, SSN, contact details or address."""
    member, benefits = rt.member(), rt.benefits()
    product = benefits["product_code"]
    acc = benefits.get("accumulators") or {}
    current_year = rt.kb.current_plan_year(product, rt.as_of)
    year = acc.get("plan_year") or current_year
    design = rt.kb.plan_design(product, year) if year else None

    def bucket(limit: float | None, met: float | None) -> dict[str, Any]:
        remaining = None if limit is None or met is None else max(round(limit - met, 2), 0.0)
        return {"plan_amount": usd(limit), "met_so_far": usd(met), "remaining": usd(remaining)}

    pt_limit = design.pt_visit_limit if design else None
    pt_used = acc.get("physical_therapy_visits_used")
    view: dict[str, Any] = {
        "plan_name": benefits["plan_name"],
        "product_code": product,
        "network_id": benefits["network_id"],
        "relationship_to_subscriber": member.get("relationship"),
        "coverage_status": benefits["coverage_status"],
        "coverage_effective": benefits["coverage_effective"],
        "coverage_termination": benefits.get("coverage_termination"),
        "today": rt.as_of,
        "current_plan_year": current_year,
        "available_plan_years": rt.kb.plan_years(product),
        "accumulators": {
            "plan_year": year,
            "as_of": acc.get("as_of"),
            "basis": "in-network, individual",
            "deductible": bucket(design.deductible_individual if design else None,
                                 acc.get("deductible_met_individual")),
            "out_of_pocket_maximum": bucket(design.oop_max_individual if design else None,
                                            acc.get("out_of_pocket_met_individual")),
            "physical_therapy_visits": {
                "plan_limit": pt_limit,
                "used": pt_used,
                "remaining": None if pt_limit is None or pt_used is None else max(pt_limit - pt_used, 0),
            },
            "plan_amounts_source": design.source if design else None,
        },
    }
    if benefits["coverage_status"] != "ACTIVE":
        view["coverage_note"] = (
            f"Coverage is NOT active: it ended on {benefits.get('coverage_termination')}. Benefits and "
            "cost-sharing no longer apply to services after that date."
        )
    return view


def _get_coverage(rt: ToolRuntime, _: NoArgs) -> dict[str, Any]:
    return coverage_view(rt)


class SearchPlanDocsArgs(Args):
    query: str = Field(min_length=2, max_length=300,
                       description="What to look up, e.g. 'specialist visit copay'.")
    plan_year: int | None = Field(
        default=None,
        description="Plan year to search. Omit for the current plan year. Use the next year only "
                    "when the question is about costs after renewal.")


def _search_plan_documents(rt: ToolRuntime, args: SearchPlanDocsArgs) -> dict[str, Any]:
    product = rt.benefits()["product_code"]
    current = rt.kb.current_plan_year(product, rt.as_of)
    year = args.plan_year or current
    if year is None or (year not in rt.kb.plan_years(product) and year != current):
        raise ToolError(f"No plan documents for plan year {year}. Available: {rt.kb.plan_years(product)}.")
    chunks = rt.kb.search(args.query, product_code=product, plan_year=year)
    return {
        "plan_year_searched": year,
        "is_current_plan_year": year == current,
        "passages": [c.as_context() for c in chunks],
        "note": "Passages are the member's own plan documents. If none answers the question, say the "
                "information is not available.",
    }


class VerifySsnArgs(Args):
    last4: str = Field(pattern=r"^\d{4}$", description="The four digits the caller gave.")


def _verify_ssn_last4(rt: ToolRuntime, args: VerifySsnArgs) -> dict[str, Any]:
    """Match / no-match only. The SSN itself never leaves the gateway, and attempts are limited
    so the tool cannot be used as an oracle to enumerate the digits."""
    session = rt.session
    if session.ssn_checks >= MAX_SSN_CHECKS:
        raise ToolError("Identity check limit reached for this conversation. Escalate per the "
                        "member services guide.")
    session.ssn_checks += 1
    on_file = (rt.member().get("ssn") or "")[-4:]
    return {"match": bool(on_file) and hmac.compare_digest(on_file, args.last4),
            "checks_remaining": MAX_SSN_CHECKS - session.ssn_checks}


TOOLS = [
    Tool(
        name="get_coverage",
        description="The member's plan, coverage status and dates, and this plan year's accumulators: "
                    "deductible, out-of-pocket maximum and physical-therapy visits, each with the plan "
                    "amount, the amount used and the amount REMAINING (already calculated).",
        args=NoArgs, kind=ToolKind.READ, handler=_get_coverage,
    ),
    Tool(
        name="search_plan_documents",
        description="Search the member's own plan documents (benefit summary for their product, member "
                    "services guide, claim/denial code reference) for copays, coinsurance, limits and "
                    "plan rules.",
        args=SearchPlanDocsArgs, kind=ToolKind.READ, handler=_search_plan_documents,
    ),
    Tool(
        name="verify_ssn_last4",
        description="Check whether the last four SSN digits given by the caller match the member on "
                    "file. Returns match / no match only.",
        args=VerifySsnArgs, kind=ToolKind.READ, handler=_verify_ssn_last4,
    ),
]
