"""Grievance tools: read, draft (model-callable) and file (gateway-only write)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..clients import WriteUnconfirmed
from ..gateway.actions import ActionStatus, PendingAction
from ..gateway.gateway import Tool, ToolError, ToolKind, ToolRuntime
from ..phi import find_phi, project
from ..untrusted import sanitize
from .common import Args, add_days

_GRIEVANCE_ALLOW = {"grievance_id": True, "category": True, "status": True, "received_date": True,
                    "resolved_date": True, "expedite_requested": True, "channel": True}
Category = Literal["billing", "quality_of_care", "access", "customer_service", "other"]


def _view(rt: ToolRuntime, g: dict[str, Any]) -> dict[str, Any]:
    out = project(g, _GRIEVANCE_ALLOW)
    out["description"], _ = sanitize(str(g.get("description", "")))
    days = rt.kb.policy().grievance_resolution_days
    if days and g.get("received_date"):
        deadline = add_days(g["received_date"], days)
        resolved = g.get("status") == "RESOLVED"
        past = (not resolved) and rt.as_of > deadline
        out["resolution"] = {
            "answer": (f"Yes - it is past the plan's resolution deadline, which was {deadline}." if past
                       else f"No - it is not past the plan's resolution deadline ({deadline})."),
            "rule": f"{days} calendar days from the date received, per {rt.kb.policy().source}",
            "resolution_deadline": deadline,
            "today": rt.as_of,
            "past_resolution_deadline": past,
        }
    return out


class GetGrievancesArgs(Args):
    grievance_id: str | None = Field(default=None, pattern=r"^G\d{2}-\d{4,8}$",
                                     description="Optional: only this grievance.")


def _get_grievances(rt: ToolRuntime, args: GetGrievancesArgs) -> dict[str, Any]:
    rt.member()
    items = [g for g in rt.api.list_grievances(rt.ctx.member_id)
             if g.get("member_id") == rt.ctx.member_id]
    if args.grievance_id:
        items = [g for g in items if g.get("grievance_id") == args.grievance_id]
        if not items:
            raise ToolError("No such record for this member.")
    return {"count": len(items), "grievances": [_view(rt, g) for g in items]}


class DraftGrievanceArgs(Args):
    category: Category
    description: str = Field(
        min_length=15, max_length=2000,
        description="Neutral summary of the member's complaint in the member's terms: what happened, "
                    "the service date and claim id if known. No SSN, date of birth, phone or address.")
    expedite_requested: bool = Field(
        default=False,
        description="True only if the representative explicitly asked for expedited handling.")


def _draft_grievance(rt: ToolRuntime, args: DraftGrievanceArgs) -> dict[str, Any]:
    rt.member()
    kinds = find_phi(args.description)
    if kinds:
        raise ToolError(f"Remove identifiers from the description ({', '.join(kinds)}); the grievance "
                        "is already linked to the member.")
    payload = {"category": args.category, "description": args.description.strip(),
               "expedite_requested": args.expedite_requested}
    action = rt.session.actions.add(PendingAction("file_grievance", payload, rt.session.turn))
    already = [a.result.get("grievance_id") for a in rt.session.actions.filed() if a.result]
    out: dict[str, Any] = {
        "action_id": action.action_id,
        "status": "DRAFT - NOT FILED",
        "draft": payload,
        "next_step": "Read this summary back to the representative and ask them to confirm that the "
                     "member agrees. Nothing is filed until the representative confirms. You cannot "
                     "file it yourself.",
    }
    if already:
        out["warning"] = (f"Already filed in this conversation: {', '.join(already)}. The plan allows "
                          "one grievance per issue - only continue if this is a different issue.")
    return out


class FileGrievanceArgs(Args):
    action_id: str


def _file_grievance(rt: ToolRuntime, args: FileGrievanceArgs) -> dict[str, Any]:
    """Execute a rep-confirmed draft. Reached only through ``ToolGateway.confirm``."""
    action = rt.session.actions.get(args.action_id)
    ctx = rt.ctx
    key = action.idempotency_key(ctx.tenant_id, ctx.member_id, ctx.conversation_id)
    audit = rt.audit
    common = dict(tenant_id=ctx.tenant_id, rep_id=ctx.rep_id, member_id=ctx.member_id,
                  conversation_id=ctx.conversation_id)
    try:
        record, replayed = rt.api.create_grievance(ctx.member_id, action.payload, key)
    except WriteUnconfirmed as exc:
        action.status = ActionStatus.UNCONFIRMED
        rt.session.tracer.emit("write", "file_grievance", "error", attempts=exc.attempts,
                               idempotency_key=key, outcome="unconfirmed")
        audit.record(**common, action="grievance.file", outcome="unconfirmed",
                     details={"action_id": action.action_id, "idempotency_key": key,
                              "attempts": exc.attempts})
        return {"status": "UNCONFIRMED", "idempotency_key": key}
    if record.get("member_id") != ctx.member_id:
        raise ToolError("The system of record returned a grievance for a different member.")
    action.status = ActionStatus.FILED
    action.result = project(record, _GRIEVANCE_ALLOW)
    rt.session.tracer.emit("write", "file_grievance", "ok", idempotency_key=key, replayed=replayed,
                           grievance_id=record.get("grievance_id"), category=record.get("category"))
    audit.record(**common, action="grievance.file", outcome="filed",
                 details={"action_id": action.action_id, "idempotency_key": key,
                          "grievance_id": record.get("grievance_id"), "replayed": replayed})
    return {"status": "FILED", **action.result}


TOOLS = [
    Tool(
        name="get_grievances",
        description="Grievances on file for the member, each with its status, the plan's resolution "
                    "deadline and whether it is past that deadline (already calculated).",
        args=GetGrievancesArgs, kind=ToolKind.READ, handler=_get_grievances,
    ),
    Tool(
        name="draft_grievance",
        description="Prepare a grievance for the member when the representative asks to file one. This "
                    "only creates a DRAFT: nothing is filed. After drafting, read the summary back and "
                    "ask the representative to confirm; the platform files it once they confirm.",
        args=DraftGrievanceArgs, kind=ToolKind.DRAFT, handler=_draft_grievance,
    ),
    Tool(
        name="file_grievance",
        description="(gateway only) File a rep-confirmed grievance draft.",
        args=FileGrievanceArgs, kind=ToolKind.WRITE, handler=_file_grievance, model_visible=False,
    ),
]
