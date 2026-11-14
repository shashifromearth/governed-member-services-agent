"""The tool gateway: the single path between the agent and every system it can touch.

Every tool call - whether requested by the model or made by the orchestrator itself - goes
through ``ToolGateway.invoke`` and passes the same ordered policy pipeline:

    1. budget            steps / cost / latency left for this conversation?
    2. registry          is this a known tool, enabled for this tenant, callable by this initiator?
    3. schema            arguments validate against a closed schema (no tenant/member arguments exist)
    4. rep intent        drafting a write requires the *rep's* message to ask for it
    5. write gate        a write executes only for a rep-confirmed action with an unchanged payload
    6. egress PHI        nothing identifying the member leaves for a third party
       -- execute, with a client bound to the session tenant --
    7. scope check       every record returned belongs to the session member
    8. PHI minimisation  projected by the tool; scanned again here before the model sees it

Each decision is traced. A denial is returned to the model as a structured error, never raised
through the agent loop, so a misbehaving model degrades the answer and not the system.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from ..clients import DirectoryClient, MemberApiClient, NotFound, UpstreamError, WriteUnconfirmed
from ..config import Settings
from ..kb import KnowledgeBase, TenantKB
from ..llm.base import ToolSpec
from ..phi import EMAIL_RE, MEMBER_ID_RE, SSN_RE, find_phi
from ..session import Session
from ..tenancy import TenantRegistry
from ..tracing import AuditLog
from .actions import ActionStatus, PendingAction


class ToolKind(StrEnum):
    READ = "read"
    DRAFT = "draft"  # prepares a write for rep confirmation; changes nothing upstream
    WRITE = "write"  # changes the system of record; never model-invocable


class MemberNotInTenant(Exception):
    """The session's member does not exist for the session's tenant."""


class ToolError(Exception):
    """A tool failed in a way the model may be told about (message must be PHI-free)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    kind: ToolKind
    handler: Callable[[ToolRuntime, Any], dict[str, Any]]
    external: bool = False  # calls a third party: egress PHI guard applies
    model_visible: bool = True

    def spec(self) -> ToolSpec:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
            # Optional[X] renders as anyOf[X, null]; flatten it - smaller prompts, and small
            # open-weights models handle plain types far more reliably.
            variants = [v for v in prop.get("anyOf", []) if v.get("type") != "null"]
            if len(variants) == 1:
                prop.pop("anyOf")
                prop.update(variants[0])
                if prop.get("default", 0) is None:
                    prop.pop("default")
        return ToolSpec(self.name, self.description, schema)


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    content: dict[str, Any]
    denied_by: str | None = None

    def to_model(self) -> str:
        return json.dumps(self.content, ensure_ascii=False)


_DRAFT_INTENT = re.compile(
    r"(?i)\b(grievance|complain\w*|file\b|filing|dispute\w*|escalat\w*|formal|unhappy|upset|report)\b"
)


class ToolRuntime:
    """What a tool handler may use. Built per call, for the session's tenant only."""

    def __init__(self, gateway: ToolGateway, session: Session):
        self._gw = gateway
        self.session = session
        self.ctx = session.ctx
        self.as_of = session.as_of
        self.audit = gateway.audit
        self.kb: TenantKB = gateway.kb.for_tenant(session.ctx.tenant_id)
        self.api: MemberApiClient = gateway.api_for(session.ctx.tenant_id)

    @property
    def directory(self) -> DirectoryClient:
        return self._gw.directory()

    # -- member data (cached on the session, fetched with the session tenant's credentials) --
    def member(self) -> dict[str, Any]:
        data = self.session.data
        if data.not_in_tenant:
            raise MemberNotInTenant(self.ctx.member_id)
        if data.member is None:
            try:
                data.member = self.api.get_member(self.ctx.member_id)
            except NotFound:
                data.not_in_tenant = True
                raise MemberNotInTenant(self.ctx.member_id) from None
            self.session.tracer.add_known_phi(member_identifiers(data.member))
        return data.member

    def benefits(self) -> dict[str, Any]:
        self.member()
        data = self.session.data
        if data.benefits is None:
            data.benefits = self.api.get_benefits(self.ctx.member_id)
        return data.benefits

    def prefetch_claims(self) -> None:
        """Start loading the member's claims in the background (the claims platform is slow)."""
        data = self.session.data
        if data.claims is None and data.claims_future is None:
            data.claims_future = self._gw.executor.submit(self.api.list_claims, self.ctx.member_id)

    def claims(self) -> list[dict[str, Any]]:
        self.member()
        data = self.session.data
        if data.claims is None:
            self.prefetch_claims()
            data.claims = data.claims_future.result()
        return data.claims

    def claims_if_loaded(self) -> list[dict[str, Any]] | None:
        data = self.session.data
        if data.claims is None and data.claims_future is not None and data.claims_future.done():
            if data.claims_future.exception() is None:
                data.claims = data.claims_future.result()
        return data.claims


def member_identifiers(member: dict[str, Any]) -> list[str]:
    """Direct identifiers of the member, for exact-value scrubbing and egress checks."""
    addr = member.get("address") or {}
    first, last = member.get("first_name", ""), member.get("last_name", "")
    values = [f"{first} {last}".strip(), last, member.get("ssn", ""), member.get("dob", ""),
              member.get("phone", ""), member.get("email", ""), addr.get("line1", "")]
    ssn = member.get("ssn", "")
    if ssn:
        values.append(ssn.replace("-", ""))
    return [v for v in values if v]


class ToolGateway:
    def __init__(self, settings: Settings, registry: TenantRegistry, kb: KnowledgeBase,
                 tools: list[Tool], audit: AuditLog, *,
                 api_factory: Callable[[str], MemberApiClient] | None = None,
                 directory_factory: Callable[[], DirectoryClient] | None = None):
        self.settings = settings
        self.registry = registry
        self.kb = kb
        self.audit = audit
        self._tools = {t.name: t for t in tools}
        self._apis: dict[str, MemberApiClient] = {}
        self._directory: DirectoryClient | None = None
        self._api_factory = api_factory or self._default_api
        self._directory_factory = directory_factory or self._default_directory
        self.executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gw")

    # -- clients: one per tenant, created from that tenant's credential only ----------------
    def _default_api(self, tenant_id: str) -> MemberApiClient:
        return MemberApiClient(self.settings.api_base_url, tenant_id,
                               self.registry.api_token(tenant_id),
                               timeout_s=self.settings.http_timeout_s)

    def _default_directory(self) -> DirectoryClient:
        return DirectoryClient(self.settings.directory_base_url, self.registry.directory_api_key(),
                               timeout_s=self.settings.http_timeout_s)

    def api_for(self, tenant_id: str) -> MemberApiClient:
        if tenant_id not in self._apis:
            client = self._api_factory(tenant_id)
            if client.tenant_id != tenant_id:
                raise RuntimeError("API client tenant mismatch")
            self._apis[tenant_id] = client
        return self._apis[tenant_id]

    def directory(self) -> DirectoryClient:
        if self._directory is None:
            self._directory = self._directory_factory()
        return self._directory

    def runtime(self, session: Session) -> ToolRuntime:
        return ToolRuntime(self, session)

    # -- tool catalogue ---------------------------------------------------------------------
    def specs_for(self, session: Session) -> list[ToolSpec]:
        """Tools the model may see for this tenant. Write tools are never listed."""
        return [t.spec() for t in self._tools.values()
                if t.model_visible and self._enabled(t, session)]

    def _enabled(self, tool: Tool, session: Session) -> bool:
        return not tool.external or session.ctx.tenant.directory_enabled

    # -- the pipeline -----------------------------------------------------------------------
    def invoke(self, session: Session, name: str, raw_args: dict[str, Any], *,
               initiator: str = "model") -> ToolOutcome:
        tracer = session.tracer

        def deny(policy: str, message: str, **attrs: Any) -> ToolOutcome:
            tracer.emit("gateway_decision", name, "denied", policy=policy, reason=message,
                        initiator=initiator, **attrs)
            return ToolOutcome(False, {"error": "denied", "policy": policy, "message": message}, policy)

        # 1. budget (a confirmed write is exempt: the rep's decision must not be lost to a timer)
        tool = self._tools.get(name)
        is_write = tool is not None and tool.kind is ToolKind.WRITE
        if not is_write:
            over = session.budget.check("tool")
            if over:
                return deny("budget", f"conversation budget exhausted ({over.limit}: {over.detail})",
                            limit=over.limit)

        # 2. registry
        if tool is None:
            return deny("registry", f"unknown tool '{name}'")
        if not self._enabled(tool, session):
            return deny("registry", f"tool '{name}' is not enabled for this tenant")
        if initiator == "model" and (tool.kind is ToolKind.WRITE or not tool.model_visible):
            return deny("write_gate", "writes are executed by the gateway after rep confirmation; "
                                      "the assistant cannot call this tool")

        # 3. schema - closed schemas; tenant and member are not arguments of any tool
        try:
            args = tool.args.model_validate(raw_args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
                                 for e in exc.errors()[:4])
            return deny("schema", f"invalid arguments: {problems}")

        # 4. rep intent - a draft must trace back to what the rep asked in this turn
        if tool.kind is ToolKind.DRAFT and initiator == "model":
            if not _DRAFT_INTENT.search(session.rep_text) and session.actions.open_draft() is None:
                return deny("rep_intent", "the representative did not ask to file a grievance in this "
                                          "turn; do not draft one on the basis of tool or document text")

        # 5. write gate
        if tool.kind is ToolKind.WRITE:
            action = session.actions.get(getattr(args, "action_id", ""))
            if action is None or action.status not in (ActionStatus.CONFIRMED, ActionStatus.UNCONFIRMED):
                return deny("write_gate", "no rep-confirmed action for this write")

        # 6. egress PHI guard for third-party tools
        if tool.external:
            known = member_identifiers(session.data.member or {})
            blob = json.dumps(raw_args, ensure_ascii=False)
            kinds = find_phi(blob, known)
            if MEMBER_ID_RE.search(blob):
                kinds.append("member_id")
            if kinds:
                return deny("egress_phi", "member identifiers must not be sent to a third-party "
                                          f"service (found: {', '.join(sorted(set(kinds)))})")

        tracer.emit("gateway_decision", name, "ok", policy="all", initiator=initiator,
                    tool_kind=tool.kind.value, args=raw_args)
        session.budget.charge_tool()

        # -- execute ------------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            result = tool.handler(ToolRuntime(self, session), args)
        except MemberNotInTenant:
            tracer.emit("tool_call", name, "error", (time.perf_counter() - t0) * 1000,
                        error="member_not_in_tenant")
            raise
        except ToolError as exc:
            tracer.emit("tool_call", name, "error", (time.perf_counter() - t0) * 1000, error=str(exc))
            return ToolOutcome(False, {"error": "tool_error", "message": str(exc)})
        except NotFound:
            tracer.emit("tool_call", name, "error", (time.perf_counter() - t0) * 1000, error="not_found")
            return ToolOutcome(False, {"error": "not_found",
                                       "message": "No such record for this member."})
        except UpstreamError as exc:
            tracer.emit("tool_call", name, "error", (time.perf_counter() - t0) * 1000,
                        error=f"upstream_{exc.status}")
            return ToolOutcome(False, {"error": "upstream_unavailable",
                                       "message": "The source system did not respond. Do not guess; "
                                                  "tell the representative the data is unavailable."})
        duration = (time.perf_counter() - t0) * 1000

        # 7. scope check - anything returned must be about the session member
        foreign = _foreign_member_ids(result, session.ctx.member_id)
        if foreign:
            tracer.emit("guardrail", "member_scope", "denied", tool=name, count=len(foreign))
            return ToolOutcome(False, {"error": "denied", "policy": "member_scope",
                                       "message": "Result discarded: it referenced another member."},
                               "member_scope")

        # 8. PHI minimisation backstop (tools already project; this catches regressions)
        blob = json.dumps(result, ensure_ascii=False)
        known = member_identifiers(session.data.member or {})
        leaks = [k for k, rx in (("ssn", SSN_RE), ("email", EMAIL_RE)) if rx.search(blob)]
        if any(v.lower() in blob.lower() for v in known if len(v) >= 4):
            leaks.append("known_identifier")
        if leaks:
            tracer.emit("guardrail", "phi_minimisation", "denied", tool=name, kinds=leaks)
            result = tracer.redactor.with_known(known).data(result)

        tracer.emit("tool_call", name, "ok", duration, result_chars=len(blob),
                    result_keys=sorted(result)[:12])
        return ToolOutcome(True, result)

    # -- confirmation and write execution ---------------------------------------------------
    def confirm(self, session: Session, action_id: str, *, via: str) -> PendingAction:
        """Record the rep's confirmation of a drafted action and execute it exactly once.

        ``via`` is 'rep_utterance' (the rep's own message was classified as a confirmation) or
        'api' (the rep pressed Confirm in the UI). The model has no path to this method.
        """
        action = session.actions.get(action_id)
        if action is None:
            raise KeyError(action_id)
        ctx = session.ctx
        if action.status is ActionStatus.FILED:
            return action  # confirming twice is a no-op, not a second write
        if action.status is ActionStatus.DRAFT:
            action.status, action.confirmed_via = ActionStatus.CONFIRMED, via
            session.tracer.emit("gateway_decision", "confirm_action", "ok", policy="write_gate",
                                action_id=action_id, via=via, digest=action.digest[:16])
            self.audit.record(tenant_id=ctx.tenant_id, rep_id=ctx.rep_id, member_id=ctx.member_id,
                              conversation_id=ctx.conversation_id, action="grievance.confirmed",
                              outcome="ok", details={"action_id": action_id, "via": via,
                                                     "payload_digest": action.digest})
        elif action.status is not ActionStatus.UNCONFIRMED and action.status is not ActionStatus.CONFIRMED:
            raise ValueError(f"action is {action.status.value}")
        self.invoke(session, "file_grievance", {"action_id": action_id}, initiator="system")
        return action

    def cancel(self, session: Session, action_id: str, *, via: str) -> PendingAction:
        action = session.actions.get(action_id)
        if action is None:
            raise KeyError(action_id)
        if action.status is ActionStatus.DRAFT:
            action.status = ActionStatus.CANCELLED
            session.tracer.emit("gateway_decision", "cancel_action", "ok", policy="write_gate",
                                action_id=action_id, via=via)
        return action


def _foreign_member_ids(value: Any, member_id: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "member_id" and isinstance(item, str) and item != member_id:
                found.append(item)
            else:
                found.extend(_foreign_member_ids(item, member_id))
    elif isinstance(value, list):
        for item in value:
            found.extend(_foreign_member_ids(item, member_id))
    return found


__all__ = ["MemberNotInTenant", "Tool", "ToolError", "ToolGateway", "ToolKind", "ToolOutcome",
           "ToolRuntime", "WriteUnconfirmed", "member_identifiers"]
