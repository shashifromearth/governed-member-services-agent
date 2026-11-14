"""The agent: one rep turn in, one reply out.

The orchestration is intentionally thin. The model plans and words the answer; everything that
must be *right* is deterministic and lives outside it:

* member resolution and tenant binding (before the model is called at all);
* confirmation of writes (classified from the rep's own words, executed by the gateway);
* retrieval scoping (tenant, product, plan year);
* arithmetic and date maths (in the tools);
* input and output guards (full SSNs are masked on the way in; identifiers are blocked on the
  way out).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..clients import UpstreamError, upstream_clock
from ..config import Settings
from ..gateway.actions import ActionStatus, PendingAction
from ..gateway.budget import BudgetTracker
from ..gateway.gateway import MemberNotInTenant, ToolGateway, member_identifiers
from ..kb import KnowledgeBase
from ..llm import ModelError, ModelProvider, build_provider, cost_usd
from ..phi import SSN_RE, Redactor, find_phi
from ..session import Session
from ..tenancy import TenantContext, TenantRegistry
from ..tools import build_tools
from ..tracing import AuditLog, JsonlTraceSink, Tracer, TraceSink
from . import prompts
from .confirmation import Decision, classify


@dataclass
class Reply:
    conversation_id: str
    turn: int
    text: str
    outcome: str  # answered | awaiting_confirmation | filed | unconfirmed | cancelled |
    #               member_not_found | budget_exhausted | unavailable
    pending_action: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ConversationNotFound(Exception):
    pass


class AgentService:
    """Composition root and conversation API used by the HTTP layer, the CLI and the eval."""

    def __init__(self, settings: Settings, registry: TenantRegistry, kb: KnowledgeBase,
                 gateway: ToolGateway, model: ModelProvider, audit: AuditLog, *,
                 as_of: str, sinks: list[TraceSink] | None = None):
        self.settings, self.registry, self.kb = settings, registry, kb
        self.gateway, self.model, self.audit = gateway, model, audit
        self.as_of = as_of
        self._sinks = sinks or []
        self._sessions: dict[str, Session] = {}

    @classmethod
    def build(cls, settings: Settings | None = None, model: ModelProvider | None = None,
              **gateway_kwargs: Any) -> AgentService:
        settings = settings or Settings.from_env()
        registry = TenantRegistry.load(settings)
        kb = KnowledgeBase.load(settings.pack_dir, registry)
        audit = AuditLog(settings.var_dir / "audit")
        gateway = ToolGateway(settings, registry, kb, build_tools(), audit, **gateway_kwargs)
        as_of = settings.as_of or upstream_clock(settings.api_base_url) or date.today().isoformat()
        return cls(settings, registry, kb, gateway, model or build_provider(settings.llm), audit,
                   as_of=as_of, sinks=[JsonlTraceSink(settings.var_dir / "traces")])

    # -- conversations ------------------------------------------------------------------------
    def start(self, tenant_id: str, member_id: str, rep_id: str = "rep") -> Session:
        tenant = self.registry.get(tenant_id)  # UnknownTenant if the rep's tenant is not registered
        ctx = TenantContext(tenant, member_id, rep_id, f"conv_{uuid.uuid4().hex[:16]}")
        tracer = Tracer(tenant_id, ctx.conversation_id, Redactor(self.settings.log_hmac_key), self._sinks)
        session = Session(ctx=ctx, tracer=tracer, budget=BudgetTracker(self.settings.budgets),
                          as_of=self.as_of)
        self._sessions[ctx.conversation_id] = session
        tracer.emit("turn", "conversation_started", "info", rep_id=rep_id, member_ref=member_id,
                    model=self.model.model, provider=self.model.name, as_of=self.as_of)
        return session

    def get(self, conversation_id: str, tenant_id: str) -> Session:
        """Look a conversation up *within a tenant*: another tenant's id is simply not found."""
        session = self._sessions.get(conversation_id)
        if session is None or session.ctx.tenant_id != tenant_id:
            raise ConversationNotFound(conversation_id)
        return session

    # -- one rep turn ---------------------------------------------------------------------------
    def send(self, session: Session, rep_text: str) -> Reply:
        with session.lock:
            t0 = time.perf_counter()
            session.turn += 1
            session.tracer.turn = session.turn
            session.budget.start_turn()
            reply = self._turn(session, rep_text)
            reply.usage = {**session.budget.snapshot(),
                           "turn_latency_s": round(time.perf_counter() - t0, 3)}
            session.tracer.emit("turn", "reply", "ok", (time.perf_counter() - t0) * 1000,
                                outcome=reply.outcome, text=reply.text[:2000], **session.budget.snapshot())
            return reply

    def _turn(self, session: Session, rep_text: str) -> Reply:
        tracer, ctx = session.tracer, session.ctx

        # Input guard: a full SSN typed by the rep is masked before it can reach the model or logs.
        clean, masked = SSN_RE.subn("[SSN removed]", rep_text.strip())
        if masked:
            tracer.emit("guardrail", "input_ssn_masked", "denied", count=masked)
        session.rep_text = clean

        # 1. Resolve the member inside the rep's tenant - before any model call.
        try:
            coverage = self.gateway.invoke(session, "get_coverage", {}, initiator="system")
        except MemberNotInTenant:
            # No member record, so nothing to scrub by value: log the size of the message only.
            tracer.emit("turn", "rep_message", "info", chars=len(clean))
            if session.turn == 1:
                self.audit.record(tenant_id=ctx.tenant_id, rep_id=ctx.rep_id, member_id=ctx.member_id,
                                  conversation_id=ctx.conversation_id, action="member.lookup",
                                  outcome="not_found_in_tenant")
            return self._reply(session, prompts.MEMBER_NOT_FOUND.format(tenant_name=ctx.tenant.name),
                               "member_not_found", record=False)
        except UpstreamError:
            tracer.emit("turn", "rep_message", "info", chars=len(clean))
            return self._reply(session, prompts.MODEL_UNAVAILABLE, "unavailable", record=False)
        # Traced only now: the member's identifiers are known, so anything the rep typed about
        # the member (name, phone, ...) is scrubbed by value before it is written.
        tracer.emit("turn", "rep_message", "info", text=clean)
        if not coverage.ok:
            limit = coverage.content.get("message", "limit reached")
            return self._reply(session, prompts.BUDGET_EXHAUSTED.format(limit=limit), "budget_exhausted",
                               record=False)
        if session.turn == 1:
            self.audit.record(tenant_id=ctx.tenant_id, rep_id=ctx.rep_id, member_id=ctx.member_id,
                              conversation_id=ctx.conversation_id, action="member.access", outcome="ok")
            self.gateway.runtime(session).prefetch_claims()  # warm the slow claims platform

        # 2. A drafted write is waiting: the rep's own words decide, not the model.
        draft, stuck = session.actions.open_draft(), session.actions.unconfirmed()
        if draft is not None:
            decision = classify(clean)
            tracer.emit("gateway_decision", "rep_confirmation", "info", decision=decision.value,
                        action_id=draft.action_id)
            if decision is Decision.CONFIRM:
                return self._after_write(session, self.gateway.confirm(session, draft.action_id,
                                                                       via="rep_utterance"), clean)
            if decision is Decision.CANCEL:
                self.gateway.cancel(session, draft.action_id, via="rep_utterance")
                return self._reply(session, prompts.CANCELLED, "cancelled", rep_text=clean)
        elif stuck is not None and (classify(clean) is Decision.CONFIRM or "retry" in clean.lower()):
            return self._after_write(session, self.gateway.confirm(session, stuck.action_id,
                                                                   via="rep_utterance"), clean)

        # 3. Retrieval for this turn, scoped by the gateway to tenant / product / plan year.
        passages: list[dict[str, Any]] = []
        plan_year = coverage.content.get("current_plan_year")
        found = self.gateway.invoke(session, "search_plan_documents", {"query": clean[:300]},
                                    initiator="system")
        if found.ok:
            passages = found.content["passages"]
            tracer.emit("retrieval", "plan_documents", "ok", plan_year=plan_year,
                        sources=sorted({p["source"] for p in passages}), passages=len(passages))

        session.transcript.append({"role": "user", "content": prompts.user_turn(
            clean, coverage=coverage.content if session.turn == 1 or not self._has_coverage(session)
            else None, passages=passages, plan_year=plan_year)})

        # 4. Model / tool loop.
        text, outcome = self._loop(session)

        # 5. Output guards.
        text = self._guard_output(session, text)
        new_draft = session.actions.open_draft()
        if new_draft is not None and new_draft.drafted_turn == session.turn:
            text = text.rstrip() + prompts.DRAFT_READBACK.format(
                category=new_draft.payload["category"], description=new_draft.payload["description"],
                expedite="yes" if new_draft.payload["expedite_requested"] else "no")
            outcome = "awaiting_confirmation"
        elif new_draft is not None and outcome == "answered":
            outcome = "awaiting_confirmation"
        return self._reply(session, text, outcome, record=False,
                           pending=new_draft.public_view() if new_draft else None)

    def _has_coverage(self, session: Session) -> bool:
        return any(m["role"] == "user" and "<member_coverage>" in m["content"]
                   for m in session.transcript)

    def _loop(self, session: Session) -> tuple[str, str]:
        tracer = session.tracer
        system = prompts.SYSTEM_TEMPLATE.format(tenant_name=session.ctx.tenant.name, as_of=session.as_of)
        specs = self.gateway.specs_for(session)
        llm = self.settings.llm
        while True:
            over = session.budget.check("model")
            if over:
                tracer.emit("gateway_decision", "model_call", "denied", policy="budget",
                            limit=over.limit, reason=over.detail)
                text = prompts.BUDGET_EXHAUSTED.format(limit=f"{over.limit}: {over.detail}")
                session.transcript.append({"role": "assistant", "text": text, "tool_calls": []})
                return text, "budget_exhausted"
            t0 = time.perf_counter()
            try:
                resp = self.model.complete(system, session.transcript, specs)
            except ModelError as exc:
                tracer.emit("model_call", self.model.model, "error", (time.perf_counter() - t0) * 1000,
                            error=str(exc)[:300])
                session.transcript.append({"role": "assistant", "text": prompts.MODEL_UNAVAILABLE,
                                           "tool_calls": []})
                return prompts.MODEL_UNAVAILABLE, "unavailable"
            cost = cost_usd(self.model.model, resp.usage, llm.price_in_per_mtok, llm.price_out_per_mtok)
            session.budget.charge_model(cost)
            tracer.emit("model_call", self.model.model, "ok", (time.perf_counter() - t0) * 1000,
                        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                        cached_input_tokens=resp.usage.cached_input_tokens, cost_usd=round(cost, 6),
                        stop_reason=resp.stop_reason, tool_calls=[c.name for c in resp.tool_calls],
                        text=resp.text[:1500])
            session.transcript.append(resp.as_message())
            if not resp.tool_calls:
                return (resp.text or "I couldn't produce an answer to that. Nothing was changed."), "answered"
            results = []
            for call in resp.tool_calls:
                try:
                    out = self.gateway.invoke(session, call.name, call.arguments, initiator="model")
                    content, is_error = out.to_model(), not out.ok
                except MemberNotInTenant:
                    content, is_error = '{"error": "not_found"}', True
                results.append({"tool_call_id": call.id, "name": call.name, "content": content,
                                "is_error": is_error})
            session.transcript.append({"role": "tool", "results": results})

    # -- guards and replies ---------------------------------------------------------------------
    def _guard_output(self, session: Session, text: str) -> str:
        known = member_identifiers(session.data.member or {})
        kinds = find_phi(text, [v for v in known if len(v) >= 5])
        if "ssn" in kinds or "known_identifier" in kinds or "email" in kinds:
            session.tracer.emit("guardrail", "output_phi", "denied", kinds=kinds)
            return prompts.SSN_REFUSAL
        return text

    def _after_write(self, session: Session, action: PendingAction, rep_text: str) -> Reply:
        if action.status is ActionStatus.FILED and action.result:
            fields = {k: action.result.get(k, "n/a")
                      for k in ("grievance_id", "category", "received_date", "status")}
            return self._reply(session, prompts.FILED.format(**fields), "filed",
                               rep_text=rep_text, pending=action.public_view())
        return self._reply(session, prompts.UNCONFIRMED, "unconfirmed", rep_text=rep_text,
                           pending=action.public_view())

    def _reply(self, session: Session, text: str, outcome: str, *, record: bool = True,
               rep_text: str | None = None, pending: dict[str, Any] | None = None) -> Reply:
        if record and rep_text is not None:
            # Deterministic turns are added to the transcript so the model keeps the full picture.
            session.transcript.append({"role": "user", "content":
                                       f"<rep_message>\n{rep_text}\n</rep_message>"})
            session.transcript.append({"role": "assistant", "text":
                                       f"[platform message] {text}", "tool_calls": []})
        return Reply(session.conversation_id, session.turn, text, outcome, pending)

    # -- explicit confirmation (UI button) ------------------------------------------------------
    def confirm(self, session: Session, action_id: str) -> Reply:
        with session.lock:
            return self._after_write(session, self.gateway.confirm(session, action_id, via="api"),
                                     "(confirmed with the Confirm button)")

    def cancel(self, session: Session, action_id: str) -> Reply:
        with session.lock:
            self.gateway.cancel(session, action_id, via="api")
            return self._reply(session, prompts.CANCELLED, "cancelled",
                               rep_text="(cancelled with the Cancel button)")
