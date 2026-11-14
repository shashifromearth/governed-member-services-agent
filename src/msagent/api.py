"""HTTP API for the rep desktop.

The rep's tenant comes from their credential, never from the request body: a rep key maps to
exactly one tenant, conversations are looked up within that tenant, and a conversation id from
another tenant is indistinguishable from one that does not exist (404).

    POST /v1/conversations                                   {member_id}        -> conversation
    POST /v1/conversations/{id}/messages                     {text}             -> reply
    POST /v1/conversations/{id}/actions/{action_id}/confirm                     -> reply
    POST /v1/conversations/{id}/actions/{action_id}/cancel                      -> reply
    GET  /v1/conversations/{id}/trace                                           -> trace events
    GET  /healthz

In this slice rep keys are static (``MSA_REP_KEYS``); in production the same dependency
validates an OIDC token and reads the tenant claim.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent.service import AgentService, ConversationNotFound
from .session import Session


class StartConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    member_id: str = Field(pattern=r"^[A-Za-z0-9-]{3,32}$")


class RepMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)


class RepIdentity(BaseModel):
    tenant_id: str
    rep_id: str


def create_app(service: AgentService | None = None) -> FastAPI:
    app = FastAPI(title="Member services agent", version="0.1.0")
    state: dict[str, AgentService] = {}

    def svc() -> AgentService:
        if "svc" not in state:
            state["svc"] = service or AgentService.build()
        return state["svc"]

    def rep(authorization: Annotated[str | None, Header()] = None,
            agent: AgentService = Depends(svc)) -> RepIdentity:  # noqa: B008
        token = (authorization or "").removeprefix("Bearer ").strip()
        entry = agent.settings.rep_keys.get(token) if token else None
        if not entry:
            raise HTTPException(401, "missing or invalid rep credential")
        return RepIdentity(tenant_id=entry["tenant"], rep_id=entry.get("rep_id", "rep"))

    def conversation(conversation_id: str, who: RepIdentity = Depends(rep),  # noqa: B008
                     agent: AgentService = Depends(svc)) -> Session:  # noqa: B008
        try:
            return agent.get(conversation_id, who.tenant_id)
        except ConversationNotFound:
            raise HTTPException(404, "conversation not found") from None

    @app.get("/healthz")
    def healthz(agent: AgentService = Depends(svc)) -> dict[str, Any]:  # noqa: B008
        return {"status": "ok", "as_of": agent.as_of, "model": agent.model.model,
                "provider": agent.model.name, "tenants": agent.registry.ids(),
                "kb_quarantined_documents": len(agent.kb.quarantined)}

    @app.post("/v1/conversations", status_code=201)
    def start(body: StartConversation, who: RepIdentity = Depends(rep),  # noqa: B008
              agent: AgentService = Depends(svc)) -> dict[str, Any]:  # noqa: B008
        session = agent.start(who.tenant_id, body.member_id, who.rep_id)
        return {"conversation_id": session.conversation_id, "tenant_id": who.tenant_id}

    @app.post("/v1/conversations/{conversation_id}/messages")
    def message(body: RepMessage, session: Session = Depends(conversation),  # noqa: B008
                agent: AgentService = Depends(svc)) -> dict[str, Any]:  # noqa: B008
        return asdict(agent.send(session, body.text))

    @app.post("/v1/conversations/{conversation_id}/actions/{action_id}/confirm")
    def confirm(action_id: str, session: Session = Depends(conversation),  # noqa: B008
                agent: AgentService = Depends(svc)) -> dict[str, Any]:  # noqa: B008
        try:
            return asdict(agent.confirm(session, action_id))
        except KeyError:
            raise HTTPException(404, "action not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/v1/conversations/{conversation_id}/actions/{action_id}/cancel")
    def cancel(action_id: str, session: Session = Depends(conversation),  # noqa: B008
               agent: AgentService = Depends(svc)) -> dict[str, Any]:  # noqa: B008
        try:
            return asdict(agent.cancel(session, action_id))
        except KeyError:
            raise HTTPException(404, "action not found") from None

    @app.get("/v1/conversations/{conversation_id}/trace")
    def trace(session: Session = Depends(conversation)) -> dict[str, Any]:  # noqa: B008
        return {"conversation_id": session.conversation_id, "events": session.tracer.export()}

    return app


app = create_app()
