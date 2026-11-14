"""Conversation state.

Everything a conversation knows lives on its ``Session`` object: the security context, the
trace, the budget, drafted actions and a member-data cache. There are no process-wide caches of
member data, so data fetched for one tenant's conversation cannot be served to another (NFR-4).
In production the same object is persisted under a ``{tenant}:{conversation}`` key.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from .gateway.actions import ActionStore
from .gateway.budget import BudgetTracker
from .llm.base import Message
from .tenancy import TenantContext
from .tracing import Tracer


@dataclass
class MemberData:
    """Upstream records fetched for this conversation. Raw records never leave the gateway."""

    member: dict[str, Any] | None = None
    benefits: dict[str, Any] | None = None
    claims: list[dict[str, Any]] | None = None
    claims_future: Future | None = None
    not_in_tenant: bool = False


@dataclass
class Session:
    ctx: TenantContext
    tracer: Tracer
    budget: BudgetTracker
    as_of: str
    actions: ActionStore = field(default_factory=ActionStore)
    data: MemberData = field(default_factory=MemberData)
    transcript: list[Message] = field(default_factory=list)
    rep_text: str = ""  # the rep's message for the turn in progress (used by intent policy)
    turn: int = 0
    ssn_checks: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def conversation_id(self) -> str:
        return self.ctx.conversation_id
