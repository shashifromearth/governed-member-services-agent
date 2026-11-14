"""Pending write actions (two-phase writes).

A write is never executed from a model tool call. The model can only *draft* an action; the
draft is frozen (payload + digest) and shown to the rep. Only an explicit rep confirmation -
captured outside the model - moves it to CONFIRMED, and the gateway then executes exactly the
frozen payload with an idempotency key derived from it.

    DRAFT --rep confirms--> CONFIRMED --201/200--> FILED
      |                         '--no ack-------> UNCONFIRMED --retry (same key)--> FILED
      '--rep cancels / new draft--> CANCELLED
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActionStatus(StrEnum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    FILED = "filed"
    UNCONFIRMED = "unconfirmed"
    CANCELLED = "cancelled"


def payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class PendingAction:
    kind: str  # "file_grievance"
    payload: dict[str, Any]
    drafted_turn: int
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:12]}")
    status: ActionStatus = ActionStatus.DRAFT
    result: dict[str, Any] | None = None
    confirmed_via: str | None = None  # "rep_utterance" | "api"

    @property
    def digest(self) -> str:
        return payload_digest(self.payload)

    def idempotency_key(self, tenant_id: str, member_id: str, conversation_id: str) -> str:
        """Stable for (conversation, frozen payload): every retry of this action reuses it."""
        raw = f"{tenant_id}|{member_id}|{conversation_id}|{self.digest}"
        return "grv-" + hashlib.sha256(raw.encode()).hexdigest()[:40]

    def public_view(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "kind": self.kind, "status": self.status.value,
                "payload": self.payload, "result": self.result}


class ActionStore:
    """Actions of one conversation."""

    def __init__(self) -> None:
        self._actions: dict[str, PendingAction] = {}

    def add(self, action: PendingAction) -> PendingAction:
        for other in self._actions.values():  # at most one open draft at a time
            if other.status is ActionStatus.DRAFT:
                other.status = ActionStatus.CANCELLED
        self._actions[action.action_id] = action
        return action

    def get(self, action_id: str) -> PendingAction | None:
        return self._actions.get(action_id)

    def open_draft(self) -> PendingAction | None:
        return next((a for a in self._actions.values() if a.status is ActionStatus.DRAFT), None)

    def unconfirmed(self) -> PendingAction | None:
        return next((a for a in self._actions.values() if a.status is ActionStatus.UNCONFIRMED), None)

    def filed(self) -> list[PendingAction]:
        return [a for a in self._actions.values() if a.status is ActionStatus.FILED]

    def all(self) -> list[PendingAction]:
        return list(self._actions.values())
