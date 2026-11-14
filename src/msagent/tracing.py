"""Conversation tracing and the audit log.

Two sinks with different purposes and retention:

* **Trace** - every step of a conversation (model calls, tool calls, gateway decisions,
  guardrails). Operational data: payloads are redacted and member ids pseudonymised before they
  are written. One JSONL file per tenant per day, so a tenant's traces can be stored, retained
  and deleted independently.
* **Audit log** - who did what to which member (access and writes). Append-only and
  hash-chained so tampering is detectable; this is the record retained for six years (NFR-5).
  It holds identifiers (it must, to be an audit trail) but never clinical or claim content.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .phi import Redactor


@dataclass
class TraceEvent:
    ts: str
    tenant_id: str
    conversation_id: str
    turn: int
    seq: int
    kind: str  # turn | model_call | tool_call | gateway_decision | guardrail | retrieval | write
    name: str
    status: str  # ok | denied | error | info
    duration_ms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class JsonlTraceSink:
    """``<dir>/<tenant>/<YYYY-MM-DD>.jsonl`` - tenant-partitioned by construction."""

    def __init__(self, root: Path):
        self._root = root
        self._lock = threading.Lock()

    def write(self, event: TraceEvent) -> None:
        path = self._root / event.tenant_id / f"{event.ts[:10]}.jsonl"
        line = json.dumps(asdict(event), default=str, ensure_ascii=False)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class Tracer:
    """Per-conversation tracer. Redaction happens here, once, before any sink sees the event."""

    def __init__(self, tenant_id: str, conversation_id: str, redactor: Redactor,
                 sinks: list[TraceSink] | None = None):
        self.tenant_id = tenant_id
        self.conversation_id = conversation_id
        self.redactor = redactor
        self.turn = 0
        self.events: list[TraceEvent] = []
        self._sinks = sinks or []
        self._seq = 0
        self._lock = threading.Lock()

    def add_known_phi(self, values: list[str]) -> None:
        """Register the session member's identifiers so they are scrubbed by exact value."""
        self.redactor = self.redactor.with_known(values)

    def emit(self, kind: str, name: str, status: str = "ok", duration_ms: float | None = None,
             **attrs: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
                tenant_id=self.tenant_id,
                conversation_id=self.conversation_id,
                turn=self.turn,
                seq=self._seq,
                kind=kind,
                name=name,
                status=status,
                duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
                attrs=self.redactor.data(attrs),
            )
            self.events.append(event)
        for sink in self._sinks:
            sink.write(event)
        return event

    def span(self, kind: str, name: str, **attrs: Any) -> _Span:
        return _Span(self, kind, name, attrs)

    def export(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]


class _Span:
    """``with tracer.span(...) as s: ...; s.set(k=v)`` - emits one event with the duration."""

    def __init__(self, tracer: Tracer, kind: str, name: str, attrs: dict[str, Any]):
        self._tracer, self._kind, self._name, self._attrs = tracer, kind, name, dict(attrs)
        self.status = "ok"

    def set(self, **attrs: Any) -> None:
        self._attrs.update(attrs)

    def __enter__(self) -> _Span:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.status = "error"
            self._attrs["error"] = f"{exc_type.__name__}: {exc}"
        self._tracer.emit(self._kind, self._name, self.status,
                          (time.perf_counter() - self._t0) * 1000, **self._attrs)
        return False


class AuditLog:
    """Append-only, hash-chained audit trail, one file per tenant.

    Each record carries the SHA-256 of the previous record, so removing or editing a line breaks
    the chain (``verify``). In production the same records go to WORM storage with a 6-year
    retention lock; the chain makes the export independently verifiable.
    """

    GENESIS = "0" * 64

    def __init__(self, root: Path):
        self._root = root
        self._lock = threading.Lock()
        self._last: dict[str, str] = {}

    def _path(self, tenant_id: str) -> Path:
        return self._root / f"{tenant_id}.jsonl"

    def _tail_hash(self, tenant_id: str) -> str:
        if tenant_id in self._last:
            return self._last[tenant_id]
        path = self._path(tenant_id)
        last = self.GENESIS
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                last = json.loads(lines[-1])["hash"]
        return last

    def record(self, *, tenant_id: str, rep_id: str, member_id: str, conversation_id: str,
               action: str, outcome: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            body = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "tenant_id": tenant_id,
                "rep_id": rep_id,
                "member_id": member_id,
                "conversation_id": conversation_id,
                "action": action,
                "outcome": outcome,
                "details": details or {},
                "prev_hash": self._tail_hash(tenant_id),
            }
            body["hash"] = hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            path = self._path(tenant_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(body, ensure_ascii=False) + "\n")
            self._last[tenant_id] = body["hash"]
            return body

    def verify(self, tenant_id: str) -> bool:
        path = self._path(tenant_id)
        if not path.is_file():
            return True
        prev = self.GENESIS
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            claimed = rec.pop("hash")
            if rec["prev_hash"] != prev:
                return False
            actual = hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if actual != claimed:
                return False
            prev = claimed
        return True
