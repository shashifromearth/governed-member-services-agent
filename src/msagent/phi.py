"""PHI minimisation primitives.

Three mechanisms, used at different trust boundaries:

* ``project``   - allow-list projection. Upstream records are reduced to the fields a tool may
  expose *before* anything reaches the model. Allow-lists fail closed: a new upstream field is
  invisible until someone adds it deliberately.
* ``Redactor``  - scrubbing for everything written to logs/traces: patterns (SSN, phone, email,
  date of birth), exact-value scrubbing of the session member's known identifiers, and keyed
  pseudonyms for member ids so log lines stay correlatable.
* ``find_phi``  - detection used by the egress guard (third-party tool arguments) and the
  output guard (final answers).
"""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable, Mapping
from typing import Any

SSN_RE = re.compile(r"(?<![\dA-Za-z])\d{3}[- ]?\d{2}[- ]?\d{4}(?![\dA-Za-z])")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DOB_RE = re.compile(
    r"(?i)\b(?:dob|date of birth|born)\b[^0-9]{0,12}(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})"
)
MEMBER_ID_RE = re.compile(r"\bM\d{6}\b")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", SSN_RE),
    ("email", EMAIL_RE),
    ("phone", PHONE_RE),
    ("dob", DOB_RE),
]


def project(record: Mapping[str, Any], allow: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only allow-listed fields. ``allow`` maps field -> True or a nested allow-map."""
    out: dict[str, Any] = {}
    for key, rule in allow.items():
        if key not in record:
            continue
        value = record[key]
        if rule is True:
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = project(value, rule)
        elif isinstance(value, list):
            out[key] = [project(v, rule) if isinstance(v, Mapping) else v for v in value]
        elif value is None:
            out[key] = None
    return out


def find_phi(text: str, known_values: Iterable[str] = ()) -> list[str]:
    """Return the kinds of PHI detected in ``text`` (an empty list means clean)."""
    kinds = [kind for kind, pattern in _PATTERNS if pattern.search(text)]
    lowered = text.lower()
    if any(v and len(v) >= 3 and v.lower() in lowered for v in known_values):
        kinds.append("known_identifier")
    return kinds


def pseudonym(value: str, key: str, prefix: str = "mbr") -> str:
    """Stable keyed pseudonym: log lines correlate without holding the identifier."""
    digest = hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()[:12]
    return f"{prefix}_{digest}"


class Redactor:
    """Scrubs text and JSON-like structures on their way to a log sink."""

    def __init__(self, hmac_key: str, known_values: Iterable[str] = ()):
        self._key = hmac_key
        # Longest first so a full name is replaced before its first-name substring.
        self._known = sorted({v for v in known_values if v and len(v) >= 3}, key=len, reverse=True)

    def with_known(self, values: Iterable[str]) -> Redactor:
        return Redactor(self._key, [*self._known, *values])

    def text(self, value: str) -> str:
        out = value
        for known in self._known:
            out = re.sub(re.escape(known), "[REDACTED]", out, flags=re.IGNORECASE)
        out = SSN_RE.sub("[SSN]", out)
        out = EMAIL_RE.sub("[EMAIL]", out)
        out = PHONE_RE.sub("[PHONE]", out)
        out = DOB_RE.sub(lambda m: m.group(0).replace(m.group(1), "[DOB]"), out)
        out = MEMBER_ID_RE.sub(lambda m: pseudonym(m.group(0), self._key), out)
        return out

    def data(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {k: self.data(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.data(v) for v in value]
        return value
