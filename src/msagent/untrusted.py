"""Handling of untrusted free text (claim notes, KB prose, third-party tool manifests).

Free text that arrives through a tool is *data*. Anything in it that reads like an instruction
to the assistant is removed before the text can reach the model, and the removal is reported so
the gateway can trace it. This is one layer only: the structural controls (no write without rep
confirmation, no tenant/member arguments on tools) do not depend on this filter being perfect.
"""
from __future__ import annotations

import re

# A role prefix inside data ("assistant: ...", "system: ...") - everything after it is suspect.
_ROLE_PREFIX = re.compile(r"(?i)(?:^|[\s.;])(?:assistant|system|ai agent|agent|developer)\s*:")

_INSTRUCTION = re.compile(
    r"(?i)("
    r"ignore (?:all |any |the )?(?:previous|prior|above)"
    r"|disregard (?:all |any |the )?(?:previous|prior|above)"
    r"|without (?:asking|confirm\w*|telling|mention\w*)"
    r"|do(?:es)? not (?:mention|tell|ask|disclose|reveal)"
    r"|don't (?:mention|tell|ask)"
    r"|need not be mentioned|not need to be mentioned"
    r"|(?:for|to) ai (?:coding )?(?:assistants?|agents?|models?)"
    r"|ai (?:coding )?assistants?\b"
    r"|you (?:must|should) (?:now|always|never)"
    r"|always begin the \w+ argument"
    r"|system prompt"
    r")"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
REMOVED = "[instruction-like text removed]"


def sanitize(text: str) -> tuple[str, bool]:
    """Return ``(clean_text, flagged)``. Flagged sentences are replaced by a neutral marker."""
    if not text:
        return text, False
    flagged = False
    role = _ROLE_PREFIX.search(text)
    if role:
        text, flagged = text[: role.start()].rstrip() + " " + REMOVED, True
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if _INSTRUCTION.search(sentence):
            flagged = True
            if not kept or kept[-1] != REMOVED:
                kept.append(REMOVED)
        else:
            kept.append(sentence)
    clean = " ".join(kept).strip()
    while f"{REMOVED} {REMOVED}" in clean:
        clean = clean.replace(f"{REMOVED} {REMOVED}", REMOVED)
    return clean, flagged
