"""Deciding whether the rep confirmed, cancelled or said something else.

This runs on the **rep's own words only** - never on model output or tool text - so nothing a
document, claim note or third party says can count as a confirmation. It is deliberately
conservative: a message is a confirmation only if it clearly affirms and contains no sign of
hesitation; anything else leaves the draft unfiled and the conversation continues.

The HTTP API also exposes explicit confirm/cancel endpoints for a UI button, which is the
preferred production path; this classifier covers the typed/voice case.
"""
from __future__ import annotations

import re
from enum import StrEnum


class Decision(StrEnum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNCLEAR = "unclear"


_CANCEL = re.compile(
    r"(?i)\b(hold off|hold on|don'?t|do not|cancel\w*|stop|wait|not yet|not now|never ?mind|"
    r"no longer|changed (?:his|her|their|my) mind|scrap|discard|withdraw\w*|forget it|"
    r"instead|first|before (?:we|you)|rather not|declin\w+)\b|^\s*no\b"
)
_CONFIRM = re.compile(
    r"(?i)^\s*(yes|yep|yeah|confirm(?:ed)?|correct|approved?|agreed|ok(?:ay)?|sure|please do|"
    r"go ahead|proceed|file it|submit it|do it|that'?s (?:right|correct))\b"
    r"|\b(go ahead|file it|submit it|please file|proceed with (?:it|filing)|confirmed|i confirm|"
    r"member (?:agrees|agreed|confirms|confirmed|approved)|you can file)\b"
)
_EDIT = re.compile(r"(?i)\b(change|edit|update|reword|rephrase|add|remove|correct the|actually it)\b")


def classify(rep_text: str) -> Decision:
    text = rep_text.strip()
    if not text:
        return Decision.UNCLEAR
    if _CANCEL.search(text):
        return Decision.CANCEL if not _CONFIRM.search(text) or _hesitates(text) else Decision.UNCLEAR
    if _EDIT.search(text):
        return Decision.UNCLEAR  # the rep wants the draft changed: re-draft, then ask again
    if _CONFIRM.search(text) and not text.rstrip().endswith("?"):
        return Decision.CONFIRM
    return Decision.UNCLEAR


def _hesitates(text: str) -> bool:
    """Mixed signals ('yes ... but hold off'): hesitation wins, nothing is filed."""
    return bool(re.search(r"(?i)\b(hold off|hold on|don'?t|do not|cancel\w*|not yet|wait|never ?mind)\b",
                          text))
