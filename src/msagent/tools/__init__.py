"""Tool catalogue. Every tool here is reachable only through ``ToolGateway.invoke``."""
from __future__ import annotations

from ..gateway.gateway import Tool
from . import claims, coverage, grievances, providers


def build_tools() -> list[Tool]:
    return [*coverage.TOOLS, *claims.TOOLS, *grievances.TOOLS, *providers.TOOLS]
