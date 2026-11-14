"""Shared helpers for tool handlers."""
from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict


class Args(BaseModel):
    """Base for tool arguments: closed schema, so unknown fields (e.g. ``tenant``) are rejected."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


def usd(amount: float | int | None) -> str | None:
    """Format money once, in the gateway, so the model never reformats or rounds a figure."""
    return None if amount is None else f"${amount:,.2f}"


def add_days(iso_date: str, days: int) -> str:
    return (date.fromisoformat(iso_date) + timedelta(days=days)).isoformat()
