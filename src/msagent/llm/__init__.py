"""Model providers. ``build_provider`` is the only place that knows the concrete classes."""
from __future__ import annotations

from ..config import LLMSettings
from .base import ModelError, ModelProvider, ModelResponse, ToolCall, ToolSpec, Usage, cost_usd

__all__ = ["ModelError", "ModelProvider", "ModelResponse", "ToolCall", "ToolSpec", "Usage",
           "build_provider", "cost_usd"]


def build_provider(settings: LLMSettings) -> ModelProvider:
    if settings.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if settings.provider == "openai":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings)
    raise ValueError(f"unknown MSA_LLM_PROVIDER '{settings.provider}' (use anthropic | openai | ollama)")
