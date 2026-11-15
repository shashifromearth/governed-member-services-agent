"""Runtime configuration.

Everything environment-specific (pack location, service URLs, credentials, model provider,
budgets) is resolved here and nowhere else, from environment variables with an optional
``.env`` file. Secrets are only ever read through :meth:`Settings.secret` so that the
production swap to a secret manager is a one-function change.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; the existing environment wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


LOCAL_MODEL_DEFAULT = "qwen2.5:7b"
OLLAMA_URL_DEFAULT = "http://localhost:11434/v1"
HOSTED_MODEL_DEFAULTS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


def resolve_llm(env: Mapping[str, str]) -> tuple[str, str, str | None, str | None, bool]:
    """Decide which model the agent uses. Returns (provider, model, base_url, api_key, is_local).

    ``MSA_LLM_PROVIDER=auto`` (the default) is **local first**: the agent runs on an open-weights
    model served by Ollama and needs no account or key. A hosted provider is used only when the
    operator has supplied its API key - Anthropic if ``ANTHROPIC_API_KEY`` is set, otherwise
    OpenAI if ``OPENAI_API_KEY`` is set. Setting the provider explicitly (anthropic | openai |
    ollama) skips the detection.
    """
    choice = (env.get("MSA_LLM_PROVIDER") or "auto").lower()
    explicit_model = env.get("MSA_LLM_MODEL") or None
    base_url = env.get("MSA_LLM_BASE_URL") or None
    generic_key = env.get("MSA_LLM_API_KEY") or None
    if choice == "auto":
        if env.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif env.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            choice = "ollama"
    if choice == "ollama":
        return ("openai", explicit_model or env.get("MSA_LOCAL_MODEL") or LOCAL_MODEL_DEFAULT,
                base_url or env.get("MSA_OLLAMA_URL") or OLLAMA_URL_DEFAULT, generic_key, True)
    if choice == "anthropic":
        return ("anthropic", explicit_model or HOSTED_MODEL_DEFAULTS["anthropic"], base_url,
                generic_key or env.get("ANTHROPIC_API_KEY") or None, False)
    if choice == "openai":
        return ("openai", explicit_model or HOSTED_MODEL_DEFAULTS["openai"], base_url,
                generic_key or env.get("OPENAI_API_KEY") or None, False)
    raise ValueError(f"unknown MSA_LLM_PROVIDER '{choice}' (use auto | ollama | anthropic | openai)")


@dataclass(frozen=True)
class Budgets:
    """Per-conversation limits enforced by the gateway (see gateway/budget.py)."""

    max_steps: int = 40  # model calls + tool calls over the whole conversation
    max_model_calls_per_turn: int = 6
    max_cost_usd: float = 0.10  # hard stop; the NFR-7 *average* target is $0.03
    turn_deadline_s: float = 30.0  # wall clock per rep turn


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "anthropic"  # anthropic | openai (OpenAI-compatible: Ollama, OpenAI, Azure, vLLM)
    model: str = "claude-haiku-5-5"
    base_url: str | None = None
    api_key: str | None = None
    max_output_tokens: int = 1024
    timeout_s: float = 120.0
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None


@dataclass(frozen=True)
class Settings:
    pack_dir: Path
    api_base_url: str
    directory_base_url: str
    tenants_file: Path
    var_dir: Path  # traces, audit log (never committed)
    as_of: str | None  # ISO date override; default is the upstream clock
    budgets: Budgets
    llm: LLMSettings
    rep_keys: dict[str, dict[str, str]] = field(default_factory=dict)
    log_hmac_key: str = "dev-only-key"
    http_timeout_s: float = 10.0

    @staticmethod
    def secret(name: str) -> str | None:
        """Single choke point for secret access (env here; secret manager in production)."""
        return os.environ.get(name) or None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file or REPO_ROOT / ".env")
        env = os.environ

        def _f(name: str, default: float) -> float:
            return float(env.get(name) or default)

        def _opt_f(name: str) -> float | None:
            return float(env[name]) if env.get(name) else None

        api_base = env.get("MSA_API_BASE_URL", "http://127.0.0.1:8090").rstrip("/")
        provider, model, base_url, api_key, local = resolve_llm(env)
        return cls(
            pack_dir=Path(env.get("MSA_PACK_DIR") or str(REPO_ROOT)).resolve(),
            api_base_url=api_base,
            directory_base_url=(env.get("MSA_DIRECTORY_BASE_URL") or api_base + "/directory/v3").rstrip("/"),
            tenants_file=Path(env.get("MSA_TENANTS_FILE") or str(REPO_ROOT / "config" / "tenants.json")),
            var_dir=Path(env.get("MSA_VAR_DIR") or str(REPO_ROOT / "var")),
            as_of=env.get("MSA_AS_OF") or None,
            budgets=Budgets(
                max_steps=int(env.get("MSA_BUDGET_MAX_STEPS") or 40),
                max_model_calls_per_turn=int(env.get("MSA_BUDGET_MAX_MODEL_CALLS_PER_TURN") or 6),
                max_cost_usd=_f("MSA_BUDGET_MAX_COST_USD", 0.10),
                # A CPU-hosted local model needs minutes, not seconds, per call.
                turn_deadline_s=_f("MSA_BUDGET_TURN_DEADLINE_S", 3600.0 if local else 30.0),
            ),
            llm=LLMSettings(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_output_tokens=int(env.get("MSA_LLM_MAX_OUTPUT_TOKENS") or 1024),
                timeout_s=_f("MSA_LLM_TIMEOUT_S", 1200.0 if local else 120.0),
                price_in_per_mtok=_opt_f("MSA_LLM_PRICE_IN_PER_MTOK"),
                price_out_per_mtok=_opt_f("MSA_LLM_PRICE_OUT_PER_MTOK"),
            ),
            rep_keys=json.loads(env.get("MSA_REP_KEYS") or "{}"),
            log_hmac_key=env.get("MSA_LOG_HMAC_KEY") or "dev-only-key",
            http_timeout_s=_f("MSA_HTTP_TIMEOUT_S", 10.0),
        )
