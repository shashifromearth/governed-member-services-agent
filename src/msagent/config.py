"""Runtime configuration.

Everything environment-specific (pack location, service URLs, credentials, model provider,
budgets) is resolved here and nowhere else, from environment variables with an optional
``.env`` file. Secrets are only ever read through :meth:`Settings.secret` so that the
production swap to a secret manager is a one-function change.
"""
from __future__ import annotations

import json
import os
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


@dataclass(frozen=True)
class Budgets:
    """Per-conversation limits enforced by the gateway (see gateway/budget.py)."""

    max_steps: int = 40  # model calls + tool calls over the whole conversation
    max_model_calls_per_turn: int = 6
    max_cost_usd: float = 0.10  # hard stop; the NFR-7 *average* target is $0.03
    turn_deadline_s: float = 30.0  # wall clock per rep turn


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "anthropic"  # anthropic | openai (OpenAI-compatible: OpenAI, Azure, Ollama, vLLM)
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

        api_base = env.get("MSA_API_BASE_URL", "http://localhost:8090").rstrip("/")
        provider = env.get("MSA_LLM_PROVIDER", "anthropic").lower()
        base_url = env.get("MSA_LLM_BASE_URL") or None
        if provider == "ollama":  # convenience alias for the OpenAI-compatible endpoint
            provider = "openai"
            base_url = base_url or "http://localhost:11434/v1"
        default_model = "claude-haiku-5-5" if provider == "anthropic" else "gpt-4o-mini"
        api_key = env.get("MSA_LLM_API_KEY") or (
            env.get("ANTHROPIC_API_KEY") if provider == "anthropic" else env.get("OPENAI_API_KEY")
        )
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
                turn_deadline_s=_f("MSA_BUDGET_TURN_DEADLINE_S", 30.0),
            ),
            llm=LLMSettings(
                provider=provider,
                model=env.get("MSA_LLM_MODEL") or default_model,
                base_url=base_url,
                api_key=api_key or None,
                max_output_tokens=int(env.get("MSA_LLM_MAX_OUTPUT_TOKENS") or 1024),
                timeout_s=_f("MSA_LLM_TIMEOUT_S", 120.0),
                price_in_per_mtok=_opt_f("MSA_LLM_PRICE_IN_PER_MTOK"),
                price_out_per_mtok=_opt_f("MSA_LLM_PRICE_OUT_PER_MTOK"),
            ),
            rep_keys=json.loads(env.get("MSA_REP_KEYS") or "{}"),
            log_hmac_key=env.get("MSA_LOG_HMAC_KEY") or "dev-only-key",
            http_timeout_s=_f("MSA_HTTP_TIMEOUT_S", 10.0),
        )
