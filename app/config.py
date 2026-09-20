"""Runtime settings, loaded from policy.yaml then overridden by env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_POLICY_FILE = ROOT / "policy.yaml"


def _load_yaml() -> dict[str, Any]:
    try:
        import yaml  # pyyaml; optional at import time so tests without it still work
        with open(_POLICY_FILE) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        return {}


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _build_settings() -> "Settings":
    p = _load_yaml()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        groq_fallback_models=tuple(
            m.strip() for m in os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",") if m.strip()),
        db_path=ROOT / os.getenv("DB_PATH", "data.db"),
        max_steps=int(os.getenv("MAX_STEPS", _get(p, "agent", "max_steps", default=12))),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", _get(p, "agent", "max_tool_calls", default=20))),
        tool_retry_limit=int(os.getenv("TOOL_RETRY_LIMIT", _get(p, "agent", "tool_retry_limit", default=2))),
        circuit_breaker_failures=int(os.getenv("CIRCUIT_BREAKER_FAILURES", _get(p, "agent", "circuit_breaker_failures", default=3))),
        refund_auto_limit=float(os.getenv("REFUND_AUTO_LIMIT", _get(p, "money", "refund_auto_limit", default=50.0))),
        refund_approval_limit=float(os.getenv("REFUND_APPROVAL_LIMIT", _get(p, "money", "refund_approval_limit", default=500.0))),
        refund_run_cap=float(os.getenv("REFUND_RUN_CAP", _get(p, "money", "refund_run_cap", default=500.0))),
        internal_email_domain=os.getenv("INTERNAL_EMAIL_DOMAIN",
                                        _get(p, "comms", "internal_email_domain", default="support.acme-shop.example")),
    )


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_provider: str = "auto"
    groq_fallback_models: tuple[str, ...] = ()
    db_path: Path = ROOT / "data.db"

    # --- agent budget ---------------------------------------------------
    max_steps: int = 12
    max_tool_calls: int = 20
    tool_retry_limit: int = 2
    circuit_breaker_failures: int = 3

    # --- money guardrails (USD) ------------------------------------------
    refund_auto_limit: float = 50.0
    refund_approval_limit: float = 500.0
    refund_run_cap: float = 500.0

    # --- comms ------------------------------------------------------------
    internal_email_domain: str = "support.acme-shop.example"

    @property
    def provider_name(self) -> str:
        if self.llm_provider == "mock":
            return "mock"
        if self.llm_provider == "groq" or (self.llm_provider == "auto" and self.groq_api_key):
            return "groq"
        return "mock"


settings = _build_settings()


def reload_settings() -> None:
    """Reload settings from policy.yaml; env vars still override."""
    global settings
    settings = _build_settings()
