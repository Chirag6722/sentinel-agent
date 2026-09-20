"""Runtime settings, loaded from .env / environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    db_path: Path = ROOT / os.getenv("DB_PATH", "data.db")

    # --- agent budget ---------------------------------------------------
    max_steps: int = 12            # LLM round-trips per run
    max_tool_calls: int = 20       # total tool invocations per run
    tool_retry_limit: int = 2      # retries on transient tool failure
    circuit_breaker_failures: int = 3  # failures of one tool before it is fused for the run

    # --- money guardrails (USD) ------------------------------------------
    refund_auto_limit: float = 50.0      # <= this: agent may refund on its own
    refund_approval_limit: float = 500.0 # <= this: human approval; above: hard deny
    refund_run_cap: float = 500.0        # cumulative refunds in one run

    # --- comms ------------------------------------------------------------
    internal_email_domain: str = "support.acme-shop.example"

    @property
    def provider_name(self) -> str:
        if self.llm_provider == "mock":
            return "mock"
        if self.llm_provider == "groq" or (self.llm_provider == "auto" and self.groq_api_key):
            return "groq"
        return "mock"


settings = Settings()
