from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    google_model: str
    default_benchmark: str
    max_assets: int
    trading_days: int


def get_settings() -> Settings:
    """Read settings at call time so Streamlit secrets/env changes are respected."""
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        google_model=os.getenv("GOOGLE_MODEL", "gemini-3.7-flash"),
        default_benchmark=os.getenv("BENCHMARK_TICKER", "SPY"),
        max_assets=int(os.getenv("MAX_ASSETS", "12")),
        trading_days=int(os.getenv("TRADING_DAYS", "252")),
    )
