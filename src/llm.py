from __future__ import annotations

import json
from typing import Any

from .config import get_settings


def _pct(x: Any) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "n/a"


def fallback_summary(context: dict[str, Any]) -> str:
    m = context.get("metrics", {})
    worst = context.get("worst_scenario", {})
    compliance = context.get("compliance", {})
    status = "passed" if compliance.get("passed") else "requires review"
    return (
        f"The portfolio's historical annualized return is {_pct(m.get('annualized_return'))} with "
        f"annualized volatility of {_pct(m.get('annualized_volatility'))} and a maximum drawdown of "
        f"{_pct(m.get('max_drawdown'))}. The most adverse modeled scenario is "
        f"{worst.get('name', 'n/a')} at approximately {_pct(worst.get('impact'))}. "
        f"The proposed allocation guardrails {status}. Review the target-weight changes, assumptions, "
        "and stress-test limitations before approving the simulated rebalance plan."
    )


def generate_executive_summary(context: dict[str, Any]) -> tuple[str, bool]:
    settings = get_settings()
    if not settings.google_api_key:
        return fallback_summary(context), False

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
            max_retries=2,
        )
        prompt = """
You are the Executive Insight Agent in an educational treasury and portfolio-risk application.
Use ONLY the supplied JSON facts. Do not invent prices, returns, news, or forecasts.
Do not give individualized financial advice and do not tell the user to buy or sell a security.
Write a concise executive brief with: (1) portfolio condition, (2) key risk, (3) stress-test insight,
(4) what the proposed rebalance is trying to improve, and (5) what the human approver should verify.
Keep it under 180 words and explicitly call the rebalance a simulated proposal.

FACTS:
""" + json.dumps(context, default=str, indent=2)
        response = model.invoke(prompt)
        text = getattr(response, "text", None)
        if callable(text):
            text = text()
        if not text:
            content = getattr(response, "content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
        text = str(text or "").strip()
        if not text:
            return fallback_summary(context), False
        return text, True
    except Exception:
        return fallback_summary(context), False
