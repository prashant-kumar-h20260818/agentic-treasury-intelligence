from __future__ import annotations

from typing import Literal

import pandas as pd
from langgraph.types import Command, interrupt

from .allocation import build_rebalance_plan, propose_target_weights
from .compliance import run_guardrails
from .llm import generate_executive_summary
from .market_data import fetch_market_bundle
from .risk import calculate_risk_metrics
from .scenarios import run_stress_tests
from .state import PortfolioState


def supervisor_agent(state: PortfolioState) -> dict:
    return {
        "agent_log": [
            "Supervisor Agent: validated the request and launched the market-data, risk, stress, allocation, governance and executive-insight workflow."
        ]
    }


def market_data_agent(state: PortfolioState) -> dict:
    try:
        bundle = fetch_market_bundle(
            state["tickers"], state["benchmark"], state["lookback"], state.get("include_news", False)
        )
        return {
            "prices": bundle.prices,
            "returns": bundle.returns,
            "benchmark_returns": bundle.benchmark_returns,
            "latest_prices": bundle.latest_prices,
            "news": bundle.news,
            "errors": bundle.warnings,
            "agent_log": [
                f"Market Data Agent: downloaded and aligned history for {len(bundle.prices.columns)} portfolio asset(s)."
            ],
        }
    except Exception as exc:
        return {
            "errors": [f"Market Data Agent failed: {exc}"],
            "agent_log": ["Market Data Agent: unable to build a usable market-data bundle."],
        }


def risk_agent(state: PortfolioState) -> dict:
    if "returns" not in state or state["returns"] is None:
        return {"errors": ["Risk Agent skipped because market data was unavailable."]}
    try:
        metrics, asset_metrics, correlation, _ = calculate_risk_metrics(
            returns=state["returns"],
            tickers=state["tickers"],
            weights=state["weights"],
            benchmark_returns=state.get("benchmark_returns", pd.Series(dtype=float)),
            risk_free_rate=state["risk_free_rate"],
        )
        return {
            "metrics": metrics,
            "asset_metrics": asset_metrics,
            "correlation": correlation,
            "agent_log": [
                "Risk Agent: calculated return, volatility, Sharpe, Sortino, VaR/CVaR, drawdown, beta and concentration metrics."
            ],
        }
    except Exception as exc:
        return {"errors": [f"Risk Agent failed: {exc}"]}


def scenario_agent(state: PortfolioState) -> dict:
    if "returns" not in state or "metrics" not in state:
        return {"errors": ["Scenario Agent skipped because risk inputs were unavailable."]}
    try:
        current = dict(zip(state["tickers"], state["weights"]))
        scenarios = run_stress_tests(state["returns"], state.get("benchmark_returns"), current)
        return {
            "scenarios": scenarios,
            "agent_log": ["Scenario Agent: ran beta- and volatility-sensitive market stress scenarios."],
        }
    except Exception as exc:
        return {"errors": [f"Scenario Agent failed: {exc}"]}


def allocation_agent(state: PortfolioState) -> dict:
    if "returns" not in state:
        return {"errors": ["Allocation Agent skipped because return history was unavailable."]}
    try:
        current = dict(zip(state["tickers"], state["weights"]))
        target = propose_target_weights(state["returns"], current, state["risk_profile"])
        plan = build_rebalance_plan(
            current_weights=current,
            target_weights=target,
            capital=state["capital"],
            latest_prices=state.get("latest_prices", {}),
        )
        return {
            "target_weights": target,
            "rebalance_plan": plan,
            "agent_log": [
                "Allocation Agent: generated a capped inverse-volatility blend and a simulated rebalance proposal."
            ],
        }
    except Exception as exc:
        return {"errors": [f"Allocation Agent failed: {exc}"]}


def governance_agent(state: PortfolioState) -> dict:
    if "target_weights" not in state or "metrics" not in state:
        return {"errors": ["Governance Agent skipped because proposal inputs were unavailable."]}
    compliance = run_guardrails(
        target_weights=state["target_weights"],
        risk_profile=state["risk_profile"],
        metrics=state["metrics"],
        data_warnings=state.get("errors", []),
    )
    return {
        "compliance": compliance,
        "agent_log": [
            "Governance Agent: checked allocation totals, concentration limits, data sufficiency and market-data quality."
        ],
    }


def executive_insight_agent(state: PortfolioState) -> dict:
    scenarios = state.get("scenarios")
    worst_name, worst_impact = "n/a", None
    if scenarios is not None and not scenarios.empty:
        idx = scenarios["Portfolio Impact"].idxmin()
        worst_name = str(idx)
        worst_impact = float(scenarios.loc[idx, "Portfolio Impact"])

    context = {
        "risk_profile": state["risk_profile"],
        "investment_objective": state["investment_objective"],
        "metrics": state.get("metrics", {}),
        "worst_scenario": {"name": worst_name, "impact": worst_impact},
        "target_weights": state.get("target_weights", {}),
        "compliance": state.get("compliance", {}),
        "news_headlines": [n.get("title", "") for n in state.get("news", [])[:8]],
    }
    summary, llm_used = generate_executive_summary(context)
    return {
        "executive_summary": summary,
        "llm_used": llm_used,
        "agent_log": [
            "Executive Insight Agent: synthesized an evidence-grounded management brief"
            + (" using Gemini." if llm_used else " using deterministic fallback logic (no API key or model unavailable).")
        ],
    }


def failure_agent(state: PortfolioState) -> dict:
    errors = state.get("errors", [])
    detail = errors[-1] if errors else "The workflow could not produce the required analytical inputs."
    return {
        "approval_status": "FAILED",
        "executive_summary": f"Analysis stopped before human approval because a required agent failed. {detail}",
        "action_result": {
            "status": "failed",
            "message": "No proposal was produced and no action was taken.",
        },
        "agent_log": ["Supervisor Agent: terminated the workflow safely because required analytical state was unavailable."],
    }


def approval_agent(state: PortfolioState) -> Command[Literal["execute_proposal", "reject_proposal"]]:
    compliance = state.get("compliance", {})
    payload = {
        "question": "Approve the simulated rebalance proposal?",
        "risk_profile": state["risk_profile"],
        "guardrails_passed": bool(compliance.get("passed", False)),
        "hard_failures": compliance.get("hard_failures", []),
        "proposal": state.get("rebalance_plan", []),
        "note": "Approval only records a simulation decision. No broker or bank transaction is executed.",
    }
    approved = interrupt(payload)
    return Command(goto="execute_proposal" if bool(approved) else "reject_proposal")


def execute_proposal(state: PortfolioState) -> dict:
    return {
        "approval_status": "APPROVED",
        "action_result": {
            "status": "simulated",
            "message": "Proposal approved and recorded. No real trade was submitted.",
            "approved_plan": state.get("rebalance_plan", []),
        },
        "agent_log": ["Action Agent: recorded human approval for the simulated proposal; no external trade was executed."],
    }


def reject_proposal(state: PortfolioState) -> dict:
    return {
        "approval_status": "REJECTED",
        "action_result": {
            "status": "cancelled",
            "message": "Proposal rejected by the human approver. No action was taken.",
        },
        "agent_log": ["Action Agent: human approver rejected the simulated proposal."],
    }
