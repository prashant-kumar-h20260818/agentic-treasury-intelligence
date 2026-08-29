from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired
from typing_extensions import TypedDict


class PortfolioState(TypedDict):
    tickers: list[str]
    weights: list[float]
    benchmark: str
    lookback: str
    risk_free_rate: float
    risk_profile: str
    investment_objective: str
    capital: float
    include_news: bool
    agent_log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    prices: NotRequired[Any]
    returns: NotRequired[Any]
    benchmark_returns: NotRequired[Any]
    latest_prices: NotRequired[dict[str, float]]
    news: NotRequired[list[dict[str, str]]]

    metrics: NotRequired[dict[str, Any]]
    asset_metrics: NotRequired[Any]
    correlation: NotRequired[Any]
    scenarios: NotRequired[Any]
    target_weights: NotRequired[dict[str, float]]
    rebalance_plan: NotRequired[list[dict[str, Any]]]
    compliance: NotRequired[dict[str, Any]]
    executive_summary: NotRequired[str]
    llm_used: NotRequired[bool]

    approval_status: NotRequired[str]
    action_result: NotRequired[dict[str, Any]]
