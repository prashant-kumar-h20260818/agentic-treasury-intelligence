from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("Weights must be a non-empty 1-D sequence.")
    if np.any(arr < 0):
        raise ValueError("Negative weights are not supported in this portfolio demo.")
    total = arr.sum()
    if total <= 0:
        raise ValueError("Portfolio weights must sum to more than zero.")
    return arr / total


def max_drawdown(return_series: pd.Series) -> float:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else float("nan")


def annualized_return(return_series: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    clean = return_series.dropna()
    if clean.empty:
        return float("nan")
    cumulative = float((1.0 + clean).prod())
    years = len(clean) / trading_days
    if cumulative <= 0 or years <= 0:
        return float("nan")
    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(return_series: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    return float(return_series.dropna().std(ddof=1) * math.sqrt(trading_days))


def historical_var_cvar(return_series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    clean = return_series.dropna()
    if clean.empty:
        return float("nan"), float("nan")
    q = float(clean.quantile(1.0 - confidence))
    tail = clean[clean <= q]
    var = max(0.0, -q)
    cvar = max(0.0, -float(tail.mean())) if not tail.empty else var
    return var, cvar


def beta_to_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if benchmark_returns is None or benchmark_returns.empty:
        return float("nan")
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 20 or aligned["benchmark"].var(ddof=1) == 0:
        return float("nan")
    return float(aligned.cov().loc["portfolio", "benchmark"] / aligned["benchmark"].var(ddof=1))


def calculate_risk_metrics(
    returns: pd.DataFrame,
    tickers: list[str],
    weights: list[float],
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    available = [t for t in tickers if t in returns.columns]
    if not available:
        raise ValueError("None of the requested tickers have usable returns.")

    original = dict(zip(tickers, weights))
    w = normalize_weights([original.get(t, 0.0) for t in available])
    aligned_returns = returns[available].dropna()
    portfolio_returns = aligned_returns.mul(w, axis=1).sum(axis=1)

    ann_ret = annualized_return(portfolio_returns)
    ann_vol = annualized_volatility(portfolio_returns)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol and np.isfinite(ann_vol) else float("nan")

    downside = portfolio_returns[portfolio_returns < 0]
    downside_vol = float(downside.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(downside) > 1 else float("nan")
    sortino = (ann_ret - risk_free_rate) / downside_vol if downside_vol and np.isfinite(downside_vol) else float("nan")

    var95, cvar95 = historical_var_cvar(portfolio_returns, confidence=0.95)
    mdd = max_drawdown(portfolio_returns)
    beta = beta_to_benchmark(portfolio_returns, benchmark_returns)
    hhi = float(np.square(w).sum())
    effective_names = float(1.0 / hhi) if hhi > 0 else float("nan")

    correlation = aligned_returns.corr()
    corr_values = correlation.to_numpy()
    if len(available) > 1:
        avg_corr = float((corr_values.sum() - len(available)) / (len(available) * (len(available) - 1)))
    else:
        avg_corr = float("nan")

    metrics = {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "daily_var_95": var95,
        "daily_cvar_95": cvar95,
        "max_drawdown": mdd,
        "beta": beta,
        "hhi": hhi,
        "effective_number_of_positions": effective_names,
        "average_correlation": avg_corr,
        "observations": int(len(portfolio_returns)),
    }

    asset_rows = []
    for ticker, weight in zip(available, w):
        s = aligned_returns[ticker]
        asset_rows.append(
            {
                "Ticker": ticker,
                "Weight": float(weight),
                "Annual Return": annualized_return(s),
                "Annual Volatility": annualized_volatility(s),
                "Daily VaR 95%": historical_var_cvar(s, 0.95)[0],
            }
        )
    asset_metrics = pd.DataFrame(asset_rows).set_index("Ticker")
    return metrics, asset_metrics, correlation, portfolio_returns


def cumulative_growth(return_series: pd.Series) -> pd.Series:
    return (1.0 + return_series.fillna(0.0)).cumprod()
