from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS = {
    "Market crash": {"market_shock": -0.20, "vol_multiplier": 1.35},
    "Mild recession": {"market_shock": -0.10, "vol_multiplier": 1.15},
    "Risk-off / rate shock": {"market_shock": -0.06, "vol_multiplier": 1.25},
    "Bull recovery": {"market_shock": 0.15, "vol_multiplier": 0.90},
}


def _asset_betas(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    if benchmark_returns is None or benchmark_returns.empty:
        return pd.Series(1.0, index=returns.columns)
    aligned = returns.join(benchmark_returns.rename("__benchmark__"), how="inner").dropna()
    if len(aligned) < 20 or aligned["__benchmark__"].var(ddof=1) == 0:
        return pd.Series(1.0, index=returns.columns)
    bvar = aligned["__benchmark__"].var(ddof=1)
    return pd.Series(
        {
            col: aligned[[col, "__benchmark__"]].cov().loc[col, "__benchmark__"] / bvar
            for col in returns.columns
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def run_stress_tests(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: dict[str, float],
) -> pd.DataFrame:
    cols = [c for c in returns.columns if c in weights]
    if not cols:
        raise ValueError("No overlapping assets available for stress testing.")
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    w = w / w.sum()
    betas = _asset_betas(returns[cols], benchmark_returns)
    vols = returns[cols].std(ddof=1) * np.sqrt(252)
    median_vol = float(vols.median()) if len(vols) else 0.2

    rows = []
    for name, spec in SCENARIOS.items():
        market_shock = spec["market_shock"]
        vol_multiplier = spec["vol_multiplier"]
        # CAPM-like market sensitivity plus a volatility-sensitive risk-off penalty/benefit.
        systematic = betas * market_shock
        vol_adjustment = -(vols - median_vol).clip(lower=0) * 0.20 if market_shock < 0 else 0.0
        asset_shocks = (systematic + vol_adjustment).clip(lower=-0.60, upper=0.50)
        portfolio_impact = float((asset_shocks * w).sum())
        rows.append(
            {
                "Scenario": name,
                "Portfolio Impact": portfolio_impact,
                "Implied Volatility Multiplier": vol_multiplier,
            }
        )
    return pd.DataFrame(rows).set_index("Scenario")
