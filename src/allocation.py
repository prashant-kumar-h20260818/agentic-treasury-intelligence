from __future__ import annotations

import numpy as np
import pandas as pd


PROFILE_CAPS = {
    "Conservative": 0.25,
    "Balanced": 0.35,
    "Growth": 0.45,
}

PROFILE_BLEND = {
    "Conservative": 0.80,
    "Balanced": 0.60,
    "Growth": 0.40,
}


def _cap_and_renormalize(weights: pd.Series, cap: float, max_iter: int = 20) -> pd.Series:
    w = weights.clip(lower=0.0).astype(float)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    w /= w.sum()
    for _ in range(max_iter):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        room = (cap - w[under]).clip(lower=0.0)
        if room.sum() <= 0:
            break
        w.loc[under] += excess * (room / room.sum())
        w /= w.sum()
    return w / w.sum()


def propose_target_weights(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    risk_profile: str,
) -> dict[str, float]:
    cols = [c for c in returns.columns if c in current_weights]
    current = pd.Series({c: current_weights[c] for c in cols}, dtype=float)
    current /= current.sum()
    vols = returns[cols].std(ddof=1).replace(0, np.nan)
    inv_vol = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv_vol.sum() <= 0:
        inv_vol = pd.Series(1.0, index=cols)
    inv_vol /= inv_vol.sum()

    blend = PROFILE_BLEND.get(risk_profile, PROFILE_BLEND["Balanced"])
    target = blend * inv_vol + (1.0 - blend) * current
    cap = PROFILE_CAPS.get(risk_profile, PROFILE_CAPS["Balanced"])
    target = _cap_and_renormalize(target, cap=cap)
    return {k: float(v) for k, v in target.items()}


def build_rebalance_plan(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    capital: float,
    latest_prices: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    tickers = sorted(set(current_weights) | set(target_weights))
    for ticker in tickers:
        current = float(current_weights.get(ticker, 0.0))
        target = float(target_weights.get(ticker, 0.0))
        delta = target - current
        notional = delta * capital
        price = float(latest_prices.get(ticker, float("nan")))
        shares = notional / price if price and np.isfinite(price) else float("nan")
        if abs(delta) < 0.0025:
            action = "HOLD"
        elif delta > 0:
            action = "INCREASE"
        else:
            action = "REDUCE"
        rows.append(
            {
                "ticker": ticker,
                "current_weight": current,
                "target_weight": target,
                "delta_weight": delta,
                "action": action,
                "notional_change": float(notional),
                "estimated_shares": float(shares) if np.isfinite(shares) else None,
            }
        )
    return rows
