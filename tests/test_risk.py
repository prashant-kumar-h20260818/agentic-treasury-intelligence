import numpy as np
import pandas as pd

from src.risk import calculate_risk_metrics, historical_var_cvar, normalize_weights


def test_normalize_weights():
    w = normalize_weights([20, 30, 50])
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, [0.2, 0.3, 0.5])


def test_var_cvar_non_negative_loss_numbers():
    s = pd.Series([-0.05, -0.02, -0.01, 0.0, 0.01, 0.02] * 20)
    var, cvar = historical_var_cvar(s, 0.95)
    assert var >= 0
    assert cvar >= var


def test_calculate_risk_metrics_synthetic():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0005, 0.01, 300),
            "BBB": rng.normal(0.0003, 0.008, 300),
        },
        index=idx,
    )
    benchmark = pd.Series(rng.normal(0.0004, 0.009, 300), index=idx)
    metrics, assets, corr, portfolio = calculate_risk_metrics(
        returns, ["AAA", "BBB"], [0.6, 0.4], benchmark, 0.03
    )
    assert metrics["observations"] == 300
    assert np.isfinite(metrics["annualized_volatility"])
    assert set(assets.index) == {"AAA", "BBB"}
    assert corr.shape == (2, 2)
    assert len(portfolio) == 300
