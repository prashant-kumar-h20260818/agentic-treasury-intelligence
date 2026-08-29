import numpy as np
import pandas as pd

from src.allocation import build_rebalance_plan, propose_target_weights
from src.compliance import run_guardrails
from src.scenarios import run_stress_tests


def synthetic_returns():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=260, freq="B")
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0005, 0.012, 260),
            "BBB": rng.normal(0.0003, 0.008, 260),
            "CCC": rng.normal(0.0004, 0.010, 260),
            "DDD": rng.normal(0.0002, 0.007, 260),
        },
        index=idx,
    )
    benchmark = pd.Series(rng.normal(0.0004, 0.01, 260), index=idx)
    return returns, benchmark


def test_target_weights_sum_to_one():
    returns, _ = synthetic_returns()
    current = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    target = propose_target_weights(returns, current, "Balanced")
    assert np.isclose(sum(target.values()), 1.0)
    assert max(target.values()) <= 0.35 + 1e-8


def test_stress_tests_shape():
    returns, benchmark = synthetic_returns()
    current = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    result = run_stress_tests(returns, benchmark, current)
    assert len(result) == 4
    assert "Portfolio Impact" in result.columns


def test_rebalance_and_guardrails():
    current = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    target = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    plan = build_rebalance_plan(current, target, 100000, {k: 100 for k in current})
    assert len(plan) == 4
    guardrails = run_guardrails(target, "Balanced", {"observations": 250}, [])
    assert guardrails["passed"] is True
