from src.llm import fallback_summary


def test_fallback_summary_uses_supplied_facts():
    text = fallback_summary(
        {
            "metrics": {
                "annualized_return": 0.10,
                "annualized_volatility": 0.15,
                "max_drawdown": -0.20,
            },
            "worst_scenario": {"name": "Market crash", "impact": -0.18},
            "compliance": {"passed": True},
        }
    )
    assert "10.0%" in text
    assert "Market crash" in text
    assert "simulated" in text
