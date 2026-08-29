from __future__ import annotations


LIMITS = {
    "Conservative": {"max_position": 0.25, "max_hhi": 0.22},
    "Balanced": {"max_position": 0.35, "max_hhi": 0.28},
    "Growth": {"max_position": 0.45, "max_hhi": 0.35},
}


def run_guardrails(
    target_weights: dict[str, float],
    risk_profile: str,
    metrics: dict,
    data_warnings: list[str],
) -> dict:
    limits = LIMITS.get(risk_profile, LIMITS["Balanced"])
    max_weight = max(target_weights.values()) if target_weights else 1.0
    hhi = float(sum(v * v for v in target_weights.values())) if target_weights else 1.0

    checks = [
        {
            "check": "Weights sum to 100%",
            "passed": abs(sum(target_weights.values()) - 1.0) < 1e-6,
            "detail": f"Total = {sum(target_weights.values()):.2%}",
        },
        {
            "check": "Single-position concentration",
            "passed": max_weight <= limits["max_position"] + 1e-9,
            "detail": f"Largest target = {max_weight:.2%}; limit = {limits['max_position']:.0%}",
        },
        {
            "check": "Portfolio concentration (HHI)",
            "passed": hhi <= limits["max_hhi"],
            "detail": f"Target HHI = {hhi:.3f}; threshold = {limits['max_hhi']:.2f}",
        },
        {
            "check": "Sufficient return history",
            "passed": int(metrics.get("observations", 0)) >= 60,
            "detail": f"Observations = {int(metrics.get('observations', 0))}",
        },
        {
            "check": "Market-data quality",
            "passed": len(data_warnings) == 0,
            "detail": "No warnings" if not data_warnings else f"{len(data_warnings)} warning(s)",
        },
    ]
    hard_failures = [c for c in checks[:3] if not c["passed"]]
    return {
        "passed": len(hard_failures) == 0,
        "checks": checks,
        "hard_failures": [c["check"] for c in hard_failures],
        "disclaimer": "Educational portfolio analytics only. No orders are sent to a broker or financial institution.",
    }
