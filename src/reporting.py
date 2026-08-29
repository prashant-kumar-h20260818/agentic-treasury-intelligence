from __future__ import annotations

import json
from datetime import datetime, timezone


def build_audit_record(state: dict, thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": state.get("tickers", []),
        "weights": state.get("weights", []),
        "risk_profile": state.get("risk_profile"),
        "metrics": state.get("metrics", {}),
        "target_weights": state.get("target_weights", {}),
        "compliance": state.get("compliance", {}),
        "approval_status": state.get("approval_status", "PENDING"),
        "action_result": state.get("action_result", {}),
        "agent_log": state.get("agent_log", []),
        "errors": state.get("errors", []),
    }


def audit_json(state: dict, thread_id: str) -> str:
    return json.dumps(build_audit_record(state, thread_id), indent=2, default=str)
