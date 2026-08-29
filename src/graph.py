from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .agents import (
    allocation_agent,
    approval_agent,
    executive_insight_agent,
    execute_proposal,
    failure_agent,
    governance_agent,
    market_data_agent,
    reject_proposal,
    risk_agent,
    scenario_agent,
    supervisor_agent,
)
from .state import PortfolioState


def _after_market_data(state: PortfolioState) -> str:
    return "risk" if state.get("returns") is not None else "failure"


def _after_risk(state: PortfolioState) -> str:
    return "scenarios" if state.get("metrics") is not None else "failure"


def _after_allocation(state: PortfolioState) -> str:
    return "governance" if state.get("target_weights") else "failure"


def build_graph():
    builder = StateGraph(PortfolioState)
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("market_data", market_data_agent)
    builder.add_node("risk", risk_agent)
    builder.add_node("scenarios", scenario_agent)
    builder.add_node("allocation", allocation_agent)
    builder.add_node("governance", governance_agent)
    builder.add_node("executive_insight", executive_insight_agent)
    builder.add_node("approval", approval_agent)
    builder.add_node("execute_proposal", execute_proposal)
    builder.add_node("reject_proposal", reject_proposal)
    builder.add_node("failure", failure_agent)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "market_data")
    builder.add_conditional_edges("market_data", _after_market_data, ["risk", "failure"])
    builder.add_conditional_edges("risk", _after_risk, ["scenarios", "failure"])
    builder.add_edge("scenarios", "allocation")
    builder.add_conditional_edges("allocation", _after_allocation, ["governance", "failure"])
    builder.add_edge("governance", "executive_insight")
    builder.add_edge("executive_insight", "approval")
    builder.add_edge("execute_proposal", END)
    builder.add_edge("reject_proposal", END)
    builder.add_edge("failure", END)

    # Pandas DataFrames are part of graph state. LangGraph documents pickle_fallback
    # for these unsupported msgpack types. This saver is process-local and receives
    # only application-generated state; use a hardened durable serializer in production.
    serde = JsonPlusSerializer(pickle_fallback=True)
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
