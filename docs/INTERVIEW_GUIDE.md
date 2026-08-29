# Interview Guide

## 30-second project pitch

> I built an Agentic Treasury and Portfolio Risk Intelligence platform using LangGraph and Streamlit. Instead of relying on an LLM for financial calculations, I separated responsibilities across specialist agents: market data, quantitative risk, stress testing, allocation, governance and executive synthesis. The graph calculates VaR/CVaR, Sharpe/Sortino, drawdown, beta and concentration deterministically, then generates a governed allocation proposal. Before any action is recorded, LangGraph pauses using a real human-in-the-loop interrupt and resumes only after approval or rejection. I also added an audit trail, tests, Docker and CI so the project demonstrates both AI orchestration and production engineering.

## Why LangGraph?

Because this is a stateful workflow with explicit sequencing, governance and pause/resume requirements. A graph makes control flow, state transitions and human approval more explicit than a linear chain or one large prompt.

## Why not let Gemini calculate VaR?

Numerical risk should be deterministic, reproducible and testable. The LLM is used only for synthesis. This reduces hallucination risk and makes governance easier.

## What makes it agentic?

- specialized nodes with distinct responsibilities
- shared evolving state
- tool/data interaction
- decision workflow rather than one-shot generation
- persistent checkpoint boundary
- human-in-the-loop control
- action path after approval
- auditability

## What happens without Gemini?

The app still works. Market data, all calculations, stress tests, allocation, governance, HITL and audit are deterministic. Only the narrative summary falls back to a template.

## How is the allocation generated?

A transparent inverse-volatility allocation is blended with the existing portfolio and capped according to the selected risk profile. It is a proposal-generation heuristic, not a claim of optimality.

## How would you productionize it at HSBC?

- licensed bank-grade market data
- policy/risk-limit service APIs
- authenticated roles and maker-checker approval
- durable LangGraph persistence
- model gateway and approved-model registry
- PII/data-classification controls
- immutable audit logging
- operational monitoring and SLA metrics
- independent model-risk validation

## How would you adapt it for Vodafone Idea?

Keep the graph pattern but replace tools/agents:

`Alert → Network KPI Agent → Log Analysis Agent → RCA Agent → Remediation Agent → Governance → Human Approval → Action`

The same HITL and governance architecture applies to potentially disruptive network changes.

## How would you position it for Accenture Strategy?

The project demonstrates more than a chatbot. It shows:

- redesign of a decision workflow
- decomposition of work between agents and humans
- decision rights and controls
- explainable quantitative logic
- scalable operating-model thinking
- governance and auditability
- a reusable pattern across industries

## Likely technical questions

1. Why use a StateGraph rather than a simple function pipeline?
2. How does `interrupt()` preserve workflow state?
3. Why is `thread_id` necessary?
4. How would you replace `InMemorySaver` in production?
5. Why use historical VaR and what are its weaknesses?
6. What is the difference between VaR and CVaR?
7. How is beta estimated and when can it be unstable?
8. What are limitations of inverse-volatility allocation?
9. What happens with missing ticker data?
10. How would you backtest the recommendation logic without leakage?
11. How would you evaluate the LLM summary?
12. How would you secure external tool calls?
13. How would you add maker-checker authorization?
14. How would you detect market-data drift/errors?
15. What would you log for regulatory auditability?

## Improvement roadmap you can discuss

- add portfolio optimization with explicit constraints
- add Black-Litterman / risk-parity alternatives
- add factor exposure analysis
- add Monte Carlo simulation
- add portfolio backtesting
- add institutional market data
- add RAG over risk policy documents
- add MCP tools for enterprise systems
- add persistent memory and Postgres checkpointer
- add observability and evaluation dashboards
- add authenticated approval roles
