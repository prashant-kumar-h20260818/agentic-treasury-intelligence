from __future__ import annotations

import math
import uuid
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from src.config import get_settings
from src.graph import build_graph
from src.reporting import audit_json
from src.risk import cumulative_growth

load_dotenv()

st.set_page_config(
    page_title="Agentic Treasury Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Community Cloud exposes secrets through st.secrets rather than a local .env file.
# Copy supported keys into the process environment without ever printing them.
for _key in ["GOOGLE_API_KEY", "GOOGLE_MODEL", "BENCHMARK_TICKER", "MAX_ASSETS", "TRADING_DAYS"]:
    try:
        if _key in st.secrets and st.secrets[_key]:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass
SETTINGS = get_settings()


@st.cache_resource
def get_graph():
    return build_graph()


def sample_portfolio() -> pd.DataFrame:
    return pd.read_csv("data/sample_portfolio.csv")


def fmt_pct(value, digits: int = 1) -> str:
    try:
        value = float(value)
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}%}"
    except Exception:
        return "n/a"


def fmt_num(value, digits: int = 2) -> str:
    try:
        value = float(value)
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    except Exception:
        return "n/a"


def normalized_input(df: pd.DataFrame) -> tuple[list[str], list[float]]:
    cleaned = df.copy()
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip().str.upper()
    cleaned["weight"] = pd.to_numeric(cleaned["weight"], errors="coerce").fillna(0.0)
    cleaned = cleaned[(cleaned["ticker"] != "") & (cleaned["weight"] > 0)]
    cleaned = cleaned.groupby("ticker", as_index=False)["weight"].sum()
    if cleaned.empty:
        raise ValueError("Add at least one ticker with a positive weight.")
    if (cleaned["weight"] < 0).any():
        raise ValueError("Negative portfolio weights are not supported in this demo.")
    total = float(cleaned["weight"].sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to more than zero.")
    cleaned["weight"] = cleaned["weight"] / total
    if len(cleaned) > SETTINGS.max_assets:
        raise ValueError(f"Use at most {SETTINGS.max_assets} assets for this interactive demo.")
    return cleaned["ticker"].tolist(), cleaned["weight"].astype(float).tolist()


def interrupt_payload(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    try:
        return interrupts[0].value
    except Exception:
        return {"question": "Human approval required.", "details": str(interrupts)}


def current_weights_df(state: dict) -> pd.DataFrame:
    return pd.DataFrame({"Ticker": state.get("tickers", []), "Current": state.get("weights", [])})


def allocation_comparison(state: dict) -> pd.DataFrame:
    current = dict(zip(state.get("tickers", []), state.get("weights", [])))
    target = state.get("target_weights", {})
    tickers = sorted(set(current) | set(target))
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Current": [current.get(t, 0.0) for t in tickers],
            "Target": [target.get(t, 0.0) for t in tickers],
        }
    )


def portfolio_growth(state: dict) -> pd.DataFrame:
    returns = state.get("returns")
    if returns is None or returns.empty:
        return pd.DataFrame()
    current = dict(zip(state.get("tickers", []), state.get("weights", [])))
    cols = [c for c in returns.columns if c in current]
    if not cols:
        return pd.DataFrame()
    w = pd.Series({c: current[c] for c in cols}, dtype=float)
    w /= w.sum()
    p = returns[cols].mul(w, axis=1).sum(axis=1)
    growth = cumulative_growth(p)
    out = growth.rename("Portfolio").to_frame()
    benchmark = state.get("benchmark_returns")
    if benchmark is not None and not benchmark.empty:
        out[state.get("benchmark", "Benchmark")] = cumulative_growth(benchmark).reindex(out.index).ffill()
    return out.dropna(how="all")


if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = sample_portfolio()
if "analysis_state" not in st.session_state:
    st.session_state.analysis_state = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

st.title("🏦 Agentic Treasury & Portfolio Risk Intelligence Platform")
st.caption(
    "A LangGraph + Streamlit multi-agent workflow for market analytics, risk, stress testing, allocation governance and human-approved simulated actions."
)

with st.sidebar:
    st.header("Analysis Controls")
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Growth"], index=1)
    lookback = st.selectbox("Market-data lookback", ["6mo", "1y", "2y", "5y"], index=1)
    benchmark = st.text_input("Benchmark ticker", value=SETTINGS.default_benchmark).strip().upper() or "SPY"
    risk_free_rate_pct = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.25)
    capital = st.number_input("Illustrative portfolio value", min_value=1000.0, value=100000.0, step=5000.0)
    investment_objective = st.text_area(
        "Investment objective",
        value="Balance long-term capital growth with controlled drawdown and diversified risk exposure.",
        height=100,
    )
    include_news = st.toggle("Collect recent ticker headlines", value=False)

    st.divider()
    st.subheader("Optional Gemini layer")
    if SETTINGS.google_api_key:
        st.success(f"Gemini enabled: {SETTINGS.google_model}")
    else:
        st.info("No GOOGLE_API_KEY detected. The app remains fully functional and uses a deterministic executive-summary fallback.")

    st.divider()
    uploaded = st.file_uploader("Load portfolio CSV", type=["csv"])
    if uploaded is not None and st.button("Use uploaded portfolio", use_container_width=True):
        try:
            loaded = pd.read_csv(uploaded)
            rename_map = {c.lower(): c for c in loaded.columns}
            if "ticker" not in rename_map or "weight" not in rename_map:
                raise ValueError("CSV must contain ticker and weight columns.")
            st.session_state.portfolio_df = loaded[[rename_map["ticker"], rename_map["weight"]]].rename(
                columns={rename_map["ticker"]: "ticker", rename_map["weight"]: "weight"}
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Reset sample portfolio", use_container_width=True):
        st.session_state.portfolio_df = sample_portfolio()
        st.session_state.analysis_state = None
        st.session_state.thread_id = None
        st.rerun()

st.subheader("1. Portfolio Input")
st.write("Edit tickers and weights below. Weights can be entered as percentages; the app normalizes them automatically.")
editor = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", help="Examples: MSFT, JPM, RELIANCE.NS"),
        "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, step=1.0, format="%.2f"),
    },
    key="portfolio_editor",
)
st.session_state.portfolio_df = editor

run_col, info_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("🚀 Run Agentic Analysis", type="primary", use_container_width=True)
with info_col:
    st.caption("The graph pauses before the simulated allocation action. You must explicitly approve or reject it.")

if run_clicked:
    try:
        tickers, weights = normalized_input(editor)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "tickers": tickers,
            "weights": weights,
            "benchmark": benchmark,
            "lookback": lookback,
            "risk_free_rate": risk_free_rate_pct / 100.0,
            "risk_profile": risk_profile,
            "investment_objective": investment_objective.strip(),
            "capital": float(capital),
            "include_news": bool(include_news),
            "agent_log": [],
            "errors": [],
        }
        with st.spinner("Agents are collecting data, measuring risk, running stress tests and preparing a governed proposal..."):
            result = get_graph().invoke(initial_state, config=config)
        st.session_state.analysis_state = result
        st.session_state.thread_id = thread_id
        st.rerun()
    except Exception as exc:
        st.error(f"Could not start analysis: {exc}")

state = st.session_state.analysis_state
thread_id = st.session_state.thread_id

if not state:
    st.info("Run the agentic analysis to populate the dashboard.")
    st.stop()

warnings = state.get("errors", [])
if warnings:
    with st.expander(f"⚠️ Data/agent warnings ({len(warnings)})", expanded=False):
        for item in warnings:
            st.warning(item)

payload = interrupt_payload(state)
if payload:
    st.warning("⏸️ LangGraph is paused at the Human Approval Agent. Review the proposal below and approve or reject it.")
elif state.get("approval_status") == "APPROVED":
    st.success("✅ Simulated proposal approved and recorded. No broker trade was executed.")
elif state.get("approval_status") == "REJECTED":
    st.error("❌ Simulated proposal rejected. No action was taken.")

metrics = state.get("metrics", {})

st.subheader("2. Executive Risk Dashboard")
card_cols = st.columns(6)
card_cols[0].metric("Annual Return", fmt_pct(metrics.get("annualized_return")))
card_cols[1].metric("Annual Volatility", fmt_pct(metrics.get("annualized_volatility")))
card_cols[2].metric("Sharpe", fmt_num(metrics.get("sharpe_ratio")))
card_cols[3].metric("Daily VaR 95%", fmt_pct(metrics.get("daily_var_95")))
card_cols[4].metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))
card_cols[5].metric("Beta", fmt_num(metrics.get("beta")))

summary = state.get("executive_summary", "")
if summary:
    label = "Gemini Executive Insight Agent" if state.get("llm_used") else "Executive Insight Agent (deterministic fallback)"
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.write(summary)

overview_tab, risk_tab, stress_tab, agents_tab, approval_tab, audit_tab = st.tabs(
    ["📈 Performance", "🧮 Risk", "🧪 Stress Tests", "🤖 Agent Workflow", "✅ Human Approval", "🧾 Audit"]
)

with overview_tab:
    growth = portfolio_growth(state)
    if not growth.empty:
        fig = px.line(growth, x=growth.index, y=growth.columns, title="Growth of $1 over the selected lookback")
        fig.update_layout(yaxis_title="Growth multiple", xaxis_title="Date", legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)

    alloc = allocation_comparison(state)
    if not alloc.empty:
        melted = alloc.melt(id_vars="Ticker", var_name="Allocation", value_name="Weight")
        fig = px.bar(melted, x="Ticker", y="Weight", color="Allocation", barmode="group", title="Current vs proposed target allocation")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if state.get("news"):
        st.markdown("#### Recent market headlines")
        for item in state["news"]:
            title = item.get("title", "Untitled")
            provider = item.get("provider", "")
            url = item.get("url", "")
            if url:
                st.markdown(f"- **{item.get('ticker', '')}** — [{title}]({url}) {f'— {provider}' if provider else ''}")
            else:
                st.markdown(f"- **{item.get('ticker', '')}** — {title} {f'— {provider}' if provider else ''}")

with risk_tab:
    left, right = st.columns([1, 1])
    with left:
        asset_metrics = state.get("asset_metrics")
        if asset_metrics is not None:
            display = asset_metrics.copy()
            for col in ["Weight", "Annual Return", "Annual Volatility", "Daily VaR 95%"]:
                if col in display:
                    display[col] = display[col].map(lambda x: f"{x:.2%}")
            st.markdown("#### Asset-level risk profile")
            st.dataframe(display, use_container_width=True)
    with right:
        corr = state.get("correlation")
        if corr is not None and not corr.empty:
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, title="Return correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

    detail = pd.DataFrame(
        {
            "Metric": ["Sortino Ratio", "Daily CVaR 95%", "HHI", "Effective Positions", "Average Correlation", "Observations"],
            "Value": [
                fmt_num(metrics.get("sortino_ratio")),
                fmt_pct(metrics.get("daily_cvar_95")),
                fmt_num(metrics.get("hhi"), 3),
                fmt_num(metrics.get("effective_number_of_positions"), 1),
                fmt_num(metrics.get("average_correlation"), 2),
                str(metrics.get("observations", "n/a")),
            ],
        }
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)

with stress_tab:
    scenarios = state.get("scenarios")
    if scenarios is not None and not scenarios.empty:
        scenario_display = scenarios.reset_index()
        fig = px.bar(
            scenario_display,
            x="Scenario",
            y="Portfolio Impact",
            text_auto=".1%",
            title="Modeled portfolio impact under stress scenarios",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        table = scenarios.copy()
        table["Portfolio Impact"] = table["Portfolio Impact"].map(lambda x: f"{x:.2%}")
        table["Implied Volatility Multiplier"] = table["Implied Volatility Multiplier"].map(lambda x: f"{x:.2f}x")
        st.dataframe(table, use_container_width=True)
        st.caption("Scenario estimates are transparent analytical proxies, not forecasts. They use market-beta and volatility sensitivity from the selected historical window.")

with agents_tab:
    st.markdown("#### LangGraph execution trace")
    for idx, log in enumerate(state.get("agent_log", []), start=1):
        st.markdown(f"**{idx}.** {log}")
    st.markdown("#### Workflow")
    st.code(
        "Supervisor → Market Data → Risk → Scenario → Allocation → Governance → Executive Insight → HUMAN INTERRUPT → Action/Audit",
        language="text",
    )

with approval_tab:
    compliance = state.get("compliance", {})
    st.markdown("#### Governance checks")
    if compliance:
        checks = pd.DataFrame(compliance.get("checks", []))
        if not checks.empty:
            checks["Status"] = checks["passed"].map(lambda x: "PASS" if x else "REVIEW")
            st.dataframe(checks[["check", "Status", "detail"]], hide_index=True, use_container_width=True)
        if compliance.get("passed"):
            st.success("Core allocation guardrails passed.")
        else:
            st.warning("One or more core guardrails require review before approving the simulation.")

    plan = state.get("rebalance_plan", [])
    if plan:
        plan_df = pd.DataFrame(plan)
        show = plan_df.copy()
        for col in ["current_weight", "target_weight", "delta_weight"]:
            show[col] = show[col].map(lambda x: f"{x:.2%}")
        show["notional_change"] = show["notional_change"].map(lambda x: f"{x:,.2f}")
        show["estimated_shares"] = show["estimated_shares"].map(lambda x: "n/a" if x is None else f"{x:,.2f}")
        st.markdown("#### Simulated rebalance proposal")
        st.dataframe(show, hide_index=True, use_container_width=True)

    if payload:
        st.info(payload.get("note", "Human decision required."))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve simulated proposal", type="primary", use_container_width=True):
                config = {"configurable": {"thread_id": thread_id}}
                with st.spinner("Resuming LangGraph from the approval checkpoint..."):
                    result = get_graph().invoke(Command(resume=True), config=config)
                st.session_state.analysis_state = result
                st.rerun()
        with c2:
            if st.button("❌ Reject proposal", use_container_width=True):
                config = {"configurable": {"thread_id": thread_id}}
                with st.spinner("Recording rejection and closing the workflow..."):
                    result = get_graph().invoke(Command(resume=False), config=config)
                st.session_state.analysis_state = result
                st.rerun()
    else:
        action = state.get("action_result", {})
        if action:
            st.write(action.get("message", "Workflow complete."))

with audit_tab:
    st.markdown("#### Reproducible audit record")
    audit = audit_json(state, thread_id or "unknown")
    st.download_button(
        "Download audit JSON",
        data=audit,
        file_name=f"agentic_treasury_audit_{thread_id or 'run'}.json",
        mime="application/json",
    )
    with st.expander("Preview audit JSON"):
        st.code(audit, language="json")

st.divider()
st.caption(
    "Educational and portfolio-demonstration software. Market data may be delayed or incomplete. The system does not connect to a broker, place orders, or provide individualized investment advice."
)
