from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class MarketBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_returns: pd.Series
    latest_prices: dict[str, float]
    news: list[dict[str, str]]
    warnings: list[str]


def _clean_ticker(value: str) -> str:
    return value.strip().upper()


def _history_for_ticker(ticker: str, period: str) -> pd.Series:
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No price history returned for {ticker}")
    close = hist["Close"].dropna().astype(float)
    if close.empty:
        raise ValueError(f"No usable closing prices returned for {ticker}")
    close.name = ticker
    return close


def _extract_news(ticker: str, limit: int = 3) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return rows

    for item in items[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or content.get("headline") or "Untitled"
        provider_obj = content.get("provider") or {}
        provider = (
            provider_obj.get("displayName")
            if isinstance(provider_obj, dict)
            else str(provider_obj or "")
        )
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(canonical, dict):
            url = canonical.get("url", "")
        else:
            url = str(canonical or "")
        rows.append({"ticker": ticker, "title": str(title), "provider": str(provider), "url": str(url)})
    return rows


def fetch_market_bundle(
    tickers: Iterable[str],
    benchmark: str,
    period: str,
    include_news: bool = False,
) -> MarketBundle:
    clean = list(dict.fromkeys(_clean_ticker(t) for t in tickers if str(t).strip()))
    benchmark = _clean_ticker(benchmark)
    warnings: list[str] = []
    series: list[pd.Series] = []

    for ticker in clean:
        try:
            series.append(_history_for_ticker(ticker, period))
        except Exception as exc:
            warnings.append(f"{ticker}: {exc}")

    if not series:
        raise ValueError("No valid portfolio price history could be downloaded.")

    prices = pd.concat(series, axis=1).sort_index().ffill().dropna(how="all")
    # Keep only assets that have enough common observations for portfolio math.
    min_obs = min(30, max(5, int(len(prices) * 0.3)))
    prices = prices.dropna(axis=1, thresh=min_obs).ffill().dropna()
    if prices.empty or len(prices.columns) == 0:
        raise ValueError("Not enough overlapping market data across the selected assets.")

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Could not calculate portfolio returns from downloaded data.")

    try:
        benchmark_prices = _history_for_ticker(benchmark, period)
        benchmark_returns = benchmark_prices.pct_change(fill_method=None).dropna()
    except Exception as exc:
        warnings.append(f"Benchmark {benchmark}: {exc}")
        benchmark_returns = pd.Series(dtype=float, name=benchmark)

    latest_prices = {col: float(prices[col].dropna().iloc[-1]) for col in prices.columns}

    news: list[dict[str, str]] = []
    if include_news:
        for ticker in list(prices.columns)[:6]:
            news.extend(_extract_news(ticker, limit=2))

    return MarketBundle(
        prices=prices,
        returns=returns,
        benchmark_returns=benchmark_returns,
        latest_prices=latest_prices,
        news=news,
        warnings=warnings,
    )
