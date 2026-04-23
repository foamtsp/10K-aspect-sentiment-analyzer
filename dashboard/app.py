"""Streamlit dashboard — async pull-based, API-driven.

Flow:
    1. User types a ticker.
    2. Dashboard calls GET /tickers/{t}/filings to populate the dropdown.
    3. User picks a filing and hits Analyze (or Force refresh).
    4. Dashboard POSTs to trigger async scoring.
       - 200: already cached → render immediately.
       - 202: scoring started → poll GET every 5 s until 200.
"""
from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASPECTS = (
    "revenue", "cash_flow", "margins", "ebitda",
    "future_plans", "risk_factors", "guidance",
)
ASPECT_LABELS = {
    "revenue": "Revenue",
    "cash_flow": "Cash Flow",
    "margins": "Margins",
    "ebitda": "EBITDA",
    "future_plans": "Future Plans",
    "risk_factors": "Risk Factors",
    "guidance": "Guidance",
}

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = 5   # seconds between cache polls
POLL_TIMEOUT  = 180 # give up after 3 minutes

st.set_page_config(page_title="10-K Signal Dashboard", layout="wide")


def api_get(path: str, timeout: float = 10.0) -> httpx.Response:
    return httpx.get(f"{API_BASE}{path}", timeout=timeout)


def api_post(path: str, timeout: float = 35.0) -> httpx.Response:
    return httpx.post(f"{API_BASE}{path}", timeout=timeout)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_filings(ticker: str) -> list[dict]:
    r = api_get(f"/tickers/{ticker}/filings")
    r.raise_for_status()
    return r.json().get("filings", [])


def render_metrics(analysis: dict, cached: bool) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filing date", str(analysis["filing_date"]))
    prediction = (analysis.get("prediction") or "").upper()
    arrow = "UP ▲" if prediction == "UP" else "DOWN ▼" if prediction == "DOWN" else "—"
    m2.metric(f"{analysis.get('horizon_days', 30)}-day prediction", arrow)
    proba = analysis.get("probability_up")
    m3.metric("Model confidence", f"{float(proba):.1%}" if proba is not None else "—")
    m4.metric(
        "Model version",
        analysis.get("model_version") or "—",
        delta="cached" if cached else "fresh",
        delta_color="off",
    )


def render_radar(scores: dict[str, float]) -> None:
    labels = [ASPECT_LABELS[a] for a in ASPECTS]
    values = [float(scores.get(a, 0.0)) for a in ASPECTS]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="royalblue"),
            name="Aspect score",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-1, 1], tickfont=dict(size=10))
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def color_score(v: float) -> str:
    if v > 0.1:
        return "color: #2ecc71;"
    if v < -0.1:
        return "color: #e74c3c;"
    return "color: #bdc3c7;"


def render_table(scores: dict[str, float], deltas: dict[str, float | None]) -> None:
    rows = []
    for a in ASPECTS:
        d = deltas.get(a)
        rows.append({
            "Aspect": ASPECT_LABELS[a],
            "Score": float(scores.get(a, 0.0)),
            "YoY Delta": float(d) if d is not None else float("nan"),
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    def fmt_delta(v):
        return "n/a" if pd.isna(v) else f"{v:+.2f}"

    def color_delta(v):
        return "color: #7f8c8d;" if pd.isna(v) else color_score(v)

    styled = (
        df.style
        .map(color_score, subset=["Score"])
        .map(color_delta, subset=["YoY Delta"])
        .format({"Score": "{:+.2f}", "YoY Delta": fmt_delta})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_analysis(analysis: dict, cached: bool) -> None:
    render_metrics(analysis, cached)
    st.divider()
    viz_col, table_col = st.columns((3, 2))
    with viz_col:
        st.subheader("Aspect sentiment radar")
        render_radar(analysis.get("aspects", {}))
    with table_col:
        st.subheader("Aspect scores & YoY deltas")
        render_table(analysis.get("aspects", {}), analysis.get("deltas", {}))
    st.caption(
        f"Accession: `{analysis['accession']}`  ·  "
        f"n_sentences: {analysis.get('n_sentences', 0)}"
    )


def run_analysis(ticker: str, accession: str, force: bool) -> None:
    """POST to trigger, then poll until result is ready or timeout."""
    qs = f"?accession={accession}" + ("&refresh=1" if force else "")

    # --- trigger ---
    try:
        r = api_post(f"/tickers/{ticker}/analysis{qs}")
    except Exception as e:
        st.error(f"Request failed: {e}")
        return

    if r.status_code == 200:
        data = r.json()
        render_analysis(data["analysis"], cached=bool(data.get("cached")))
        return

    if r.status_code != 202:
        st.error(f"API error {r.status_code}: {r.text[:400]}")
        return

    # --- poll ---
    status_box = st.empty()
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        dots = "." * ((elapsed // POLL_INTERVAL) % 4 + 1)
        status_box.info(f"Scoring in progress{dots}  ({elapsed}s elapsed — typically 20–30s)")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        try:
            r = api_get(f"/tickers/{ticker}/analysis?accession={accession}")
        except Exception as e:
            status_box.error(f"Poll failed: {e}")
            return

        if r.status_code == 200:
            status_box.empty()
            data = r.json()
            render_analysis(data["analysis"], cached=bool(data.get("cached")))
            return

        if r.status_code != 202:
            status_box.error(f"API error {r.status_code}: {r.text[:400]}")
            return

    status_box.error(
        "Timed out waiting for analysis. "
        "Check CloudWatch logs for /aws/lambda/tenk-scorer."
    )


def sidebar() -> tuple[str, dict | None, bool, bool]:
    st.sidebar.title("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()

    filings: list[dict] = []
    error: str | None = None
    if ticker:
        try:
            filings = fetch_filings(ticker)
        except httpx.HTTPStatusError as e:
            error = f"{e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:  # noqa: BLE001
            error = str(e)

    if error:
        st.sidebar.error(f"Could not list filings: {error}")

    selected: dict | None = None
    if filings:
        labels = [f"{f['filing_date']} · {f['form']} · {f['accession']}" for f in filings]
        idx = st.sidebar.selectbox(
            "Select 10-K", range(len(filings)), format_func=lambda i: labels[i]
        )
        selected = filings[idx]
    elif ticker and not error:
        st.sidebar.info("No 10-K filings found for this ticker.")

    st.sidebar.divider()
    col1, col2 = st.sidebar.columns(2)
    analyze = col1.button(
        "Analyze", type="primary", use_container_width=True, disabled=selected is None
    )
    refresh = col2.button(
        "Force Refresh", use_container_width=True, disabled=selected is None
    )
    return ticker, selected, analyze, refresh


def main() -> None:
    ticker, selected, analyze, refresh = sidebar()
    st.title(f"{ticker or '—'} 10-K Analysis" if ticker else "10-K Analysis")

    if not (analyze or refresh) or not selected:
        st.info("Enter a ticker, pick a 10-K from the sidebar, then click **Analyze**.")
        return

    run_analysis(ticker, selected["accession"], force=refresh)


if __name__ == "__main__":
    main()
