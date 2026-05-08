"""AI Coach page — generate a simulation-enriched prompt to paste into Claude desktop."""
from __future__ import annotations
import streamlit as st
from analytics.portfolio import build_portfolio
from analytics.price_enrichment import enrich_all
from analytics.simulator import run_simulation
from analytics.prompt_builder import build_prompt


@st.cache_data(ttl=300, show_spinner=False)
def _build_report():
    """Fetch trades, enrich with price data, run simulations. Cached 5 min."""
    _, closed = build_portfolio()
    if not closed:
        return None, None
    enriched = enrich_all(closed)
    report = run_simulation(closed, enriched)
    return closed, report


def render() -> None:
    st.header("AI Coach")
    st.caption(
        "Generates a detailed coaching prompt — enriched with actual market data — "
        "ready to paste into Claude desktop (claude.ai)."
    )

    _, closed = build_portfolio()
    if not closed:
        st.info("No closed trades yet — import and approve orders first.")
        return

    # ── Simulation controls ───────────────────────────────────────────────────
    with st.expander("Simulation settings (optional)", expanded=False):
        col1, col2 = st.columns(2)
        stop_min = col1.slider("Min stop %", 2, 8, 3)
        stop_max = col2.slider("Max stop %", 6, 20, 15)
        rr_max = col1.slider("Max R:R target", 2, 8, 4)
        trail_max = col2.slider("Max trailing stop %", 5, 20, 10)

    # ── Custom question ───────────────────────────────────────────────────────
    st.subheader("Your question for Claude")
    suggested = [
        "Is my stop loss too tight? Show me which trades I was shaken out of.",
        "Should I take profit earlier or hold longer? What's the data say?",
        "Am I entering too early? How much would I save by waiting 1-2 days?",
        "Should I sell into strength or weakness? What's been working for me?",
        "What's my biggest single improvement — show me the $ impact.",
    ]
    selected = st.selectbox("Quick questions", ["Custom..."] + suggested)
    if selected == "Custom...":
        question = st.text_area("Type your question", height=80)
    else:
        question = st.text_area("Type your question", value=selected, height=80)

    # ── Generate ──────────────────────────────────────────────────────────────
    if st.button("Generate Coaching Prompt", type="primary"):
        with st.spinner("Fetching market data and running simulations... (30-60 seconds)"):
            stop_pcts = [s / 100 for s in range(stop_min, stop_max + 1)]
            rr_targets = [r * 0.5 for r in range(1, rr_max * 2 + 1)]
            trailing_pcts = [t / 100 for t in range(3, trail_max + 1, 2)]

            enriched = enrich_all(closed)
            if not enriched:
                st.error("Could not fetch price data. Check your internet connection.")
                return

            report = run_simulation(
                closed, enriched,
                stop_pcts=stop_pcts,
                rr_targets=rr_targets,
                trailing_pcts=trailing_pcts,
            )
            prompt = build_prompt(closed, report, user_question=question)

        st.success("Done! Copy the prompt below and paste it into Claude desktop.")

        # ── Copy box ──────────────────────────────────────────────────────────
        st.text_area(
            "Coaching prompt (select all → copy)",
            value=prompt,
            height=500,
            label_visibility="collapsed",
        )

        char_count = len(prompt)
        token_est = char_count // 4
        st.caption(
            f"~{token_est:,} tokens · {char_count:,} characters  |  "
            "Paste into **claude.ai** → Projects → your trading journal project"
        )

        # ── Quick stats preview ───────────────────────────────────────────────
        st.divider()
        st.subheader("Simulation preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Trades analysed", len(report.stop_replays))
        stopped = sum(1 for r in report.stop_replays if r.was_stopped_out)
        col2.metric("Stop-outs replayed", stopped)
        early = sum(1 for r in report.entry_timing if r.better_entry_available)
        col3.metric("Early entries found", early)

        # Best stop improvement
        best_stop_gain = 0.0
        for r in report.stop_replays:
            if r.was_stopped_out:
                for s in r.scenarios:
                    if not s['would_be_stopped'] and s['pnl_difference'] > best_stop_gain:
                        best_stop_gain = s['pnl_difference']

        # Biggest left-on-table trade
        most_left = max(
            (r for r in report.take_profit if r.mfe_pnl > r.actual_pnl),
            key=lambda r: r.mfe_pnl - r.actual_pnl,
            default=None,
        )

        if best_stop_gain > 0:
            st.metric(
                "Best stop adjustment (single trade)",
                f"+{best_stop_gain:,.0f}",
                help="Max PnL improvement from changing stop on one stopped-out trade",
            )
        if most_left:
            left = most_left.mfe_pnl - most_left.actual_pnl
            st.metric(
                f"Most left on table — {most_left.trade.symbol}",
                f"{left:,.0f}",
                help="Difference between your actual exit and the MFE on this trade",
            )
