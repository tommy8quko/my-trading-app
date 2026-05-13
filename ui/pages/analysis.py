"""Analysis — Statistics + AI Coach tabs."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio
from analytics.metrics import breakdown_by_direction, breakdown_by_dow


@st.cache_data(ttl=120)
def _load():
    return build_portfolio()


def render() -> None:
    _, closed = _load()

    tab_stats, tab_coach = st.tabs(["Statistics", "AI Coach"])

    with tab_stats:
        if not closed:
            st.info("No closed trades yet.")
        else:
            currencies = sorted({t.currency for t in closed})
            if len(currencies) > 1:
                ccy = st.radio("Currency", currencies, horizontal=True)
                trades = [t for t in closed if t.currency == ccy]
            else:
                trades = closed

            col_dir, col_dow = st.columns(2)

            with col_dir:
                st.subheader("Long vs Short")
                df_dir = (
                    pd.DataFrame(breakdown_by_direction(trades)).T
                    .reset_index().rename(columns={"index": "Direction"})
                )
                st.dataframe(df_dir, use_container_width=True, hide_index=True)

            with col_dow:
                st.subheader("Entry Day of Week")
                dow = breakdown_by_dow(trades)
                if dow:
                    df_dow = (
                        pd.DataFrame(dow).T
                        .reset_index().rename(columns={"index": "Day"})
                    )
                    st.dataframe(df_dow, use_container_width=True, hide_index=True)

    with tab_coach:
        from analytics.price_enrichment import enrich_all
        from analytics.simulator import run_simulation
        from analytics.prompt_builder import build_prompt

        if not closed:
            st.info("No closed trades yet — import and approve orders first.")
        else:
            st.caption("Generates a coaching prompt enriched with market data — paste into claude.ai.")

            with st.expander("Simulation settings", expanded=False):
                c1, c2 = st.columns(2)
                stop_min = c1.slider("Min stop %", 2, 8, 3)
                stop_max = c2.slider("Max stop %", 6, 20, 15)
                rr_max = c1.slider("Max R:R target", 2, 8, 4)
                trail_max = c2.slider("Max trailing stop %", 5, 20, 10)

            suggested = [
                "Is my stop loss too tight? Show me which trades I was shaken out of.",
                "Should I take profit earlier or hold longer? What's the data say?",
                "Am I entering too early? How much would I save by waiting 1-2 days?",
                "What's my biggest single improvement — show me the $ impact.",
            ]
            selected = st.selectbox("Quick questions", ["Custom..."] + suggested)
            question = st.text_area(
                "Your question",
                value="" if selected == "Custom..." else selected,
                height=80,
            )

            if st.button("Generate Coaching Prompt", type="primary"):
                with st.spinner("Fetching market data and running simulations…"):
                    stop_pcts = [s / 100 for s in range(stop_min, stop_max + 1)]
                    rr_targets = [r * 0.5 for r in range(1, rr_max * 2 + 1)]
                    trailing_pcts = [t / 100 for t in range(3, trail_max + 1, 2)]
                    enriched = enrich_all(closed)
                    if not enriched:
                        st.error("Could not fetch price data.")
                    else:
                        report = run_simulation(closed, enriched,
                                                stop_pcts=stop_pcts, rr_targets=rr_targets,
                                                trailing_pcts=trailing_pcts)
                        prompt = build_prompt(closed, report, user_question=question)
                        st.success("Done! Copy the prompt below and paste into claude.ai.")
                        st.text_area("Prompt", value=prompt, height=400, label_visibility="collapsed")
                        st.caption(f"~{len(prompt)//4:,} tokens · {len(prompt):,} chars")
