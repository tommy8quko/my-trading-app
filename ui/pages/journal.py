"""Journal — Trade Log + Trade Replay tabs."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio


@st.cache_data(ttl=120)
def _load():
    return build_portfolio()


def render() -> None:
    _, closed = _load()

    tab_log, tab_replay = st.tabs(["Trade Log", "Trade Replay"])

    # ── Trade Log ─────────────────────────────────────────────────────────────
    with tab_log:
        if not closed:
            st.info("No closed trades yet.")
        else:
            currencies = sorted({t.currency for t in closed})
            if len(currencies) > 1:
                ccy = st.radio("Currency", currencies, horizontal=True)
                trades = [t for t in closed if t.currency == ccy]
            else:
                trades = closed

            rows = [{
                "Symbol": t.symbol,
                "Ccy": t.currency,
                "Direction": t.direction,
                "Entry": t.entry_date,
                "Exit": t.exit_date,
                "Qty": t.quantity,
                "Avg Entry": t.avg_entry,
                "Avg Exit": t.avg_exit,
                "Fees": round(t.fees, 2),
                "PnL": round(t.realized_pnl, 2),
                "R": f"{t.r_multiple:.2f}" if t.r_multiple is not None else "",
            } for t in sorted(trades, key=lambda x: x.exit_date, reverse=True)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Trade Replay ──────────────────────────────────────────────────────────
    with tab_replay:
        if not closed:
            st.info("No closed trades to replay.")
        else:
            symbols = sorted({t.symbol for t in closed})
            selected = st.selectbox("Filter by symbol", ["All"] + symbols)
            trades = closed if selected == "All" else [t for t in closed if t.symbol == selected]
            trades = sorted(trades, key=lambda t: t.exit_date, reverse=True)

            for t in trades:
                r_label = f"  R: **{t.r_multiple:+.2f}R**" if t.r_multiple is not None else ""
                with st.expander(
                    f"{t.exit_date}  |  {t.direction} {t.symbol}  |  "
                    f"PnL: **{t.realized_pnl:+,.0f} {t.currency}**{r_label}"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Entry", f"{t.avg_entry:.4f}")
                    c2.metric("Avg Exit", f"{t.avg_exit:.4f}")
                    c3.metric("Quantity", f"{t.quantity:,.0f}")
                    c4.metric("Fees", f"{t.fees:.2f}")

                    c5, c6, c7 = st.columns(3)
                    c5.metric("Entry Date", str(t.entry_date))
                    c6.metric("Exit Date", str(t.exit_date))
                    c7.metric("Holding Days", (t.exit_date - t.entry_date).days)

                    if t.initial_stop:
                        risk = abs(t.avg_entry - t.initial_stop)
                        st.caption(f"Initial Stop: {t.initial_stop:.4f}  |  Risk/share: {risk:.4f}")
