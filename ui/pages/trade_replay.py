"""Trade replay / journal page — view individual closed trades in detail."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio, ClosedTrade


@st.cache_data(ttl=120)
def _load():
    return build_portfolio()


def render() -> None:
    st.header("Trade Replay")

    _, closed = _load()
    if not closed:
        st.info("No closed trades to replay.")
        return

    symbols = sorted({t.symbol for t in closed})
    selected_symbol = st.selectbox("Filter by symbol (optional)", ["All"] + symbols)

    trades = closed if selected_symbol == "All" else [t for t in closed if t.symbol == selected_symbol]
    trades = sorted(trades, key=lambda t: t.exit_date, reverse=True)

    for t in trades:
        r_label = f"  R: **{t.r_multiple:+.2f}R**" if t.r_multiple is not None else ""
        pnl_color = "green" if t.realized_pnl >= 0 else "red"
        with st.expander(
            f"{t.exit_date}  |  {t.direction} {t.symbol}  |  "
            f"PnL: **{t.realized_pnl:+,.0f} {t.currency}**{r_label}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Entry", f"{t.avg_entry:.4f}")
            c2.metric("Avg Exit", f"{t.avg_exit:.4f}")
            c3.metric("Quantity", f"{t.quantity}")
            c4.metric("Fees", f"{t.fees:.2f}")

            col5, col6, col7 = st.columns(3)
            col5.metric("Entry Date", str(t.entry_date))
            col6.metric("Exit Date", str(t.exit_date))
            hold = (t.exit_date - t.entry_date).days
            col7.metric("Holding Days", hold)

            if t.initial_stop:
                risk_per_share = abs(t.avg_entry - t.initial_stop)
                st.write(f"**Initial Stop:** {t.initial_stop:.4f}  |  Risk/share: {risk_per_share:.4f}")
