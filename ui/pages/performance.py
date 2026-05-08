"""Performance dashboard page."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio, ClosedTrade
from analytics.metrics import summary_dict, cumulative_pnl, breakdown_by_direction, breakdown_by_dow


@st.cache_data(ttl=120)
def _load() -> tuple[list, list]:
    return build_portfolio()


def render() -> None:
    st.header("Performance")

    _, closed = _load()

    if not closed:
        st.info("No closed trades yet. Import orders to get started.")
        return

    # ── Currency filter ───────────────────────────────────────────────────────
    currencies = sorted({t.currency for t in closed})
    if len(currencies) > 1:
        selected_currency = st.radio(
            "Currency", currencies, horizontal=True,
            help="Metrics and PnL are in the selected currency only — mixing USD and HKD would be meaningless.",
        )
        closed = [t for t in closed if t.currency == selected_currency]
    else:
        selected_currency = currencies[0] if currencies else "USD"

    # ── Summary metrics ───────────────────────────────────────────────────────
    m = summary_dict(closed)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", m["total_trades"])
    col2.metric("Win Rate", f"{m['win_rate']:.1%}")
    col3.metric("Profit Factor", f"{m['profit_factor']:.2f}")
    col4.metric(
        "Expectancy (R)",
        f"{m['expectancy_r']:.2f}R" if m["expectancy_r"] is not None else "N/A",
    )

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total PnL", f"{m['total_pnl']:,.0f}")
    col6.metric("Avg Win", f"{m['avg_win']:,.0f}")
    col7.metric("Avg Loss", f"{m['avg_loss']:,.0f}")
    col8.metric("Max Drawdown", f"{m['max_drawdown']:,.0f}")

    st.divider()

    # ── Equity curve ─────────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    cum_data = cumulative_pnl(closed)
    if cum_data:
        df_equity = pd.DataFrame(cum_data, columns=["date", "cumulative_pnl"])
        st.line_chart(df_equity.set_index("date")["cumulative_pnl"])

    # ── Direction breakdown ───────────────────────────────────────────────────
    st.subheader("Long vs Short")
    dir_data = breakdown_by_direction(closed)
    df_dir = pd.DataFrame(dir_data).T.reset_index().rename(columns={"index": "Direction"})
    st.dataframe(df_dir, use_container_width=True, hide_index=True)

    # ── Day of week ───────────────────────────────────────────────────────────
    st.subheader("Entry Day of Week")
    dow_data = breakdown_by_dow(closed)
    if dow_data:
        df_dow = pd.DataFrame(dow_data).T.reset_index().rename(columns={"index": "Day"})
        st.dataframe(df_dow, use_container_width=True, hide_index=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader("Closed Trades")
    rows = [
        {
            "Symbol": t.symbol,
            "Ccy": t.currency,
            "Direction": t.direction,
            "Entry": t.entry_date,
            "Exit": t.exit_date,
            "Qty": t.quantity,
            "Avg Entry": t.avg_entry,
            "Avg Exit": t.avg_exit,
            "Fees": t.fees,
            "PnL": round(t.realized_pnl, 2),
            "R": f"{t.r_multiple:.2f}" if t.r_multiple is not None else "",
        }
        for t in sorted(closed, key=lambda x: x.exit_date, reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
