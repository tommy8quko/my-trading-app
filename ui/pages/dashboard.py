"""Dashboard — KPI strip + equity curve + open positions."""
from __future__ import annotations
from datetime import date
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio
from analytics.metrics import summary_dict, cumulative_pnl
from ingestion.review_queue import fetch_pending
from db.client import get_supabase


@st.cache_data(ttl=60)
def _load():
    return build_portfolio()


def _save_lot(order_id: str, action: str, qty: float, stop_loss: float | None) -> None:
    get_supabase().table("orders").update({
        "action": action,
        "quantity": qty,
        "stop_loss_at_entry": stop_loss if stop_loss and stop_loss > 0 else None,
    }).eq("id", order_id).execute()


def _delete_order(order_id: str) -> None:
    get_supabase().table("orders").delete().eq("id", order_id).execute()


def _fetch_order_meta(order_id: str) -> dict:
    return get_supabase().table("orders").select("account_id, instrument_id").eq("id", order_id).single().execute().data


def _insert_order(account_id, instrument_id, action, order_date, price, quantity, fees, currency, stop_loss) -> None:
    get_supabase().table("orders").insert({
        "account_id": account_id,
        "instrument_id": instrument_id,
        "action": action,
        "order_date": order_date.isoformat(),
        "order_time": None,
        "price": price,
        "quantity": quantity,
        "fees": fees,
        "currency": currency,
        "stop_loss_at_entry": stop_loss if stop_loss and stop_loss > 0 else None,
        "external_order_id": f"MANUAL_{action}_{order_date}_{quantity}_{price}",
    }).execute()


def render() -> None:
    open_pos, closed = _load()

    # ── Currency toggle ───────────────────────────────────────────────────────
    currencies = sorted({t.currency for t in closed}) if closed else ["USD"]
    col_ccy, _ = st.columns([2, 8])
    with col_ccy:
        currency = st.radio("Currency", currencies, horizontal=True, label_visibility="collapsed") if len(currencies) > 1 else currencies[0]

    filtered = [t for t in closed if t.currency == currency]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    m = summary_dict(filtered) if filtered else {}
    pending_count = len(fetch_pending(limit=200))

    k = st.columns(8)
    k[0].metric("Total PnL", f"{m.get('total_pnl', 0):,.0f} {currency}")
    k[1].metric("Win Rate", f"{m.get('win_rate', 0):.1%}")
    k[2].metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
    k[3].metric("Expectancy", f"{m.get('expectancy_r') or 0:.2f}R")
    k[4].metric("Avg Win", f"{m.get('avg_win', 0):,.0f}")
    k[5].metric("Avg Loss", f"{m.get('avg_loss', 0):,.0f}")
    k[6].metric("Open Positions", len(open_pos))
    k[7].metric("Pending Review", pending_count)

    # ── Equity curve ──────────────────────────────────────────────────────────
    cum = cumulative_pnl(filtered)
    if cum:
        df_eq = pd.DataFrame(cum, columns=["date", "PnL"])
        st.line_chart(df_eq.set_index("date")["PnL"], height=180)

    # ── Open positions table ──────────────────────────────────────────────────
    if not open_pos:
        st.info("No open positions.")
        return

    sorted_pos = sorted(open_pos, key=lambda x: x.symbol)
    rows = [{
        "Symbol": p.symbol,
        "Ccy": p.currency,
        "Direction": p.direction,
        "Qty": p.total_quantity,
        "Avg Entry": round(p.avg_entry_price, 4),
        "Stop Loss": p.initial_stop_loss or "",
        "Entry Date": p.earliest_entry_date,
        "Lots": len(p.lots),
    } for p in sorted_pos]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Position editing (collapsed by default) ───────────────────────────────
    with st.expander("Edit / add orders"):
        selected_symbol = st.selectbox("Position", [p.symbol for p in sorted_pos])
        pos = next(p for p in sorted_pos if p.symbol == selected_symbol)

        # Stop loss shortcut
        c1, c2 = st.columns([3, 1])
        with c1:
            new_sl = st.number_input("Set stop loss (all lots)", value=float(pos.initial_stop_loss or 0),
                                     min_value=0.0, step=0.01, format="%.4f", label_visibility="collapsed")
        with c2:
            if st.button("Apply to all lots", use_container_width=True):
                for lot in pos.lots:
                    _save_lot(lot.order_id, lot.action, lot.quantity, new_sl)
                st.cache_data.clear()
                st.success("Updated.")
                st.rerun()

        # Lot editor
        lot_data = [{"order_id": l.order_id, "Date": str(l.order_date), "Action": l.action,
                     "Qty": l.quantity, "Entry Price": l.entry_price, "Stop Loss": l.stop_loss_at_entry or 0.0}
                    for l in pos.lots]
        edited_df = st.data_editor(
            pd.DataFrame(lot_data),
            column_config={
                "Action": st.column_config.SelectboxColumn("Action", options=["BUY", "SELL", "SHORT", "COVER"], required=True),
                "order_id": st.column_config.TextColumn("order_id", disabled=True),
                "Date": st.column_config.TextColumn("Date", disabled=True),
                "Entry Price": st.column_config.NumberColumn("Entry Price", disabled=True),
            },
            use_container_width=True, hide_index=True, key=f"lot_editor_{selected_symbol}",
        )

        ca, cb, cc = st.columns(3)
        if ca.button("💾 Save lot changes", type="primary", use_container_width=True):
            for _, row in edited_df.iterrows():
                _save_lot(row["order_id"], row["Action"], float(row["Qty"]), float(row["Stop Loss"]))
            st.cache_data.clear()
            st.success("Saved.")
            st.rerun()
        if cb.button("🗑️ Delete position", use_container_width=True):
            for lot in pos.lots:
                _delete_order(lot.order_id)
            st.cache_data.clear()
            st.rerun()

        # Add order
        st.caption(f"Add order to {selected_symbol}:")
        with st.form(key=f"add_{selected_symbol}"):
            f1, f2 = st.columns(2)
            add_action = f1.selectbox("Action", ["SELL", "BUY", "COVER", "SHORT"])
            add_date = f1.date_input("Date", value=date.today())
            add_qty = f1.number_input("Qty", min_value=0.0, step=1.0, format="%.0f")
            add_price = f2.number_input("Price", min_value=0.0, step=0.01, format="%.4f")
            add_fees = f2.number_input("Fees", min_value=0.0, step=0.01, value=0.0)
            add_sl = f2.number_input("Stop loss", min_value=0.0, step=0.01, value=0.0)
            if st.form_submit_button("Add order", type="primary", use_container_width=True):
                if add_qty > 0 and add_price > 0:
                    meta = _fetch_order_meta(pos.lots[0].order_id)
                    _insert_order(meta["account_id"], meta["instrument_id"], add_action,
                                  add_date, add_price, add_qty, add_fees, pos.currency, add_sl)
                    st.cache_data.clear()
                    st.success(f"Added {add_action} {add_qty} @ {add_price}")
                    st.rerun()
                else:
                    st.error("Qty and price must be > 0")
