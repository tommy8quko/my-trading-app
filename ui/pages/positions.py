"""Open positions page."""
from __future__ import annotations
from datetime import date
import pandas as pd
import streamlit as st
from analytics.portfolio import build_portfolio
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
    """Return account_id and instrument_id from an existing order."""
    result = (
        get_supabase().table("orders")
        .select("account_id, instrument_id")
        .eq("id", order_id)
        .single()
        .execute()
    )
    return result.data


def _insert_order(account_id: str, instrument_id: str, action: str,
                  order_date: date, price: float, quantity: float,
                  fees: float, currency: str, stop_loss: float | None) -> None:
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
    st.header("Open Positions")

    open_pos, _ = _load()

    if not open_pos:
        st.info("No open positions.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for p in sorted(open_pos, key=lambda x: x.symbol):
        rows.append({
            "Symbol": p.symbol,
            "Exchange": p.exchange,
            "Currency": p.currency,
            "Direction": p.direction,
            "Qty": p.total_quantity,
            "Avg Entry": round(p.avg_entry_price, 4),
            "Stop Loss": p.initial_stop_loss or "",
            "Entry Date": p.earliest_entry_date,
            "Lots": len(p.lots),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Edit panel ────────────────────────────────────────────────────────────
    st.subheader("Edit Position")

    sorted_pos = sorted(open_pos, key=lambda x: x.symbol)
    selected_symbol = st.selectbox("Select position", [p.symbol for p in sorted_pos])
    pos = next(p for p in sorted_pos if p.symbol == selected_symbol)

    # Stop loss shortcut — apply same value to every lot at once
    st.caption("Set stop loss across all lots:")
    col1, col2 = st.columns([3, 1])
    with col1:
        current_sl = pos.initial_stop_loss or 0.0
        new_sl = st.number_input(
            "Stop loss price", value=float(current_sl),
            min_value=0.0, step=0.01, format="%.4f",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("Apply to all lots", use_container_width=True):
            for lot in pos.lots:
                _save_lot(lot.order_id, lot.action, lot.quantity, new_sl)
            st.cache_data.clear()
            st.success("Stop loss updated.")
            st.rerun()

    # Lot-level editor
    st.caption("Edit individual lots — change Qty or Stop Loss, then click Save:")
    lot_data = [{
        "order_id": l.order_id,
        "Date": str(l.order_date),
        "Action": l.action,
        "Qty": l.quantity,
        "Entry Price": l.entry_price,
        "Stop Loss": l.stop_loss_at_entry or 0.0,
    } for l in pos.lots]

    edited_df = st.data_editor(
        pd.DataFrame(lot_data),
        column_config={
            "Action": st.column_config.SelectboxColumn(
                "Action", options=["BUY", "SELL", "SHORT", "COVER"], required=True
            ),
            "order_id": st.column_config.TextColumn("order_id", disabled=True),
            "Date": st.column_config.TextColumn("Date", disabled=True),
            "Entry Price": st.column_config.NumberColumn("Entry Price", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key=f"lot_editor_{selected_symbol}",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save lot changes", type="primary", use_container_width=True):
            for _, row in edited_df.iterrows():
                _save_lot(row["order_id"], row["Action"], float(row["Qty"]), float(row["Stop Loss"]))
            st.cache_data.clear()
            st.success("Saved.")
            st.rerun()

    with col2:
        if st.button("🗑️ Delete entire position", type="secondary", use_container_width=True):
            for lot in pos.lots:
                _delete_order(lot.order_id)
            st.cache_data.clear()
            st.warning(f"{selected_symbol} deleted.")
            st.rerun()

    # Delete a single lot
    if len(pos.lots) > 1:
        st.caption("Delete one lot:")
        lot_labels = [f"{l.order_date}  ×{l.quantity}  @{l.entry_price}" for l in pos.lots]
        del_idx = st.selectbox("Lot to delete", range(len(lot_labels)),
                               format_func=lambda i: lot_labels[i],
                               label_visibility="collapsed")
        if st.button("Delete this lot"):
            _delete_order(pos.lots[del_idx].order_id)
            st.cache_data.clear()
            st.warning("Lot deleted.")
            st.rerun()

    st.divider()

    # ── Add order ─────────────────────────────────────────────────────────────
    with st.expander("➕ Add order to this position"):
        st.caption(f"Manually record a trade for **{selected_symbol}** (e.g. a SELL you forgot to import)")
        with st.form(key=f"add_order_{selected_symbol}"):
            c1, c2 = st.columns(2)
            with c1:
                add_action = st.selectbox("Action", ["SELL", "BUY", "COVER", "SHORT"])
                add_date = st.date_input("Trade date", value=date.today())
                add_qty = st.number_input("Quantity (shares)", min_value=0.0, step=1.0, format="%.0f")
            with c2:
                add_price = st.number_input("Price", min_value=0.0, step=0.01, format="%.4f")
                add_fees = st.number_input("Fees", min_value=0.0, step=0.01, format="%.2f", value=0.0)
                add_sl = st.number_input("Stop loss (optional)", min_value=0.0, step=0.01, format="%.4f", value=0.0)

            submitted = st.form_submit_button("Add order", type="primary", use_container_width=True)
            if submitted:
                if add_qty <= 0 or add_price <= 0:
                    st.error("Quantity and price must be greater than 0.")
                else:
                    meta = _fetch_order_meta(pos.lots[0].order_id)
                    _insert_order(
                        account_id=meta["account_id"],
                        instrument_id=meta["instrument_id"],
                        action=add_action,
                        order_date=add_date,
                        price=add_price,
                        quantity=add_qty,
                        fees=add_fees,
                        currency=pos.currency,
                        stop_loss=add_sl,
                    )
                    st.cache_data.clear()
                    st.success(f"{add_action} {add_qty} {selected_symbol} @ {add_price} added.")
                    st.rerun()
