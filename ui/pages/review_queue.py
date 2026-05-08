"""Review queue page — approve or reject staged imports."""
from __future__ import annotations
import json
import pandas as pd
import streamlit as st
from ingestion.review_queue import fetch_pending, approve_item, reject_item


def render() -> None:
    st.header("Review Queue")

    pending = fetch_pending(limit=200)

    if not pending:
        st.success("No pending items — all clear.")
        return

    st.info(f"{len(pending)} order(s) awaiting review.")

    col_approve_all, col_reject_all, _ = st.columns([1, 1, 4])
    if col_approve_all.button("Approve All", type="primary"):
        for row in pending:
            approve_item(row["id"])
        st.success(f"Approved {len(pending)} orders.")
        st.rerun()

    if col_reject_all.button("Reject All"):
        for row in pending:
            reject_item(row["id"])
        st.warning(f"Rejected {len(pending)} orders.")
        st.rerun()

    st.divider()

    for row in pending:
        norm = row.get("normalized_json") or {}
        raw = row.get("raw_parsed_json") or {}
        symbol = raw.get("symbol", "?")
        action = raw.get("action") or norm.get("action", "?")
        quantity = raw.get("quantity") or norm.get("quantity", "?")
        broker = raw.get("broker_short_code", "?")
        source = raw.get("source_type", "?")
        confidence = row.get("confidence_score") or 0
        with st.expander(
            f"**{action} {quantity} {symbol}**"
            f"  —  {broker}  |  {source}  "
            f"|  confidence {confidence:.0%}",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            c1.write("**Normalized order**")
            c1.json(norm)
            c2.write("**Raw parsed**")
            c2.json(row.get("raw_parsed_json") or {})

            notes = st.text_input("Notes (optional)", key=f"notes_{row['id']}")
            bc1, bc2, _ = st.columns([1, 1, 4])
            if bc1.button("Approve", key=f"approve_{row['id']}", type="primary"):
                approve_item(row["id"], notes=notes)
                st.success("Approved")
                st.rerun()
            if bc2.button("Reject", key=f"reject_{row['id']}"):
                reject_item(row["id"], notes=notes)
                st.warning("Rejected")
                st.rerun()
