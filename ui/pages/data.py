"""Data — Import & Sync + Diagnostics + Review Queue tabs."""
from __future__ import annotations
from datetime import date, timedelta
import streamlit as st
from ingestion.pipeline import ingest_file
from ingestion.review_queue import fetch_pending, approve_item, reject_item

_BROKERS = ["IBKR", "CHIEF"]


def render() -> None:
    tab_import, tab_diag, tab_queue = st.tabs(["Import & Sync", "Diagnostics", "Review Queue"])

    # ── Import & Sync ─────────────────────────────────────────────────────────
    with tab_import:
        st.subheader("Sync from Gmail")
        since = st.date_input("Fetch emails since", value=date.today())
        if st.button("Sync Gmail", type="primary"):
            with st.spinner("Connecting to Gmail..."):
                try:
                    from email_connector.sync import sync_emails
                    result = sync_emails(since=since)
                    st.success(f"Done — {result.queued} new order(s) queued, {result.duplicates} duplicate(s) skipped.")
                    st.info(
                        f"**Pipeline breakdown:**  \n"
                        f"Already processed: {result.skipped_already_processed}  \n"
                        f"No broker adapter: {result.skipped_no_adapter}  \n"
                        f"No order found: {result.skipped_no_order}  \n"
                        f"No account: {result.skipped_no_account}  \n"
                        f"Errors: {len(result.errors)}"
                    )
                    if result.errors:
                        for e in result.errors[:5]:
                            st.code(e)
                    senders = getattr(result, "senders_seen", [])
                    if senders:
                        with st.expander(f"📬 {len(senders)} emails scanned"):
                            for s in sorted(set(senders)):
                                st.code(s)
                except Exception as exc:
                    st.error(f"Gmail sync failed: {exc}")

        st.divider()

        st.subheader("Upload CSV / XLSX")
        broker = st.selectbox("Broker", _BROKERS)
        uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded and st.button("Import File"):
            with st.spinner("Parsing..."):
                result = ingest_file(uploaded.read(), uploaded.name, broker)
            if result.errors:
                for e in result.errors:
                    st.error(e)
            else:
                st.success(f"Done — {result.queued} order(s) queued, {result.duplicates} duplicate(s) skipped.")
                if result.queued:
                    st.info("Switch to the Review Queue tab to approve them.")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with tab_diag:
        st.subheader("Preview broker email subjects")
        diag_since = st.date_input("Since", value=date.today() - timedelta(days=60), key="diag_since")
        if st.button("Show broker email subjects"):
            with st.spinner("Fetching..."):
                try:
                    import pandas as pd
                    from email_connector.connector import GmailConnector
                    from email_connector.sync import BROKER_SENDER_FILTERS
                    rows = []
                    with GmailConnector() as gmail:
                        for sf in BROKER_SENDER_FILTERS:
                            for raw in gmail.fetch_emails(since=diag_since, sender_filter=sf, max_results=50):
                                rows.append({
                                    "from": raw["from_address"],
                                    "subject": raw["subject"],
                                    "date": str(raw["received_at"].date()),
                                    "body_preview": raw["body"][:120].replace("\n", " "),
                                })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.warning("No emails found.")
                except Exception as exc:
                    st.error(str(exc))

        st.divider()

        st.subheader("Inspect one Chief order email")
        if st.button("Show Chief email (parsed text + regex check)"):
            with st.spinner("Fetching..."):
                try:
                    from email_connector.connector import GmailConnector
                    from ingestion.brokers.chief import _strip_html, _INTRO_RE, _parse_kv
                    with GmailConnector() as gmail:
                        for raw in gmail.fetch_emails(
                            since=date.today() - timedelta(days=60),
                            sender_filter="cs@chiefgroup.com.hk",
                            max_results=10,
                        ):
                            if "Fully" in raw["subject"] or "成交" in raw["subject"]:
                                st.write(f"**From:** {raw['from_address']}")
                                st.write(f"**Subject:** {raw['subject']}")
                                body = raw["body"]
                                is_html = "<html" in body.lower() or "<!doctype" in body.lower()
                                stripped = _strip_html(body) if is_html else body
                                st.code(stripped[:1500])
                                intro = _INTRO_RE.search(stripped)
                                st.write(f"**Intro match:** {intro.groups() if intro else 'NO MATCH'}")
                                st.write(f"**KV pairs:** {_parse_kv(stripped)}")
                                break
                except Exception as exc:
                    st.error(str(exc))

    # ── Review Queue ──────────────────────────────────────────────────────────
    with tab_queue:
        pending = fetch_pending(limit=200)
        if not pending:
            st.success("No pending items — all clear.")
        else:
            st.info(f"{len(pending)} order(s) awaiting review.")
            ca, cb, _ = st.columns([1, 1, 4])
            if ca.button("Approve All", type="primary"):
                for row in pending:
                    approve_item(row["id"])
                st.success(f"Approved {len(pending)} orders.")
                st.rerun()
            if cb.button("Reject All"):
                for row in pending:
                    reject_item(row["id"])
                st.warning(f"Rejected {len(pending)} orders.")
                st.rerun()

            st.divider()
            for row in pending:
                raw = row.get("raw_parsed_json") or {}
                norm = row.get("normalized_json") or {}
                symbol = raw.get("symbol", "?")
                action = raw.get("action") or norm.get("action", "?")
                quantity = raw.get("quantity") or norm.get("quantity", "?")
                broker = raw.get("broker_short_code", "?")
                confidence = row.get("confidence_score") or 0
                with st.expander(f"**{action} {quantity} {symbol}**  —  {broker}  |  {confidence:.0%}"):
                    c1, c2 = st.columns(2)
                    c1.write("**Normalized**")
                    c1.json(norm)
                    c2.write("**Raw parsed**")
                    c2.json(raw)
                    notes = st.text_input("Notes", key=f"notes_{row['id']}")
                    ba, br, _ = st.columns([1, 1, 4])
                    if ba.button("Approve", key=f"approve_{row['id']}", type="primary"):
                        approve_item(row["id"], notes=notes)
                        st.success("Approved")
                        st.rerun()
                    if br.button("Reject", key=f"reject_{row['id']}"):
                        reject_item(row["id"], notes=notes)
                        st.warning("Rejected")
                        st.rerun()
