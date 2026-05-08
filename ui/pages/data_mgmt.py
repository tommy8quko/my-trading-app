"""Data management page — manual file upload + Gmail sync trigger."""
from __future__ import annotations
from datetime import date, timedelta
import streamlit as st
from ingestion.pipeline import ingest_file


_BROKERS = ["IBKR", "CHIEF"]


def render() -> None:
    st.header("Data Management")

    # ── Gmail sync ────────────────────────────────────────────────────────────
    st.subheader("Sync from Gmail")
    since = st.date_input("Fetch emails since", value=date.today())
    if st.button("Sync Gmail", type="primary"):
        with st.spinner("Connecting to Gmail..."):
            try:
                from email_connector.sync import sync_emails
                result = sync_emails(since=since)
                st.success(
                    f"Done — {result.queued} new order(s) queued, "
                    f"{result.duplicates} duplicate(s) skipped."
                )
                # Always show diagnostic breakdown
                st.info(
                    f"**Pipeline breakdown:**  \n"
                    f"Already processed (skipped): {result.skipped_already_processed}  \n"
                    f"No broker adapter matched: {result.skipped_no_adapter}  \n"
                    f"No order found in email: {result.skipped_no_order}  \n"
                    f"No account found for broker: {result.skipped_no_account}  \n"
                    f"Errors: {len(result.errors)}"
                )
                if result.errors:
                    for e in result.errors[:5]:
                        st.code(e)

                # Diagnostic: show unique senders found
                senders = getattr(result, "senders_seen", [])
                if senders:
                    unique_senders = sorted(set(senders))
                    with st.expander(f"📬 {len(senders)} emails scanned — click to see senders"):
                        st.caption("These are the From addresses found. Check if your broker emails are listed:")
                        for s in unique_senders:
                            st.code(s)
            except Exception as exc:
                st.error(f"Gmail sync failed: {exc}")

    st.divider()

    # ── Email subject diagnostic ──────────────────────────────────────────────
    st.subheader("Diagnose — preview broker email subjects")
    diag_since = st.date_input("Fetch since (diagnostic)", value=date.today() - timedelta(days=60), key="diag_since")
    if st.button("Show broker email subjects"):
        with st.spinner("Fetching subjects..."):
            try:
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
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.warning("No emails found for broker senders.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    # ── Chief body inspector ──────────────────────────────────────────────────
    st.subheader("Diagnose — inspect one Chief order email body")
    if st.button("Show Chief email body (parsed text + regex check)"):
        with st.spinner("Fetching..."):
            try:
                import re
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
                            st.write(f"**Raw body length:** {len(body)} chars")
                            st.write("**Raw body (first 500 chars):**")
                            st.code(body[:500])

                            is_html = "<html" in body.lower() or "<!doctype" in body.lower()
                            st.write(f"**Is HTML:** {is_html}")

                            if is_html:
                                stripped = _strip_html(body)
                            else:
                                stripped = body

                            st.write(f"**Stripped text ({len(stripped)} chars):**")
                            st.code(stripped[:1500])

                            intro = _INTRO_RE.search(stripped)
                            st.write(f"**_INTRO_RE match:** {intro.groups() if intro else 'NO MATCH'}")

                            kv = _parse_kv(stripped)
                            st.write(f"**KV pairs found:** {kv}")
                            break
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    # ── Manual file upload ────────────────────────────────────────────────────
    st.subheader("Upload CSV / XLSX")
    broker = st.selectbox("Broker", _BROKERS)
    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded and st.button("Import File"):
        with st.spinner("Parsing..."):
            content = uploaded.read()
            result = ingest_file(content, uploaded.name, broker)

        if result.errors:
            for e in result.errors:
                st.error(e)
        else:
            st.success(
                f"Done — {result.queued} order(s) queued, "
                f"{result.duplicates} duplicate(s) skipped."
            )
            if result.queued:
                st.info("Go to the Review Queue to approve or reject them.")
