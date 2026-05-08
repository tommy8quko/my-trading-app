"""High-level email sync: fetch → pipeline → return aggregate result."""
from __future__ import annotations
import logging
from datetime import date, timedelta

from email_connector.connector import GmailConnector
from ingestion.pipeline import ingest_email, PipelineResult

log = logging.getLogger(__name__)

# Known broker sender domains — used to filter Gmail fetch for speed
BROKER_SENDER_FILTERS = [
    "interactivebrokers.com",
    "chiefgroup.com.hk",
]


def sync_emails(
    since: date | None = None,
    max_results: int = 500,
) -> PipelineResult:
    if since is None:
        since = date.today() - timedelta(days=30)

    aggregate = PipelineResult()
    aggregate.senders_seen: list[str] = []

    with GmailConnector() as gmail:
        for sender_filter in BROKER_SENDER_FILTERS:
            for raw in gmail.fetch_emails(
                since=since,
                sender_filter=sender_filter,
                max_results=max_results,
            ):
                aggregate.senders_seen.append(raw["from_address"])
                try:
                    r = ingest_email(
                        body=raw["body"],
                        gmail_message_id=raw["gmail_message_id"],
                        from_address=raw["from_address"],
                        subject=raw["subject"],
                        received_at=raw["received_at"],
                    )
                    aggregate.queued += r.queued
                    aggregate.duplicates += r.duplicates
                    aggregate.errors.extend(r.errors)
                    aggregate.queue_ids.extend(r.queue_ids)
                    aggregate.skipped_already_processed += r.skipped_already_processed
                    aggregate.skipped_no_adapter += r.skipped_no_adapter
                    aggregate.skipped_no_order += r.skipped_no_order
                    aggregate.skipped_no_account += r.skipped_no_account
                except Exception as exc:
                    log.exception("Unhandled error processing email %s", raw.get("gmail_message_id"))
                    aggregate.errors.append(str(exc))

    return aggregate
