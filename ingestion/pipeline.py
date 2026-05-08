"""
Ingestion pipeline: raw source → ParsedOrder → dedup → normalize → review queue.

Entry points:
    ingest_email(raw_email, gmail_message_id, from_address, subject, received_at)
    ingest_file(content, filename, broker_short_code)
"""
from __future__ import annotations
import logging
from datetime import datetime

from models.order import ParsedOrder
from models.review import ReviewItem
from ingestion.brokers.base import BrokerAdapter
from ingestion.brokers.ibkr import IBKRAdapter
from ingestion.brokers.chief import ChiefAdapter
from ingestion.dedup import order_exists, file_hash, file_already_imported, email_already_processed
from ingestion.normalizer import get_account_id_for_broker, normalize_order
from ingestion.review_queue import save_to_review_queue
from db.client import get_supabase

log = logging.getLogger(__name__)

_ADAPTERS: list[type[BrokerAdapter]] = [IBKRAdapter, ChiefAdapter]


# ── helpers ──────────────────────────────────────────────────────────────────

def _detect_adapter(from_address: str) -> type[BrokerAdapter] | None:
    for adapter in _ADAPTERS:
        if adapter.matches_sender(from_address):
            return adapter
    return None


def _get_broker_id(short_code: str) -> str | None:
    client = get_supabase()
    result = client.table("brokers").select("id").eq("short_code", short_code).execute()
    return result.data[0]["id"] if result.data else None


def _record_email(gmail_message_id: str, from_address: str, subject: str,
                  received_at: datetime | None, broker_short_code: str) -> str:
    """Upsert an email_records row; return its UUID."""
    client = get_supabase()
    broker_id = _get_broker_id(broker_short_code)
    result = (
        client.table("email_records")
        .upsert(
            {
                "gmail_message_id": gmail_message_id,
                "from_address": from_address,
                "subject": subject,
                "received_at": received_at.isoformat() if received_at else None,
                "broker_id": broker_id,
                "processed": False,
            },
            on_conflict="gmail_message_id",
        )
        .execute()
    )
    return result.data[0]["id"]


def _mark_email_processed(email_record_id: str) -> None:
    get_supabase().table("email_records").update(
        {"processed": True}
    ).eq("id", email_record_id).execute()


def _record_document(filename: str, sha256: str, source_type: str) -> str:
    client = get_supabase()
    result = client.table("imported_documents").insert(
        {"original_reference": filename, "sha256_hash": sha256, "source_type": source_type}
    ).execute()
    return result.data[0]["id"]


def _queue_order(
    order: ParsedOrder,
    source_type: str,
    account_id: str,
    document_id: str | None,
    email_record_id: str | None,
    confidence: float,
) -> str | None:
    """Dedup-check then push to review queue. Returns queue UUID or None if duplicate."""
    if order_exists(order, account_id):
        log.info("Duplicate skipped: %s / %s", order.external_order_id, order.symbol)
        return None

    normalized = normalize_order(order, account_id, document_id)
    enriched_raw = {
        **order.raw_data,
        "symbol": order.symbol,
        "exchange": order.exchange,
        "action": order.action,
        "quantity": order.quantity,
        "price": order.price,
        "currency": order.currency,
        "source_type": source_type,
        "broker_short_code": order.broker_short_code,
    }
    item = ReviewItem(
        id=None,
        source_type=source_type,
        broker_short_code=order.broker_short_code,
        raw_parsed=enriched_raw,
        normalized=normalized,
        confidence=confidence,
        email_record_id=email_record_id,
        document_id=document_id,
    )
    return save_to_review_queue(item)


# ── public API ────────────────────────────────────────────────────────────────

class PipelineResult:
    def __init__(self) -> None:
        self.queued: int = 0
        self.duplicates: int = 0
        self.errors: list[str] = []
        self.queue_ids: list[str] = []
        # diagnostic counters
        self.skipped_already_processed: int = 0
        self.skipped_no_adapter: int = 0
        self.skipped_no_order: int = 0
        self.skipped_no_account: int = 0


def ingest_email(
    body: str,
    gmail_message_id: str,
    from_address: str,
    subject: str,
    received_at: datetime | None,
) -> PipelineResult:
    """Parse a single email and push any order to the review queue."""
    result = PipelineResult()

    if email_already_processed(gmail_message_id):
        result.skipped_already_processed += 1
        return result

    adapter_cls = _detect_adapter(from_address)
    if adapter_cls is None:
        result.skipped_no_adapter += 1
        return result

    email_record_id = _record_email(
        gmail_message_id, from_address, subject, received_at, adapter_cls.short_code
    )

    order: ParsedOrder | None = adapter_cls.parse_subject(subject, received_at)
    if order is None:
        order = adapter_cls.parse_body(body, subject, received_at)

    if order is None:
        result.skipped_no_order += 1
        _mark_email_processed(email_record_id)
        return result

    account_id = get_account_id_for_broker(adapter_cls.short_code)
    if account_id is None:
        result.skipped_no_account += 1
        result.errors.append(f"No active account found for broker {adapter_cls.short_code}")
        return result

    queue_id = _queue_order(
        order=order,
        source_type="email",
        account_id=account_id,
        document_id=None,
        email_record_id=email_record_id,
        confidence=0.95,
    )
    if queue_id:
        result.queued += 1
        result.queue_ids.append(queue_id)
    else:
        result.duplicates += 1

    _mark_email_processed(email_record_id)
    return result


def ingest_file(
    content: bytes,
    filename: str,
    broker_short_code: str,
) -> PipelineResult:
    """Parse a CSV or XLSX file and push each order to the review queue."""
    from ingestion.parsers.csv_parser import CSVParser
    from ingestion.parsers.xlsx_parser import XLSXParser

    result = PipelineResult()
    sha256 = file_hash(content)

    if file_already_imported(sha256):
        log.info("File already imported: %s (%s)", filename, sha256)
        return result

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "xlsx":
        parser = XLSXParser()
    else:
        parser = CSVParser()

    try:
        orders = parser.parse(content, broker_short_code)
    except Exception as exc:
        result.errors.append(f"Parse error: {exc}")
        return result

    if not orders:
        result.errors.append("No orders found in file")
        return result

    document_id = _record_document(filename, sha256, ext or "csv")
    account_id = get_account_id_for_broker(broker_short_code)
    if account_id is None:
        result.errors.append(f"No active account for broker {broker_short_code}")
        return result

    for order in orders:
        try:
            queue_id = _queue_order(
                order=order,
                source_type=ext or "csv",
                account_id=account_id,
                document_id=document_id,
                email_record_id=None,
                confidence=0.80,
            )
            if queue_id:
                result.queued += 1
                result.queue_ids.append(queue_id)
            else:
                result.duplicates += 1
        except Exception as exc:
            result.errors.append(f"Order error ({order.symbol}): {exc}")

    return result
