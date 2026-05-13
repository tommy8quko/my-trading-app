"""CRUD operations for the review_queue table."""
from __future__ import annotations
from typing import Any
from models.review import ReviewItem
from db.client import get_supabase


def save_to_review_queue(item: ReviewItem) -> str:
    """Insert a ReviewItem into review_queue; return its new UUID."""
    client = get_supabase()
    row = {
        "raw_parsed_json": {**item.raw_parsed, "source_type": item.source_type, "broker_short_code": item.broker_short_code},
        "normalized_json": item.normalized,
        "confidence_score": item.confidence,
        "email_record_id": item.email_record_id,
        "document_id": item.document_id,
        "status": item.status,
        "review_notes": item.review_notes,
    }
    result = client.table("review_queue").insert(row).execute()
    return result.data[0]["id"]


def fetch_pending(limit: int = 100) -> list[dict[str, Any]]:
    """Return up to `limit` rows with status='pending', oldest first."""
    client = get_supabase()
    result = (
        client.table("review_queue")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data


def approve_item(queue_id: str, notes: str = "", stop_loss: float | None = None) -> None:
    """Mark a review_queue row as approved and insert into orders."""
    client = get_supabase()
    row = client.table("review_queue").select("*").eq("id", queue_id).single().execute().data
    if not row:
        raise ValueError(f"Review item {queue_id} not found")

    normalized = dict(row["normalized_json"] or {})
    if stop_loss is not None and stop_loss > 0:
        normalized["stop_loss_at_entry"] = stop_loss

    # ignore_duplicates=True: if this order was already approved via a duplicate
    # queue entry (same email matched two sender filters), skip silently.
    client.table("orders").upsert(
        normalized,
        on_conflict="account_id,external_order_id",
        ignore_duplicates=True,
    ).execute()
    client.table("review_queue").update(
        {"status": "approved", "review_notes": notes}
    ).eq("id", queue_id).execute()


def reject_item(queue_id: str, notes: str = "") -> None:
    """Mark a review_queue row as rejected."""
    client = get_supabase()
    client.table("review_queue").update(
        {"status": "rejected", "review_notes": notes}
    ).eq("id", queue_id).execute()


def auto_approve_all_pending() -> int:
    """Approve every pending item without review; returns count approved."""
    pending = fetch_pending(limit=1000)
    for row in pending:
        approve_item(row["id"], notes="auto_approved")
    return len(pending)
