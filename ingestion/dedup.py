"""Duplicate detection for imported orders."""
from __future__ import annotations
import hashlib
from models.order import ParsedOrder
from db.client import get_supabase


def order_exists(order: ParsedOrder, account_id: str) -> bool:
    """Return True if this order is already in the DB (same account + external_order_id)."""
    if not order.external_order_id:
        return False
    client = get_supabase()
    result = (
        client.table("orders")
        .select("id")
        .eq("account_id", account_id)
        .eq("external_order_id", order.external_order_id)
        .execute()
    )
    return len(result.data) > 0


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_already_imported(sha256: str) -> bool:
    """Return True if a document with this hash was already processed."""
    client = get_supabase()
    result = (
        client.table("imported_documents")
        .select("id")
        .eq("sha256_hash", sha256)
        .execute()
    )
    return len(result.data) > 0


def email_already_processed(gmail_message_id: str) -> bool:
    client = get_supabase()
    result = (
        client.table("email_records")
        .select("id")
        .eq("gmail_message_id", gmail_message_id)
        .eq("processed", True)
        .execute()
    )
    return len(result.data) > 0
