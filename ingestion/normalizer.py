"""Convert ParsedOrder → DB-ready dicts, resolving account + instrument UUIDs."""
from __future__ import annotations
from datetime import date
from models.order import ParsedOrder
from db.client import get_supabase


def get_or_create_instrument(symbol: str, exchange: str, currency: str) -> str:
    """Return instrument UUID, creating the row if it doesn't exist."""
    client = get_supabase()
    result = (
        client.table("instruments")
        .select("id")
        .eq("symbol", symbol)
        .eq("exchange", exchange)
        .execute()
    )
    if result.data:
        return result.data[0]["id"]

    insert = (
        client.table("instruments")
        .insert({"symbol": symbol, "exchange": exchange, "currency": currency, "asset_class": "equity"})
        .execute()
    )
    return insert.data[0]["id"]


def get_account_id_for_broker(broker_short_code: str) -> str | None:
    """Return the first active account UUID for a broker short code."""
    client = get_supabase()
    result = (
        client.table("accounts")
        .select("id, broker_id, brokers(short_code)")
        .eq("is_active", True)
        .execute()
    )
    for row in result.data:
        broker = row.get("brokers") or {}
        if broker.get("short_code") == broker_short_code:
            return row["id"]
    return None


def normalize_order(order: ParsedOrder, account_id: str, document_id: str | None) -> dict:
    """Return a dict ready for INSERT into the `orders` table."""
    instrument_id = get_or_create_instrument(order.symbol, order.exchange, order.currency)
    return {
        "account_id": account_id,
        "instrument_id": instrument_id,
        "document_id": document_id,
        "external_order_id": order.external_order_id,
        "order_date": order.order_date.isoformat(),
        "order_time": str(order.order_time) if order.order_time else None,
        "action": order.action,
        "price": order.price,
        "quantity": order.quantity,
        "fees": order.fees,
        "currency": order.currency,
        "stop_loss_at_entry": order.stop_loss_at_entry,
        "raw_data_json": order.raw_data,
    }
