"""
IBKR email parser.

IBKR sends order confirmations with subjects like:
    SOLD 215 PAYP @ 20.91 (UXXX29872)
    BOUGHT 100 AAPL @ 183.50 (UABC12345)

Pattern: {ACTION} {QTY} {SYMBOL} @ {PRICE} ({ORDER_ID})
"""
from __future__ import annotations
import re
from datetime import datetime, date
from models.order import ParsedOrder
from ingestion.brokers.base import BrokerAdapter

_SUBJECT_RE = re.compile(
    r"^(BOUGHT|SOLD|SHORTED|COVERED)\s+"
    r"([\d,]+(?:\.\d+)?)\s+"
    r"([A-Z0-9.]+)\s+"
    r"@\s*([\d.]+)\s+"
    r"\((\w+)\)",
    re.IGNORECASE,
)

_ACTION_MAP = {
    "BOUGHT": "BUY",
    "SOLD":   "SELL",
    "SHORTED": "SHORT",
    "COVERED": "COVER",
}


class IBKRAdapter(BrokerAdapter):
    short_code = "IBKR"
    sender_patterns = ["interactivebrokers.com", "ibkr.com"]

    @classmethod
    def parse_subject(cls, subject: str, received_at: datetime | None) -> ParsedOrder | None:
        m = _SUBJECT_RE.match(subject.strip())
        if not m:
            return None

        raw_action, raw_qty, symbol, raw_price, order_id = m.groups()
        qty = float(raw_qty.replace(",", ""))
        price = float(raw_price)
        action = _ACTION_MAP.get(raw_action.upper(), raw_action.upper())

        exchange = "HKEX" if symbol.endswith(".HK") else "NYSE"
        currency = "HKD" if symbol.endswith(".HK") else "USD"

        order_date: date = received_at.date() if received_at else date.today()
        order_time = received_at.time() if received_at else None

        # order_id in the subject is the account number (same for all orders).
        # Build a unique ID from symbol + date + time + price instead.
        unique_id = f"IBKR_{symbol}_{order_date}_{order_time}_{price}_{qty}"

        return ParsedOrder(
            broker_short_code=cls.short_code,
            external_order_id=unique_id,
            action=action,
            symbol=symbol.upper(),
            exchange=exchange,
            currency=currency,
            price=price,
            quantity=qty,
            order_date=order_date,
            order_time=order_time,
            raw_data={"subject": subject},
        )

    @classmethod
    def parse_body(cls, body: str, subject: str, received_at: datetime | None) -> ParsedOrder | None:
        # IBKR order info is fully in the subject; body parsing not needed yet
        return None
