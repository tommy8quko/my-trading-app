"""Generic CSV parser: maps column names to ParsedOrder fields."""
from __future__ import annotations
import csv
import io
from datetime import date, time
from models.order import ParsedOrder
from ingestion.parsers.base import FileParser

# Default column name aliases → ParsedOrder field
_DEFAULT_MAP: dict[str, str] = {
    "date": "order_date",
    "trade_date": "order_date",
    "time": "order_time",
    "trade_time": "order_time",
    "action": "action",
    "side": "action",
    "symbol": "symbol",
    "ticker": "symbol",
    "exchange": "exchange",
    "currency": "currency",
    "price": "price",
    "filled_price": "price",
    "avg_price": "price",
    "quantity": "quantity",
    "qty": "quantity",
    "shares": "quantity",
    "fees": "fees",
    "commission": "fees",
    "order_id": "external_order_id",
    "external_order_id": "external_order_id",
    "ref_no": "external_order_id",
    "stop_loss": "stop_loss_at_entry",
    "stop_loss_at_entry": "stop_loss_at_entry",
}

_ACTION_ALIASES = {
    "buy": "BUY", "bought": "BUY",
    "sell": "SELL", "sold": "SELL",
    "short": "SHORT", "shorted": "SHORT",
    "cover": "COVER", "covered": "COVER",
}


def _parse_date(val: str) -> date:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            from datetime import datetime
            return datetime.strptime(val.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {val!r}")


def _parse_time(val: str) -> time | None:
    if not val or not val.strip():
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            from datetime import datetime
            return datetime.strptime(val.strip(), fmt).time()
        except ValueError:
            continue
    return None


class CSVParser(FileParser):
    def __init__(self, column_map: dict[str, str] | None = None):
        self._col_map = {k.lower(): v for k, v in (_DEFAULT_MAP | (column_map or {})).items()}

    def parse(self, content: bytes, broker_short_code: str) -> list[ParsedOrder]:
        text = content.decode("utf-8-sig")  # strip BOM
        reader = csv.DictReader(io.StringIO(text))
        orders: list[ParsedOrder] = []

        for row in reader:
            mapped: dict[str, str] = {}
            for col, val in row.items():
                field = self._col_map.get((col or "").strip().lower())
                if field:
                    mapped[field] = (val or "").strip()

            try:
                symbol = mapped.get("symbol", "").upper()
                action_raw = mapped.get("action", "BUY").upper()
                action = _ACTION_ALIASES.get(action_raw.lower(), action_raw)
                price = float(mapped.get("price", "0").replace(",", ""))
                quantity = float(mapped.get("quantity", "0").replace(",", ""))
                if not symbol or price <= 0 or quantity <= 0:
                    continue

                exchange = mapped.get("exchange") or ("HKEX" if symbol.endswith(".HK") else "NYSE")
                currency = mapped.get("currency") or ("HKD" if symbol.endswith(".HK") else "USD")
                order_date = _parse_date(mapped.get("order_date", ""))
                order_time = _parse_time(mapped.get("order_time", ""))
                fees = float(mapped.get("fees", "0").replace(",", ""))
                sl = mapped.get("stop_loss_at_entry")
                stop_loss = float(sl) if sl else None
                ext_id = mapped.get("external_order_id") or f"{broker_short_code}_{symbol}_{order_date}_{quantity}"

                orders.append(ParsedOrder(
                    broker_short_code=broker_short_code,
                    external_order_id=ext_id,
                    action=action,
                    symbol=symbol,
                    exchange=exchange,
                    currency=currency,
                    price=price,
                    quantity=quantity,
                    order_date=order_date,
                    order_time=order_time,
                    fees=fees,
                    stop_loss_at_entry=stop_loss,
                    raw_data=dict(row),
                ))
            except (ValueError, KeyError):
                continue

        return orders
