from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any


@dataclass
class ParsedOrder:
    """Raw order data extracted by a broker parser — not yet written to DB."""
    broker_short_code: str
    external_order_id: str          # broker's own ref, used for dedup
    action: str                     # 'BUY' | 'SELL' | 'SHORT' | 'COVER'
    symbol: str
    exchange: str                   # 'NYSE'|'NASDAQ'|'HKEX'|''
    currency: str
    price: float
    quantity: float
    order_date: date
    order_time: time | None = None
    fees: float = 0.0
    stop_loss_at_entry: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.action in ("BUY",)

    @property
    def is_sell(self) -> bool:
        return self.action in ("SELL", "COVER")

    @property
    def direction(self) -> str:
        return "short" if self.action == "SHORT" else "long"
