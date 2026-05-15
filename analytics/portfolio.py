"""
FIFO position tracking engine.

Reconstructs current open positions and closed trades from the orders table.
All monetary values are in the order's native currency unless noted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from db.client import get_supabase


@dataclass
class Lot:
    """A single entry into a position (one BUY or SHORT fill)."""
    order_id: str
    order_date: date
    action: str
    quantity: float
    entry_price: float
    fees: float
    currency: str
    stop_loss_at_entry: float | None


@dataclass
class PartialExit:
    """One exit fill (SELL or COVER) accumulated until position fully closes."""
    exit_date: date
    quantity: float
    exit_price: float
    exit_fees: float
    avg_entry: float          # weighted avg entry price for the consumed lots
    entry_fees_allocated: float


@dataclass
class ClosedTrade:
    symbol: str
    exchange: str
    currency: str
    direction: str          # 'LONG' | 'SHORT'
    entry_date: date
    exit_date: date
    quantity: float
    avg_entry: float
    avg_exit: float
    fees: float
    realized_pnl: float     # in trade currency
    initial_stop: float | None
    r_multiple: float | None
    exit_time: str | None = None  # HH:MM:SS from order_time of the closing order


@dataclass
class OpenPosition:
    symbol: str
    exchange: str
    currency: str
    direction: str          # 'LONG' | 'SHORT'
    lots: list[Lot] = field(default_factory=list)
    partial_exits: list[PartialExit] = field(default_factory=list)

    @property
    def total_quantity(self) -> float:
        return sum(l.quantity for l in self.lots)

    @property
    def avg_entry_price(self) -> float:
        total_q = self.total_quantity
        if total_q == 0:
            return 0.0
        return sum(l.quantity * l.entry_price for l in self.lots) / total_q

    @property
    def total_fees(self) -> float:
        return sum(l.fees for l in self.lots)

    @property
    def earliest_entry_date(self) -> date | None:
        return min((l.order_date for l in self.lots), default=None)

    @property
    def initial_stop_loss(self) -> float | None:
        if self.lots:
            return self.lots[0].stop_loss_at_entry
        return None


def _fetch_orders() -> list[dict[str, Any]]:
    client = get_supabase()
    result = (
        client.table("orders")
        .select(
            "id, account_id, order_date, order_time, action, price, quantity, fees, "
            "currency, stop_loss_at_entry, instrument_id, "
            "instruments(symbol, exchange, currency)"
        )
        .order("order_date", desc=False)
        .order("order_time", desc=False)
        .execute()
    )
    return result.data


def build_portfolio() -> tuple[list[OpenPosition], list[ClosedTrade]]:
    """
    Process all orders FIFO and return (open_positions, closed_trades).
    A ClosedTrade is only emitted when the entire position quantity reaches zero.
    """
    orders = _fetch_orders()
    open_positions: dict[str, OpenPosition] = {}   # key: symbol
    closed_trades: list[ClosedTrade] = []
    # Track first-lot metadata separately because lots are consumed during exits
    position_meta: dict[str, dict] = {}  # symbol → {entry_date, initial_stop}

    for o in orders:
        instr = o.get("instruments") or {}
        symbol = instr.get("symbol", "?")
        exchange = instr.get("exchange", "?")
        currency = o.get("currency") or instr.get("currency", "USD")
        action = o["action"]
        qty = float(o["quantity"])
        price = float(o["price"])
        fees = float(o.get("fees") or 0)
        sl = o.get("stop_loss_at_entry")
        order_date = date.fromisoformat(o["order_date"])

        if action == "BUY":
            pos = open_positions.setdefault(
                symbol,
                OpenPosition(symbol=symbol, exchange=exchange, currency=currency, direction="LONG"),
            )
            lot = Lot(
                order_id=o["id"],
                order_date=order_date,
                action=action,
                quantity=qty,
                entry_price=price,
                fees=fees,
                currency=currency,
                stop_loss_at_entry=float(sl) if sl else None,
            )
            pos.lots.append(lot)
            if symbol not in position_meta:
                position_meta[symbol] = {
                    "entry_date": order_date,
                    "initial_stop": float(sl) if sl else None,
                }

        elif action == "SHORT":
            pos = open_positions.setdefault(
                symbol,
                OpenPosition(symbol=symbol, exchange=exchange, currency=currency, direction="SHORT"),
            )
            lot = Lot(
                order_id=o["id"],
                order_date=order_date,
                action=action,
                quantity=qty,
                entry_price=price,
                fees=fees,
                currency=currency,
                stop_loss_at_entry=float(sl) if sl else None,
            )
            pos.lots.append(lot)
            if symbol not in position_meta:
                position_meta[symbol] = {
                    "entry_date": order_date,
                    "initial_stop": float(sl) if sl else None,
                }

        elif action in ("SELL", "COVER"):
            pos = open_positions.get(symbol)
            if pos is None:
                continue  # orphaned exit — skip

            remaining_exit = qty
            lots_used: list[tuple[Lot, float]] = []

            while remaining_exit > 0 and pos.lots:
                lot = pos.lots[0]
                consume = min(lot.quantity, remaining_exit)
                lots_used.append((lot, consume))
                remaining_exit -= consume
                lot.quantity -= consume
                if lot.quantity <= 0:
                    pos.lots.pop(0)

            if not lots_used:
                continue

            exit_qty = sum(q for _, q in lots_used)
            batch_avg_entry = sum(l.entry_price * q for l, q in lots_used) / exit_qty

            # Proportional entry fees for the consumed shares
            allocated_entry_fees = 0.0
            for lot, q in lots_used:
                original_qty = lot.quantity + q  # restored before this loop consumed it
                allocated_entry_fees += lot.fees * (q / original_qty) if original_qty else 0

            pos.partial_exits.append(PartialExit(
                exit_date=order_date,
                quantity=exit_qty,
                exit_price=price,
                exit_fees=fees,
                avg_entry=batch_avg_entry,
                entry_fees_allocated=allocated_entry_fees,
            ))

            # Only emit ClosedTrade when the whole position is gone
            if pos.total_quantity <= 0:
                all_exits = pos.partial_exits
                total_closed_qty = sum(e.quantity for e in all_exits)

                avg_exit = sum(e.exit_price * e.quantity for e in all_exits) / total_closed_qty
                avg_entry = sum(e.avg_entry * e.quantity for e in all_exits) / total_closed_qty

                total_exit_fees = sum(e.exit_fees for e in all_exits)
                total_entry_fees = sum(e.entry_fees_allocated for e in all_exits)
                total_fees = total_exit_fees + total_entry_fees

                meta = position_meta.get(symbol, {})
                entry_date = meta.get("entry_date", order_date)
                initial_stop = meta.get("initial_stop")

                direction = pos.direction
                if direction == "LONG":
                    realized_pnl = (avg_exit - avg_entry) * total_closed_qty - total_fees
                else:
                    realized_pnl = (avg_entry - avg_exit) * total_closed_qty - total_fees

                risk_per_share = abs(avg_entry - initial_stop) if initial_stop else None
                r_multiple = None
                if risk_per_share and risk_per_share > 0:
                    pnl_per_share = realized_pnl / total_closed_qty
                    r_multiple = pnl_per_share / risk_per_share

                closed_trades.append(ClosedTrade(
                    symbol=symbol,
                    exchange=exchange,
                    currency=currency,
                    direction=direction,
                    entry_date=entry_date,
                    exit_date=order_date,
                    quantity=total_closed_qty,
                    avg_entry=avg_entry,
                    avg_exit=avg_exit,
                    fees=total_fees,
                    realized_pnl=realized_pnl,
                    initial_stop=initial_stop,
                    r_multiple=r_multiple,
                    exit_time=o.get("order_time"),
                ))
                del open_positions[symbol]
                position_meta.pop(symbol, None)

    return list(open_positions.values()), closed_trades
