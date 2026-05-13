"""
Aggregate performance metrics over a list of ClosedTrade objects.

All functions accept a list[ClosedTrade] and return plain Python types
so they can be easily serialized / displayed in Streamlit.
"""
from __future__ import annotations
import math
from datetime import date
from typing import Any
from analytics.portfolio import ClosedTrade, OpenPosition


def _winners(trades: list[ClosedTrade]) -> list[ClosedTrade]:
    return [t for t in trades if t.realized_pnl > 0]


def _losers(trades: list[ClosedTrade]) -> list[ClosedTrade]:
    return [t for t in trades if t.realized_pnl <= 0]


def win_rate(trades: list[ClosedTrade]) -> float:
    if not trades:
        return 0.0
    return len(_winners(trades)) / len(trades)


def profit_factor(trades: list[ClosedTrade]) -> float:
    gross_win = sum(t.realized_pnl for t in _winners(trades))
    gross_loss = abs(sum(t.realized_pnl for t in _losers(trades)))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy_r(trades: list[ClosedTrade]) -> float | None:
    """Return expectancy in R units. None if no trades have R multiples."""
    r_trades = [t for t in trades if t.r_multiple is not None]
    if not r_trades:
        return None
    win_r = [t.r_multiple for t in r_trades if t.r_multiple > 0]
    loss_r = [abs(t.r_multiple) for t in r_trades if t.r_multiple <= 0]
    wr = len(win_r) / len(r_trades)
    lr = 1 - wr
    avg_win_r = sum(win_r) / len(win_r) if win_r else 0.0
    avg_loss_r = sum(loss_r) / len(loss_r) if loss_r else 0.0
    return (wr * avg_win_r) - (lr * avg_loss_r)


def avg_win(trades: list[ClosedTrade]) -> float:
    w = _winners(trades)
    return sum(t.realized_pnl for t in w) / len(w) if w else 0.0


def avg_loss(trades: list[ClosedTrade]) -> float:
    l = _losers(trades)
    return sum(t.realized_pnl for t in l) / len(l) if l else 0.0


def avg_holding_days(trades: list[ClosedTrade]) -> float:
    if not trades:
        return 0.0
    days = [(t.exit_date - t.entry_date).days for t in trades]
    return sum(days) / len(days)


def max_drawdown(trades: list[ClosedTrade]) -> float:
    """Return max drawdown as a positive number (worst equity dip)."""
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_date):
        equity += t.realized_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def cumulative_pnl(trades: list[ClosedTrade]) -> list[tuple[date, float]]:
    """Return [(exit_date, cumulative_pnl)] sorted by date."""
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    cum = 0.0
    result = []
    for t in sorted_trades:
        cum += t.realized_pnl
        result.append((t.exit_date, cum))
    return result


def equity_curve_full(
    closed: list[ClosedTrade],
    open_pos: list[OpenPosition],
    prices: dict[str, float | None],
) -> list[tuple[date, float]]:
    """
    Equity curve including:
    - realized P&L from fully closed trades
    - realized P&L from partial exits on still-open positions
    - current unrealized P&L appended as today's point
    """
    from datetime import date as date_cls
    events: list[tuple[date, float]] = []

    for t in closed:
        events.append((t.exit_date, t.realized_pnl))

    for pos in open_pos:
        for pe in pos.partial_exits:
            if pos.direction == "LONG":
                pnl = (pe.exit_price - pe.avg_entry) * pe.quantity - pe.exit_fees - pe.entry_fees_allocated
            else:
                pnl = (pe.avg_entry - pe.exit_price) * pe.quantity - pe.exit_fees - pe.entry_fees_allocated
            events.append((pe.exit_date, pnl))

    events.sort(key=lambda x: x[0])
    cum = 0.0
    result: list[tuple[date, float]] = []
    for d, pnl in events:
        cum += pnl
        result.append((d, cum))

    unrealized = 0.0
    for pos in open_pos:
        price = prices.get(pos.symbol)
        if price is not None:
            if pos.direction == "LONG":
                unrealized += (price - pos.avg_entry_price) * pos.total_quantity
            else:
                unrealized += (pos.avg_entry_price - price) * pos.total_quantity

    if unrealized != 0:
        today = date_cls.today()
        base = result[-1][1] if result else 0.0
        result.append((today, base + unrealized))

    return result


def breakdown_by_direction(trades: list[ClosedTrade]) -> dict[str, Any]:
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]
    return {
        "LONG": {
            "count": len(long_trades),
            "win_rate": win_rate(long_trades),
            "total_pnl": sum(t.realized_pnl for t in long_trades),
        },
        "SHORT": {
            "count": len(short_trades),
            "win_rate": win_rate(short_trades),
            "total_pnl": sum(t.realized_pnl for t in short_trades),
        },
    }


def breakdown_by_dow(trades: list[ClosedTrade]) -> dict[str, Any]:
    """Group by day-of-week of entry date (Monday=0)."""
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    groups: dict[str, list[ClosedTrade]] = {d: [] for d in dow_names}
    for t in trades:
        groups[dow_names[t.entry_date.weekday()]].append(t)
    return {
        day: {
            "count": len(ts),
            "win_rate": win_rate(ts),
            "total_pnl": sum(t.realized_pnl for t in ts),
        }
        for day, ts in groups.items()
        if ts
    }


_HOLD_BUCKETS = [
    ("Intraday",  0,  0),
    ("1–3 days",  1,  3),
    ("4–10 days", 4, 10),
    ("11–30 days",11, 30),
    ("31+ days",  31, 9999),
]


def breakdown_by_holding_period(trades: list[ClosedTrade]) -> dict[str, Any]:
    groups: dict[str, list[ClosedTrade]] = {label: [] for label, *_ in _HOLD_BUCKETS}
    for t in trades:
        days = (t.exit_date - t.entry_date).days
        for label, lo, hi in _HOLD_BUCKETS:
            if lo <= days <= hi:
                groups[label].append(t)
                break
    return {
        label: {
            "count": len(ts),
            "win_rate": win_rate(ts),
            "total_pnl": sum(t.realized_pnl for t in ts),
            "avg_days": round(sum((t.exit_date - t.entry_date).days for t in ts) / len(ts), 1) if ts else 0,
        }
        for label, ts in groups.items()
        if ts
    }


def summary_dict(trades: list[ClosedTrade]) -> dict[str, Any]:
    """All key metrics in one dict for easy display."""
    return {
        "total_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "expectancy_r": expectancy_r(trades),
        "avg_win": avg_win(trades),
        "avg_loss": avg_loss(trades),
        "total_pnl": sum(t.realized_pnl for t in trades),
        "avg_holding_days": avg_holding_days(trades),
        "max_drawdown": max_drawdown(trades),
    }
