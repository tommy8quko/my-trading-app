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


def rr_ratio(trades: list[ClosedTrade]) -> float | None:
    """Avg winning R / avg losing R. None if insufficient data."""
    r_trades = [t for t in trades if t.r_multiple is not None]
    win_r = [t.r_multiple for t in r_trades if t.r_multiple > 0]
    loss_r = [abs(t.r_multiple) for t in r_trades if t.r_multiple <= 0]
    if not win_r or not loss_r:
        return None
    return (sum(win_r) / len(win_r)) / (sum(loss_r) / len(loss_r))


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
    sorted_trades = sorted(trades, key=lambda t: (t.exit_date, t.exit_time or ''))
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


def current_streak(trades: list[ClosedTrade]) -> dict[str, Any]:
    """Return {'count': N, 'type': 'W'|'L'} for the most recent consecutive run."""
    if not trades:
        return {"count": 0, "type": None}
    sorted_trades = sorted(trades, key=lambda t: (t.exit_date, t.exit_time or ''))
    last_type = "W" if sorted_trades[-1].realized_pnl > 0 else "L"
    count = 0
    for t in reversed(sorted_trades):
        t_type = "W" if t.realized_pnl > 0 else "L"
        if t_type == last_type:
            count += 1
        else:
            break
    return {"count": count, "type": last_type}


def longest_win_streak(trades: list[ClosedTrade]) -> int:
    best, cur = 0, 0
    for t in sorted(trades, key=lambda t: (t.exit_date, t.exit_time or '')):
        if t.realized_pnl > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def longest_loss_streak(trades: list[ClosedTrade]) -> int:
    best, cur = 0, 0
    for t in sorted(trades, key=lambda t: (t.exit_date, t.exit_time or '')):
        if t.realized_pnl <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summary_dict(trades: list[ClosedTrade]) -> dict[str, Any]:
    """All key metrics in one dict — single sort, single partition, no repeated iteration."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy_r": None, "rr_ratio": None, "avg_win": 0.0, "avg_loss": 0.0,
            "total_pnl": 0.0, "avg_holding_days": 0.0, "max_drawdown": 0.0,
            "current_streak": {"count": 0, "type": None},
            "longest_win_streak": 0, "longest_loss_streak": 0,
            "avg_hold_winners": 0, "avg_hold_losers": 0,
            "avg_win_pct": None, "avg_loss_pct": None, "kelly": None,
            "biggest_win_pct": None, "biggest_loss_pct": None,
        }

    # Sort once — reused for streaks, drawdown, equity curve
    sorted_trades = sorted(trades, key=lambda t: (t.exit_date, t.exit_time or ''))

    # Partition once into winners/losers
    winners, losers = [], []
    total_pnl = 0.0
    for t in sorted_trades:
        total_pnl += t.realized_pnl
        (winners if t.realized_pnl > 0 else losers).append(t)

    nw, nl, n = len(winners), len(losers), len(sorted_trades)
    wr = nw / n

    # Avg win/loss
    aw = sum(t.realized_pnl for t in winners) / nw if nw else 0.0
    al = sum(t.realized_pnl for t in losers)  / nl if nl else 0.0

    # Profit factor
    gross_win  = sum(t.realized_pnl for t in winners)
    gross_loss = abs(sum(t.realized_pnl for t in losers))
    pf = gross_win / gross_loss if gross_loss else (float("inf") if gross_win > 0 else 0.0)

    # Holding days
    hold_w = sum((t.exit_date - t.entry_date).days for t in winners)
    hold_l = sum((t.exit_date - t.entry_date).days for t in losers)
    avg_hold_days_all = sum((t.exit_date - t.entry_date).days for t in sorted_trades) / n

    # % return per trade (single pass)
    def _pct(t):
        if not t.avg_entry:
            return None
        return (t.avg_exit - t.avg_entry) / t.avg_entry * 100 if t.direction == "LONG" \
               else (t.avg_entry - t.avg_exit) / t.avg_entry * 100

    win_pcts, loss_pcts, all_pcts = [], [], []
    for t in sorted_trades:
        p = _pct(t)
        if p is not None:
            all_pcts.append(p)
            (win_pcts if t.realized_pnl > 0 else loss_pcts).append(p)

    # Expectancy R and R/R (single pass over r_multiple trades)
    r_trades = [t for t in sorted_trades if t.r_multiple is not None]
    exp_r = rr = None
    if r_trades:
        win_r  = [t.r_multiple for t in r_trades if t.r_multiple > 0]
        loss_r = [abs(t.r_multiple) for t in r_trades if t.r_multiple <= 0]
        rwr = len(win_r) / len(r_trades)
        avg_wr = sum(win_r)  / len(win_r)  if win_r  else 0.0
        avg_lr = sum(loss_r) / len(loss_r) if loss_r else 0.0
        exp_r = (rwr * avg_wr) - ((1 - rwr) * avg_lr)
        if win_r and loss_r:
            rr = avg_wr / avg_lr

    # Max drawdown (single pass over sorted trades)
    equity = peak = max_dd = 0.0
    for t in sorted_trades:
        equity += t.realized_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Streaks (single pass over sorted trades)
    last_type = "W" if sorted_trades[-1].realized_pnl > 0 else "L"
    streak_count = 0
    for t in reversed(sorted_trades):
        if ("W" if t.realized_pnl > 0 else "L") == last_type:
            streak_count += 1
        else:
            break

    best_w = cur_w = best_l = cur_l = 0
    for t in sorted_trades:
        if t.realized_pnl > 0:
            cur_w += 1; best_w = max(best_w, cur_w); cur_l = 0
        else:
            cur_l += 1; best_l = max(best_l, cur_l); cur_w = 0

    kelly = round(wr - (1 - wr) / (aw / abs(al)), 4) if aw and al else None

    return {
        "total_trades":        n,
        "win_rate":            wr,
        "profit_factor":       pf,
        "expectancy_r":        exp_r,
        "rr_ratio":            rr,
        "avg_win":             aw,
        "avg_loss":            al,
        "total_pnl":           total_pnl,
        "avg_holding_days":    avg_hold_days_all,
        "max_drawdown":        max_dd,
        "current_streak":      {"count": streak_count, "type": last_type},
        "longest_win_streak":  best_w,
        "longest_loss_streak": best_l,
        "avg_hold_winners":    round(hold_w / nw, 1) if nw else 0,
        "avg_hold_losers":     round(hold_l / nl, 1) if nl else 0,
        "avg_win_pct":         round(sum(win_pcts)  / len(win_pcts),  2) if win_pcts  else None,
        "avg_loss_pct":        round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else None,
        "kelly":               kelly,
        "biggest_win_pct":     round(max(all_pcts), 2) if all_pcts else None,
        "biggest_loss_pct":    round(min(all_pcts), 2) if all_pcts else None,
    }
