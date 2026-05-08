"""
What-if simulations: replay each trade against actual price data
to find optimal stop levels, profit targets, and exit timing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from analytics.portfolio import ClosedTrade
from analytics.price_enrichment import TradePrice


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StopReplayResult:
    trade: ClosedTrade
    original_stop_pct: float
    original_outcome: float          # realized PnL
    was_stopped_out: bool
    scenarios: list[dict] = field(default_factory=list)
    # Each scenario: {stop_pct, would_be_stopped, simulated_pnl, note}


@dataclass
class TakeProfitResult:
    trade: ClosedTrade
    actual_pnl: float
    mfe_pct: float                   # best possible % gain
    mfe_pnl: float                   # PnL if exited at MFE
    left_on_table_pct: float         # how much of MFE was captured
    exit_style: str                  # 'strength' | 'weakness' | 'neutral'
    exit_day_position: float | None  # 0=low, 1=high of exit day
    trailing_scenarios: list[dict] = field(default_factory=list)
    fixed_target_scenarios: list[dict] = field(default_factory=list)
    post_exit_continued: bool = False   # did price keep going in your favour?
    post_exit_drift_5d: float = 0.0
    post_exit_drift_20d: float = 0.0


@dataclass
class EntryTimingResult:
    trade: ClosedTrade
    entry_price: float
    entry_day_low: float | None
    entry_day_high: float | None
    days_adverse_before_recovery: int   # how many days went against you before recovering
    better_entry_available: bool        # a better entry existed within 3 days
    better_entry_price: float | None
    better_entry_saving_pct: float      # % improvement if waited


@dataclass
class SimulationReport:
    stop_replays: list[StopReplayResult]
    take_profit: list[TakeProfitResult]
    entry_timing: list[EntryTimingResult]
    stop_levels_tested: list[float]
    rr_targets_tested: list[float]
    trailing_stops_tested: list[float]


# ── Stop loss replay ──────────────────────────────────────────────────────────

def _was_stopped(df: pd.DataFrame, entry_date: date, entry_price: float,
                 stop_price: float, direction: str) -> tuple[bool, date | None, float | None]:
    """Return (was_stopped, stop_date, recovery_price_20d_after_stop)."""
    hold = df.loc[df.index >= entry_date]
    for row_date, row in hold.iterrows():
        if direction == "LONG" and row["Low"] <= stop_price:
            # Estimate fill at stop price (conservative)
            recovery = None
            post = df.loc[df.index > row_date]
            if len(post) >= 20:
                recovery = post.iloc[19]["Close"]
            return True, row_date, recovery
        if direction == "SHORT" and row["High"] >= stop_price:
            recovery = None
            post = df.loc[df.index > row_date]
            if len(post) >= 20:
                recovery = post.iloc[19]["Close"]
            return True, row_date, recovery
    return False, None, None


def simulate_stop_levels(
    trade: ClosedTrade,
    tp: TradePrice,
    stop_pcts: list[float] = None,
) -> StopReplayResult | None:
    if stop_pcts is None:
        stop_pcts = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]

    df = tp.ohlcv
    if df.empty:
        return None

    entry = trade.avg_entry
    direction = trade.direction
    qty = trade.quantity

    # Detect if original trade was stopped out (exit below entry for LONG, above for SHORT)
    if direction == "LONG":
        was_stopped_out = trade.avg_exit < entry * 0.99  # rough heuristic
    else:
        was_stopped_out = trade.avg_exit > entry * 1.01

    # Calculate original stop % from stop_loss_at_entry if available
    if trade.initial_stop:
        orig_stop_pct = abs(entry - trade.initial_stop) / entry * 100
    else:
        orig_stop_pct = abs(entry - trade.avg_exit) / entry * 100 if was_stopped_out else 5.0

    result = StopReplayResult(
        trade=trade,
        original_stop_pct=orig_stop_pct,
        original_outcome=trade.realized_pnl,
        was_stopped_out=was_stopped_out,
    )

    for stop_pct in stop_pcts:
        if direction == "LONG":
            stop_price = entry * (1 - stop_pct)
        else:
            stop_price = entry * (1 + stop_pct)

        stopped, stop_date, recovery = _was_stopped(df, trade.entry_date, entry, stop_price, direction)

        if stopped:
            sim_pnl = (stop_price - entry) * qty if direction == "LONG" else (entry - stop_price) * qty
            note = f"Still stopped out on {stop_date}"
            if recovery:
                if direction == "LONG":
                    post_move = (recovery - stop_price) / stop_price * 100
                else:
                    post_move = (stop_price - recovery) / stop_price * 100
                note += f"; price moved {post_move:+.1f}% in next 20 days after stop"
        else:
            # Trade survived — use actual exit price
            sim_pnl = (trade.avg_exit - entry) * qty if direction == "LONG" else (entry - trade.avg_exit) * qty
            note = "Would NOT have been stopped out → held to original exit"
            if was_stopped_out:
                note = f"Would NOT have been stopped out → price reached {trade.avg_exit:.2f} (original stop fill)"

        result.scenarios.append({
            "stop_pct": stop_pct,
            "stop_price": round(stop_price, 4),
            "would_be_stopped": stopped,
            "simulated_pnl": round(sim_pnl, 2),
            "pnl_difference": round(sim_pnl - trade.realized_pnl, 2),
            "note": note,
        })

    return result


# ── Take profit analysis ──────────────────────────────────────────────────────

def _trailing_stop_exit(df: pd.DataFrame, entry_date: date, direction: str,
                        trail_pct: float) -> tuple[float, date] | None:
    """Simulate a trailing stop from MFE. Return (exit_price, exit_date)."""
    hold = df.loc[df.index >= entry_date]
    if hold.empty:
        return None

    peak = hold.iloc[0]["Close"]
    for row_date, row in hold.iterrows():
        if direction == "LONG":
            if row["High"] > peak:
                peak = row["High"]
            trail_price = peak * (1 - trail_pct)
            if row["Low"] <= trail_price:
                return trail_price, row_date
        else:
            if row["Low"] < peak:
                peak = row["Low"]
            trail_price = peak * (1 + trail_pct)
            if row["High"] >= trail_price:
                return trail_price, row_date
    return None


def simulate_take_profit(
    trade: ClosedTrade,
    tp: TradePrice,
    rr_targets: list[float] = None,
    trailing_pcts: list[float] = None,
) -> TakeProfitResult | None:
    if rr_targets is None:
        rr_targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    if trailing_pcts is None:
        trailing_pcts = [0.03, 0.05, 0.07, 0.10]

    df = tp.ohlcv
    if df.empty:
        return None

    entry = trade.avg_entry
    qty = trade.quantity
    direction = trade.direction

    mfe_price = tp.mfe_price or trade.avg_exit
    if direction == "LONG":
        mfe_pnl = (mfe_price - entry) * qty
    else:
        mfe_pnl = (entry - mfe_price) * qty

    left_pct = (trade.realized_pnl / mfe_pnl * 100) if mfe_pnl > 0 else 100.0

    # Exit style: where in the exit day's range did they sell?
    pos = tp.exit_day_position
    if pos is None:
        exit_style = "unknown"
    elif pos >= 0.7:
        exit_style = "strength"  # sold near the high of exit day
    elif pos <= 0.3:
        exit_style = "weakness"  # sold near the low of exit day
    else:
        exit_style = "neutral"

    result = TakeProfitResult(
        trade=trade,
        actual_pnl=trade.realized_pnl,
        mfe_pct=tp.mfe_pct,
        mfe_pnl=round(mfe_pnl, 2),
        left_on_table_pct=round(left_pct, 1),
        exit_style=exit_style,
        exit_day_position=round(pos, 2) if pos is not None else None,
        post_exit_drift_5d=tp.post_exit_drift_5d,
        post_exit_drift_20d=tp.post_exit_drift_20d,
    )

    # Did price continue in your favour after exit?
    if direction == "LONG":
        result.post_exit_continued = tp.post_exit_drift_5d > 2.0
    else:
        result.post_exit_continued = tp.post_exit_drift_5d < -2.0

    # Fixed R:R target scenarios
    initial_risk = abs(entry - trade.initial_stop) if trade.initial_stop else entry * 0.05
    for rr in rr_targets:
        if direction == "LONG":
            target_price = entry + initial_risk * rr
            hit = tp.mfe_price >= target_price if tp.mfe_price else False
        else:
            target_price = entry - initial_risk * rr
            hit = tp.mfe_price <= target_price if tp.mfe_price else False

        if hit:
            sim_pnl = (target_price - entry) * qty if direction == "LONG" else (entry - target_price) * qty
        else:
            sim_pnl = trade.realized_pnl  # didn't reach target, kept original exit

        result.fixed_target_scenarios.append({
            "rr": rr,
            "target_price": round(target_price, 4),
            "target_hit": hit,
            "simulated_pnl": round(sim_pnl, 2),
            "vs_actual": round(sim_pnl - trade.realized_pnl, 2),
        })

    # Trailing stop scenarios
    for trail_pct in trailing_pcts:
        exit_info = _trailing_stop_exit(df, trade.entry_date, direction, trail_pct)
        if exit_info:
            trail_exit_price, trail_exit_date = exit_info
            sim_pnl = (trail_exit_price - entry) * qty if direction == "LONG" else (entry - trail_exit_price) * qty
        else:
            trail_exit_price = trade.avg_exit
            trail_exit_date = trade.exit_date
            sim_pnl = trade.realized_pnl

        result.trailing_scenarios.append({
            "trail_pct": trail_pct,
            "exit_price": round(trail_exit_price, 4),
            "exit_date": str(trail_exit_date),
            "simulated_pnl": round(sim_pnl, 2),
            "vs_actual": round(sim_pnl - trade.realized_pnl, 2),
        })

    return result


# ── Entry timing ──────────────────────────────────────────────────────────────

def simulate_entry_timing(
    trade: ClosedTrade,
    tp: TradePrice,
    delay_days: list[int] = None,
) -> EntryTimingResult | None:
    if delay_days is None:
        delay_days = [1, 2, 3]

    df = tp.ohlcv
    if df.empty:
        return None

    entry = trade.avg_entry
    direction = trade.direction

    # Entry day data
    entry_rows = df.loc[df.index == trade.entry_date]
    entry_day_low = float(entry_rows.iloc[0]["Low"]) if not entry_rows.empty else None
    entry_day_high = float(entry_rows.iloc[0]["High"]) if not entry_rows.empty else None

    # How many days was trade adverse before recovering?
    hold = df.loc[df.index >= trade.entry_date]
    adverse_days = 0
    for _, row in hold.iterrows():
        if direction == "LONG" and row["Close"] < entry:
            adverse_days += 1
        elif direction == "SHORT" and row["Close"] > entry:
            adverse_days += 1
        else:
            break

    # Best entry within next 3 days (lowest low for LONG, highest high for SHORT)
    next_3 = df.loc[
        (df.index > trade.entry_date) &
        (df.index <= trade.entry_date)
    ]
    # Get first 3 trading days after entry
    after_entry = df.loc[df.index > trade.entry_date].head(3)

    better_entry = None
    saving_pct = 0.0

    if not after_entry.empty:
        if direction == "LONG":
            best_price = after_entry["Low"].min()
            if best_price < entry:
                better_entry = best_price
                saving_pct = (entry - best_price) / entry * 100
        else:
            best_price = after_entry["High"].max()
            if best_price > entry:
                better_entry = best_price
                saving_pct = (best_price - entry) / entry * 100

    return EntryTimingResult(
        trade=trade,
        entry_price=entry,
        entry_day_low=entry_day_low,
        entry_day_high=entry_day_high,
        days_adverse_before_recovery=adverse_days,
        better_entry_available=better_entry is not None,
        better_entry_price=round(better_entry, 4) if better_entry else None,
        better_entry_saving_pct=round(saving_pct, 2),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_simulation(
    trades: list[ClosedTrade],
    enriched: dict[str, "TradePrice"],
    stop_pcts: list[float] = None,
    rr_targets: list[float] = None,
    trailing_pcts: list[float] = None,
) -> SimulationReport:
    stop_replays = []
    take_profit = []
    entry_timing = []

    for t in trades:
        key = f"{t.symbol}_{t.entry_date}_{t.exit_date}"
        tp = enriched.get(key)
        if not tp:
            continue

        sr = simulate_stop_levels(t, tp, stop_pcts)
        if sr:
            stop_replays.append(sr)

        tpr = simulate_take_profit(t, tp, rr_targets, trailing_pcts)
        if tpr:
            take_profit.append(tpr)

        etr = simulate_entry_timing(t, tp)
        if etr:
            entry_timing.append(etr)

    return SimulationReport(
        stop_replays=stop_replays,
        take_profit=take_profit,
        entry_timing=entry_timing,
        stop_levels_tested=stop_pcts or [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15],
        rr_targets_tested=rr_targets or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        trailing_stops_tested=trailing_pcts or [0.03, 0.05, 0.07, 0.10],
    )
