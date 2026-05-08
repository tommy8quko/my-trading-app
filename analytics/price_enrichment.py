"""
Fetch actual OHLCV price data for each trade's window via yfinance.

For each trade we fetch from (entry_date - 5 days) to (exit_date + 30 days)
so simulations can see what happened both before and after the trade.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from analytics.portfolio import ClosedTrade

log = logging.getLogger(__name__)


@dataclass
class TradePrice:
    """OHLCV data covering one trade's window."""
    symbol: str
    trade: ClosedTrade
    ohlcv: pd.DataFrame          # columns: Open High Low Close Volume; index: date
    entry_day_row: pd.Series | None = None
    exit_day_row: pd.Series | None = None

    # Derived fields populated by enrich()
    mae_pct: float = 0.0         # Max Adverse Excursion % during hold
    mfe_pct: float = 0.0         # Max Favorable Excursion % during hold
    mae_price: float = 0.0
    mfe_price: float = 0.0
    post_exit_drift_5d: float = 0.0   # % move 5 days after exit
    post_exit_drift_10d: float = 0.0
    post_exit_drift_20d: float = 0.0
    exit_day_position: float | None = None  # 0=low, 1=high of exit day range


def _yf_symbol(symbol: str, exchange: str) -> str:
    """Convert internal symbol to yfinance ticker."""
    if exchange == "HKEX":
        # yfinance wants e.g. "0700.HK"
        if not symbol.endswith(".HK"):
            return symbol.lstrip("0") + ".HK" if symbol.isdigit() else symbol + ".HK"
    return symbol


def _fetch_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
    )
    if df.empty:
        return df
    df.index = df.index.date  # strip timezone, keep date
    return df[["Open", "High", "Low", "Close", "Volume"]]


def enrich_trade(trade: ClosedTrade) -> TradePrice | None:
    """Fetch price data and compute MAE/MFE/post-exit drift for one trade."""
    yf_sym = _yf_symbol(trade.symbol, trade.exchange)
    fetch_start = trade.entry_date - timedelta(days=7)
    fetch_end = trade.exit_date + timedelta(days=30)

    try:
        df = _fetch_ohlcv(yf_sym, fetch_start, fetch_end)
    except Exception as exc:
        log.warning("yfinance fetch failed for %s: %s", yf_sym, exc)
        return None

    if df.empty:
        log.warning("No price data for %s", yf_sym)
        return None

    tp = TradePrice(symbol=yf_sym, trade=trade, ohlcv=df)

    # Rows during hold period
    hold = df.loc[
        (df.index >= trade.entry_date) & (df.index <= trade.exit_date)
    ]
    if hold.empty:
        return tp

    entry_price = trade.avg_entry
    direction = trade.direction

    if direction == "LONG":
        worst = hold["Low"].min()
        best = hold["High"].max()
        tp.mae_pct = (entry_price - worst) / entry_price * 100
        tp.mfe_pct = (best - entry_price) / entry_price * 100
    else:  # SHORT
        worst = hold["High"].max()
        best = hold["Low"].min()
        tp.mae_pct = (worst - entry_price) / entry_price * 100
        tp.mfe_pct = (entry_price - best) / entry_price * 100

    tp.mae_price = worst
    tp.mfe_price = best

    # Exit day position in range (0 = sold at low, 1 = sold at high)
    exit_rows = df.loc[df.index == trade.exit_date]
    if not exit_rows.empty:
        row = exit_rows.iloc[0]
        day_range = row["High"] - row["Low"]
        if day_range > 0:
            tp.exit_day_position = (trade.avg_exit - row["Low"]) / day_range
        tp.exit_day_row = row

    # Post-exit drift
    post = df.loc[df.index > trade.exit_date]
    for days, attr in [(5, "post_exit_drift_5d"), (10, "post_exit_drift_10d"), (20, "post_exit_drift_20d")]:
        if len(post) >= days:
            future_close = post.iloc[days - 1]["Close"]
            drift = (future_close - trade.avg_exit) / trade.avg_exit * 100
            setattr(tp, attr, drift if direction == "LONG" else -drift)

    return tp


def enrich_all(trades: list[ClosedTrade]) -> dict[str, TradePrice]:
    """Return {trade_key: TradePrice} for all trades. Silently skips failures."""
    result = {}
    for t in trades:
        key = f"{t.symbol}_{t.entry_date}_{t.exit_date}"
        tp = enrich_trade(t)
        if tp:
            result[key] = tp
    return result
