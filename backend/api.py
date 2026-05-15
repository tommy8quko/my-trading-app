"""FastAPI backend — serves all data to the React frontend."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import date
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analytics.portfolio import build_portfolio
from analytics.metrics import summary_dict, breakdown_by_direction, breakdown_by_dow, breakdown_by_holding_period, cumulative_pnl, equity_curve_full
from ingestion.review_queue import fetch_pending, approve_item, reject_item
from ingestion.pipeline import ingest_file
from db.client import get_supabase

# Symbols excluded from all metrics/coach (cash-parking instruments, not trades)
EXCLUDED_SYMBOLS = {"SGOV", "BOXX"}

# FX rate used to convert HKD → USD in the "ALL" combined view
_FX_HKD_USD = 7.8

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Live price helper ─────────────────────────────────────────────────────────

import time as _time
_price_cache: dict[str, float | None] = {}
_price_cache_ts: float = 0.0
_PRICE_TTL = 300  # seconds (5 minutes)

def _symbol_to_yf_ticker(symbol: str, exchange: str) -> str:
    """Convert a stored symbol to the yfinance ticker format."""
    ex = (exchange or "").upper()
    # HK Exchange: strip leading zeros from numeric symbols, append .HK
    if ex in ("HKEX", "SEHK", "HKG", "HK") or (symbol.isdigit() and ex not in ("NYSE", "NASDAQ", "AMEX")):
        return f"{symbol.lstrip('0') or '0'}.HK"
    return symbol


def _fetch_prices(positions: list[tuple[str, str]]) -> dict[str, float | None]:
    """Fetch last close via yfinance; cached for 5 minutes to avoid slow repeated calls."""
    global _price_cache, _price_cache_ts
    if not positions:
        return {}

    now = _time.time()
    symbols_requested = {sym for sym, _ in positions}

    # Return cache if fresh and covers all requested symbols
    if now - _price_cache_ts < _PRICE_TTL and symbols_requested <= _price_cache.keys():
        return {sym: _price_cache[sym] for sym in symbols_requested}

    ticker_map: dict[str, str] = {
        _symbol_to_yf_ticker(sym, ex): sym for sym, ex in positions
    }
    tickers = list(ticker_map.keys())
    try:
        import yfinance as yf
        import pandas as pd
        data = yf.download(tickers, period="2d", progress=False,
                           auto_adjust=True, threads=False)
        close = data["Close"] if not isinstance(data.columns, pd.MultiIndex) else data["Close"]
        result: dict[str, float | None] = {}
        for ticker, orig_sym in ticker_map.items():
            try:
                series = close[ticker] if len(tickers) > 1 else close
                val = series.dropna().iloc[-1]
                result[orig_sym] = float(val) if not pd.isna(val) else None
            except Exception:
                result[orig_sym] = None
        _price_cache = result
        _price_cache_ts = now
        return result
    except Exception:
        return {sym: None for sym, _ in positions}


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    open_pos, closed = build_portfolio()

    prices = _fetch_prices([(p.symbol, p.exchange) for p in open_pos])

    def _pos_pnl(p, price):
        if price is None:
            return None, None
        if p.direction == "LONG":
            pnl = (price - p.avg_entry_price) * p.total_quantity
        else:
            pnl = (p.avg_entry_price - price) * p.total_quantity
        cost = p.avg_entry_price * p.total_quantity
        pct  = (pnl / cost * 100) if cost else None
        return round(pnl, 2), round(pct, 2) if pct is not None else None

    return {
        "open_positions": [
            {
                "symbol":        p.symbol,
                "exchange":      p.exchange,
                "currency":      p.currency,
                "direction":     p.direction,
                "quantity":      p.total_quantity,
                "avg_entry":     round(p.avg_entry_price, 4),
                "stop_loss":     p.initial_stop_loss,
                "entry_date":    str(p.earliest_entry_date) if p.earliest_entry_date else None,
                "current_price": round(prices.get(p.symbol), 4) if prices.get(p.symbol) else None,
                "unrealized_pnl": _pos_pnl(p, prices.get(p.symbol))[0],
                "pct_return":     _pos_pnl(p, prices.get(p.symbol))[1],
                "lots": [
                    {
                        "order_id":   l.order_id,
                        "date":       str(l.order_date),
                        "action":     l.action,
                        "quantity":   l.quantity,
                        "entry_price":l.entry_price,
                        "stop_loss":  l.stop_loss_at_entry,
                    }
                    for l in p.lots
                ],
            }
            for p in open_pos
        ],
        "closed_trades": [
            {
                "symbol":       t.symbol,
                "exchange":     t.exchange,
                "currency":     t.currency,
                "direction":    t.direction,
                "entry_date":   str(t.entry_date),
                "exit_date":    str(t.exit_date),
                "quantity":     t.quantity,
                "avg_entry":    t.avg_entry,
                "avg_exit":     t.avg_exit,
                "fees":         round(t.fees, 2),
                "pnl":          round(t.realized_pnl, 2),
                "pct_return":   round(
                    ((t.avg_exit - t.avg_entry) / t.avg_entry * 100) if t.direction == "LONG"
                    else ((t.avg_entry - t.avg_exit) / t.avg_entry * 100), 2
                ) if t.avg_entry else None,
                "r_multiple":   round(t.r_multiple, 2) if t.r_multiple is not None else None,
                "initial_stop": t.initial_stop,
            }
            for t in sorted(closed, key=lambda x: (x.exit_date, x.exit_time or ''), reverse=True)
        ],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics(currency: str = "USD"):
    import copy
    open_pos, closed = build_portfolio()
    pending = len(fetch_pending(limit=200))
    src_currencies = sorted({t.currency for t in closed}) if closed else ["USD"]
    currencies_out = (["ALL"] if len(src_currencies) > 1 else []) + src_currencies

    if currency == "ALL":
        # Convert non-USD trades to USD at the fixed FX rate, then compute combined metrics
        def _to_usd(t):
            if t.currency == "USD":
                return t
            tc = copy.copy(t)
            tc.realized_pnl = t.realized_pnl / _FX_HKD_USD
            return tc
        filtered = [_to_usd(t) for t in closed if t.symbol not in EXCLUDED_SYMBOLS]
        # Equity curve uses only closed trades (open-position unrealized P&L omitted for ALL)
        cum = equity_curve_full(filtered, [], {})
    else:
        filtered = [t for t in closed if t.currency == currency and t.symbol not in EXCLUDED_SYMBOLS]
        open_filtered = [p for p in open_pos if p.currency == currency]
        prices = _fetch_prices([(p.symbol, p.exchange) for p in open_filtered]) if open_filtered else {}
        cum = equity_curve_full(filtered, open_filtered, prices)

    m = summary_dict(filtered) if filtered else {}
    return {
        "total_pnl":           round(m.get("total_pnl", 0), 2),
        "win_rate":            round(m.get("win_rate", 0), 4),
        "profit_factor":       round(m.get("profit_factor", 0), 2),
        "expectancy_r":        round(m.get("expectancy_r") or 0, 2),
        "rr_ratio":            round(m.get("rr_ratio"), 2) if m.get("rr_ratio") is not None else None,
        "avg_win":             round(m.get("avg_win", 0), 2),
        "avg_loss":            round(m.get("avg_loss", 0), 2),
        "max_drawdown":        round(m.get("max_drawdown", 0), 2),
        "total_trades":        m.get("total_trades", 0),
        "pending_review":      pending,
        "current_streak":      m.get("current_streak", {"count": 0, "type": None}),
        "longest_win_streak":  m.get("longest_win_streak", 0),
        "longest_loss_streak": m.get("longest_loss_streak", 0),
        "avg_hold_winners":    m.get("avg_hold_winners", 0),
        "avg_hold_losers":     m.get("avg_hold_losers", 0),
        "equity_curve":        [{"date": str(d), "pnl": round(v, 2)} for d, v in cum],
        "currencies":          currencies_out,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats(currency: str = "USD"):
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency and t.symbol not in EXCLUDED_SYMBOLS]
    return {
        "by_direction": breakdown_by_direction(filtered),
        "by_dow": breakdown_by_dow(filtered),
        "by_holding": breakdown_by_holding_period(filtered),
    }


# ── Review Queue ──────────────────────────────────────────────────────────────

@app.get("/api/review/pending")
def get_pending():
    rows = fetch_pending(limit=200)
    result = []
    for row in rows:
        raw = row.get("raw_parsed_json") or {}
        norm = row.get("normalized_json") or {}
        result.append({
            "id": row["id"],
            "symbol": raw.get("symbol", "?"),
            "action": raw.get("action") or norm.get("action", "?"),
            "quantity": raw.get("quantity") or norm.get("quantity", "?"),
            "broker": raw.get("broker_short_code", "?"),
            "confidence": row.get("confidence_score") or 0,
            "raw": raw,
            "normalized": norm,
        })
    return result


class ApproveRequest(BaseModel):
    notes: str = ""
    stop_loss: float | None = None

@app.post("/api/review/{queue_id}/approve")
async def approve(queue_id: str, request: Request):
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        notes = body.get("notes", "")
        sl = body.get("stop_loss")
        stop_loss = float(sl) if sl else None
        approve_item(queue_id, notes=notes, stop_loss=stop_loss)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/review/{queue_id}/reject")
def reject(queue_id: str, notes: str = ""):
    try:
        reject_item(queue_id, notes=notes)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))


# ── Gmail Sync ────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    since: str  # ISO date string

@app.post("/api/sync/gmail")
def sync_gmail(req: SyncRequest):
    from email_connector.sync import sync_emails
    since = date.fromisoformat(req.since)
    result = sync_emails(since=since)
    return {
        "queued": result.queued,
        "duplicates": result.duplicates,
        "skipped_already_processed": result.skipped_already_processed,
        "skipped_no_adapter": result.skipped_no_adapter,
        "skipped_no_order": result.skipped_no_order,
        "skipped_no_account": result.skipped_no_account,
        "errors": result.errors[:10],
        "senders_seen": sorted(set(getattr(result, "senders_seen", []))),
    }


# ── File Import ───────────────────────────────────────────────────────────────

@app.post("/api/import/file")
async def import_file(file: UploadFile = File(...), broker: str = Form(...)):
    content = await file.read()
    result = ingest_file(content, file.filename, broker)
    return {
        "queued": result.queued,
        "duplicates": result.duplicates,
        "errors": result.errors,
    }


# ── Orders (edit / delete / add) ─────────────────────────────────────────────

class OrderUpdate(BaseModel):
    action: str
    quantity: float
    stop_loss: float | None = None

class OrderInsert(BaseModel):
    account_id: str
    instrument_id: str
    action: str
    order_date: str
    price: float
    quantity: float
    fees: float = 0.0
    currency: str
    stop_loss: float | None = None

class StopLossUpdate(BaseModel):
    stop_loss: float | None = None

@app.patch("/api/orders/{order_id}/stop_loss")
def update_stop_loss(order_id: str, body: StopLossUpdate):
    get_supabase().table("orders").update({
        "stop_loss_at_entry": body.stop_loss if body.stop_loss and body.stop_loss > 0 else None,
    }).eq("id", order_id).execute()
    return {"ok": True}

@app.put("/api/orders/{order_id}")
def update_order(order_id: str, body: OrderUpdate):
    get_supabase().table("orders").update({
        "action": body.action,
        "quantity": body.quantity,
        "stop_loss_at_entry": body.stop_loss if body.stop_loss and body.stop_loss > 0 else None,
    }).eq("id", order_id).execute()
    return {"ok": True}

@app.delete("/api/orders/{order_id}")
def delete_order(order_id: str):
    get_supabase().table("orders").delete().eq("id", order_id).execute()
    return {"ok": True}

@app.post("/api/orders")
def insert_order(body: OrderInsert):
    get_supabase().table("orders").insert({
        "account_id": body.account_id,
        "instrument_id": body.instrument_id,
        "action": body.action,
        "order_date": body.order_date,
        "order_time": None,
        "price": body.price,
        "quantity": body.quantity,
        "fees": body.fees,
        "currency": body.currency,
        "stop_loss_at_entry": body.stop_loss if body.stop_loss and body.stop_loss > 0 else None,
        "external_order_id": f"MANUAL_{body.action}_{body.order_date}_{body.quantity}_{body.price}",
    }).execute()
    return {"ok": True}

@app.get("/api/orders/{order_id}/meta")
def get_order_meta(order_id: str):
    result = get_supabase().table("orders").select("account_id, instrument_id").eq("id", order_id).single().execute()
    return result.data


# ── Diagnostics ──────────────────────────────────────────────────────────────

@app.get("/api/diagnostics/subjects")
def get_diagnostic_subjects(since: str = None):
    from email_connector.connector import GmailConnector
    from email_connector.sync import BROKER_SENDER_FILTERS
    if since:
        since_date = date.fromisoformat(since)
    else:
        from datetime import timedelta
        since_date = date.today() - timedelta(days=30)
    subjects = []
    with GmailConnector() as gmail:
        for sender_filter in BROKER_SENDER_FILTERS:
            for raw in gmail.fetch_emails(since=since_date, sender_filter=sender_filter, max_results=200):
                subjects.append({
                    "subject": raw.get("subject", ""),
                    "from": raw.get("from_address", ""),
                    "received_at": str(raw.get("received_at", "")),
                })
    return subjects


@app.get("/api/diagnostics/chief-email")
def get_chief_email_diagnostic(since: str = None):
    from email_connector.connector import GmailConnector
    from ingestion.brokers.chief import _strip_html, _INTRO_RE
    if since:
        since_date = date.fromisoformat(since)
    else:
        from datetime import timedelta
        since_date = date.today() - timedelta(days=7)
    results = []
    with GmailConnector() as gmail:
        for raw in gmail.fetch_emails(since=since_date, sender_filter="chiefgroup.com.hk", max_results=20):
            stripped = _strip_html(raw["body"])
            m = _INTRO_RE.search(stripped)
            results.append({
                "subject": raw.get("subject", ""),
                "stripped_text": stripped[:500],
                "intro_match": m.group(0) if m else None,
            })
    return results


# ── Trade Reviews (tags, notes, MAE/MFE) ─────────────────────────────────────

class TradeReviewBody(BaseModel):
    symbol: str
    entry_date: str
    setup_tag: str | None = None
    outcome_reason: str | None = None
    market_condition: str | None = None
    emotional_notes: str | None = None
    mae: float | None = None
    mfe: float | None = None

@app.get("/api/trades/review")
def get_trade_review(symbol: str, entry_date: str):
    result = get_supabase().table("trade_reviews").select("*") \
        .eq("symbol", symbol).eq("entry_date", entry_date).execute()
    return result.data[0] if result.data else {}

@app.post("/api/trades/review")
def save_trade_review(body: TradeReviewBody):
    row = {k: v for k, v in body.model_dump().items() if v is not None}
    get_supabase().table("trade_reviews").upsert(
        row, on_conflict="symbol,entry_date"
    ).execute()
    return {"ok": True}

@app.get("/api/trades/maemfe")
def get_maemfe(symbol: str, exchange: str, entry_date: str, exit_date: str,
               direction: str, entry_price: float):
    try:
        import yfinance as yf
        ticker = _symbol_to_yf_ticker(symbol, exchange)
        hist = yf.Ticker(ticker).history(start=entry_date, end=exit_date,
                                         interval="1d", auto_adjust=True)
        if hist.empty:
            return {"mae": None, "mfe": None}
        low_min  = float(hist["Low"].min())
        high_max = float(hist["High"].max())
        if direction == "LONG":
            mae = round(max(0.0, entry_price - low_min), 4)
            mfe = round(max(0.0, high_max - entry_price), 4)
        else:
            mae = round(max(0.0, high_max - entry_price), 4)
            mfe = round(max(0.0, entry_price - low_min), 4)
        return {"mae": mae, "mfe": mfe}
    except Exception:
        return {"mae": None, "mfe": None}


# ── AI Coach ──────────────────────────────────────────────────────────────────

FOCUS_INSTRUCTIONS = {
    "general":    "Provide a comprehensive analysis covering all aspects of performance.",
    "psychology": "Focus on psychological patterns: revenge trading, tilt, overconfidence after wins, fear after losses, and whether the trader holds losers longer than winners.",
    "entries":    "Focus on entry and exit quality: are entries too early or late, are winners cut too short (MFE >> exit), are losers held too long (MAE >> loss).",
    "sizing":     "Focus on position sizing: is it consistent, does performance correlate with size, are larger positions managed differently?",
}

class CoachRequest(BaseModel):
    notes: str = ""
    focus: str = "general"

@app.post("/api/coach/generate")
def generate_coach(req: CoachRequest):
    _, closed_all = build_portfolio()
    closed = [t for t in closed_all if t.symbol not in EXCLUDED_SYMBOLS]
    m = summary_dict(closed) if closed else {}
    if not closed:
        return {"response": "No closed trades yet — nothing to analyse."}

    sorted_trades = sorted(closed, key=lambda t: t.exit_date)
    winners = [t for t in closed if t.realized_pnl > 0]
    losers  = [t for t in closed if t.realized_pnl <= 0]

    # All trades table — include currency so AI knows USD vs HKD
    trade_lines = "\n".join(
        f"{t.symbol}({t.currency}) {t.direction} | {t.entry_date}→{t.exit_date} "
        f"| hold={(t.exit_date-t.entry_date).days}d "
        f"| entry={t.avg_entry:.4f} exit={t.avg_exit:.4f} "
        f"| pnl={t.realized_pnl:.2f}{t.currency} | R={f'{t.r_multiple:.2f}' if t.r_multiple is not None else 'N/A'}"
        for t in sorted_trades
    )

    # Per-currency summary metrics
    currencies = sorted({t.currency for t in closed})
    def _trade_str(t):
        return f"{t.symbol}({t.currency}) {t.direction} {t.exit_date} PnL={t.realized_pnl:.2f}{t.currency} R={t.r_multiple or 'N/A'}"

    ccy_sections = []
    for ccy in currencies:
        ccy_trades = [t for t in closed if t.currency == ccy]
        cm = summary_dict(ccy_trades)
        r_trades = [t for t in ccy_trades if t.r_multiple is not None]
        rank_key = (lambda t: t.r_multiple) if r_trades else (lambda t: t.realized_pnl)
        rank_set = r_trades if r_trades else ccy_trades
        best5  = sorted(rank_set, key=rank_key, reverse=True)[:5]
        worst5 = sorted(rank_set, key=rank_key)[:5]
        sym_pnl: dict[str, list[float]] = {}
        for t in ccy_trades:
            sym_pnl.setdefault(t.symbol, []).append(t.realized_pnl)
        sym_lines = "\n".join(
            f"  {sym}: {len(pnls)} trades, total={sum(pnls):.2f}{ccy}, win%={100*sum(1 for p in pnls if p>0)/len(pnls):.0f}%"
            for sym, pnls in sorted(sym_pnl.items(), key=lambda x: sum(x[1]), reverse=True)
        )
        ccy_sections.append(f"""
--- {ccy} TRADES ({len(ccy_trades)} trades, US stocks use USD / HK stocks use HKD) ---
Win rate: {(cm['win_rate'] or 0):.1%} | Profit factor: {(cm['profit_factor'] or 0):.2f} | Expectancy: {(cm['expectancy_r'] or 0):.2f}R
Avg win: {cm['avg_win']:.2f}{ccy} | Avg loss: {cm['avg_loss']:.2f}{ccy} | Max drawdown: {cm['max_drawdown']:.2f}{ccy}
Avg hold winners: {cm['avg_hold_winners']}d | Avg hold losers: {cm['avg_hold_losers']}d

Best 5: {' | '.join(_trade_str(t) for t in best5)}
Worst 5: {' | '.join(_trade_str(t) for t in worst5)}

By symbol:
{sym_lines}""")

    streak = m.get("current_streak", {})
    focus_text = FOCUS_INSTRUCTIONS.get(req.focus, FOCUS_INSTRUCTIONS["general"])

    prompt = f"""You are a professional quantitative trading performance analyst reviewing a trader's complete history.
Note: this trader trades both US stocks (P&L in USD) and HK stocks (P&L in HKD). These are separate currencies — do not add them together.

FOCUS: {focus_text}

=== OVERALL METRICS (all currencies combined for streak/pattern analysis) ===
Total trades: {m['total_trades']} | Current streak: {streak.get('count',0)}{streak.get('type','')} | Best win streak: {m['longest_win_streak']} | Worst loss streak: {m['longest_loss_streak']}

=== METRICS BY CURRENCY ===
{''.join(ccy_sections)}

=== ALL TRADES (chronological, currency shown per trade) ===
{trade_lines}

=== TRADER NOTES ===
{req.notes or 'None provided'}

Respond with 4–6 specific, data-driven insights. Reference actual numbers and currencies. Analyse USD and HKD performance separately where relevant. End with 3 concrete actions the trader should take next week."""

    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"response": message.content[0].text}
    except Exception as e:
        return {"response": f"AI coach unavailable: {e}\n\nPrompt sent:\n\n{prompt}"}


# ── SPY benchmark ─────────────────────────────────────────────────────────────

_spy_cache: list = []
_spy_cache_ts: float = 0.0
_SPY_TTL = 3600  # 1-hour cache for historical data

def _fetch_spy(from_date: str) -> list[dict]:
    global _spy_cache, _spy_cache_ts
    now = _time.time()
    if _spy_cache and now - _spy_cache_ts < _SPY_TTL:
        return [p for p in _spy_cache if p["date"] >= from_date]
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").history(start=from_date, period="max", auto_adjust=True)
        if spy.empty:
            return []
        base = float(spy["Close"].iloc[0])
        _spy_cache = [
            {"date": str(d.date()), "pct": round((float(c) / base - 1) * 100, 2)}
            for d, c in zip(spy.index, spy["Close"])
        ]
        _spy_cache_ts = now
        return _spy_cache
    except Exception:
        return []


@app.get("/api/stats/spy")
def get_spy(from_date: str = None):
    if not from_date:
        from datetime import timedelta
        from_date = str(date.today() - timedelta(days=730))
    return _fetch_spy(from_date)


# ── Setup tag performance ─────────────────────────────────────────────────────

@app.get("/api/stats/setup_tags")
def get_setup_tag_stats(currency: str = "USD"):
    from collections import defaultdict
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency and t.symbol not in EXCLUDED_SYMBOLS]
    trade_lookup = {(t.symbol, str(t.entry_date)): t for t in filtered}

    reviews = get_supabase().table("trade_reviews").select("symbol,entry_date,setup_tag").execute().data

    tag_trades: dict[str, list] = defaultdict(list)
    for rev in reviews:
        tag = rev.get("setup_tag")
        if not tag:
            continue
        trade = trade_lookup.get((rev["symbol"], rev["entry_date"]))
        if trade:
            tag_trades[tag].append(trade)

    result = {}
    for tag, trades in tag_trades.items():
        winners = [t for t in trades if t.realized_pnl > 0]
        r_trades = [t for t in trades if t.r_multiple is not None]
        result[tag] = {
            "count": len(trades),
            "win_rate": round(len(winners) / len(trades), 4),
            "avg_r": round(sum(t.r_multiple for t in r_trades) / len(r_trades), 2) if r_trades else None,
            "total_pnl": round(sum(t.realized_pnl for t in trades), 2),
            "avg_pnl": round(sum(t.realized_pnl for t in trades) / len(trades), 2),
        }
    return result


# ── Market condition performance ─────────────────────────────────────────────

@app.get("/api/stats/market_condition")
def get_market_condition_stats(currency: str = "USD"):
    from collections import defaultdict
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency and t.symbol not in EXCLUDED_SYMBOLS]
    trade_lookup = {(t.symbol, str(t.entry_date)): t for t in filtered}
    reviews = get_supabase().table("trade_reviews").select("symbol,entry_date,market_condition").execute().data
    cond_trades: dict[str, list] = defaultdict(list)
    for rev in reviews:
        cond = rev.get("market_condition")
        if not cond:
            continue
        trade = trade_lookup.get((rev["symbol"], rev["entry_date"]))
        if trade:
            cond_trades[cond].append(trade)
    result = {}
    for cond, trades in cond_trades.items():
        winners = [t for t in trades if t.realized_pnl > 0]
        result[cond] = {
            "count": len(trades),
            "win_rate": round(len(winners) / len(trades), 4),
            "total_pnl": round(sum(t.realized_pnl for t in trades), 2),
            "avg_pnl": round(sum(t.realized_pnl for t in trades) / len(trades), 2),
        }
    return result


# ── MAE/MFE aggregate analysis ────────────────────────────────────────────────

@app.get("/api/stats/maemfe_analysis")
def get_maemfe_analysis(currency: str = "USD"):
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency and t.symbol not in EXCLUDED_SYMBOLS]
    trade_lookup = {(t.symbol, str(t.entry_date)): t for t in filtered}

    reviews = get_supabase().table("trade_reviews").select("*").execute().data

    result = []
    for rev in reviews:
        mae = rev.get("mae")
        mfe = rev.get("mfe")
        if mae is None or mfe is None:
            continue
        trade = trade_lookup.get((rev["symbol"], rev["entry_date"]))
        if not trade:
            continue

        mae_f, mfe_f = float(mae), float(mfe)
        pps = trade.realized_pnl / trade.quantity if trade.quantity else 0
        risk = abs(trade.avg_entry - trade.initial_stop) if trade.initial_stop else None

        result.append({
            "symbol": trade.symbol,
            "entry_date": str(trade.entry_date),
            "exit_date": str(trade.exit_date),
            "setup_tag": rev.get("setup_tag"),
            "direction": trade.direction,
            "mae": mae_f,
            "mfe": mfe_f,
            "actual_pps": round(pps, 4),
            "mfe_efficiency": round(pps / mfe_f * 100, 1) if mfe_f > 0 else None,
            "mae_r": round(mae_f / risk, 2) if risk and risk > 0 else None,
            "mfe_r": round(mfe_f / risk, 2) if risk and risk > 0 else None,
            "r_multiple": trade.r_multiple,
            "pnl": round(trade.realized_pnl, 2),
        })

    return sorted(result, key=lambda x: x["exit_date"], reverse=True)


# ── Weekly review ─────────────────────────────────────────────────────────────

def _build_weekly_prompt() -> str | None:
    """Build the weekly review prompt. Returns None if no trades this week."""
    from datetime import timedelta
    _, closed_all = build_portfolio()
    closed = [t for t in closed_all if t.symbol not in EXCLUDED_SYMBOLS]

    today = date.today()
    week_start = today - timedelta(days=7)
    week_trades = [t for t in closed if t.exit_date >= week_start]

    if not week_trades:
        return None

    m = summary_dict(week_trades)
    trade_lines = "\n".join(
        f"  {t.symbol}({t.currency}) {t.direction} | exit={t.exit_date}"
        f" | pnl={t.realized_pnl:.2f}{t.currency}"
        f" | R={f'{t.r_multiple:.2f}' if t.r_multiple is not None else 'N/A'}"
        for t in sorted(week_trades, key=lambda t: t.exit_date)
    )

    return f"""You are a professional trading coach reviewing a trader's week.

=== THIS WEEK ({week_start} to {today}) ===
Trades: {m['total_trades']} | Win rate: {m['win_rate']:.1%} | Profit factor: {m['profit_factor']:.2f}
Avg win: {m['avg_win']:.2f} | Avg loss: {m['avg_loss']:.2f} | Expectancy: {(m.get('expectancy_r') or 0):.2f}R

{trade_lines}

Write a concise weekly review with these 4 sections:
1. What worked well (specific trades/patterns, reference actual numbers)
2. What didn't work (specific trades/mistakes)
3. Key execution observation for this week
4. One concrete focus for next week

Keep each point to 2-3 sentences. Be specific and direct."""


@app.get("/api/coach/weekly_review_prompt")
def get_weekly_review_prompt():
    """Return the raw prompt text — copy and paste into any LLM."""
    prompt = _build_weekly_prompt()
    if prompt is None:
        return {"prompt": "No trades closed in the past 7 days — nothing to review."}
    return {"prompt": prompt}


@app.post("/api/coach/weekly_review")
def generate_weekly_review():
    prompt = _build_weekly_prompt()
    if prompt is None:
        return {"response": "No trades closed in the past 7 days — nothing to review."}
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"response": message.content[0].text}
    except Exception as e:
        return {"response": f"AI unavailable: {e}"}


# ── Static frontend ───────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{path:path}")
def catch_all(path: str):
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
