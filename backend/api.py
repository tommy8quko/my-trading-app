"""FastAPI backend — serves all data to the React frontend."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import date
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analytics.portfolio import build_portfolio
from analytics.metrics import summary_dict, breakdown_by_direction, breakdown_by_dow, breakdown_by_holding_period, cumulative_pnl
from ingestion.review_queue import fetch_pending, approve_item, reject_item
from ingestion.pipeline import ingest_file
from db.client import get_supabase

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Live price helper ─────────────────────────────────────────────────────────

def _symbol_to_yf_ticker(symbol: str, exchange: str) -> str:
    """Convert a stored symbol to the yfinance ticker format."""
    ex = (exchange or "").upper()
    # HK Exchange: strip leading zeros from numeric symbols, append .HK
    if ex in ("HKEX", "SEHK", "HKG", "HK") or (symbol.isdigit() and ex not in ("NYSE", "NASDAQ", "AMEX")):
        return f"{symbol.lstrip('0') or '0'}.HK"
    return symbol


def _fetch_prices(positions: list[tuple[str, str]]) -> dict[str, float | None]:
    """Fetch last close via yfinance for (symbol, exchange) pairs; keyed by original symbol."""
    if not positions:
        return {}
    # Build ticker→original_symbol mapping (multiple symbols may share a ticker in edge cases)
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
                "symbol": t.symbol,
                "currency": t.currency,
                "direction": t.direction,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "quantity": t.quantity,
                "avg_entry": t.avg_entry,
                "avg_exit": t.avg_exit,
                "fees": round(t.fees, 2),
                "pnl": round(t.realized_pnl, 2),
                "r_multiple": round(t.r_multiple, 2) if t.r_multiple is not None else None,
                "initial_stop": t.initial_stop,
            }
            for t in sorted(closed, key=lambda x: x.exit_date, reverse=True)
        ],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics(currency: str = "USD"):
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency]
    m = summary_dict(filtered) if filtered else {}
    pending = len(fetch_pending(limit=200))
    cum = cumulative_pnl(filtered)
    return {
        "total_pnl": round(m.get("total_pnl", 0), 2),
        "win_rate": round(m.get("win_rate", 0), 4),
        "profit_factor": round(m.get("profit_factor", 0), 2),
        "expectancy_r": round(m.get("expectancy_r") or 0, 2),
        "avg_win": round(m.get("avg_win", 0), 2),
        "avg_loss": round(m.get("avg_loss", 0), 2),
        "max_drawdown": round(m.get("max_drawdown", 0), 2),
        "total_trades": m.get("total_trades", 0),
        "pending_review": pending,
        "equity_curve": [{"date": str(d), "pnl": round(v, 2)} for d, v in cum],
        "currencies": sorted({t.currency for t in closed}) if closed else ["USD"],
    }


# ── Statistics ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats(currency: str = "USD"):
    _, closed = build_portfolio()
    filtered = [t for t in closed if t.currency == currency]
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
def approve(queue_id: str, req: ApproveRequest = None):
    req = req or ApproveRequest()
    try:
        approve_item(queue_id, notes=req.notes, stop_loss=req.stop_loss)
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


# ── AI Coach ──────────────────────────────────────────────────────────────────

class CoachRequest(BaseModel):
    notes: str = ""

@app.post("/api/coach/generate")
def generate_coach(req: CoachRequest):
    _, closed = build_portfolio()
    m = summary_dict(closed) if closed else {}
    recent = sorted(closed, key=lambda t: t.exit_date, reverse=True)[:10]
    trade_lines = "\n".join(
        f"- {t.symbol} {t.direction} exit {t.exit_date}: PnL={t.realized_pnl:.2f} R={t.r_multiple or 'N/A'}"
        for t in recent
    )
    prompt = (
        f"Trading performance summary:\n"
        f"Win rate: {(m.get('win_rate') or 0):.1%}, "
        f"Profit factor: {(m.get('profit_factor') or 0):.2f}, "
        f"Expectancy: {(m.get('expectancy_r') or 0):.2f}R, "
        f"Total trades: {m.get('total_trades') or 0}\n\n"
        f"Recent trades:\n{trade_lines}\n\n"
        f"Trader notes: {req.notes or 'None provided'}\n\n"
        f"As a trading coach, give 3-5 concise actionable insights based on these results."
    )
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
        return {"response": f"AI coach unavailable: {e}\n\nPrompt that would have been sent:\n\n{prompt}"}


# ── Static frontend ───────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{path:path}")
def catch_all(path: str):
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
