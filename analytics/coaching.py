"""
AI coaching via Claude API (claude-opus-4-7, adaptive thinking, prompt caching).

Cache layout:
  - system block: static persona + rules   → cached (rarely changes)
  - user block 1: trade history as CSV     → cached (changes once per session refresh)
  - user block 2: the question             → NOT cached (different every call)
"""
from __future__ import annotations
import csv
import io
import logging
from typing import Any

import anthropic

from analytics.portfolio import ClosedTrade
from config import get_secret

log = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are an elite trading coach and performance analyst for a personal equities trader.

Your role:
- Analyse the trader's closed trade history provided in each conversation
- Identify patterns in winning vs losing trades (instrument, direction, hold time, day of week, R-multiple)
- Give specific, data-driven feedback — reference actual trades by symbol and date
- Ask one clarifying question per response to deepen your understanding of the trader's process
- Never give generic advice; every insight must be grounded in the data
- Always express risk in R-multiples when relevant (1R = initial risk per share × position size)
- Maintain a coach's tone: honest, direct, encouraging, never sycophantic

You have deep knowledge of: technical analysis, position sizing, risk management, trading psychology,
FIFO accounting, and equity markets (NYSE + HKEX).

Format: Use markdown. Lead with the most important insight. End with one targeted question."""


def _trades_to_csv(trades: list[ClosedTrade]) -> str:
    """Serialize closed trades to CSV for the cached context block."""
    if not trades:
        return "No closed trades yet."

    buf = io.StringIO()
    fieldnames = [
        "symbol", "exchange", "currency", "direction",
        "entry_date", "exit_date", "quantity",
        "avg_entry", "avg_exit", "fees",
        "realized_pnl", "initial_stop", "r_multiple",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for t in sorted(trades, key=lambda x: x.exit_date):
        writer.writerow({
            "symbol": t.symbol,
            "exchange": t.exchange,
            "currency": t.currency,
            "direction": t.direction,
            "entry_date": t.entry_date.isoformat(),
            "exit_date": t.exit_date.isoformat(),
            "quantity": t.quantity,
            "avg_entry": round(t.avg_entry, 4),
            "avg_exit": round(t.avg_exit, 4),
            "fees": round(t.fees, 2),
            "realized_pnl": round(t.realized_pnl, 2),
            "initial_stop": round(t.initial_stop, 4) if t.initial_stop else "",
            "r_multiple": round(t.r_multiple, 2) if t.r_multiple is not None else "",
        })
    return buf.getvalue()


def get_coaching(
    question: str,
    trades: list[ClosedTrade],
    stream: bool = True,
) -> str:
    """
    Ask the AI coach a question about the trade history.

    Args:
        question: The user's free-form question.
        trades:   Closed trade list from analytics.portfolio.build_portfolio().
        stream:   Use streaming (recommended for long responses).

    Returns:
        The assistant's response as a string.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    trade_csv = _trades_to_csv(trades)
    trade_context = (
        f"Here is the complete closed trade history in CSV format:\n\n```csv\n{trade_csv}\n```"
    )

    messages = [
        {
            "role": "user",
            "content": [
                # Block 1: trade history — cached (stable across calls in same session)
                {
                    "type": "text",
                    "text": trade_context,
                    "cache_control": {"type": "ephemeral"},
                },
                # Block 2: the question — NOT cached (changes every call)
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    if stream:
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        ) as s:
            return s.get_final_message().content[0].text
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        )
        # Extract text from content blocks (skip thinking blocks)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks)
