"""
Assembles a complete coaching prompt from simulation results.
Output is plain text — copy and paste into Claude desktop.
"""
from __future__ import annotations
from analytics.portfolio import ClosedTrade
from analytics.metrics import summary_dict
from analytics.simulator import SimulationReport


_PERSONA = """You are an elite trading coach analysing a personal equities trader's performance.
You have been given:
1. Their complete trade history with key metrics
2. Simulation data showing what would have happened with different stop losses, profit targets, and entry timing — all calculated against ACTUAL market price data

Your job:
- Give specific, data-driven advice grounded in the numbers below
- Quote actual trade examples (symbol + date) when making a point
- Calculate exact improvements where the data supports it (e.g. "Moving your stop to 7% would have converted 3 losses into wins, improving your total PnL by $X")
- Cover: stop loss sizing, profit-taking method (strength vs weakness, trailing vs fixed target), entry timing
- Rank your findings by impact (biggest $ improvement first)
- End with 3 concrete rules the trader should apply from their next trade onwards

Do NOT give generic advice. Every point must reference the actual numbers below."""


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def _fmt_money(v: float) -> str:
    return f"{v:+,.0f}"


def build_prompt(
    trades: list[ClosedTrade],
    report: SimulationReport,
    user_question: str = "",
) -> str:
    sections: list[str] = [_PERSONA, ""]

    # ── Summary ────────────────────────────────────────────────────────────────
    m = summary_dict(trades)
    sections.append("=" * 60)
    sections.append("PORTFOLIO SUMMARY")
    sections.append("=" * 60)
    sections.append(f"Total closed trades : {m['total_trades']}")
    sections.append(f"Win rate            : {m['win_rate']:.1%}")
    sections.append(f"Profit factor       : {m['profit_factor']:.2f}")
    exp = m['expectancy_r']
    sections.append(f"Expectancy          : {f'{exp:.2f}R' if exp is not None else 'N/A'}")
    sections.append(f"Total PnL           : {m['total_pnl']:,.0f}")
    sections.append(f"Avg win             : {m['avg_win']:,.0f}")
    sections.append(f"Avg loss            : {m['avg_loss']:,.0f}")
    sections.append(f"Max drawdown        : {m['max_drawdown']:,.0f}")
    sections.append("")

    # ── Stop loss replay ───────────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("STOP LOSS REPLAY — WHAT IF YOU USED A WIDER/TIGHTER STOP?")
    sections.append("(Calculated against actual daily price data)")
    sections.append("=" * 60)

    stopped_trades = [r for r in report.stop_replays if r.was_stopped_out]
    not_stopped = [r for r in report.stop_replays if not r.was_stopped_out]

    sections.append(f"\nTrades stopped out: {len(stopped_trades)} of {len(report.stop_replays)}")
    sections.append("")

    for r in stopped_trades:
        t = r.trade
        sections.append(
            f"TRADE: {t.direction} {t.symbol}  |  Entry {t.entry_date} @ {t.avg_entry:.4f}  "
            f"|  Stopped @ {t.avg_exit:.4f}  |  PnL: {_fmt_money(t.realized_pnl)}"
        )
        sections.append(f"  Original stop: ~{r.original_stop_pct:.1f}%")
        sections.append("  Stop level scenarios:")
        for s in r.scenarios:
            marker = " ←" if abs(s['stop_pct'] - r.original_stop_pct / 100) < 0.005 else ""
            stopped_label = "STOPPED" if s['would_be_stopped'] else "SURVIVED"
            sections.append(
                f"    {s['stop_pct']*100:.0f}% stop ({s['stop_price']:.4f}): "
                f"{stopped_label}  |  PnL: {_fmt_money(s['simulated_pnl'])}  "
                f"[diff: {_fmt_money(s['pnl_difference'])}]{marker}"
            )
            if s['note']:
                sections.append(f"      → {s['note']}")
        sections.append("")

    # ── Take profit analysis ───────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("TAKE PROFIT ANALYSIS — DID YOU EXIT AT THE RIGHT TIME?")
    sections.append("=" * 60)

    for r in report.take_profit:
        t = r.trade
        sections.append(
            f"\nTRADE: {t.direction} {t.symbol}  |  Entry {t.entry_date} @ {t.avg_entry:.4f}  "
            f"|  Exit {t.exit_date} @ {t.avg_exit:.4f}  |  PnL: {_fmt_money(r.actual_pnl)}"
        )
        sections.append(
            f"  Max Favorable Excursion (MFE): {r.mfe_pct:.1f}%  |  Best possible PnL: {_fmt_money(r.mfe_pnl)}"
        )
        sections.append(f"  You captured {r.left_on_table_pct:.0f}% of the available move")

        # Exit style
        pos_label = f"{r.exit_day_position:.0%}" if r.exit_day_position is not None else "unknown"
        sections.append(
            f"  Exit style: {r.exit_style.upper()}  "
            f"(sold at {pos_label} of exit day's price range — 0%=low, 100%=high)"
        )

        # Post-exit drift
        drift5 = r.post_exit_drift_5d
        drift20 = r.post_exit_drift_20d
        continued = "YES — price kept moving in your favour" if r.post_exit_continued else "No significant continuation"
        sections.append(f"  Post-exit drift: 5d={_fmt_pct(drift5)}, 20d={_fmt_pct(drift20)}  →  {continued}")

        # Fixed targets
        sections.append("  Fixed R:R target scenarios:")
        for s in r.fixed_target_scenarios:
            hit = "HIT ✓" if s['target_hit'] else "not reached"
            sections.append(
                f"    {s['rr']}R target @ {s['target_price']:.4f}: {hit}  |  "
                f"PnL: {_fmt_money(s['simulated_pnl'])}  [diff: {_fmt_money(s['vs_actual'])}]"
            )

        # Trailing stops
        sections.append("  Trailing stop scenarios (from peak):")
        for s in r.trailing_scenarios:
            sections.append(
                f"    {s['trail_pct']*100:.0f}% trail: exit @ {s['exit_price']:.4f} on {s['exit_date']}  |  "
                f"PnL: {_fmt_money(s['simulated_pnl'])}  [diff: {_fmt_money(s['vs_actual'])}]"
            )
        sections.append("")

    # ── Entry timing ───────────────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("ENTRY TIMING ANALYSIS — DID YOU ENTER TOO EARLY?")
    sections.append("=" * 60)

    early_entries = [r for r in report.entry_timing if r.better_entry_available]
    sections.append(
        f"\n{len(early_entries)} of {len(report.entry_timing)} trades had a better entry "
        f"available within 3 days of actual entry.\n"
    )

    for r in report.entry_timing:
        if not r.better_entry_available and r.days_adverse_before_recovery == 0:
            continue  # skip clean entries to keep prompt shorter
        t = r.trade
        sections.append(
            f"TRADE: {t.direction} {t.symbol}  |  Entered {t.entry_date} @ {r.entry_price:.4f}"
        )
        if r.entry_day_low:
            sections.append(f"  Entry day range: low {r.entry_day_low:.4f}  high {r.entry_day_high:.4f}")
        sections.append(f"  Days adverse before recovery: {r.days_adverse_before_recovery}")
        if r.better_entry_available:
            sections.append(
                f"  Better entry available within 3 days: {r.better_entry_price:.4f}  "
                f"(would save {r.better_entry_saving_pct:.2f}% on entry)"
            )
        sections.append("")

    # ── User question ──────────────────────────────────────────────────────────
    sections.append("=" * 60)
    sections.append("MY QUESTION")
    sections.append("=" * 60)
    if user_question.strip():
        sections.append(user_question.strip())
    else:
        sections.append(
            "Based on all the data above, give me your most important findings. "
            "What are the top 3 changes I should make to improve my performance? "
            "Show me the exact $ impact of each change."
        )

    return "\n".join(sections)
