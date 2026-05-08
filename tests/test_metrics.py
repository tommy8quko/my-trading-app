"""Unit tests for analytics.metrics."""
import pytest
from datetime import date
from analytics.portfolio import ClosedTrade
from analytics.metrics import (
    win_rate, profit_factor, expectancy_r,
    avg_win, avg_loss, max_drawdown, summary_dict,
)


def _trade(pnl: float, r: float | None = None, direction: str = "LONG",
           entry: str = "2024-01-01", exit_: str = "2024-01-05") -> ClosedTrade:
    return ClosedTrade(
        symbol="TEST",
        exchange="NYSE",
        currency="USD",
        direction=direction,
        entry_date=date.fromisoformat(entry),
        exit_date=date.fromisoformat(exit_),
        quantity=100,
        avg_entry=10.0,
        avg_exit=10.0 + pnl / 100,
        fees=0,
        realized_pnl=pnl,
        initial_stop=None,
        r_multiple=r,
    )


TRADES = [
    _trade(200, r=2.0),
    _trade(150, r=1.5),
    _trade(-100, r=-1.0),
    _trade(-80, r=-0.8),
    _trade(300, r=3.0),
]


def test_win_rate():
    assert win_rate(TRADES) == pytest.approx(3 / 5)


def test_profit_factor():
    gross_win = 200 + 150 + 300
    gross_loss = 100 + 80
    assert profit_factor(TRADES) == pytest.approx(gross_win / gross_loss)


def test_expectancy_r():
    # (3/5 * mean(2, 1.5, 3)) - (2/5 * mean(1, 0.8))
    wr = 3 / 5
    lr = 2 / 5
    avg_w = (2 + 1.5 + 3) / 3
    avg_l = (1 + 0.8) / 2
    expected = wr * avg_w - lr * avg_l
    assert expectancy_r(TRADES) == pytest.approx(expected)


def test_expectancy_r_no_r_multiples():
    trades = [_trade(100), _trade(-50)]
    assert expectancy_r(trades) is None


def test_avg_win():
    assert avg_win(TRADES) == pytest.approx((200 + 150 + 300) / 3)


def test_avg_loss():
    assert avg_loss(TRADES) == pytest.approx((-100 + -80) / 2)


def test_max_drawdown():
    # equity: 200 → 350 → 250 → 170 → 470
    # peak at 350; dip to 170 = dd of 180
    assert max_drawdown(TRADES) == pytest.approx(180)


def test_empty_trades():
    m = summary_dict([])
    assert m["total_trades"] == 0
    assert m["win_rate"] == 0.0
