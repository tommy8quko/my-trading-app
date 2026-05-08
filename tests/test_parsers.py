"""Unit tests for broker email parsers."""
import pytest
from datetime import datetime, timezone
from ingestion.brokers.ibkr import IBKRAdapter
from ingestion.brokers.chief import ChiefAdapter

DT = datetime(2024, 6, 15, 9, 30, 0, tzinfo=timezone.utc)


class TestIBKRAdapter:
    def test_sold(self):
        o = IBKRAdapter.parse_subject("SOLD 215 PAYP @ 20.91 (UXXX29872)", DT)
        assert o is not None
        assert o.action == "SELL"
        assert o.symbol == "PAYP"
        assert o.quantity == 215
        assert o.price == 20.91
        assert o.external_order_id == "UXXX29872"
        assert o.exchange == "NYSE"
        assert o.currency == "USD"

    def test_bought(self):
        o = IBKRAdapter.parse_subject("BOUGHT 100 AAPL @ 183.50 (UABC12345)", DT)
        assert o is not None
        assert o.action == "BUY"
        assert o.quantity == 100

    def test_short(self):
        o = IBKRAdapter.parse_subject("SHORTED 500 TSLA @ 245.10 (UXYZ99001)", DT)
        assert o is not None
        assert o.action == "SHORT"

    def test_cover(self):
        o = IBKRAdapter.parse_subject("COVERED 500 TSLA @ 230.00 (UXYZ99002)", DT)
        assert o is not None
        assert o.action == "COVER"

    def test_hk_stock(self):
        o = IBKRAdapter.parse_subject("BOUGHT 200 0700.HK @ 345.00 (UHKG00001)", DT)
        assert o is not None
        assert o.exchange == "HKEX"
        assert o.currency == "HKD"

    def test_no_match(self):
        assert IBKRAdapter.parse_subject("Account Statement for June 2024", DT) is None

    def test_qty_with_comma(self):
        o = IBKRAdapter.parse_subject("BOUGHT 1,000 SPY @ 520.00 (UAAA11111)", DT)
        assert o is not None
        assert o.quantity == 1000.0

    def test_matches_sender(self):
        assert IBKRAdapter.matches_sender("ibkralerts@ibkr.com")
        assert not IBKRAdapter.matches_sender("noreply@fidelity.com")


class TestChiefAdapter:
    _SELL_BODY = """閣下沽出#GLW (康寧)的成交，詳情如下：

參考編號    89585908
客戶編號    M****04
貨幣        USD
價格        180.2000
已成交數量  45
待成交數量  0
成交金額    $8,109.00"""

    _BUY_HK_BODY = """閣下買入#700 (騰訊)的成交，詳情如下：

參考編號    12345678
客戶編號    M****04
貨幣        HKD
價格        345.0000
已成交數量  200
待成交數量  0
成交金額    $69,000.00"""

    def test_sell_usd(self):
        o = ChiefAdapter.parse_body(self._SELL_BODY, "", DT)
        assert o is not None
        assert o.action == "SELL"
        assert o.symbol == "GLW"
        assert o.price == 180.2
        assert o.quantity == 45
        assert o.currency == "USD"
        assert o.external_order_id == "89585908"
        assert o.exchange == "NYSE"

    def test_buy_hk(self):
        o = ChiefAdapter.parse_body(self._BUY_HK_BODY, "", DT)
        assert o is not None
        assert o.action == "BUY"
        assert o.symbol == "0700.HK"
        assert o.exchange == "HKEX"
        assert o.currency == "HKD"

    def test_no_match(self):
        assert ChiefAdapter.parse_body("Hello, your statement is ready.", "", DT) is None

    def test_parse_subject_returns_none(self):
        assert ChiefAdapter.parse_subject("any subject", DT) is None

    def test_matches_sender(self):
        assert ChiefAdapter.matches_sender("noreply@chiefgroup.com.hk")
        assert not ChiefAdapter.matches_sender("alerts@ibkr.com")
