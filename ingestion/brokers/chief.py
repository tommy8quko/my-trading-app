"""
Chief Securities HK email parser.

Email body format (Traditional Chinese):
    閣下沽出#GLW (康寧)的成交，詳情如下：

    參考編號    89585908
    客戶編號    M****04
    貨幣        USD
    價格        180.2000
    已成交數量  45
    待成交數量  0
    成交金額    $8,109.00

Action keywords:
    沽出 → SELL
    買入 → BUY
"""
from __future__ import annotations
import html as _html_mod
import re
from datetime import datetime, date
from html.parser import HTMLParser
from models.order import ParsedOrder
from ingestion.brokers.base import BrokerAdapter

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class _TextExtractor(HTMLParser):
    """Extract readable text from HTML, preserving table structure as tab-separated lines."""
    _BLOCK = {"tr", "p", "div", "br", "li", "h1", "h2", "h3", "h4", "blockquote"}
    _CELL = {"td", "th"}
    _SKIP = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._in_skip: int = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in self._SKIP:
            self._in_skip += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")
        elif tag in self._CELL:
            self._parts.append("\t")

    def handle_endtag(self, tag: str):
        if tag in self._SKIP:
            self._in_skip = max(0, self._in_skip - 1)
        elif tag in self._CELL:
            self._parts.append("\t")

    def handle_data(self, data: str):
        if not self._in_skip:
            self._parts.append(data)

    def handle_entityref(self, name: str):
        if not self._in_skip:
            self._parts.append(_html_mod.unescape(f"&{name};"))

    def handle_charref(self, name: str):
        if not self._in_skip:
            try:
                c = chr(int(name[1:], 16) if name.startswith("x") else int(name))
                self._parts.append(c)
            except (ValueError, OverflowError):
                pass

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            # collapse runs of tabs/spaces to a single tab, strip surrounding tabs
            line = re.sub(r"[ \t]+", "\t", line).strip("\t")
            if line.strip():
                cleaned.append(line)
        return "\n".join(cleaned)


def _strip_html(text: str) -> str:
    extractor = _TextExtractor()
    try:
        extractor.feed(text)
        return extractor.get_text()
    except Exception:
        # Fallback: crude tag removal
        return _HTML_TAG_RE.sub(" ", text)


# Allow optional whitespace between action and #SYMBOL, in case cells are split
_INTRO_RE = re.compile(
    r"閣下(沽出|買入)\s*#\s*([A-Z0-9.]+)",
    re.UNICODE,
)

_ACTION_MAP = {"沽出": "SELL", "買入": "BUY"}

# Matches a line that is purely Chinese characters (Traditional/Simplified CJK range)
_CHINESE_ONLY_RE = re.compile(r"^[一-鿿㐀-䶿]+$")


def _parse_kv(body: str) -> dict[str, str]:
    """
    Extract Chinese key → value pairs from stripped Chief email text.

    Handles two layouts produced by their HTML table:
      • Same-line:       "價格\\t20.6700"
      • Consecutive:     "價格\\n20.6700"   (each <td> on its own HTML line)
    """
    result: dict[str, str] = {}
    lines = [l.strip() for l in body.split("\n") if l.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Same-line tab-separated  key\tvalue
        if "\t" in line:
            parts = line.split("\t", 1)
            key, val = parts[0].strip(), parts[1].strip()
            if _CHINESE_ONLY_RE.match(key) and val and not _CHINESE_ONLY_RE.match(val):
                result[key] = val.lstrip("$").replace(",", "")
                i += 1
                continue

        # Pure-Chinese key — value is on the next non-Chinese, non-empty line
        if _CHINESE_ONLY_RE.match(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_val = lines[j].strip()
                if not _CHINESE_ONLY_RE.match(next_val):
                    result[line] = next_val.lstrip("$").replace(",", "")
                    i = j + 1
                    continue

        i += 1
    return result


class ChiefAdapter(BrokerAdapter):
    short_code = "CHIEF"
    sender_patterns = ["chiefgroup.com", "cs@chiefgroup.com.hk"]

    @classmethod
    def parse_subject(cls, subject: str, received_at: datetime | None) -> ParsedOrder | None:
        return None

    @classmethod
    def parse_body(cls, body: str, subject: str, received_at: datetime | None) -> ParsedOrder | None:
        if "<html" in body.lower() or "<!doctype" in body.lower():
            body = _strip_html(body)

        intro = _INTRO_RE.search(body)
        if not intro:
            return None

        raw_action, symbol = intro.group(1), intro.group(2).upper()
        action = _ACTION_MAP.get(raw_action, "BUY")

        kv = _parse_kv(body)

        try:
            price = float(kv.get("價格", "0"))
            quantity = float(kv.get("已成交數量", "0"))
        except ValueError:
            return None

        if price <= 0 or quantity <= 0:
            return None

        currency = kv.get("貨幣", "HKD").strip()
        ref_no = kv.get("參考編號", "")

        if currency == "HKD":
            exchange = "HKEX"
            if symbol.isdigit() and len(symbol) <= 4:
                symbol = symbol.zfill(4) + ".HK"
        else:
            exchange = "NYSE"

        order_date: date = received_at.date() if received_at else date.today()
        order_time = received_at.time() if received_at else None

        # Chief commission schedule:
        # HK stocks: 0.0668% of trade value, min HKD 20
        # US stocks: USD $0.008/share, min USD 0.99
        trade_value = price * quantity
        if currency == "HKD":
            fees = max(trade_value * 0.000668, 20.0)
        else:
            fees = max(quantity * 0.008, 0.99)

        return ParsedOrder(
            broker_short_code=cls.short_code,
            external_order_id=ref_no or f"CHIEF_{symbol}_{order_date}_{quantity}",
            action=action,
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            price=price,
            quantity=quantity,
            fees=round(fees, 4),
            order_date=order_date,
            order_time=order_time,
            raw_data={"subject": subject, "body_kv": kv},
        )
