"""
Gmail IMAP connector.

Fetches unprocessed order-confirmation emails from Gmail using IMAP + App Password.
Yields raw email data dicts to the caller; does not parse or persist anything itself.
"""
from __future__ import annotations
import email
import imaplib
import logging
from datetime import datetime, date, timedelta, timezone
from email.header import decode_header
from typing import Iterator

from config import get_secret

log = logging.getLogger(__name__)

GMAIL_IMAP_HOST = "imap.gmail.com"
GMAIL_IMAP_PORT = 993


def _decode_header_value(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = decode_header(raw)
    out = []
    for chunk, enc in parts:
        if isinstance(chunk, bytes):
            out.append(chunk.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out)


def _charset_from_html(payload: bytes) -> str | None:
    """Read charset from HTML <meta charset=...> tag in the first 2KB."""
    import re
    m = re.search(rb'charset=["\']?\s*([a-zA-Z0-9_-]+)', payload[:2000], re.IGNORECASE)
    return m.group(1).decode("ascii", errors="ignore") if m else None


def _decode_payload(payload: bytes, mime_charset: str | None, is_html: bool = False) -> str:
    """Decode bytes using the best available charset."""
    # For HTML, prefer charset from <meta> tag over MIME header (more reliable)
    charset = (_charset_from_html(payload) if is_html else None) or mime_charset or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def _extract_body(msg: email.message.Message) -> str:
    """Return plain text body. Falls back to HTML→text if no text/plain part."""
    html_fallback: str = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            if ct == "text/plain":
                return _decode_payload(payload, part.get_content_charset())
            if ct == "text/html" and not html_fallback:
                html_fallback = _decode_payload(payload, part.get_content_charset(), is_html=True)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return _decode_payload(payload, msg.get_content_charset())

    if html_fallback:
        return html_fallback
    return ""


class GmailConnector:
    """Thin wrapper around imaplib for reading Gmail."""

    def __init__(self, email_address: str | None = None, app_password: str | None = None):
        self._address = email_address or get_secret("GMAIL_ADDRESS")
        self._password = app_password or get_secret("GMAIL_APP_PASSWORD")
        self._conn: imaplib.IMAP4_SSL | None = None

    def connect(self) -> None:
        self._conn = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
        self._conn.login(self._address, self._password)
        log.info("Connected to Gmail as %s", self._address)

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "GmailConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    def fetch_emails(
        self,
        folder: str = "INBOX",
        since: date | None = None,
        sender_filter: str | None = None,
        max_results: int = 200,
    ) -> Iterator[dict]:
        """
        Yield dicts with keys: gmail_message_id, from_address, subject,
        received_at (datetime UTC), body (str).

        Args:
            folder: IMAP folder name (default INBOX).
            since: Only fetch emails on or after this date.
            sender_filter: Optional substring to match against From header.
            max_results: Cap to avoid runaway fetches.
        """
        if self._conn is None:
            raise RuntimeError("Not connected — call connect() first")

        self._conn.select(folder, readonly=True)

        criteria: list[str] = []
        if since:
            # IMAP date format: "01-Jan-2024"
            imap_date = since.strftime("%d-%b-%Y")
            criteria.append(f'SINCE "{imap_date}"')
        if sender_filter:
            criteria.append(f'FROM "{sender_filter}"')

        search_string = " ".join(criteria) if criteria else "ALL"
        _, data = self._conn.search(None, search_string)
        message_ids = data[0].split() if data[0] else []

        # Most recent first; cap at max_results
        message_ids = message_ids[-max_results:][::-1]

        for uid in message_ids:
            _, msg_data = self._conn.fetch(uid, "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            message_id_header = msg.get("Message-ID", "").strip()
            if not message_id_header:
                continue

            from_raw = msg.get("From", "")
            from_address = _decode_header_value(from_raw)
            subject = _decode_header_value(msg.get("Subject", ""))
            date_str = msg.get("Date", "")
            try:
                received_at = email.utils.parsedate_to_datetime(date_str)
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=timezone.utc)
                received_at = received_at.astimezone(timezone.utc)
            except Exception:
                received_at = datetime.now(timezone.utc)

            body = _extract_body(msg)

            yield {
                "gmail_message_id": message_id_header,
                "from_address": from_address,
                "subject": subject,
                "received_at": received_at,
                "body": body,
            }
