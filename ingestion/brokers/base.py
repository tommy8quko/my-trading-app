from __future__ import annotations
from abc import ABC, abstractmethod
from models.order import ParsedOrder


class BrokerAdapter(ABC):
    """Base class for all broker email parsers."""

    short_code: str = ""                  # e.g. 'IBKR'
    sender_patterns: list[str] = []       # substrings to match against From: header

    @classmethod
    def matches_sender(cls, from_address: str) -> bool:
        addr = from_address.lower()
        return any(p.lower() in addr for p in cls.sender_patterns)

    @classmethod
    @abstractmethod
    def parse_subject(cls, subject: str, received_at) -> ParsedOrder | None:
        """Try to extract an order from the email subject line.
        Return None if this email is not an order notification."""

    @classmethod
    @abstractmethod
    def parse_body(cls, body: str, subject: str, received_at) -> ParsedOrder | None:
        """Try to extract an order from the email body.
        Called as fallback when parse_subject returns None."""
