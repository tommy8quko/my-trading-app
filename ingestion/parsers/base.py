"""Abstract interface for file-based parsers (CSV, XLSX)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from models.order import ParsedOrder


class FileParser(ABC):
    """Parse a file's raw bytes into a list of ParsedOrders."""

    @abstractmethod
    def parse(self, content: bytes, broker_short_code: str) -> list[ParsedOrder]:
        """Return zero or more orders extracted from `content`."""
