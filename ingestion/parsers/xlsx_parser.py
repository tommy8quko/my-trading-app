"""XLSX parser: delegates to CSVParser after converting to CSV in memory."""
from __future__ import annotations
import io
import csv
from models.order import ParsedOrder
from ingestion.parsers.base import FileParser
from ingestion.parsers.csv_parser import CSVParser


class XLSXParser(FileParser):
    def __init__(self, sheet_index: int = 0, column_map: dict[str, str] | None = None):
        self._sheet_index = sheet_index
        self._csv_parser = CSVParser(column_map)

    def parse(self, content: bytes, broker_short_code: str) -> list[ParsedOrder]:
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError("openpyxl is required for XLSX parsing: pip install openpyxl") from exc

        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.worksheets[self._sheet_index]

        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in ws.iter_rows(values_only=True):
            writer.writerow([str(c) if c is not None else "" for c in row])

        return self._csv_parser.parse(buf.getvalue().encode(), broker_short_code)
