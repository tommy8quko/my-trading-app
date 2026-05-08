from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ReviewItem:
    """One pending item in the review queue."""
    id: str | None               # UUID once saved to DB
    source_type: str             # 'email' | 'csv' | 'xlsx'
    broker_short_code: str
    raw_parsed: dict[str, Any]
    normalized: dict[str, Any]
    confidence: float            # 0.0 – 1.0
    email_record_id: str | None = None
    document_id: str | None = None
    status: str = "pending"      # 'pending'|'approved'|'rejected'|'auto_approved'
    review_notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
