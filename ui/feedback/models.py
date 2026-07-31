from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional


FeedbackLabel = Literal["positive", "negative"]


@dataclass(frozen=True)
class FeedbackEvent:
    """Immutable feedback record persisted for each rated chatbot response."""

    message_id: str
    label: FeedbackLabel
    query: str
    answer: str
    confidence: Optional[float] = None
    matched_question: str = ""
    faq_id: str = ""
    category: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_csv_row(self) -> Dict[str, Any]:
        row = asdict(self)
        if row["confidence"] is not None:
            row["confidence"] = round(float(row["confidence"]), 6)
        return row

