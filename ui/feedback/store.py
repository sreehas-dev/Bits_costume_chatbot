from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Set

from .models import FeedbackEvent


class CsvFeedbackStore:
    """Append-only CSV repository for chatbot feedback events."""

    FIELDNAMES = [
        "timestamp",
        "message_id",
        "label",
        "query",
        "answer",
        "confidence",
        "matched_question",
        "faq_id",
        "category",
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def save(self, event: FeedbackEvent) -> None:
        """Persist one feedback event safely with required CSV headers."""
        with self._lock:
            self._ensure_file()
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES, extrasaction="ignore")
                writer.writerow(event.to_csv_row())

    def get_rated_message_ids(self) -> Set[str]:
        """Return message IDs that already have feedback in the CSV."""
        if not self.csv_path.exists():
            return set()

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return {row["message_id"] for row in reader if row.get("message_id")}

    def count(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            return max(sum(1 for _ in file) - 1, 0)
