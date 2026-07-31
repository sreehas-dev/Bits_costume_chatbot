from __future__ import annotations

from typing import Any, Dict, MutableMapping

import streamlit as st

from .models import FeedbackEvent, FeedbackLabel
from .store import CsvFeedbackStore


class FeedbackRenderer:
    """Streamlit UI renderer for thumbs up/down response feedback."""

    POSITIVE_ICON = "👍"
    NEGATIVE_ICON = "👎"

    def __init__(self, store: CsvFeedbackStore):
        self.store = store

    @staticmethod
    def is_feedback_eligible(message: Dict[str, Any]) -> bool:
        return message.get("role") == "assistant" and bool(message.get("feedback_enabled"))

    def render(self, message: Dict[str, Any], session_state: MutableMapping[str, Any]) -> None:
        """Render feedback buttons below one assistant answer and persist on click."""
        if not self.is_feedback_eligible(message):
            return

        message_id = message["message_id"]
        feedback_key = f"feedback_{message_id}"
        existing_label = session_state.get(feedback_key)

        if existing_label == "saved":
            st.caption("Feedback already saved")
            return

        if isinstance(existing_label, str) and existing_label:
            icon = self._icon_for_label(str(existing_label))
            st.caption(f"Feedback saved: {icon}")
            return

        st.caption("Was this answer helpful?")
        positive_col, negative_col, _ = st.columns([0.08, 0.08, 0.84])

        with positive_col:
            if st.button(self.POSITIVE_ICON, key=f"positive_{message_id}", help="Helpful answer"):
                self._save_feedback("positive", message, session_state, feedback_key)
                st.rerun()

        with negative_col:
            if st.button(self.NEGATIVE_ICON, key=f"negative_{message_id}", help="Not helpful answer"):
                self._save_feedback("negative", message, session_state, feedback_key)
                st.rerun()

    def _save_feedback(
        self,
        label: FeedbackLabel,
        message: Dict[str, Any],
        session_state: MutableMapping[str, Any],
        feedback_key: str,
    ) -> None:
        event = FeedbackEvent(
            message_id=message["message_id"],
            label=label,
            query=message.get("query", ""),
            answer=message.get("content", ""),
            confidence=message.get("confidence"),
            matched_question=message.get("matched_question", ""),
            faq_id=message.get("faq_id", ""),
            category=message.get("category", ""),
        )
        self.store.save(event)
        session_state[feedback_key] = label
        st.toast("Feedback saved. Thank you!", icon=self._icon_for_label(label))

    def _icon_for_label(self, label: str) -> str:
        return self.POSITIVE_ICON if label == "positive" else self.NEGATIVE_ICON
