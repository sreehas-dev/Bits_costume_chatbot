from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


ALLOWED_EMAIL_DOMAIN = "@wilp.bits-pilani.ac.in"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UserRecord:
    """CSV-safe user account record."""

    email: str
    password_hash: str
    is_verified: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    verified_at: str = ""
    last_login_at: str = ""
    otp_hash: str = ""
    otp_expires_at: str = ""
    otp_sent_at: str = ""
    otp_attempts: int = 0

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "email": self.email.strip().lower(),
            "password_hash": self.password_hash,
            "is_verified": "true" if self.is_verified else "false",
            "created_at": self.created_at,
            "verified_at": self.verified_at,
            "last_login_at": self.last_login_at,
            "otp_hash": self.otp_hash,
            "otp_expires_at": self.otp_expires_at,
            "otp_sent_at": self.otp_sent_at,
            "otp_attempts": str(int(self.otp_attempts or 0)),
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, Optional[str]]) -> "UserRecord":
        return cls(
            email=(row.get("email") or "").strip().lower(),
            password_hash=row.get("password_hash") or "",
            is_verified=(row.get("is_verified") or "").strip().lower() == "true",
            created_at=row.get("created_at") or "",
            verified_at=row.get("verified_at") or "",
            last_login_at=row.get("last_login_at") or "",
            otp_hash=row.get("otp_hash") or "",
            otp_expires_at=row.get("otp_expires_at") or "",
            otp_sent_at=row.get("otp_sent_at") or "",
            otp_attempts=int(row.get("otp_attempts") or 0),
        )

    def with_updates(self, **updates: Any) -> "UserRecord":
        values = self.to_csv_row()
        values.update(updates)
        values["is_verified"] = str(values.get("is_verified", "false")).lower() == "true"
        values["otp_attempts"] = int(values.get("otp_attempts") or 0)
        return UserRecord(**values)


def normalize_email(email: object) -> str:
    return "" if email is None else str(email).strip().lower()


def is_allowed_email(email: object) -> bool:
    return normalize_email(email).endswith(ALLOWED_EMAIL_DOMAIN)
