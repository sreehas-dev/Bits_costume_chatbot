from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Mapping, Optional

from .models import UserRecord, utc_now_iso
from .store import CsvUserStore


OTP_TTL_MINUTES = 10
OTP_RESEND_SECONDS = 60
MAX_OTP_ATTEMPTS = 5
OTP_LENGTH = 6


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool = True
    sender: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.host and self.port and self.username and self.password)


def get_secret_value(secrets_source: Optional[Mapping[str, Any]], key: str, default: str = "") -> str:
    if secrets_source is not None:
        try:
            value = secrets_source[key]
            if value is not None:
                return str(value)
        except Exception:
            pass
    return os.getenv(key, default)


def load_smtp_config(secrets_source: Optional[Mapping[str, Any]] = None) -> SmtpConfig:
    username = get_secret_value(secrets_source, "SMTP_USERNAME")
    return SmtpConfig(
        host=get_secret_value(secrets_source, "SMTP_HOST", "smtp.gmail.com"),
        port=int(get_secret_value(secrets_source, "SMTP_PORT", "587") or 587),
        username=username,
        password=get_secret_value(secrets_source, "SMTP_PASSWORD"),
        use_tls=get_secret_value(secrets_source, "SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"},
        sender=get_secret_value(secrets_source, "SMTP_FROM", username),
    )


class EmailOtpService:
    """Handles verification OTP creation, storage, and email delivery."""

    def __init__(self, user_store: CsvUserStore, smtp_config: SmtpConfig):
        self.user_store = user_store
        self.smtp_config = smtp_config

    def send_signup_otp(self, user: UserRecord) -> None:
        if not self.smtp_config.configured:
            raise RuntimeError("SMTP is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, and SMTP_PASSWORD.")

        existing = self.user_store.get(user.email) or user
        if existing.otp_sent_at and not self._can_resend(existing.otp_sent_at):
            raise ValueError("Please wait before requesting another OTP.")

        otp = "".join(secrets.choice("0123456789") for _ in range(OTP_LENGTH))
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)
        updated = existing.with_updates(
            otp_hash=self._hash_otp(otp),
            otp_expires_at=expires_at.isoformat(),
            otp_sent_at=utc_now_iso(),
            otp_attempts=0,
        )
        self.user_store.upsert(updated)
        self._send_email(updated.email, otp)

    def verify_otp(self, email: str, otp: str) -> bool:
        user = self.user_store.get(email)
        if not user or not user.otp_hash:
            return False

        if user.otp_attempts >= MAX_OTP_ATTEMPTS:
            raise ValueError("Too many invalid OTP attempts. Request a new OTP.")

        if self._is_expired(user.otp_expires_at):
            raise ValueError("OTP expired. Request a new OTP.")

        if not hmac.compare_digest(self._hash_otp(otp.strip()), user.otp_hash):
            self.user_store.upsert(user.with_updates(otp_attempts=user.otp_attempts + 1))
            return False

        self.user_store.upsert(
            user.with_updates(
                is_verified="true",
                verified_at=utc_now_iso(),
                otp_hash="",
                otp_expires_at="",
                otp_sent_at="",
                otp_attempts=0,
            )
        )
        return True

    def _send_email(self, to_email: str, otp: str) -> None:
        message = EmailMessage()
        message["Subject"] = "Verify your BITS Pilani Assistant account"
        message["From"] = self.smtp_config.sender or self.smtp_config.username
        message["To"] = to_email
        message.set_content(
            f"Your verification OTP is {otp}.\n\n"
            f"This OTP expires in {OTP_TTL_MINUTES} minutes. "
            "If you did not request this, ignore this email."
        )

        with smtplib.SMTP(self.smtp_config.host, self.smtp_config.port, timeout=20) as smtp:
            if self.smtp_config.use_tls:
                smtp.starttls()
            smtp.login(self.smtp_config.username, self.smtp_config.password)
            smtp.send_message(message)

    @staticmethod
    def _hash_otp(otp: str) -> str:
        return hashlib.sha256(otp.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    @classmethod
    def _is_expired(cls, value: str) -> bool:
        parsed = cls._parse_datetime(value)
        return parsed is None or parsed <= datetime.now(timezone.utc)

    @classmethod
    def _can_resend(cls, sent_at: str) -> bool:
        parsed = cls._parse_datetime(sent_at)
        if parsed is None:
            return True
        return datetime.now(timezone.utc) - parsed >= timedelta(seconds=OTP_RESEND_SECONDS)
