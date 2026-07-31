from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import os
import secrets
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Optional

from .models import UserRecord, normalize_email, utc_now_iso


class CsvUserStore:
    """CSV-backed authenticated user repository.

    Passwords are stored as PBKDF2-HMAC-SHA256 hashes. CSV writes are guarded by a
    process-local lock and replace the file atomically to avoid partial writes.
    """

    FIELDNAMES = [
        "email",
        "password_hash",
        "is_verified",
        "created_at",
        "verified_at",
        "last_login_at",
        "otp_hash",
        "otp_expires_at",
        "otp_sent_at",
        "otp_attempts",
    ]
    PBKDF2_ITERATIONS = 260_000

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.FIELDNAMES).writeheader()

    def get(self, email: str) -> Optional[UserRecord]:
        users = self._read_users()
        return users.get(normalize_email(email))

    def create_user(self, email: str, password: str) -> UserRecord:
        email = normalize_email(email)
        with self._lock:
            users = self._read_users_unlocked()
            if email in users:
                raise ValueError("An account already exists for this email.")
            user = UserRecord(email=email, password_hash=self.hash_password(password))
            users[email] = user
            self._write_users_unlocked(users.values())
            return user

    def upsert(self, user: UserRecord) -> None:
        with self._lock:
            users = self._read_users_unlocked()
            users[normalize_email(user.email)] = user
            self._write_users_unlocked(users.values())

    def verify_password(self, email: str, password: str) -> bool:
        user = self.get(email)
        if not user or not user.password_hash:
            return False
        return self.check_password(password, user.password_hash)

    def mark_login(self, email: str) -> None:
        user = self.get(email)
        if user:
            self.upsert(user.with_updates(last_login_at=utc_now_iso()))

    def _read_users(self) -> Dict[str, UserRecord]:
        with self._lock:
            return self._read_users_unlocked()

    def _read_users_unlocked(self) -> Dict[str, UserRecord]:
        self._ensure_file()
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return {
                user.email: user
                for user in (UserRecord.from_csv_row(row) for row in reader)
                if user.email
            }

    def _write_users_unlocked(self, users: Iterable[UserRecord]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=str(self.csv_path.parent),
            delete=False,
        ) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for user in sorted(users, key=lambda item: item.email):
                writer.writerow(user.to_csv_row())
            temp_name = tmp.name
        os.replace(temp_name, self.csv_path)

    @classmethod
    def hash_password(cls, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            cls.PBKDF2_ITERATIONS,
        )
        return "pbkdf2_sha256${}${}${}".format(
            cls.PBKDF2_ITERATIONS,
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )

    @classmethod
    def check_password(cls, password: str, stored_hash: str) -> bool:
        try:
            algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(iterations),
            )
            return hmac.compare_digest(candidate, expected)
        except Exception:
            return False

