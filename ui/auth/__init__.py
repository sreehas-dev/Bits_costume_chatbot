"""Email verification authentication package for the Streamlit chatbot."""

from .email_otp import EmailOtpService, OTP_LENGTH, SmtpConfig, load_smtp_config
from .models import ALLOWED_EMAIL_DOMAIN, UserRecord, is_allowed_email, normalize_email
from .renderer import AuthRenderer
from .store import CsvUserStore

__all__ = [
    "ALLOWED_EMAIL_DOMAIN",
    "AuthRenderer",
    "CsvUserStore",
    "EmailOtpService",
    "OTP_LENGTH",
    "SmtpConfig",
    "UserRecord",
    "is_allowed_email",
    "load_smtp_config",
    "normalize_email",
]
