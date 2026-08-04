from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import MutableMapping

import streamlit as st

from .email_otp import EmailOtpService, OTP_LENGTH
from .models import ALLOWED_EMAIL_DOMAIN, is_allowed_email
from .store import CsvUserStore


PASSWORD_MIN_LENGTH = 8
EMAIL_PREFIX_RE = re.compile(r"^[A-Z0-9._%+-]+$", re.IGNORECASE)


class AuthRenderer:
    """Streamlit authentication gate for verified WILP email users."""

    def __init__(self, user_store: CsvUserStore, otp_service: EmailOtpService):
        self.user_store = user_store
        self.otp_service = otp_service

    @staticmethod
    def is_authenticated(session_state: MutableMapping[str, object]) -> bool:
        return bool(session_state.get("auth_user_email"))

    @staticmethod
    def logout(session_state: MutableMapping[str, object]) -> None:
        for key in ("auth_user_email", "auth_pending_email", "auth_view"):
            session_state.pop(key, None)

    def render_gate(self, session_state: MutableMapping[str, object]) -> bool:
        if self.is_authenticated(session_state):
            return True

        view = str(session_state.get("auth_view") or "login")
        if view not in {"login", "signup", "verify"}:
            view = "login"
            session_state["auth_view"] = view

        st.title(f"{self._time_greeting()}, welcome")
        st.caption(f"Access is restricted to verified `{ALLOWED_EMAIL_DOMAIN}` accounts.")

        login_col, signup_col = st.columns(2)
        with login_col:
            if st.button("Login", use_container_width=True, disabled=view == "login"):
                session_state["auth_view"] = "login"
                st.rerun()
        with signup_col:
            if st.button("Sign up", use_container_width=True, disabled=view == "signup"):
                session_state["auth_view"] = "signup"
                st.rerun()

        st.markdown("---")

        if view == "signup":
            self._render_signup(session_state)
        elif view == "verify":
            self._render_verify(session_state)
        else:
            self._render_login(session_state)

        st.stop()

    def _render_login(self, session_state: MutableMapping[str, object]) -> None:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            email = self._email_prefix_input("login")
            password = self._text_input_value("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if not submitted:
            return

        if not self._valid_email(email):
            st.error("Enter a valid WILP email ID before the fixed domain.")
            return

        user = self.user_store.get(email)
        password_matches = self.user_store.verify_password(email, password)
        if user is None or not password_matches:
            st.error("Invalid email or password.")
            return

        if not user.is_verified:
            session_state["auth_pending_email"] = email
            session_state["auth_view"] = "verify"
            st.warning("Verify your email OTP before logging in.")
            st.rerun()

        self.user_store.mark_login(email)
        session_state["auth_user_email"] = email
        st.success("Logged in successfully.")
        st.rerun()

    def _render_signup(self, session_state: MutableMapping[str, object]) -> None:
        st.subheader("Create account")
        st.caption("Enter only your email ID. The WILP domain is fixed and cannot be edited.")

        with st.form("signup_form", clear_on_submit=False):
            email = self._email_prefix_input("signup")
            password = self._text_input_value("Password", type="password", key="signup_password")
            confirm_password = self._text_input_value("Confirm password", type="password", key="signup_confirm_password")
            submitted = st.form_submit_button("Create account and send OTP", use_container_width=True)

        if not submitted:
            return

        if not self._valid_email(email):
            st.error("Enter a valid WILP email ID before the fixed domain.")
            return

        password_text = "" if password is None else str(password)
        confirm_password_text = "" if confirm_password is None else str(confirm_password)

        if len(password_text) < PASSWORD_MIN_LENGTH:
            st.error(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
            return

        if password_text != confirm_password_text:
            st.error("Passwords do not match.")
            return

        try:
            user = self.user_store.create_user(email, password_text)
            self.otp_service.send_signup_otp(user)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Could not send verification email: {exc}")
            return

        session_state["auth_pending_email"] = email
        session_state["auth_view"] = "verify"
        st.success("Account created. Enter the OTP sent to your email.")
        st.rerun()

    def _render_verify(self, session_state: MutableMapping[str, object]) -> None:
        st.subheader("Verify OTP")
        pending_email = self._normalize_email(session_state.get("auth_pending_email"))
        st.caption(f"Enter the {OTP_LENGTH}-digit OTP sent to `{pending_email or 'your WILP email'}`.")

        with st.form("verify_otp_form", clear_on_submit=False):
            email = self._email_prefix_input("verify", default_email=pending_email)
            otp = self._otp_input("verify_otp")
            verified = st.form_submit_button("Verify", use_container_width=True)

        resend_col, back_col = st.columns(2)
        with resend_col:
            resend = st.button("Resend OTP", use_container_width=True)
        with back_col:
            if st.button("Back to login", use_container_width=True):
                session_state["auth_view"] = "login"
                st.rerun()

        if resend:
            self._resend_otp(email, session_state)
            return

        if not verified:
            return

        if not self._valid_email(email):
            st.error("Enter a valid WILP email ID before the fixed domain.")
            return

        if not (otp.isdigit() and len(otp) == OTP_LENGTH):
            st.error(f"Enter the complete {OTP_LENGTH}-digit OTP.")
            return

        try:
            if self.otp_service.verify_otp(email, otp):
                session_state.pop("auth_pending_email", None)
                session_state["auth_view"] = "login"
                st.success("Email verified. Please login.")
                st.rerun()
            else:
                st.error("Invalid OTP.")
        except ValueError as exc:
            st.error(str(exc))

    def _resend_otp(self, email: str, session_state: MutableMapping[str, object]) -> None:
        if not self._valid_email(email):
            st.error("Enter a valid WILP email ID before the fixed domain.")
            return

        user = self.user_store.get(email)
        if not user:
            st.error("Create an account first.")
            session_state["auth_view"] = "signup"
            return

        if user.is_verified:
            st.info("This email is already verified. Please login.")
            session_state["auth_view"] = "login"
            return

        try:
            self.otp_service.send_signup_otp(user)
            session_state["auth_pending_email"] = email
            st.success("A new OTP was sent to your email.")
        except Exception as exc:
            st.error(str(exc))

    def _email_prefix_input(self, namespace: str, default_email: str = "") -> str:
        default_prefix = self._prefix_from_email(default_email)
        st.markdown(
            """
            <style>
                div[data-testid="stHorizontalBlock"] div[data-testid="column"]:has(input[aria-label="Email ID"]) {
                    padding-right: 0 !important;
                }
                .fixed-email-domain {
                    height: 2.55rem;
                    margin-top: 1.75rem;
                    padding: 0.55rem 0.75rem;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-left: 0;
                    border-radius: 0 0.5rem 0.5rem 0;
                    background: rgba(240, 242, 246, 0.75);
                    color: rgba(49, 51, 63, 0.8);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        prefix_col, domain_col = st.columns([0.56, 0.44], gap="small")
        with prefix_col:
            prefix = self._text_input_value(
                "Email ID",
                value=default_prefix,
                placeholder="",
                key=f"{namespace}_email_prefix",
            )
        with domain_col:
            st.markdown(f'<div class="fixed-email-domain">{ALLOWED_EMAIL_DOMAIN}</div>', unsafe_allow_html=True)
        prefix = prefix.strip().replace(ALLOWED_EMAIL_DOMAIN, "").replace("@", "")
        return f"{prefix}{ALLOWED_EMAIL_DOMAIN}" if prefix else ""

    @staticmethod
    def _otp_input(namespace: str) -> str:
        otp = st.text_input(
            f"OTP ({OTP_LENGTH} digits)",
            key=namespace,
            max_chars=OTP_LENGTH,
            placeholder="",
        )
        return "" if otp is None else str(otp).strip()

    @staticmethod
    def _valid_email(email: str) -> bool:
        if not email or not is_allowed_email(email):
            return False
        prefix = email.removesuffix(ALLOWED_EMAIL_DOMAIN)
        return bool(prefix and EMAIL_PREFIX_RE.fullmatch(prefix))

    @staticmethod
    def _prefix_from_email(email: str) -> str:
        email = AuthRenderer._normalize_email(email)
        return email.removesuffix(ALLOWED_EMAIL_DOMAIN) if email.endswith(ALLOWED_EMAIL_DOMAIN) else ""

    @staticmethod
    def _normalize_email(email: object) -> str:
        return "" if email is None else str(email).strip().lower()

    @staticmethod
    def _text_input_value(label: str, **kwargs: object) -> str:
        value = st.text_input(label, **kwargs)
        return "" if value is None else str(value)

    @staticmethod
    def _browser_hour() -> int:
        try:
            raw_hour = st.query_params.get("browser_hour", "")
            if isinstance(raw_hour, list):
                raw_hour = raw_hour[0] if raw_hour else ""
            hour = int(str(raw_hour))
            if 0 <= hour <= 23:
                return hour
        except Exception:
            pass

        try:
            raw_timestamp = st.query_params.get("browser_ts", "")
            raw_offset = st.query_params.get("browser_tz_offset", "")
            if isinstance(raw_timestamp, list):
                raw_timestamp = raw_timestamp[0] if raw_timestamp else ""
            if isinstance(raw_offset, list):
                raw_offset = raw_offset[0] if raw_offset else ""
            timestamp_ms = int(str(raw_timestamp))
            offset_minutes = int(str(raw_offset))
            client_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            client_local = client_utc - timedelta(minutes=offset_minutes)
            return client_local.hour
        except Exception:
            pass

        return 12

    @classmethod
    def _time_greeting(cls) -> str:
        hour = cls._browser_hour()
        if 5 <= hour < 12:
            return "Good morning"
        if 12 <= hour < 17:
            return "Good afternoon"
        return "Good evening"

