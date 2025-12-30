"""
ExplainFutures - Supabase Database Manager (APP.PY COMPATIBLE / MINIMAL SCOPE)

Scope of this file (as requested):
- ONLY the database I/O needed by App.py (login + demo + password reset).
- Later, we will extend page-by-page for the rest of the application.

Assumptions (based on your schema snippet):
Tables used:
- users
- user_login_history
- demo_sessions

Notes:
- Passwords are handled as plaintext to match your current legacy approach (NOT recommended for production).
- Password reset email sending is NOT implemented here (non-DB). This file only creates/stores reset token+expiry
  and returns a response that App.py can display.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import streamlit as st
from supabase import create_client, Client


class SupabaseManager:
    """Supabase manager supporting App.py authentication + demo session + password reset token storage."""

    def __init__(self) -> None:
        """Initialize Supabase client from Streamlit secrets and test connectivity."""
        try:
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            self.client: Client = create_client(self.url, self.key)

            # Optional demo IDs (App.py may rely on them)
            self.demo_user_id = st.secrets.get("app", {}).get("demo_user_id")
            self.demo_project_id = st.secrets.get("app", {}).get("demo_project_id")

            # Test connection quickly
            self.client.table("users").select("user_id").limit(1).execute()

        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            raise

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _safe_bool(v: Any, default: bool = False) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y")
        return default

    def _log_login_event(
        self,
        user_id: Optional[str],
        successful: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Write to user_login_history. Never raises."""
        try:
            payload = {
                "user_id": user_id,
                "login_timestamp": self._now_iso(),
                "ip_address": ip_address,
                "user_agent": user_agent,
                "login_successful": successful,
                "failure_reason": failure_reason,
            }
            self.client.table("user_login_history").insert(payload).execute()
        except Exception:
            # Logging must never break login UX
            pass

    # ---------------------------------------------------------------------
    # Reads used by sidebar / Home later (App.py may call these indirectly)
    # ---------------------------------------------------------------------

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Minimal user fetch (safe for sidebar, profile headers)."""
        try:
            res = (
                self.client.table("users")
                .select("user_id, username, email, full_name, subscription_tier, is_active, is_demo_user")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_id failed: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch user row by username."""
        try:
            res = (
                self.client.table("users")
                .select("*")
                .eq("username", username)
                .limit(1)
                .execute()
            )
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_username failed: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Fetch user row by email."""
        try:
            res = (
                self.client.table("users")
                .select("*")
                .eq("email", email)
                .limit(1)
                .execute()
            )
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_email failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # App.py: Login
    # ---------------------------------------------------------------------

    def verify_password(self, user: Dict[str, Any], password: str) -> bool:
        """
        Plaintext password verification (legacy).
        If you later migrate to hashing, only this function needs to change.
        """
        try:
            stored = user.get("password")
            if stored is None:
                return False
            return str(stored) == str(password)
        except Exception:
            return False

    def login_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        App.py-compatible login:
        - Fetch user by username
        - Check is_active
        - Verify password
        - Update user login metadata
        - Log event in user_login_history
        Returns the user dict on success, else None.
        """
        user = self.get_user_by_username(username)
        if not user:
            self._log_login_event(
                user_id=None,
                successful=False,
                ip_address=ip_address,
                user_agent=user_agent,
                failure_reason="user_not_found",
            )
            return None

        if not self._safe_bool(user.get("is_active", True), default=True):
            self._log_login_event(
                user_id=user.get("user_id"),
                successful=False,
                ip_address=ip_address,
                user_agent=user_agent,
                failure_reason="inactive_user",
            )
            return None

        if not self.verify_password(user, password):
            # increment failed attempts (best effort)
            try:
                failed = int(user.get("failed_login_attempts") or 0) + 1
                self.client.table("users").update(
                    {
                        "failed_login_attempts": failed,
                        "last_failed_login": self._now_iso(),
                        "updated_at": self._now_iso(),
                    }
                ).eq("user_id", user["user_id"]).execute()
            except Exception:
                pass

            self._log_login_event(
                user_id=user.get("user_id"),
                successful=False,
                ip_address=ip_address,
                user_agent=user_agent,
                failure_reason="invalid_password",
            )
            return None

        # Successful login: update metadata
        try:
            login_count = int(user.get("login_count") or 0) + 1
            self.client.table("users").update(
                {
                    "last_login": self._now_iso(),
                    "last_login_ip": ip_address,
                    "last_login_user_agent": user_agent,
                    "login_count": login_count,
                    "failed_login_attempts": 0,
                    "updated_at": self._now_iso(),
                }
            ).eq("user_id", user["user_id"]).execute()
        except Exception:
            pass

        self._log_login_event(
            user_id=user.get("user_id"),
            successful=True,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=None,
        )

        # Return the user row (App.py will store needed fields in session_state)
        return user

    # ---------------------------------------------------------------------
    # App.py: Demo user
    # ---------------------------------------------------------------------

    def is_demo_user(self, user_id: str) -> bool:
        """
        App.py uses this to decide demo session behavior.
        Primary signal is users.is_demo_user (schema includes it).
        Fallback to secrets demo_user_id if present.
        """
        try:
            u = self.get_user_by_id(user_id)
            if u and self._safe_bool(u.get("is_demo_user"), default=False):
                return True
        except Exception:
            pass

        if self.demo_user_id and str(user_id) == str(self.demo_user_id):
            return True

        return False

    def create_demo_session(
        self,
        user_id: str,
        project_id: str,
        duration_seconds: int = 1800,
    ) -> Optional[Dict[str, Any]]:
        """
        Insert demo session row and return it.
        App.py should store session_token / expires_at / session id in session_state.
        """
        try:
            session_token = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(seconds=int(duration_seconds))

            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "session_token": session_token,
                "expires_at": expires_at.isoformat(),
                "cleanup_required": True,
            }
            res = self.client.table("demo_sessions").insert(payload).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"Failed to create demo session: {e}")
            return None

    def end_demo_session(self, session_id: str) -> None:
        """
        Best-effort end; if you later create an RPC cleanup function, you can call it here.
        For now, we mark cleanup_required false and set an optional logout timestamp if present.
        """
        try:
            # Minimal: disable cleanup_required to indicate session closed
            self.client.table("demo_sessions").update(
                {
                    "cleanup_required": False,
                }
            ).eq("session_id", session_id).execute()
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # App.py: Password reset (DB-side only)
    # ---------------------------------------------------------------------

    def request_password_reset(self, email: str) -> Dict[str, Any]:
        """
        App.py calls this when the user clicks "Restore password".

        DB responsibilities here:
        - Check whether email exists in users.email
        - If exists: generate token + expiry and store in users.password_reset_token/password_reset_expires
        - Return a response that App.py can render

        Email sending is NOT done here (non-DB). We will implement that later in the app layer.
        """
        email = (email or "").strip()
        if not email:
            return {
                "success": False,
                "message": "Please enter a valid email address.",
                "email_exists": False,
            }

        try:
            user = self.get_user_by_email(email)
            if not user:
                # Safer UX (avoid enumeration): claim email sent.
                # If you want explicit existence, switch message.
                return {
                    "success": True,
                    "message": "If this email exists in our system, a reset link will be sent.",
                    "email_exists": False,
                }

            token = str(uuid.uuid4())
            expires = datetime.now() + timedelta(hours=1)

            self.client.table("users").update(
                {
                    "password_reset_token": token,
                    "password_reset_expires": expires.isoformat(),
                    "updated_at": self._now_iso(),
                }
            ).eq("user_id", user["user_id"]).execute()

            return {
                "success": True,
                "message": "If this email exists in our system, a reset link will be sent.",
                "email_exists": True,
                # App.py may optionally use these for debugging/admin flows.
                # Do NOT show token to end users in production UI.
                "reset_token": token,
                "reset_expires": expires.isoformat(),
                "user_id": user["user_id"],
            }

        except Exception as e:
            st.warning(f"request_password_reset failed: {e}")
            return {
                "success": False,
                "message": "An error occurred while processing your request. Please try again later.",
                "error": str(e),
            }


# ---------------------------------------------------------------------
# Singleton (Streamlit)
# ---------------------------------------------------------------------

@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Return cached database manager instance."""
    return SupabaseManager()
