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
# Home (01_Home.py) + Progress I/O
# Tables:
# - projects
# - project_collaborators
# - project_progress_steps
# ---------------------------------------------------------------------

def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single project (used by sidebar and project headers)."""
    try:
        res = (
            self.client.table("projects")
            .select(
                "project_id, project_name, project_code, description, owner_id, "
                "application_name, status, visibility, is_demo_project, "
                "workflow_state, current_page, completion_percentage, "
                "baseline_year, scenario_target_year, settings, tags, "
                "created_at, updated_at, last_accessed, team_size"
            )
            .eq("project_id", project_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        st.warning(f"get_project_by_id failed: {e}")
        return None


def get_user_projects(
    self,
    user_id: str,
    include_collaborations: bool = True,
    include_deleted: bool = False,
    application_name: str = "explainfutures",
) -> List[Dict[str, Any]]:
    """
    Home page: list projects owned by user + collaborated projects.
    Ensures completion_percentage is returned for progress bars.
    """
    try:
        # Owned projects
        owned_query = (
            self.client.table("projects")
            .select(
                "project_id, project_name, project_code, description, owner_id, "
                "application_name, status, visibility, is_demo_project, "
                "workflow_state, current_page, completion_percentage, "
                "baseline_year, scenario_target_year, tags, created_at, updated_at, last_accessed, team_size"
            )
            .eq("owner_id", user_id)
            .eq("application_name", application_name)
        )

        if not include_deleted:
            owned_query = owned_query.neq("status", "deleted")

        owned_res = owned_query.order("updated_at", desc=True).execute()
        owned_projects = owned_res.data if owned_res.data else []

        for p in owned_projects:
            p["access_role"] = "owner"
            p["is_owner"] = True

        if not include_collaborations:
            return owned_projects

        # Collaborated projects
        collab_query = (
            self.client.table("project_collaborators")
            .select("project_id, role, can_view, can_edit, can_delete, can_export, invitation_status, created_at")
            .eq("user_id", user_id)
        )
        collab_res = collab_query.execute()
        collabs = collab_res.data if collab_res.data else []

        if not collabs:
            return owned_projects

        project_ids = [c["project_id"] for c in collabs if c.get("project_id")]
        if not project_ids:
            return owned_projects

        proj_query = (
            self.client.table("projects")
            .select(
                "project_id, project_name, project_code, description, owner_id, "
                "application_name, status, visibility, is_demo_project, "
                "workflow_state, current_page, completion_percentage, "
                "baseline_year, scenario_target_year, tags, created_at, updated_at, last_accessed, team_size"
            )
            .in_("project_id", project_ids)
            .eq("application_name", application_name)
        )

        if not include_deleted:
            proj_query = proj_query.neq("status", "deleted")

        proj_res = proj_query.execute()
        collab_projects = proj_res.data if proj_res.data else []

        collab_dict = {c["project_id"]: c for c in collabs}
        for p in collab_projects:
            c = collab_dict.get(p["project_id"], {})
            p["access_role"] = c.get("role", "viewer")
            p["is_owner"] = False
            p["can_view"] = c.get("can_view", True)
            p["can_edit"] = c.get("can_edit", False)
            p["can_delete"] = c.get("can_delete", False)
            p["can_export"] = c.get("can_export", True)
            p["invitation_status"] = c.get("invitation_status", "accepted")

        owned_projects.extend(collab_projects)
        return owned_projects

    except Exception as e:
        st.warning(f"get_user_projects failed: {e}")
        return []


def get_project_collaborators(self, project_id: str) -> List[Dict[str, Any]]:
    """Home page: show collaborators list (minimal, safe)."""
    try:
        # Pull collaborator rows
        collab_res = (
            self.client.table("project_collaborators")
            .select(
                "collaborator_id, project_id, user_id, role, can_view, can_edit, can_delete, can_export, "
                "invitation_status, invited_by, invited_at, accepted_at, last_accessed, created_at, updated_at"
            )
            .eq("project_id", project_id)
            .execute()
        )
        collabs = collab_res.data if collab_res.data else []
        if not collabs:
            return []

        # Pull user info for these collaborators
        user_ids = list({c["user_id"] for c in collabs if c.get("user_id")})
        users_res = (
            self.client.table("users")
            .select("user_id, username, email, full_name")
            .in_("user_id", user_ids)
            .execute()
        )
        users = users_res.data if users_res.data else []
        users_dict = {u["user_id"]: u for u in users}

        out = []
        for c in collabs:
            u = users_dict.get(c["user_id"], {})
            out.append(
                {
                    **c,
                    "username": u.get("username"),
                    "email": u.get("email"),
                    "full_name": u.get("full_name"),
                }
            )
        return out

    except Exception as e:
        st.warning(f"get_project_collaborators failed: {e}")
        return []


def touch_project_last_accessed(self, project_id: str, user_id: Optional[str] = None) -> None:
    """Called when opening a project from Home to update recency fields."""
    try:
        payload = {"last_accessed": self._now_iso(), "updated_at": self._now_iso()}
        if user_id:
            payload["last_activity_by"] = user_id
        self.client.table("projects").update(payload).eq("project_id", project_id).execute()
    except Exception:
        pass


# ----------------------------
# Progress-step functions (Option 2)
# ----------------------------

def upsert_progress_step(self, project_id: str, step_key: str, step_percent: int) -> bool:
    """
    Insert/update a single step contribution in project_progress_steps.
    Requires a UNIQUE constraint on (project_id, step_key) OR uses manual update fallback.
    """
    try:
        step_percent = int(step_percent)

        # Try upsert (works if unique constraint exists in DB)
        payload = {
            "project_id": project_id,
            "step_key": step_key,
            "step_percent": step_percent,
            "updated_at": self._now_iso(),
        }

        try:
            self.client.table("project_progress_steps").upsert(payload).execute()
            return True
        except Exception:
            # Fallback: manual update then insert
            existing = (
                self.client.table("project_progress_steps")
                .select("id")
                .eq("project_id", project_id)
                .eq("step_key", step_key)
                .limit(1)
                .execute()
            )
            if existing.data:
                self.client.table("project_progress_steps").update(
                    {"step_percent": step_percent, "updated_at": self._now_iso()}
                ).eq("id", existing.data[0]["id"]).execute()
            else:
                self.client.table("project_progress_steps").insert(payload).execute()
            return True

    except Exception as e:
        st.warning(f"upsert_progress_step failed: {e}")
        return False


def recompute_and_update_project_progress(
    self,
    project_id: str,
    workflow_state: Optional[str] = None,
    current_page: Optional[int] = None,
) -> int:
    """
    Sum all step_percent from project_progress_steps and write it to projects.completion_percentage.
    Returns the computed percentage.
    """
    try:
        res = (
            self.client.table("project_progress_steps")
            .select("step_percent")
            .eq("project_id", project_id)
            .execute()
        )
        rows = res.data if res.data else []
        total = sum(int(r.get("step_percent") or 0) for r in rows)
        total = max(0, min(100, int(total)))

        update_payload = {"completion_percentage": total, "updated_at": self._now_iso()}
        if workflow_state is not None:
            update_payload["workflow_state"] = workflow_state
        if current_page is not None:
            update_payload["current_page"] = int(current_page)

        self.client.table("projects").update(update_payload).eq("project_id", project_id).execute()
        return total

    except Exception as e:
        st.warning(f"recompute_and_update_project_progress failed: {e}")
        return 0
 
  

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
