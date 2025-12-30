"""
ExplainFutures - Supabase Database Manager (FULL / APP + HOME COMPATIBLE)

Goal:
- Keep ALL important functions (do not drop anything).
- Fix class structure/indentation so App.py and Home.py work reliably.
- Include:
  - App.py: login, password reset token storage, demo sessions
  - Home: project listing, collaborators, project fetch
  - Progress: project_progress_steps + projects.completion_percentage recompute
  - Time-series: save/load/delete/summary
  - Parameters: save/get/merge/rename/delete/duplicate checks
  - Health reports: save/get/generate/update checks
  - Project management: create/delete/restore/rename, limits checks

Notes:
- Passwords are plaintext (as per your current choice).
- Email sending is intentionally NOT implemented here (DB layer only).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from supabase import Client, create_client


class SupabaseManager:
    """Supabase manager with time-series + app auth + home/progress support."""

    def __init__(self) -> None:
        """Initialize Supabase client from Streamlit secrets."""
        try:
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            self.client: Client = create_client(self.url, self.key)

            # Optional demo IDs
            self.demo_user_id = st.secrets.get("app", {}).get("demo_user_id")
            self.demo_project_id = st.secrets.get("app", {}).get("demo_project_id")

            # Test connection
            self.client.table("users").select("user_id").limit(1).execute()

        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            st.info("💡 Try updating Supabase client version if needed.")
            raise

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

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

    def convert_timestamps_to_serializable(self, obj: Any) -> Any:
        """Recursively convert pandas/datetime objects to ISO strings for JSON serialization."""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self.convert_timestamps_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_timestamps_to_serializable(i) for i in obj]
        return obj

    def _log_login_event(
        self,
        user_id: Optional[str],
        successful: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Write to user_login_history; never raises."""
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
            pass

    # =====================================================================
    # TIME-SERIES DATA MANAGEMENT
    # =====================================================================

    def save_timeseries_data(
        self,
        project_id: str,
        df_long: pd.DataFrame,
        data_source: str = "original",
        batch_size: int = 1000,
    ) -> bool:
        """
        Save time-series data in batches into timeseries_data.
        Expected columns: timestamp/time, variable, value
        """
        try:
            # Delete existing for this project+source first
            self.client.table("timeseries_data").delete().eq("project_id", project_id).eq(
                "data_source", data_source
            ).execute()

            time_col = "timestamp" if "timestamp" in df_long.columns else "time"
            records: List[Dict[str, Any]] = []

            for _, row in df_long.iterrows():
                ts = row[time_col]
                ts_str = ts.isoformat() if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).isoformat()

                val = row["value"]
                if pd.isna(val):
                    val = None
                else:
                    val = float(val)

                records.append(
                    {
                        "project_id": str(project_id),
                        "timestamp": ts_str,
                        "variable": str(row["variable"]),
                        "value": val,
                        "data_source": data_source,
                    }
                )

            inserted_count = 0
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.client.table("timeseries_data").insert(batch).execute()
                inserted_count += len(batch)

            return True

        except Exception as e:
            st.error(f"❌ Error saving time-series data: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
            return False

    def load_timeseries_data(
        self,
        project_id: str,
        data_source: str = "original",
        variables: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Load time-series data into a long DataFrame: timestamp, variable, value."""
        try:
            query = (
                self.client.table("timeseries_data")
                .select("timestamp, variable, value")
                .eq("project_id", project_id)
                .eq("data_source", data_source)
            )

            if variables:
                query = query.in_("variable", variables)

            all_data: List[Dict[str, Any]] = []
            offset = 0
            limit = 1000

            while True:
                res = query.order("timestamp").range(offset, offset + limit - 1).execute()
                if not res.data:
                    break
                all_data.extend(res.data)
                if len(res.data) < limit:
                    break
                offset += limit

            if not all_data:
                return None

            df = pd.DataFrame(all_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(["variable", "timestamp"]).reset_index(drop=True)
            return df

        except Exception as e:
            st.error(f"❌ Error loading time-series data: {str(e)}")
            return None

    def get_timeseries_summary(self, project_id: str, data_source: str = "original") -> Dict[str, Any]:
        """Return counts + variable list for timeseries_data."""
        try:
            count_res = (
                self.client.table("timeseries_data")
                .select("data_id", count="exact")
                .eq("project_id", project_id)
                .eq("data_source", data_source)
                .execute()
            )
            total_records = count_res.count if hasattr(count_res, "count") else 0

            vars_res = (
                self.client.table("timeseries_data")
                .select("variable")
                .eq("project_id", project_id)
                .eq("data_source", data_source)
                .execute()
            )
            variables = list({r["variable"] for r in (vars_res.data or [])})

            return {
                "total_records": total_records,
                "variable_count": len(variables),
                "variables": sorted(variables),
                "data_source": data_source,
            }

        except Exception as e:
            st.error(f"Error getting summary: {str(e)}")
            return {"total_records": 0, "variable_count": 0, "variables": [], "data_source": data_source}

    def delete_timeseries_data(
        self,
        project_id: str,
        data_source: Optional[str] = None,
        variables: Optional[List[str]] = None,
    ) -> bool:
        """Delete time-series data for a project (optionally filtered)."""
        try:
            q = self.client.table("timeseries_data").delete().eq("project_id", project_id)
            if data_source:
                q = q.eq("data_source", data_source)
            if variables:
                q = q.in_("variable", variables)
            q.execute()
            return True
        except Exception as e:
            st.error(f"Error deleting data: {str(e)}")
            return False

    # =====================================================================
    # USER MANAGEMENT (App.py)
    # =====================================================================

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
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
        try:
            res = self.client.table("users").select("*").eq("username", username).limit(1).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_username failed: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            res = self.client.table("users").select("*").eq("email", email).limit(1).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_email failed: {e}")
            return None

    def verify_password(self, user: Dict[str, Any], password: str) -> bool:
        """Plaintext password verification (legacy)."""
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
        """App.py-compatible login."""
        user = self.get_user_by_username(username)
        if not user:
            self._log_login_event(None, False, ip_address, user_agent, "user_not_found")
            return None

        if not self._safe_bool(user.get("is_active", True), default=True):
            self._log_login_event(user.get("user_id"), False, ip_address, user_agent, "inactive_user")
            return None

        if not self.verify_password(user, password):
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

            self._log_login_event(user.get("user_id"), False, ip_address, user_agent, "invalid_password")
            return None

        # Successful login metadata
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

        self._log_login_event(user.get("user_id"), True, ip_address, user_agent, None)
        return user

    def request_password_reset(self, email: str) -> Dict[str, Any]:
        """
        DB-side reset token+expiry storage only.
        App.py shows message; email sending will be handled later (outside DB layer).
        """
        email = (email or "").strip()
        if not email:
            return {"success": False, "message": "Please enter a valid email address.", "email_exists": False}

        try:
            user = self.get_user_by_email(email)
            # Avoid enumeration: same message either way
            msg = "If this email exists in our system, a reset link will be sent."

            if not user:
                return {"success": True, "message": msg, "email_exists": False}

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
                "message": msg,
                "email_exists": True,
                "user_id": user["user_id"],
                # keep for admin/debug (do not show in UI in production)
                "reset_token": token,
                "reset_expires": expires.isoformat(),
            }

        except Exception as e:
            st.warning(f"request_password_reset failed: {e}")
            return {"success": False, "message": "An error occurred. Please try again later.", "error": str(e)}

    # =====================================================================
    # DEMO SESSION MANAGEMENT
    # =====================================================================

    def is_demo_user(self, user_id: str) -> bool:
        """Use users.is_demo_user, with fallback to secrets demo_user_id."""
        try:
            u = self.get_user_by_id(user_id)
            if u and self._safe_bool(u.get("is_demo_user"), default=False):
                return True
        except Exception:
            pass

        return bool(self.demo_user_id and str(user_id) == str(self.demo_user_id))

    def create_demo_session(self, user_id: str, project_id: str, duration_seconds: int = 1800) -> Optional[Dict[str, Any]]:
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
        """Best effort."""
        try:
            self.client.table("demo_sessions").update({"cleanup_required": False}).eq("session_id", session_id).execute()
        except Exception:
            pass

    # =====================================================================
    # PROJECT MANAGEMENT (Home)
    # =====================================================================

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
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
        """Home page: owned projects + collaborations."""
        try:
            owned_q = (
                self.client.table("projects")
                .select(
                    "project_id, project_name, project_code, description, owner_id, "
                    "application_name, status, visibility, is_demo_project, "
                    "workflow_state, current_page, completion_percentage, total_parameters, total_scenarios, "
                    "baseline_year, scenario_target_year, tags, created_at, updated_at, last_accessed, team_size"
                )
                .eq("owner_id", user_id)
                .eq("application_name", application_name)
            )
            if not include_deleted:
                owned_q = owned_q.neq("status", "deleted")

            owned_res = owned_q.order("updated_at", desc=True).execute()
            owned = owned_res.data if owned_res.data else []
            for p in owned:
                p["access_role"] = "owner"
                p["is_owner"] = True

            if not include_collaborations:
                return owned

            collab_res = (
                self.client.table("project_collaborators")
                .select("project_id, role, can_view, can_edit, can_delete, can_export, invitation_status, created_at")
                .eq("user_id", user_id)
                .execute()
            )
            collabs = collab_res.data if collab_res.data else []
            if not collabs:
                return owned

            ids = [c["project_id"] for c in collabs if c.get("project_id")]
            if not ids:
                return owned

            proj_q = (
                self.client.table("projects")
                .select(
                    "project_id, project_name, project_code, description, owner_id, "
                    "application_name, status, visibility, is_demo_project, "
                    "workflow_state, current_page, completion_percentage, total_parameters, total_scenarios, "
                    "baseline_year, scenario_target_year, tags, created_at, updated_at, last_accessed, team_size"
                )
                .in_("project_id", ids)
                .eq("application_name", application_name)
            )
            if not include_deleted:
                proj_q = proj_q.neq("status", "deleted")

            proj_res = proj_q.execute()
            collab_projects = proj_res.data if proj_res.data else []

            collab_map = {c["project_id"]: c for c in collabs}
            for p in collab_projects:
                c = collab_map.get(p["project_id"], {})
                p["access_role"] = c.get("role", "viewer")
                p["is_owner"] = False
                p["can_view"] = c.get("can_view", True)
                p["can_edit"] = c.get("can_edit", False)
                p["can_delete"] = c.get("can_delete", False)
                p["can_export"] = c.get("can_export", True)
                p["invitation_status"] = c.get("invitation_status", "accepted")

            owned.extend(collab_projects)
            return owned

        except Exception as e:
            st.warning(f"get_user_projects failed: {e}")
            return []

    def get_project_collaborators(self, project_id: str) -> List[Dict[str, Any]]:
        """Home: collaborator list joined with users."""
        try:
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

            user_ids = list({c["user_id"] for c in collabs if c.get("user_id")})
            users_res = self.client.table("users").select("user_id, username, email, full_name").in_(
                "user_id", user_ids
            ).execute()
            users = users_res.data if users_res.data else []
            users_map = {u["user_id"]: u for u in users}

            out: List[Dict[str, Any]] = []
            for c in collabs:
                u = users_map.get(c["user_id"], {})
                out.append({**c, "username": u.get("username"), "email": u.get("email"), "full_name": u.get("full_name")})
            return out

        except Exception as e:
            st.warning(f"get_project_collaborators failed: {e}")
            return []

    def touch_project_last_accessed(self, project_id: str, user_id: Optional[str] = None) -> None:
        """When opening from Home."""
        try:
            payload = {"last_accessed": self._now_iso(), "updated_at": self._now_iso()}
            if user_id:
                payload["last_activity_by"] = user_id
            self.client.table("projects").update(payload).eq("project_id", project_id).execute()
        except Exception:
            pass

    # =====================================================================
    # PROGRESS STEPS (Option 2)
    # =====================================================================

    def upsert_progress_step(self, project_id: str, step_key: str, step_percent: int) -> bool:
        """Insert/update a step contribution in project_progress_steps."""
        try:
            step_percent = int(step_percent)
            payload = {
                "project_id": project_id,
                "step_key": step_key,
                "step_percent": step_percent,
                "updated_at": self._now_iso(),
            }

            # Prefer upsert if unique constraint exists; else fallback
            try:
                self.client.table("project_progress_steps").upsert(payload).execute()
                return True
            except Exception:
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
        """Sum step_percent and write into projects.completion_percentage."""
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

            update_payload: Dict[str, Any] = {"completion_percentage": total, "updated_at": self._now_iso()}
            if workflow_state is not None:
                update_payload["workflow_state"] = workflow_state
            if current_page is not None:
                update_payload["current_page"] = int(current_page)

            self.client.table("projects").update(update_payload).eq("project_id", project_id).execute()
            return total

        except Exception as e:
            st.warning(f"recompute_and_update_project_progress failed: {e}")
            return 0

    def update_project_progress(
        self,
        project_id: str,
        workflow_state: Optional[str] = None,
        current_page: Optional[int] = None,
        completion_percentage: Optional[int] = None,
    ) -> None:
        """Direct update (kept for backward compatibility)."""
        try:
            update_data: Dict[str, Any] = {"updated_at": self._now_iso()}
            if workflow_state:
                update_data["workflow_state"] = workflow_state
            if current_page is not None:
                update_data["current_page"] = int(current_page)
            if completion_percentage is not None:
                update_data["completion_percentage"] = int(completion_percentage)

            self.client.table("projects").update(update_data).eq("project_id", project_id).execute()
        except Exception as e:
            st.warning(f"Progress update warning: {str(e)}")

    # =====================================================================
    # LIMIT CHECKS (used by Home / creation)
    # =====================================================================

    def check_user_limits(self, user_id: str) -> Dict[str, Any]:
        try:
            user = self.client.table("users").select("*").eq("user_id", user_id).execute()
            if not user.data:
                return {"can_create_project": False, "can_upload": False, "reason": "User not found"}

            u = user.data[0]
            can_create = u.get("current_project_count", 0) < u.get("max_projects", 3)
            can_upload = u.get("uploads_this_month", 0) < u.get("max_uploads_per_month", 50)

            return {
                "can_create_project": can_create,
                "can_upload": can_upload,
                "current_projects": u.get("current_project_count", 0),
                "max_projects": u.get("max_projects", 3),
                "current_uploads": u.get("uploads_this_month", 0),
                "max_uploads": u.get("max_uploads_per_month", 50),
            }

        except Exception as e:
            st.error(f"Error checking limits: {str(e)}")
            return {"can_create_project": False, "can_upload": False, "reason": str(e)}

    # =====================================================================
    # PROJECT CREATE / DELETE / RESTORE / RENAME (kept)
    # =====================================================================

    def create_project(
        self,
        owner_id: str,
        project_name: str,
        description: str = None,
        baseline_year: int = None,
        scenario_target_year: int = None,
        application_name: str = "explainfutures",
    ) -> Optional[Dict[str, Any]]:
        try:
            import random
            import string

            project_code = f"PRJ-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

            project_data: Dict[str, Any] = {
                "owner_id": owner_id,
                "project_name": project_name,
                "project_code": project_code,
                "description": description,
                "status": "active",
                "workflow_state": "setup",
                "current_page": 2,
                "completion_percentage": 0,
                "application_name": application_name,
            }
            if baseline_year is not None:
                project_data["baseline_year"] = baseline_year
            if scenario_target_year is not None:
                project_data["scenario_target_year"] = scenario_target_year

            res = self.client.table("projects").insert(project_data).execute()
            if not res.data:
                return None

            # Update project count
            try:
                user = self.client.table("users").select("current_project_count").eq("user_id", owner_id).execute()
                if user.data:
                    new_count = user.data[0].get("current_project_count", 0) + 1
                    self.client.table("users").update({"current_project_count": new_count}).eq("user_id", owner_id).execute()
            except Exception:
                pass

            return res.data[0]

        except Exception as e:
            st.error(f"Error creating project: {str(e)}")
            return None

    def delete_project(self, project_id: str, user_id: str) -> bool:
        """Soft delete."""
        try:
            proj = self.client.table("projects").select("owner_id, project_name").eq("project_id", project_id).execute()
            if not proj.data or proj.data[0]["owner_id"] != user_id:
                st.error("Only the project owner can delete this project")
                return False

            self.client.table("projects").update(
                {
                    "status": "deleted",
                    "deleted_at": self._now_iso(),
                    "deleted_by": user_id,
                    "updated_at": self._now_iso(),
                }
            ).eq("project_id", project_id).execute()

            # Remove collaborators
            try:
                self.client.table("project_collaborators").delete().eq("project_id", project_id).execute()
            except Exception:
                pass

            # Delete timeseries
            try:
                self.delete_timeseries_data(project_id)
            except Exception:
                pass

            # Recompute user's active project count
            try:
                active = (
                    self.client.table("projects")
                    .select("project_id", count="exact")
                    .eq("owner_id", user_id)
                    .eq("status", "active")
                    .execute()
                )
                new_count = active.count if hasattr(active, "count") else 0
                self.client.table("users").update({"current_project_count": new_count}).eq("user_id", user_id).execute()
            except Exception:
                pass

            return True

        except Exception as e:
            st.error(f"Error deleting project: {str(e)}")
            return False

    def restore_project(self, project_id: str, user_id: str) -> bool:
        try:
            proj = self.client.table("projects").select("owner_id, status, project_name").eq("project_id", project_id).execute()
            if not proj.data:
                st.error("Project not found")
                return False
            if proj.data[0]["owner_id"] != user_id:
                st.error("Only the project owner can restore this project")
                return False
            if proj.data[0]["status"] != "deleted":
                st.warning("Project is not deleted")
                return False

            self.client.table("projects").update(
                {"status": "active", "deleted_at": None, "deleted_by": None, "updated_at": self._now_iso()}
            ).eq("project_id", project_id).execute()

            # update user's count
            try:
                active = (
                    self.client.table("projects")
                    .select("project_id", count="exact")
                    .eq("owner_id", user_id)
                    .eq("status", "active")
                    .execute()
                )
                new_count = active.count if hasattr(active, "count") else 0
                self.client.table("users").update({"current_project_count": new_count}).eq("user_id", user_id).execute()
            except Exception:
                pass

            return True
        except Exception as e:
            st.error(f"Error restoring project: {str(e)}")
            return False

    def rename_project(self, project_id: str, new_name: str, user_id: str) -> bool:
        try:
            proj = self.client.table("projects").select("owner_id, project_name").eq("project_id", project_id).execute()
            if not proj.data:
                st.error("Project not found")
                return False
            if proj.data[0]["owner_id"] != user_id:
                st.error("Only the project owner can rename this project")
                return False
            if not new_name or len(new_name.strip()) == 0:
                st.error("Project name cannot be empty")
                return False
            if len(new_name) > 200:
                st.error("Project name is too long (max 200 characters)")
                return False

            self.client.table("projects").update(
                {"project_name": new_name.strip(), "updated_at": self._now_iso()}
            ).eq("project_id", project_id).execute()
            return True

        except Exception as e:
            st.error(f"Error renaming project: {str(e)}")
            return False

    # =====================================================================
    # FILE UPLOAD MANAGEMENT
    # =====================================================================

    def save_uploaded_file(
        self, project_id: str, filename: str, file_size: int, file_type: str, metadata: dict = None
    ) -> Optional[Dict[str, Any]]:
        try:
            res = self.client.table("uploaded_files").insert(
                {
                    "project_id": project_id,
                    "filename": filename,
                    "file_size": file_size,
                    "file_type": file_type,
                    "metadata": json.dumps(metadata) if metadata else None,
                    "uploaded_at": self._now_iso(),
                }
            ).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.error(f"Error saving file info: {str(e)}")
            return None

    def get_uploaded_files(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            res = (
                self.client.table("uploaded_files")
                .select("*")
                .eq("project_id", project_id)
                .order("uploaded_at", desc=True)
                .execute()
            )
            return res.data if res.data else []
        except Exception as e:
            st.error(f"Error fetching uploaded files: {str(e)}")
            return []

    # =====================================================================
    # PARAMETER MANAGEMENT
    # =====================================================================

    def save_parameters(self, project_id: str, parameters: List[Dict[str, Any]]) -> bool:
        try:
            for param in parameters:
                existing = (
                    self.client.table("parameters")
                    .select("parameter_id")
                    .eq("project_id", project_id)
                    .eq("parameter_name", param["name"])
                    .execute()
                )

                param_data = {
                    "project_id": project_id,
                    "parameter_name": param["name"],
                    "data_type": param.get("data_type", "numeric"),
                    "unit": param.get("unit"),
                    "description": param.get("description"),
                    "min_value": param.get("min_value"),
                    "max_value": param.get("max_value"),
                    "mean_value": param.get("mean_value"),
                    "std_value": param.get("std_value"),
                    "missing_count": param.get("missing_count", 0),
                    "total_count": param.get("total_count", 0),
                    "updated_at": self._now_iso(),
                }

                if existing.data:
                    self.client.table("parameters").update(param_data).eq(
                        "parameter_id", existing.data[0]["parameter_id"]
                    ).execute()
                else:
                    param_data["created_at"] = self._now_iso()
                    self.client.table("parameters").insert(param_data).execute()

            return True
        except Exception as e:
            st.error(f"Error saving parameters: {str(e)}")
            return False

    def get_project_parameters(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            res = self.client.table("parameters").select("*").eq("project_id", project_id).order("parameter_name").execute()
            return res.data if res.data else []
        except Exception as e:
            st.error(f"Error fetching parameters: {str(e)}")
            return []

    def check_duplicate_parameters(self, project_id: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            params = self.get_project_parameters(project_id)
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for p in params:
                groups.setdefault(p["parameter_name"], []).append(p)
            return {name: group for name, group in groups.items() if len(group) > 1}
        except Exception as e:
            st.error(f"Error checking duplicates: {str(e)}")
            return {}

    def merge_parameters(self, parameter_ids: List[str], keep_id: str) -> bool:
        try:
            for pid in parameter_ids:
                if pid != keep_id:
                    self.client.table("parameters").delete().eq("parameter_id", pid).execute()
            return True
        except Exception as e:
            st.error(f"Error merging parameters: {str(e)}")
            return False

    def rename_parameter(self, parameter_id: str, new_name: str) -> bool:
        try:
            self.client.table("parameters").update({"parameter_name": new_name, "updated_at": self._now_iso()}).eq(
                "parameter_id", parameter_id
            ).execute()
            return True
        except Exception as e:
            st.error(f"Error renaming parameter: {str(e)}")
            return False

    def delete_parameter(self, parameter_id: str) -> bool:
        try:
            self.client.table("parameters").delete().eq("parameter_id", parameter_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting parameter: {str(e)}")
            return False

    # =====================================================================
    # STEP COMPLETION JSON (projects.step_completion)
    # =====================================================================

    def update_step_completion(self, project_id: str, step_key: str, completed: bool = True) -> bool:
        try:
            res = self.client.table("projects").select("step_completion").eq("project_id", project_id).execute()
            if not res.data:
                return False

            step_completion = res.data[0].get("step_completion") or {}
            step_completion[step_key] = completed

            self.client.table("projects").update(
                {"step_completion": step_completion, "updated_at": self._now_iso()}
            ).eq("project_id", project_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating step completion: {str(e)}")
            return False

    def get_step_completion(self, project_id: str) -> Dict[str, bool]:
        try:
            res = self.client.table("projects").select("step_completion").eq("project_id", project_id).execute()
            if res.data and res.data[0].get("step_completion"):
                return res.data[0]["step_completion"]
            return {}
        except Exception as e:
            st.error(f"Error fetching step completion: {str(e)}")
            return {}

    # =====================================================================
    # HEALTH REPORTS
    # =====================================================================

    def save_health_report(self, project_id: str, health_data: Dict[str, Any]) -> bool:
        try:
            import hashlib

            parameters = self.get_project_parameters(project_id)
            param_string = json.dumps(sorted([p["parameter_name"] for p in parameters]))
            data_hash = hashlib.md5(param_string.encode()).hexdigest()

            missing_values_detail = self.convert_timestamps_to_serializable(health_data.get("missing_values_detail", {}))
            outliers_detail = self.convert_timestamps_to_serializable(health_data.get("outliers_detail", {}))
            coverage_detail = self.convert_timestamps_to_serializable(health_data.get("coverage_detail", {}))
            issues_list = self.convert_timestamps_to_serializable(health_data.get("issues_list", []))
            time_metadata = self.convert_timestamps_to_serializable(health_data.get("time_metadata", {}))
            parameters_analyzed = self.convert_timestamps_to_serializable(health_data.get("parameters_analyzed", []))

            def to_int(v: Any) -> int:
                if v is None:
                    return 0
                if hasattr(v, "item"):
                    v = v.item()
                return int(float(v))

            def to_float(v: Any) -> float:
                if v is None:
                    return 0.0
                if hasattr(v, "item"):
                    v = v.item()
                return float(v)

            report_data: Dict[str, Any] = {
                "project_id": str(project_id),
                "health_score": to_int(health_data.get("health_score", 0)),
                "health_category": str(health_data.get("health_category", "poor")),
                "total_parameters": to_int(health_data.get("total_parameters", 0)),
                "total_data_points": to_int(health_data.get("total_data_points", 0)),
                "total_missing_values": to_int(health_data.get("total_missing_values", 0)),
                "missing_percentage": to_float(health_data.get("missing_percentage", 0)),
                "critical_issues": to_int(health_data.get("critical_issues", 0)),
                "warnings": to_int(health_data.get("warnings", 0)),
                "duplicate_timestamps": to_int(health_data.get("duplicate_timestamps", 0)),
                "outlier_count": to_int(health_data.get("outlier_count", 0)),
                "missing_values_detail": json.dumps(missing_values_detail),
                "outliers_detail": json.dumps(outliers_detail),
                "coverage_detail": json.dumps(coverage_detail),
                "issues_list": json.dumps(issues_list),
                "time_metadata": json.dumps(time_metadata),
                "parameters_analyzed": parameters_analyzed,
                "data_hash": str(data_hash),
                "updated_at": self._now_iso(),
            }

            if "conversion_info" in health_data:
                report_data["conversion_info"] = json.dumps(self.convert_timestamps_to_serializable(health_data["conversion_info"]))
            if "warnings_list" in health_data:
                report_data["warnings_list"] = json.dumps(self.convert_timestamps_to_serializable(health_data["warnings_list"]))
            if "raw_health_report" in health_data:
                report_data["raw_health_report"] = json.dumps(self.convert_timestamps_to_serializable(health_data["raw_health_report"]))

            existing = (
                self.client.table("health_reports")
                .select("report_id")
                .eq("project_id", project_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if existing.data:
                self.client.table("health_reports").update(report_data).eq("report_id", existing.data[0]["report_id"]).execute()
            else:
                self.client.table("health_reports").insert(report_data).execute()

            return True

        except Exception as e:
            st.error(f"Error saving health report: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
            return False

    def get_health_report(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            res = (
                self.client.table("health_reports")
                .select("*")
                .eq("project_id", project_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if not res.data:
                return None

            report = res.data[0]
            report["missing_values_detail"] = json.loads(report.get("missing_values_detail", "{}"))
            report["outliers_detail"] = json.loads(report.get("outliers_detail", "{}"))
            report["coverage_detail"] = json.loads(report.get("coverage_detail", "{}"))
            report["issues_list"] = json.loads(report.get("issues_list", "[]"))
            report["time_metadata"] = json.loads(report.get("time_metadata", "{}"))

            if report.get("conversion_info"):
                report["conversion_info"] = json.loads(report["conversion_info"])
            if report.get("warnings_list"):
                report["warnings_list"] = json.loads(report["warnings_list"])
            if report.get("raw_health_report"):
                report["raw_health_report"] = json.loads(report["raw_health_report"])

            return report

        except Exception as e:
            st.error(f"Error fetching health report: {str(e)}")
            return None

    def needs_health_report_update(self, project_id: str) -> bool:
        try:
            report = self.get_health_report(project_id)
            if not report:
                return True

            parameters = self.get_project_parameters(project_id)
            current_params = sorted([p["parameter_name"] for p in parameters])
            report_params = sorted(report.get("parameters_analyzed", []))

            if current_params != report_params:
                return True

            import hashlib

            current_hash = hashlib.md5(json.dumps(current_params).encode()).hexdigest()
            return current_hash != report.get("data_hash")

        except Exception as e:
            st.error(f"Error checking report status: {str(e)}")
            return True

    def generate_health_report_from_parameters(self, project_id: str) -> Dict[str, Any]:
        try:
            parameters = self.get_project_parameters(project_id)
            if not parameters:
                return {"success": False, "message": "No parameters to analyze"}

            health_score = 100
            issues: List[str] = []
            critical_issues = 0
            warnings = 0

            missing_values_detail: Dict[str, Any] = {}
            total_missing = 0
            total_count = 0

            for p in parameters:
                name = p["parameter_name"]
                missing_count = p.get("missing_count", 0) or 0
                param_total = p.get("total_count", 0) or 0

                total_missing += missing_count
                total_count += param_total

                if param_total > 0:
                    missing_pct = missing_count / param_total
                    missing_values_detail[name] = {"count": missing_count, "percentage": missing_pct, "total": param_total}

                    if missing_pct > 0.20:
                        health_score -= 15
                        issues.append(f"⚠️ {name}: {missing_pct*100:.1f}% missing (critical)")
                        critical_issues += 1
                    elif missing_pct > 0.05:
                        health_score -= 5
                        issues.append(f"⚠️ {name}: {missing_pct*100:.1f}% missing")
                        warnings += 1

            overall_missing_pct = total_missing / total_count if total_count > 0 else 0
            health_score = max(0, min(100, health_score))

            if health_score >= 85:
                category = "excellent"
            elif health_score >= 70:
                category = "good"
            elif health_score >= 50:
                category = "fair"
            else:
                category = "poor"

            report = {
                "success": True,
                "health_score": health_score,
                "health_category": category,
                "total_parameters": len(parameters),
                "total_data_points": total_count,
                "total_missing_values": total_missing,
                "missing_percentage": overall_missing_pct,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "duplicate_timestamps": 0,
                "outlier_count": 0,
                "missing_values_detail": missing_values_detail,
                "outliers_detail": {},
                "coverage_detail": {},
                "issues_list": issues,
                "time_metadata": {},
                "parameters_analyzed": [p["parameter_name"] for p in parameters],
            }

            self.save_health_report(project_id, report)
            return report

        except Exception as e:
            st.error(f"Error generating health report: {str(e)}")
            return {"success": False, "message": str(e)}


@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Get cached database manager instance."""
    return SupabaseManager()
