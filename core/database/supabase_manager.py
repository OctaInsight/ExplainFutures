"""
ExplainFutures - Supabase Database Manager
- Time-series storage
- Project progress step tracking (Option 2: per-step contributions + recompute)
⚠️ WARNING: plaintext passwords - NOT RECOMMENDED for production!
"""

import streamlit as st
from supabase import create_client
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np


class SupabaseManager:
    """Supabase manager with time-series data storage capabilities"""

    # ---------------------
    # Section 1 — Init / Client
    # ---------------------
    def __init__(self):
        try:
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            self.client = create_client(self.url, self.key)

            self.demo_user_id = st.secrets["app"].get("demo_user_id")
            self.demo_project_id = st.secrets["app"].get("demo_project_id")

            # test connection
            self.client.table("users").select("user_id").limit(1).execute()

        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            raise

    # ---------------------
    # Section 2 — Helpers (serialization)
    # ---------------------
    def convert_timestamps_to_serializable(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self.convert_timestamps_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_timestamps_to_serializable(x) for x in obj]
        return obj

    # ========================================================================
    # Section 3 — TIME-SERIES DATA MANAGEMENT
    # ========================================================================
    def save_timeseries_data(
        self,
        project_id: str,
        df_long: pd.DataFrame,
        data_source: str = "raw",
        batch_size: int = 1000
    ) -> bool:
        """
        Save time-series data to `timeseries_data`.
        Expected columns in df_long: (timestamp|time), variable, value
        """
        try:
            # delete existing rows for (project_id, data_source)
            self.client.table("timeseries_data").delete().eq("project_id", str(project_id)).eq(
                "data_source", data_source
            ).execute()

            time_col = "timestamp" if "timestamp" in df_long.columns else "time"

            records = []
            for _, row in df_long.iterrows():
                ts = row[time_col]
                ts_iso = ts.isoformat() if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).isoformat()

                val = row.get("value")
                if pd.isna(val):
                    val = None
                else:
                    val = float(val)

                records.append(
                    {
                        "project_id": str(project_id),
                        "timestamp": ts_iso,
                        "variable": str(row["variable"]),
                        "value": val,
                        "data_source": str(data_source),
                    }
                )

            for i in range(0, len(records), batch_size):
                self.client.table("timeseries_data").insert(records[i : i + batch_size]).execute()

            return True

        except Exception as e:
            st.error(f"❌ Error saving time-series data: {str(e)}")
            return False

    def load_timeseries_data(
        self,
        project_id: str,
        data_source: str = "raw",
        variables: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load time-series data from `timeseries_data` with pagination."""
        try:
            query = (
                self.client.table("timeseries_data")
                .select("timestamp, variable, value, data_source")
                .eq("project_id", str(project_id))
                .eq("data_source", str(data_source))
            )

            if variables:
                query = query.in_("variable", variables)

            all_data = []
            offset = 0
            limit = 1000

            while True:
                result = query.order("timestamp").range(offset, offset + limit - 1).execute()
                if not result.data:
                    break
                all_data.extend(result.data)
                if len(result.data) < limit:
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

    def get_timeseries_summary(self, project_id: str, data_source: str = "raw") -> Dict[str, Any]:
        try:
            count_result = (
                self.client.table("timeseries_data")
                .select("data_id", count="exact")
                .eq("project_id", str(project_id))
                .eq("data_source", str(data_source))
                .execute()
            )
            total_records = count_result.count if hasattr(count_result, "count") else 0

            vars_result = (
                self.client.table("timeseries_data")
                .select("variable")
                .eq("project_id", str(project_id))
                .eq("data_source", str(data_source))
                .execute()
            )
            variables = list({r["variable"] for r in (vars_result.data or [])})

            return {
                "total_records": total_records,
                "variable_count": len(variables),
                "variables": sorted(variables),
                "data_source": data_source,
            }

        except Exception as e:
            st.error(f"Error getting summary: {str(e)}")
            return {"total_records": 0, "variable_count": 0, "variables": [], "data_source": data_source}

    # ========================================================================
    # Section 4 — PROJECT PROGRESS (Option 2)
    # ========================================================================
    def upsert_progress_step(self, project_id: str, step_key: str, step_percent: int) -> bool:
        """
        Store each page/step contribution in `project_progress_steps`.
        If (project_id, step_key) exists -> update, else insert.
        """
        try:
            pid = str(project_id)
            key = str(step_key)
            pct = int(step_percent)

            existing = (
                self.client.table("project_progress_steps")
                .select("step_id")
                .eq("project_id", pid)
                .eq("step_key", key)
                .limit(1)
                .execute()
            )

            payload = {
                "project_id": pid,
                "step_key": key,
                "step_percent": pct,
                "updated_at": datetime.now().isoformat(),
            }

            if existing.data:
                step_id = existing.data[0]["step_id"]
                self.client.table("project_progress_steps").update(payload).eq("step_id", step_id).execute()
            else:
                payload["created_at"] = datetime.now().isoformat()
                self.client.table("project_progress_steps").insert(payload).execute()

            return True

        except Exception as e:
            st.warning(f"Progress step upsert warning: {str(e)}")
            return False

    def recompute_and_update_project_progress(
        self,
        project_id: str,
        workflow_state: Optional[str] = None,
        current_page: Optional[int] = None
    ) -> int:
        """
        Sum all steps in `project_progress_steps` and write to `projects.completion_percentage`.
        Returns the computed percentage.
        """
        pid = str(project_id)

        # Sum step percents
        steps = (
            self.client.table("project_progress_steps")
            .select("step_percent")
            .eq("project_id", pid)
            .execute()
        )

        total = 0
        for r in (steps.data or []):
            try:
                total += int(r.get("step_percent") or 0)
            except Exception:
                pass

        # clamp
        total = max(0, min(100, total))

        update_data = {
            "completion_percentage": total,
            "updated_at": datetime.now().isoformat(),
        }
        if workflow_state is not None:
            update_data["workflow_state"] = workflow_state
        if current_page is not None:
            update_data["current_page"] = int(current_page)

        self.client.table("projects").update(update_data).eq("project_id", pid).execute()
        return total

    # ========================================================================
    # Section 5 — Existing: project progress + step completion
    # ========================================================================
    def update_project_progress(
        self,
        project_id: str,
        workflow_state: str = None,
        current_page: int = None,
        completion_percentage: int = None
    ):
        """
        Legacy updater. Keep it, but prefer recompute_and_update_project_progress().
        """
        try:
            update_data = {"updated_at": datetime.now().isoformat()}

            if workflow_state:
                update_data["workflow_state"] = workflow_state
            if current_page:
                update_data["current_page"] = current_page
            if completion_percentage is not None:
                update_data["completion_percentage"] = completion_percentage

            self.client.table("projects").update(update_data).eq("project_id", str(project_id)).execute()

        except Exception as e:
            st.warning(f"Progress update warning: {str(e)}")

    def update_step_completion(self, project_id: str, step_key: str, completed: bool = True) -> bool:
        try:
            result = self.client.table("projects").select("step_completion").eq("project_id", str(project_id)).execute()
            if not result.data:
                return False

            step_completion = result.data[0].get("step_completion") or {}
            step_completion[str(step_key)] = bool(completed)

            self.client.table("projects").update(
                {"step_completion": step_completion, "updated_at": datetime.now().isoformat()}
            ).eq("project_id", str(project_id)).execute()

            return True

        except Exception as e:
            st.error(f"Error updating step completion: {str(e)}")
            return False

    def get_step_completion(self, project_id: str) -> Dict[str, bool]:
        try:
            result = self.client.table("projects").select("step_completion").eq("project_id", str(project_id)).execute()
            if result.data and result.data[0].get("step_completion"):
                return result.data[0]["step_completion"]
            return {}
        except Exception as e:
            st.error(f"Error fetching step completion: {str(e)}")
            return {}

    # ========================================================================
    # Section 6 — Keep the rest of your class as-is
    # (User management, projects, parameters, health report, etc.)
    # ========================================================================


@st.cache_resource
def get_db_manager() -> SupabaseManager:
    return SupabaseManager()
