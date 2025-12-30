"""
ExplainFutures - Supabase Database Manager
- Time-series storage + project progress helpers (step-based recompute)
⚠️ WARNING: plaintext passwords - NOT RECOMMENDED for production!
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pandas as pd
import numpy as np


class SupabaseManager:
    """Supabase manager with time-series data storage capabilities"""

    def __init__(self):
        """Initialize Supabase client from Streamlit secrets"""
        try:
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            self.client = create_client(self.url, self.key)

            # Test connection quickly
            self.client.table("projects").select("project_id").limit(1).execute()

        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            raise

    # ------------------------------------------------------------------------
    # BASIC LOOKUPS (USER / PROJECT)
    # ------------------------------------------------------------------------
    def get_project_by_id(self, project_id: str):
        """Fetch minimal project fields used by the UI (sidebar/home)."""
        try:
            res = self.client.table("projects").select(
                "project_id, project_name, project_code, "
                "workflow_state, current_page, completion_percentage"
            ).eq("project_id", project_id).limit(1).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_project_by_id failed: {e}")
            return None

    def get_user_by_id(self, user_id: str):
        """Fetch minimal user fields used by the UI."""
        try:
            res = self.client.table("users").select(
                "user_id, username, email, full_name, subscription_tier"
            ).eq("user_id", user_id).limit(1).execute()
            return res.data[0] if res.data else None
        except Exception as e:
            st.warning(f"get_user_by_id failed: {e}")
            return None

    def convert_timestamps_to_serializable(self, obj):
        """
        Recursively convert pandas Timestamps and datetime objects to ISO format strings
        for JSON serialization
        """
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self.convert_timestamps_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_timestamps_to_serializable(item) for item in obj]
        else:
            return obj

    # ========================================================================
    # TIME-SERIES DATA MANAGEMENT
    # ========================================================================

    def save_timeseries_data(
        self,
        project_id: str,
        df_long: pd.DataFrame,
        data_source: str = "original",
        batch_size: int = 1000,
    ) -> bool:
        """
        Save time-series data to database in batches.

        Expected df_long columns:
          - timestamp OR time
          - variable
          - value
        """
        try:
            # Delete existing for same project+source
            self.client.table("timeseries_data").delete().eq("project_id", str(project_id)).eq(
                "data_source", data_source
            ).execute()

            records = []

            time_col = "timestamp" if "timestamp" in df_long.columns else "time"

            for _, row in df_long.iterrows():
                ts = row[time_col]
                if isinstance(ts, pd.Timestamp):
                    ts_str = ts.isoformat()
                else:
                    ts_str = pd.Timestamp(ts).isoformat()

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

            total = len(records)
            inserted = 0

            for i in range(0, total, batch_size):
                batch = records[i : i + batch_size]
                self.client.table("timeseries_data").insert(batch).execute()
                inserted += len(batch)

            return True

        except Exception as e:
            st.error(f"❌ Error saving time-series data: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
            return False

    def load_timeseries_data(
        self, project_id: str, data_source: str = "original", variables: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load time-series data (long format) from database."""
        try:
            query = (
                self.client.table("timeseries_data")
                .select("timestamp, variable, value")
                .eq("project_id", str(project_id))
                .eq("data_source", data_source)
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

    def get_timeseries_summary(self, project_id: str, data_source: str = "original") -> Dict[str, Any]:
        """Get summary statistics for stored time-series data."""
        try:
            count_result = (
                self.client.table("timeseries_data")
                .select("data_id", count="exact")
                .eq("project_id", str(project_id))
                .eq("data_source", data_source)
                .execute()
            )

            total_records = count_result.count if hasattr(count_result, "count") else 0

            vars_result = (
                self.client.table("timeseries_data")
                .select("variable")
                .eq("project_id", str(project_id))
                .eq("data_source", data_source)
                .execute()
            )

            variables = list(set([r["variable"] for r in vars_result.data])) if vars_result.data else []

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
    # PROJECT PROGRESS (STEP-BASED)
    # ========================================================================

    def upsert_progress_step(self, project_id: str, step_key: str, step_value: int) -> bool:
        """
        Store a single step contribution as a row in progress_steps.
        Assumes table:
          progress_steps(project_id uuid/text, step_key text, step_value int, updated_at timestamp)
        """
        try:
            payload = {
                "project_id": str(project_id),
                "step_key": str(step_key),
                "step_value": int(step_value),
                "updated_at": datetime.now().isoformat(),
            }
            # Upsert on (project_id, step_key)
            self.client.table("progress_steps").upsert(payload, on_conflict="project_id,step_key").execute()
            return True
        except Exception as e:
            st.warning(f"upsert_progress_step failed: {e}")
            return False

    def recompute_and_update_project_progress(
        self,
        project_id: str,
        workflow_state: str = None,
        current_page: int = None,
    ) -> bool:
        """
        Sum all progress_steps.step_value for the project, then update projects.completion_percentage.
        """
        try:
            res = (
                self.client.table("progress_steps")
                .select("step_value")
                .eq("project_id", str(project_id))
                .execute()
            )
            total = 0
            if res.data:
                total = sum(int(r.get("step_value") or 0) for r in res.data)

            update_data = {"completion_percentage": int(total), "updated_at": datetime.now().isoformat()}
            if workflow_state:
                update_data["workflow_state"] = workflow_state
            if current_page is not None:
                update_data["current_page"] = int(current_page)

            self.client.table("projects").update(update_data).eq("project_id", str(project_id)).execute()
            return True

        except Exception as e:
            st.warning(f"recompute_and_update_project_progress failed: {e}")
            return False

    # ========================================================================
    # STEP COMPLETION (JSON FIELD ON PROJECTS)
    # ========================================================================

    def update_step_completion(self, project_id: str, step_key: str, completed: bool = True) -> bool:
        """Update completion status of a specific workflow step (projects.step_completion JSON)."""
        try:
            result = self.client.table("projects").select("step_completion").eq("project_id", str(project_id)).execute()

            if not result.data:
                return False

            step_completion = result.data[0].get("step_completion", {})
            if step_completion is None:
                step_completion = {}

            step_completion[step_key] = bool(completed)

            self.client.table("projects").update(
                {"step_completion": step_completion, "updated_at": datetime.now().isoformat()}
            ).eq("project_id", str(project_id)).execute()

            return True

        except Exception as e:
            st.error(f"Error updating step completion: {str(e)}")
            return False


@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Get cached database manager instance"""
    return SupabaseManager()
