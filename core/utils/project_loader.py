"""
Project Loader Utilities
Loads a project's related data from database into Streamlit session_state.
This file should not depend on core/utils.py (module).
It uses helpers via core/utils/__init__.py exports (or direct local import).

Update (for Page 3 alignment):
- Added lazy "ensure_*" loaders so pages can load only what they need
  (especially timeseries_data) without forcing full project reload.
- No destructive operations; only reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import streamlit as st


# ---------------------
# Section 0: Internal Helpers
# ---------------------

def _set_current_project_id(project_id: str) -> None:
    """
    Centralized setter to avoid duplication across entry points.
    Keeps behavior consistent and reduces risk of conflicting state updates.
    """
    st.session_state.current_project_id = project_id


def _default_timeseries_store_key(data_source: str) -> str:
    """
    Choose a reasonable session_state key for storing a loaded timeseries dataframe.
    This is intentionally conservative to avoid collisions with page-specific logic.
    """
    ds = (data_source or "").lower().strip()
    if ds in ("raw", "original"):
        return "df_long"
    if ds in ("cleaned",):
        # Avoid overwriting Page 3's working df_clean; store DB-cleaned separately by default.
        return "df_cleaned_db"
    return f"df_{ds}" if ds else "df_timeseries"


# ---------------------
# Section 1: Core Loader
# ---------------------

def load_complete_project_data(project_id: str, db) -> Dict[str, Any]:
    """
    Load all project-related data into session state.

    What it loads (current implementation):
    1) Project record
    2) Parameters
    3) Health report (if any)
    4) Uploaded files history
    5) Sets flags indicating whether data exists

    NOTE:
    - This function intentionally does NOT load timeseries_data by default.
      Timeseries can be large and is best loaded per-page using ensure_timeseries_loaded().

    Returns:
        dict with:
          - success: bool
          - data_loaded: bool
          - message: str
          - errors: list[str]
    """
    result: Dict[str, Any] = {
        "success": True,
        "data_loaded": False,
        "message": "",
        "errors": [],
    }

    try:
        # 1) Load Project Info
        st.write("📌 Loading project information...")
        project = db.client.table("projects").select("*").eq("project_id", project_id).execute()
        if project and getattr(project, "data", None):
            st.session_state.current_project = project.data[0]
        else:
            st.session_state.current_project = None

        # 2) Load Parameters
        st.write("📊 Loading project parameters...")
        parameters = db.get_project_parameters(project_id)
        st.session_state.project_parameters = parameters if parameters else []

        # 3) Load Health Report (latest)
        st.write("🩺 Loading data health report...")
        try:
            health_report = db.get_health_report(project_id)
            st.session_state.health_report = health_report
        except Exception:
            st.session_state.health_report = None

        # 4) Load Uploaded Files Info
        st.write("📁 Loading file history...")
        uploaded_files = db.get_uploaded_files(project_id)
        if uploaded_files:
            st.session_state.uploaded_files_history = uploaded_files
        else:
            st.session_state.uploaded_files_history = []

        # 5) Set flags
        if parameters:
            st.session_state.has_processed_data = True
            st.session_state.data_loaded = True
            result["data_loaded"] = True
            result["message"] = f"✅ Loaded {len(parameters)} parameters"
        else:
            st.session_state.has_processed_data = False
            st.session_state.data_loaded = False
            result["message"] = "ℹ️ No data found - please upload data"

        return result

    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
        result["message"] = f"❌ Error loading project data: {str(e)}"
        return result


# ---------------------
# Section 2: Open Project Entry Point
# ---------------------

def load_project_on_open(project_id: str, db) -> bool:
    """
    Main function to call when opening a project.
    Shows loading progress and populates session_state.

    Returns:
        True if successful, False otherwise
    """
    _set_current_project_id(project_id)

    with st.spinner("Loading project data..."):
        result = load_complete_project_data(project_id, db)

    if result["success"]:
        if result["data_loaded"]:
            st.success(result["message"])
        else:
            st.info(result["message"])
        return True

    st.error(result["message"])
    for error in result["errors"]:
        st.error(f"  • {error}")
    return False


# ---------------------
# Section 3: Guard / Ensure Loaded on Each Page
# ---------------------

def ensure_project_data_loaded(project_id: str, db) -> bool:
    """
    Ensure project data is loaded.
    Call this at the start of each page.

    Returns:
        True if data is available, False otherwise
    """
    # Ensure correct project is selected
    if st.session_state.get("current_project_id") != project_id:
        _set_current_project_id(project_id)
        load_complete_project_data(project_id, db)

    # If not loaded, try silent load
    if (not st.session_state.get("project_parameters")) and (not st.session_state.get("data_loaded")):
        load_complete_project_data(project_id, db)

    return bool(st.session_state.get("data_loaded", False))


# ---------------------
# Section 4: Lazy Page-Level Loaders (recommended for large objects)
# ---------------------

def ensure_parameters_loaded(project_id: str, db, force_reload: bool = False) -> bool:
    """
    Loads project parameters into st.session_state.project_parameters if missing.

    Returns:
        True if parameters are available (possibly empty list), False if error.
    """
    try:
        if force_reload or ("project_parameters" not in st.session_state) or (st.session_state.get("project_parameters") is None):
            params = db.get_project_parameters(project_id)
            st.session_state.project_parameters = params if params else []
        return True
    except Exception as e:
        st.session_state.project_parameters = st.session_state.get("project_parameters") or []
        st.error(f"Error loading parameters: {str(e)}")
        return False


def ensure_latest_health_report_loaded(project_id: str, db, force_reload: bool = False) -> bool:
    """
    Loads latest health report into st.session_state.health_report if missing.

    Returns:
        True if loaded (or none), False if error.
    """
    try:
        if force_reload or ("health_report" not in st.session_state):
            try:
                st.session_state.health_report = db.get_health_report(project_id)
            except Exception:
                st.session_state.health_report = None
        return True
    except Exception as e:
        st.session_state.health_report = st.session_state.get("health_report")
        st.error(f"Error loading health report: {str(e)}")
        return False


def ensure_timeseries_loaded(
    project_id: str,
    db,
    data_source: str = "raw",
    variables: Optional[Sequence[str]] = None,
    store_key: Optional[str] = None,
    force_reload: bool = False,
) -> bool:
    """
    Page-level loader for timeseries_data.

    Why this exists:
    - Timeseries can be large. Loading it in load_complete_project_data() slows the whole app.
    - Pages (e.g., Page 3) should load ONLY what they need.

    Behavior:
    - Loads df = db.load_timeseries_data(project_id=..., data_source=..., variables=...)
    - Stores it into st.session_state[store_key]
      (default: 'df_long' for raw/original; 'df_cleaned_db' for cleaned)
    - Does NOT merge/transform; this avoids conflicts with Page 3's in-page logic.

    Returns:
        True if dataframe was loaded and is non-empty, False otherwise.
    """
    if st.session_state.get("current_project_id") != project_id:
        _set_current_project_id(project_id)

    key = store_key or _default_timeseries_store_key(data_source)

    if (not force_reload) and (key in st.session_state) and (st.session_state.get(key) is not None):
        df_existing = st.session_state.get(key)
        try:
            return bool(getattr(df_existing, "__len__", lambda: 0)() > 0)
        except Exception:
            return True

    try:
        df = db.load_timeseries_data(
            project_id=project_id,
            data_source=data_source,
            variables=list(variables) if variables else None
        )
        st.session_state[key] = df
        return df is not None and len(df) > 0
    except Exception as e:
        st.error(f"Error loading timeseries_data ({data_source}): {str(e)}")
        st.session_state[key] = None
        return False
