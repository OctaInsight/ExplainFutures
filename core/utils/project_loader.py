# core/utils/project_loader.py

"""
Project Loader Utilities
Loads a project's related data from database into Streamlit session_state.
This file should not depend on core/utils.py (module).
It uses helpers via core/utils/__init__.py exports (or direct local import).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


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
    st.session_state.current_project_id = project_id

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
        st.session_state.current_project_id = project_id
        load_complete_project_data(project_id, db)

    # If not loaded, try silent load
    if (not st.session_state.get("project_parameters")) and (not st.session_state.get("data_loaded")):
        load_complete_project_data(project_id, db)

    return bool(st.session_state.get("data_loaded", False))
