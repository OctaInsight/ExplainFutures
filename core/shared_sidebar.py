"""
Shared Sidebar Component
This module provides the sidebar used across all pages.
NOTE: Must be called after st.set_page_config()!
"""

import streamlit as st
from datetime import datetime
import time
from pathlib import Path

from core.database.supabase_manager import get_db_manager

# Sidebar step definitions (visual workflow)
WORKFLOW_STEPS = [
    ("Upload", 2),
    ("Preprocess", 3),
    ("Explore", 4),
    ("Relationships", 5),
    ("Structure", 6),
    ("Train", 7),
    ("Evaluate", 8),
    ("Forecast", 9),
    ("Scenario NLP", 10),
    ("Scenario Space", 11),
    ("Mapping", 12),
    ("Imputation", 13),
    ("Trajectory", 14),
    ("Feedback", 15),
]


def render_app_sidebar():
    """Render sidebar UI across pages."""

    st.sidebar.image(str(Path("assets/logo_small.png")), use_container_width=True)
    st.sidebar.markdown("### ExplainFutures")

    # Try to initialize DB manager
    try:
        db = get_db_manager()
        DB_AVAILABLE = True
    except Exception:
        db = None
        DB_AVAILABLE = False

    # -------------------------------
    # User Information
    # -------------------------------
    user = st.session_state.get("user")

    if user:
        full_name = user.get("full_name") or user.get("username") or "User"
        st.sidebar.markdown(f"**Logged in:** {full_name}")
        st.sidebar.caption(f"Tier: {user.get('subscription_tier', 'free')}")
    else:
        st.sidebar.warning("⚠️ Not logged in")

    st.sidebar.markdown("---")

    # -------------------------------
    # Workflow Steps UI
    # -------------------------------
    current_page = st.session_state.get("current_page", None)
    workflow_state = st.session_state.get("workflow_state", "setup")

    st.sidebar.markdown("### Workflow")

    # Draw workflow steps with a simple indicator
    for label, page_num in WORKFLOW_STEPS:
        is_current = (current_page == page_num)
        prefix = "➡️" if is_current else "•"
        st.sidebar.write(f"{prefix} {label}")

    st.sidebar.caption(f"Current state: {str(workflow_state).title()}")

    st.sidebar.markdown("---")

    # -------------------------------
    # Logout button
    # -------------------------------
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        # Keep your existing logout behavior intact
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Logged out")
        time.sleep(1)
        st.switch_page("App.py")

    st.sidebar.markdown("---")

    # Project Information from session state and database
    # -------------------------------
    current_project_id = st.session_state.get("current_project_id")

    if DB_AVAILABLE and current_project_id:
        project = db.get_project_by_id(current_project_id)

        if project:
            progress = project.get("completion_percentage", 0) or 0

            st.sidebar.info(f"**{project['project_name']}**")
            st.sidebar.caption(f"Code: {project.get('project_code', 'N/A')}")
            st.sidebar.progress(progress / 100, text=f"Progress: {progress}%")

            st.sidebar.caption(
                f"Stage: {project.get('workflow_state', 'setup').title()}"
            )

            # Keep UI/session state consistent
            st.session_state["completion_percentage"] = progress
        else:
            st.sidebar.warning("⚠️ Project not found in database")
