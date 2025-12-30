"""
Shared Sidebar Component
This module provides the sidebar that appears on all pages

IMPORTANT: Only call render_app_sidebar() AFTER st.set_page_config()!
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import base64


# =============================================================================
# Helper: logo
# =============================================================================
def get_logo_base64():
    """Get logo as base64 string for embedding"""
    logo_path = Path(__file__).parent / "assets" / "logo_small.png"
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        try:
            logo_path = Path("assets/logo_small.png")
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None


# =============================================================================
# Workflow chart
# =============================================================================
def render_workflow_flowchart():
    """
    Updated workflow indicator with visual flowchart using tables
    NOTE:
    - Dot completion is driven by st.session_state flags.
    - We will set those flags from DB project_progress_steps.step_key
      in render_app_sidebar() before calling this function.
    """

    # Define steps (these KEYS must match what you write to project_progress_steps.step_key)
    left_steps = [
        ("data_loaded", "Data Import & Diagnostics"),
        ("data_cleaned", "Preprocessing"),
        ("data_explored", "Exploration & Visualization"),
        ("relations_analyzed", "Variable Relationships"),
        ("dim_reduction_done", "System Structure (Dimensionality Reduction)"),
        ("models_trained", "Time Modeling & Training"),
        ("models_evaluated", "Model Evaluation & Selection"),
        ("projections_done", "Forecasting (Projections)"),
    ]

    right_steps = [
        ("scenarios_analyzed", "Scenario Analysis (NLP)"),
        ("scenario_space_done", "Scenario Space"),
    ]

    center_steps = [
        ("parameter_mapping_done", "Parameter Mapping"),
        ("parameter_completion_done", "Parameter Completion & Imputation"),
        ("trajectory_space_done", "Trajectory vs Scenario Space"),
    ]

    # Colors
    done_fill = "#21c55d"
    done_border = "#0e6537"
    pending_fill = "#d1d5db"
    pending_border = "#6b7280"

    def is_done(key):
        return bool(st.session_state.get(key, False))

    def dot_cell(key, tooltip):
        done = is_done(key)
        fill = done_fill if done else pending_fill
        border = done_border if done else pending_border
        return f'<td style="padding: 0; text-align: center;"><div style="width: 10px; height: 10px; border-radius: 50%; background: {fill}; border: 2px solid {border}; margin: 0 auto;" title="{tooltip}"></div></td>'

    def line_cell(key, height="8px"):
        done = is_done(key)
        color = done_fill if done else pending_fill
        return f'<td style="padding: 0; text-align: center;"><div style="width: 2px; height: {height}; background: {color}; margin: 0 auto;"></div></td>'

    # Build left column rows
    left_rows = []
    for i, (key, tooltip) in enumerate(left_steps):
        if i > 0:
            left_rows.append(f'<tr>{line_cell(left_steps[i-1][0])}</tr>')
        left_rows.append(f'<tr>{dot_cell(key, tooltip)}</tr>')
    left_rows.append(f'<tr>{line_cell(left_steps[-1][0])}</tr>')
    left_table = "<table style='margin: 0; padding: 0; border-collapse: collapse;'>" + "".join(left_rows) + "</table>"

    # Build right column rows
    right_rows = []
    right_rows.append(f'<tr>{dot_cell(right_steps[0][0], right_steps[0][1])}</tr>')
    right_rows.append(f'<tr>{line_cell(right_steps[0][0])}</tr>')
    right_rows.append(f'<tr>{dot_cell(right_steps[1][0], right_steps[1][1])}</tr>')
    right_rows.append(f'<tr>{line_cell(right_steps[1][0], "90px")}</tr>')
    right_table = "<table style='margin: 0; padding: 0; border-collapse: collapse;'>" + "".join(right_rows) + "</table>"

    # Build center column rows
    center_rows = []
    for i, (key, tooltip) in enumerate(center_steps):
        if i > 0:
            center_rows.append(f'<tr>{line_cell(center_steps[i-1][0])}</tr>')
        center_rows.append(f'<tr>{dot_cell(key, tooltip)}</tr>')
    center_table = "<table style='margin: 0; padding: 0; border-collapse: collapse;'>" + "".join(center_rows) + "</table>"

    # Main HTML using table layout
    html = f'''
<div style="text-align: center; padding: 8px; margin: 0; background-color: #f8f9fa; border-radius: 8px;">
    <div style="font-size: 11px; font-weight: 700; color: #0e6537; margin-bottom: 8px; letter-spacing: 1px;">
        WORKFLOW
    </div>
    
    <table style="margin: 8px auto; border-collapse: collapse;">
        <tr>
            <td style="padding: 0 10px; vertical-align: top;">
                {left_table}
            </td>
            <td style="padding: 0 10px; vertical-align: top;">
                {right_table}
            </td>
        </tr>
    </table>
    
    <table style="margin: 0 auto; border-collapse: collapse;">
        <tr>
            <td style="padding: 0; vertical-align: top;">
                {center_table}
            </td>
        </tr>
    </table>
    
    <div style="font-size: 10px; color: #6b7280; margin-top: 8px;">
        <span style="color: {done_fill};">●</span> Done &nbsp;&nbsp;
        <span style="color: {pending_fill};">●</span> Pending
    </div>
</div>
'''
    
    st.sidebar.markdown(html, unsafe_allow_html=True)


# =============================================================================
# Main renderer
# =============================================================================
def render_app_sidebar():
    """
    Render the application sidebar with logo, status, and info
    Call this at the top of every page AFTER st.set_page_config()
    """

    # -------------------------------------------------------------
    # A) Pull workflow done-keys from DB and set session_state flags
    # -------------------------------------------------------------
    pid = st.session_state.get("current_project_id")
    if pid:
        try:
            from core.database.supabase_manager import get_db_manager
            db = get_db_manager()

            res = (
                db.client.table("project_progress_steps")
                .select("step_key")
                .eq("project_id", pid)
                .execute()
            )

            done_keys = {r.get("step_key") for r in (res.data or []) if r.get("step_key")}
            for k in done_keys:
                st.session_state[str(k)] = True

        except Exception:
            # Do not break the app if sidebar DB read fails
            pass

    # -------------------------------------------------------------
    # B) Workflow chart (top)
    # -------------------------------------------------------------
    render_workflow_flowchart()
    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # C) Header / Logo
    # -------------------------------------------------------------
    logo_base64 = get_logo_base64()
    if logo_base64:
        st.sidebar.markdown(
            f"""
            <div style='text-align: center; padding: 1.5rem 0 1rem 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; margin-bottom: 1rem;'>
                <img src='data:image/png;base64,{logo_base64}'
                     style='width: 80px; height: 80px; margin-bottom: 0.5rem;' alt='Logo'/>
                <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>
                    Data-Driven Future Exploration
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div style='text-align: center; padding: 1.5rem 0 1rem 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔮</div>
                <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>
                    Data-Driven Future Exploration
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # D) Auth section
    # -------------------------------------------------------------
    if st.session_state.get("authenticated", False):
        username = st.session_state.get("username", "User")
        full_name = st.session_state.get("full_name", username)
        is_demo = st.session_state.get("is_demo", False)

        if is_demo:
            st.sidebar.warning(f"🎭 Demo: **{username}**")
        else:
            st.sidebar.success(f"✅ **{full_name}**")
            st.sidebar.caption(f"@{username}")

        tier = st.session_state.get("subscription_tier", "free")
        st.sidebar.caption(f"Plan: {str(tier).upper()}")

        if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            try:
                try:
                    from core.database.supabase_manager import get_db_manager
                    db = get_db_manager()
                    if st.session_state.get("is_demo") and st.session_state.get("demo_session_id"):
                        try:
                            db.end_demo_session(st.session_state.demo_session_id)
                        except Exception:
                            pass
                except Exception:
                    pass

                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                st.switch_page("App.py")
            except Exception:
                st.rerun()
    else:
        st.sidebar.warning("⚠️ Not logged in")
        st.sidebar.caption("Please login via home page")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # E) Database status
    # -------------------------------------------------------------
    try:
        from core.database.supabase_manager import get_db_manager
        _ = get_db_manager()
        st.sidebar.success("🗄️ Database: Connected")
    except Exception:
        st.sidebar.error("🗄️ Database: Not Connected")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # F) Project Status block
    # -------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h3 style='margin: 0 0 0.5rem 0; font-size: 1rem; color: #31333F;'>📊 Project Status</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_project_id = st.session_state.get("current_project_id")

    if current_project_id:
        try:
            from core.database.supabase_manager import get_db_manager
            db = get_db_manager()

            project_result = (
                db.client.table("projects")
                .select("project_name, project_code, completion_percentage, workflow_state")
                .eq("project_id", current_project_id)
                .limit(1)
                .execute()
            )

            if project_result.data:
                project = project_result.data[0]

                project_name = project.get("project_name", "Unknown")
                project_code = project.get("project_code", "N/A")

                # ✅ Progress bar linked to projects.completion_percentage
                progress = int(project.get("completion_percentage") or 0)
                progress = max(0, min(100, progress))

                st.sidebar.info(f"**{project_name}**")
                st.sidebar.caption(f"Code: {project_code}")
                st.sidebar.progress(progress / 100, text=f"Progress: {progress}%")

                workflow_state = project.get("workflow_state", "setup")
                st.sidebar.caption(f"Stage: {str(workflow_state).title()}")

                # Keep UI state consistent
                st.session_state["completion_percentage"] = progress
            else:
                st.sidebar.caption("📁 Project not found")

        except Exception:
            # Fallback to session state (does not crash sidebar)
            project_name = st.session_state.get("project_name", "Unnamed Project")
            st.sidebar.info(f"**{project_name}**")
    else:
        st.sidebar.caption("📁 No project loaded")

    # -------------------------------------------------------------
    # G) Data status
    # -------------------------------------------------------------
    if st.session_state.get("data_loaded", False):
        st.sidebar.success("✅ Data loaded")

        if st.session_state.get("df_long") is not None:
            df = st.session_state.df_long
            try:
                n_vars = len(df["variable"].unique())
                n_points = len(df)
            except Exception:
                n_vars = 0
                n_points = 0

            col1, col2 = st.sidebar.columns(2)
            col1.metric("Variables", n_vars)
            col2.metric("Points", n_points)

            if st.session_state.get("uploaded_file_name"):
                st.sidebar.caption(f"📄 {st.session_state.uploaded_file_name}")
    else:
        st.sidebar.caption("📥 No data loaded yet")

    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # H) Quick guide
    # -------------------------------------------------------------
    with st.sidebar.expander("ℹ️ Quick Guide"):
        st.markdown(
            """
            **Workflow:**
            1. 📁 Upload data
            2. 🧹 Clean & preprocess
            3. 📊 Visualize & explore
            4. 🔬 Analyze relationships
            5. 📉 Reduce dimensions
            6. 🤖 Train models
            7. ✅ Evaluate models
            8. 🔮 Generate projections
            9. 📝 Analyze scenarios (NLP)
            10. 📊 Create scenario matrix
            11. 🎯 Build trajectory space
            """
        )

    st.sidebar.markdown("---")

    # -------------------------------------------------------------
    # I) About
    # -------------------------------------------------------------
    with st.sidebar.expander("📖 About ExplainFutures"):
        st.markdown(
            """
            **Version:** 1.0.0

            A modular platform for:
            - Time-series analysis
            - Interactive visualizations
            - Predictive modeling
            - Future exploration
            - Scenario planning
            """
        )

    with st.sidebar.expander("🏢 About Octa Insight"):
        st.markdown(
            """
            **Octa Insight** specializes in data-driven decision support systems.

            - Analytics platforms
            - AI/ML solutions
            - Consulting services

            📧 info@octainsight.com
            """
        )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.caption("ExplainFutures v1.0.0")
    st.sidebar.caption("© 2025 Octa Insight")
