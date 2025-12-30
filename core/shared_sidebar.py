"""
Shared Sidebar Component
This module provides the sidebar that appears on all pages

IMPORTANT: Only call render_app_sidebar() AFTER st.set_page_config()!
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import base64
import textwrap


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
    Updated workflow indicator
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

    def get_dot_style(key):
        is_done = bool(st.session_state.get(key, False))
        fill = done_fill if is_done else pending_fill
        border = done_border if is_done else pending_border
        return fill, border

    def get_line_color(key):
        is_done = bool(st.session_state.get(key, False))
        return done_fill if is_done else pending_fill

    # Build left column HTML
    left_html_parts = []
    for i, (key, tooltip) in enumerate(left_steps):
        fill, border = get_dot_style(key)
        if i > 0:
            line_color = get_line_color(left_steps[i - 1][0])
            left_html_parts.append(
                f'<div style="width: 1px; height: 6px; background: {line_color}; margin: 0 auto;"></div>'
            )
        left_html_parts.append(
            f'<div style="width: 6px; height: 6px; border-radius: 50%; '
            f'background: {fill}; border: 1.5px solid {border}; margin: 0 auto;" title="{tooltip}"></div>'
        )
    left_line_color = get_line_color(left_steps[-1][0])
    left_html_parts.append(
        f'<div style="width: 1px; height: 6px; background: {left_line_color}; margin: 0 auto;"></div>'
    )
    left_html = "".join(left_html_parts)

    # Build right column HTML
    right_html_parts = []
    fill1, border1 = get_dot_style(right_steps[0][0])
    right_html_parts.append(
        f'<div style="width: 6px; height: 6px; border-radius: 50%; '
        f'background: {fill1}; border: 1.5px solid {border1}; margin: 0 auto;" title="{right_steps[0][1]}"></div>'
    )
    line_color_right = get_line_color(right_steps[0][0])
    right_html_parts.append(
        f'<div style="width: 1px; height: 6px; background: {line_color_right}; margin: 0 auto;"></div>'
    )
    fill2, border2 = get_dot_style(right_steps[1][0])
    right_html_parts.append(
        f'<div style="width: 6px; height: 6px; border-radius: 50%; '
        f'background: {fill2}; border: 1.5px solid {border2}; margin: 0 auto;" title="{right_steps[1][1]}"></div>'
    )
    line_color_right2 = get_line_color(right_steps[1][0])
    remaining_height = 96 - 18
    right_html_parts.append(
        f'<div style="width: 1px; height: {remaining_height}px; background: {line_color_right2}; margin: 0 auto;"></div>'
    )
    right_html = "".join(right_html_parts)

    # Build center column HTML
    center_html_parts = []
    for i, (key, tooltip) in enumerate(center_steps):
        fill, border = get_dot_style(key)
        if i > 0:
            line_color = get_line_color(center_steps[i - 1][0])
            center_html_parts.append(
                f'<div style="width: 1px; height: 6px; background: {line_color}; margin: 0 auto;"></div>'
            )
        center_html_parts.append(
            f'<div style="width: 6px; height: 6px; border-radius: 50%; '
            f'background: {fill}; border: 1.5px solid {border}; margin: 0 auto;" title="{tooltip}"></div>'
        )
    center_html = "".join(center_html_parts)

    # Complete HTML structure
    html = f"""
    <style>
        .workflow-container {{
            text-align: center;
            padding: 0.3rem 0;
            margin: 0;
        }}
        .workflow-title {{
            font-size: 0.65rem;
            font-weight: 600;
            background: linear-gradient(135deg, #0e6537 0%, #21c55d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            letter-spacing: 0.5px;
        }}
        .workflow-columns {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 0.2rem 0;
        }}
        .workflow-column {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .workflow-legend {{
            font-size: 0.55rem;
            color: #6b7280;
            margin-top: 0.15rem;
        }}
    </style>
    
    <div class="workflow-container">
        <div class="workflow-title">
            WORKFLOW
        </div>

        <div class="workflow-columns">
            <div class="workflow-column">
                {left_html}
            </div>
            <div class="workflow-column">
                {right_html}
            </div>
        </div>

        <div style="display: flex; justify-content: center; margin-top: 0;">
            <div class="workflow-column">
                {center_html}
            </div>
        </div>

        <div class="workflow-legend">
            <span style="color: {done_fill};">●</span> Done
            <span style="color: {pending_fill}; margin-left: 0.3rem;">●</span> Pending
        </div>
    </div>
    """
    
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
