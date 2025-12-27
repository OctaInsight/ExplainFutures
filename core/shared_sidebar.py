"""
Shared Sidebar Component
This module provides the sidebar that appears on all pages

IMPORTANT: Only call render_app_sidebar() AFTER st.set_page_config()!
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import base64

# Helper function to get logo
def get_logo_base64():
    """Get logo as base64 string for embedding"""
    logo_path = Path(__file__).parent / "assets" / "logo_small.png"
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        # Try alternative path
        try:
            logo_path = Path("assets/logo_small.png")
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return None


def render_workflow_flowchart():
    """
    Ultra-minimal workflow indicator - tiny dots for ALL steps
    Left path: 8 steps (historical data)
    Right path: 2 steps (scenarios)
    Final: 1 step (trajectory)
    All with 6px dots and 1px lines
    """
    
    # Define all steps with session state keys
    left_steps = [
        ("data_loaded", "Upload & Data Diagnostics"),
        ("data_cleaned", "Data Cleaning & Preprocessing"),
        ("data_explored", "Exploration & Visualizations"),
        ("relations_analyzed", "Variable Relationships"),
        ("dim_reduction_done", "Dimensionality Reduction"),
        ("models_trained", "Time-Based Models & ML Training"),
        ("models_evaluated", "Model Evaluation & Selection"),
        ("projections_done", "Future Projections"),
    ]
    
    right_steps = [
        ("scenarios_analyzed", "Scenario Analysis (NLP)"),
        ("scenario_matrix_done", "Scenario Matrix"),
    ]
    
    final_step = ("trajectory_space_done", "Trajectory-Scenario Space")
    
    # Colors
    done_fill = "#21c55d"
    done_border = "#0e6537"
    pending_fill = "#d1d5db"
    pending_border = "#6b7280"
    
    def get_dot_html(key, tooltip):
        """Generate HTML for a single dot"""
        is_done = st.session_state.get(key, False)
        fill = done_fill if is_done else pending_fill
        border = done_border if is_done else pending_border
        
        return f"""
        <div style='width: 6px; height: 6px; border-radius: 50%; 
                   background: {fill};
                   border: 1.5px solid {border};'
             title='{tooltip}'></div>
        """
    
    def get_line_html(prev_key):
        """Generate HTML for a connecting line"""
        is_done = st.session_state.get(prev_key, False)
        color = done_fill if is_done else pending_fill
        
        return f"""
        <div style='width: 1px; height: 6px; background: {color};'></div>
        """
    
    # Build left column HTML
    left_html = ""
    for i, (key, tooltip) in enumerate(left_steps):
        if i > 0:
            # Add connecting line (color from previous step)
            left_html += get_line_html(left_steps[i-1][0])
        left_html += get_dot_html(key, tooltip)
    
    # Add line from last left step to final
    left_html += get_line_html(left_steps[-1][0])
    
    # Build right column HTML
    right_html = ""
    for i, (key, tooltip) in enumerate(right_steps):
        if i > 0:
            # Add connecting line (color from previous step)
            right_html += get_line_html(right_steps[i-1][0])
        right_html += get_dot_html(key, tooltip)
    
    # Add line from last right step to final
    right_html += get_line_html(right_steps[-1][0])
    
    # Final step dot
    final_html = get_dot_html(final_step[0], final_step[1])
    
    # Assemble complete flowchart
    html = f"""
    <div style='text-align: center; padding: 0.3rem 0; margin: 0;'>
        <div style='font-size: 0.65rem; font-weight: 600; 
                    background: linear-gradient(135deg, #0e6537 0%, #21c55d 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin-bottom: 0.25rem; letter-spacing: 0.5px;'>
            WORKFLOW
        </div>
        <div style='display: flex; justify-content: center; gap: 1rem; margin: 0.2rem 0;'>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 0;'>
                {left_html}
            </div>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 0;'>
                {right_html}
            </div>
        </div>
        <div style='display: flex; justify-content: center; margin-top: 0;'>
            {final_html}
        </div>
        <div style='font-size: 0.55rem; color: #6b7280; margin-top: 0.15rem;'>
            <span style='color: {done_fill};'>●</span> Done 
            <span style='color: {pending_fill}; margin-left: 0.3rem;'>●</span> Pending
        </div>
    </div>
    """
    
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_app_sidebar():
    """
    Render the application sidebar with logo, status, and info
    This function should be called at the top of every page AFTER st.set_page_config()
    """
    
    # === WORKFLOW PROGRESS FLOWCHART (FIRST - ABOVE LOGO) ===
    render_workflow_flowchart()
    
    st.sidebar.markdown("---")
    
    # Try to get logo as base64
    logo_base64 = get_logo_base64()
    
    # Logo and Title
    if logo_base64:
        # Method 1: Base64 embedded image (most reliable)
        st.sidebar.markdown(f"""
            <div style='text-align: center; padding: 1.5rem 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
                <img src='data:image/png;base64,{logo_base64}' style='width: 80px; height: 80px; margin-bottom: 0.5rem;' alt='Logo'/>
                <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Data-Driven Future Exploration</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: Use emoji if logo file not found
        st.sidebar.markdown("""
            <div style='text-align: center; padding: 1.5rem 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔮</div>
                <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Data-Driven Future Exploration</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Login/Logout section
    if st.session_state.get("logged_in", False):
        st.sidebar.success(f"✅ Logged in: **{st.session_state.get('username', 'User')}**")
        
        # Show login duration
        if st.session_state.get("login_time"):
            duration = datetime.now() - st.session_state.login_time
            minutes = int(duration.total_seconds() / 60)
            st.sidebar.caption(f"Session: {minutes} min")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.login_time = None
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Not logged in")
        st.sidebar.caption("Please login via home page")
    
    st.sidebar.markdown("---")
    
    # Main Menu Section
    st.sidebar.markdown("""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h3 style='margin: 0 0 0.5rem 0; font-size: 1rem; color: #31333F;'>📊 Project Status</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Project Information
    if st.session_state.get("project_name"):
        st.sidebar.info(f"**Project:** {st.session_state.project_name}")
    else:
        st.sidebar.caption("📁 No project loaded")
    
    # Data Status
    if st.session_state.get("data_loaded", False):
        st.sidebar.success("✅ Data loaded")
        
        # Show data metrics
        if st.session_state.get("df_long") is not None:
            df = st.session_state.df_long
            n_vars = len(df['variable'].unique())
            n_points = len(df)
            
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Variables", n_vars)
            col2.metric("Points", n_points)
            
            if st.session_state.get("uploaded_file_name"):
                st.sidebar.caption(f"📄 {st.session_state.uploaded_file_name}")
    else:
        st.sidebar.caption("📥 No data loaded")
    
    # Database Status
    st.sidebar.markdown("")  # Small spacer
    if st.session_state.get("database_connected", False):
        st.sidebar.success("🗄️ Database: Connected")
    else:
        st.sidebar.caption("🗄️ Database: Phase 4 (in-memory)")
    
    st.sidebar.markdown("---")
    
    # Navigation Help (collapsible)
    with st.sidebar.expander("ℹ️ Quick Guide"):
        st.markdown("""
        **Workflow:**
        1. 📁 Upload data
        2. 🧹 Clean & preprocess
        3. 📊 Visualize
        4. 🔬 Analyze
        5. 🤖 Model
        6. 🔮 Project
        """)
    
    st.sidebar.markdown("---")
    
    # About sections in sidebar
    with st.sidebar.expander("📖 About ExplainFutures"):
        st.markdown("""
        **Version:** 1.0.0-phase1
        
        A modular platform for:
        - Time-series analysis
        - Interactive visualizations
        - Predictive modeling
        - Future exploration
        - Scenario planning
        """)
    
    with st.sidebar.expander("🏢 About Octa Insight"):
        st.markdown("""
        **Octa Insight** specializes in data-driven decision support systems.
        
        - Analytics platforms
        - AI/ML solutions
        - Consulting services
        
        📧 info@octainsight.com
        """)
    
    # Footer
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    try:
        from core.config import get_config
        config_data = get_config()
        version = config_data.get('version', '1.0.0')
    except:
        version = '1.0.0'
    
    st.sidebar.caption(f"ExplainFutures v{version}")
    st.sidebar.caption("© 2024 Octa Insight")
