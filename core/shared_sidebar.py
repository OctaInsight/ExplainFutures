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
    Render two-column workflow flowchart with Graphviz
    
    Left path: Historical Data Analysis
    Right path: Scenario Planning
    Both converge to: Trajectory-Scenario Space
    
    Shows circles only (no text in chart)
    Hover shows step name via tooltip (title attribute)
    Green = completed, Grey = not completed
    """
    
    # Define workflow steps with session state keys
    # Format: (node_id, display_name, session_key)
    left_path = [
        ("L1", "Upload & Data Diagnostics", "data_loaded"),
        ("L2", "Data Cleaning & Preprocessing", "data_cleaned"),
        ("L3", "Exploration & Visualizations", "data_explored"),
        ("L4", "Variable Relationships", "relations_analyzed"),
        ("L5", "Dimensionality Reduction", "dim_reduction_done"),
        ("L6", "Time-Based Models & ML Training", "models_trained"),
        ("L7", "Model Evaluation & Selection", "models_evaluated"),
        ("L8", "Future Projections", "projections_done"),
    ]
    
    right_path = [
        ("R1", "Scenario Analysis (NLP)", "scenarios_analyzed"),
        ("R2", "Scenario Matrix", "scenario_matrix_done"),
    ]
    
    final_step = ("FINAL", "Trajectory-Scenario Space", "trajectory_space_done")
    
    # Colors
    done_color = "#33cc66"      # Green
    todo_color = "#9aa0a6"      # Grey
    
    def get_color(session_key):
        """Get color based on completion status"""
        return done_color if st.session_state.get(session_key, False) else todo_color
    
    def get_edge_color(from_key):
        """Edge is green if FROM node is completed"""
        return done_color if st.session_state.get(from_key, False) else todo_color
    
    # Build Graphviz DOT
    dot_lines = [
        "digraph Workflow {",
        "  rankdir=TB;",  # Top to bottom
        "  bgcolor=transparent;",
        "  node [shape=circle width=0.5 height=0.5 fixedsize=true fontsize=8 fontname=Arial];",
        "  edge [penwidth=2];",
        "",
        "  // Invisible spacer for alignment",
        "  {rank=same; L1; R1;}",
        "",
    ]
    
    # Left path nodes
    for node_id, name, key in left_path:
        color = get_color(key)
        # Use tooltip for hover (title attribute in SVG)
        dot_lines.append(f'  {node_id} [label="" color="{color}" fillcolor="{color}" style=filled tooltip="{name}"];')
    
    # Right path nodes
    for node_id, name, key in right_path:
        color = get_color(key)
        dot_lines.append(f'  {node_id} [label="" color="{color}" fillcolor="{color}" style=filled tooltip="{name}"];')
    
    # Final node
    final_color = get_color(final_step[2])
    dot_lines.append(f'  {final_step[0]} [label="" color="{final_color}" fillcolor="{final_color}" style=filled tooltip="{final_step[1]}"];')
    
    dot_lines.append("")
    
    # Left path edges
    for i in range(len(left_path) - 1):
        from_node = left_path[i][0]
        to_node = left_path[i + 1][0]
        from_key = left_path[i][2]
        edge_color = get_edge_color(from_key)
        dot_lines.append(f'  {from_node} -> {to_node} [color="{edge_color}"];')
    
    # Right path edges
    for i in range(len(right_path) - 1):
        from_node = right_path[i][0]
        to_node = right_path[i + 1][0]
        from_key = right_path[i][2]
        edge_color = get_edge_color(from_key)
        dot_lines.append(f'  {from_node} -> {to_node} [color="{edge_color}"];')
    
    dot_lines.append("")
    
    # Converging edges to final step
    last_left_key = left_path[-1][2]
    last_right_key = right_path[-1][2]
    
    dot_lines.append(f'  {left_path[-1][0]} -> {final_step[0]} [color="{get_edge_color(last_left_key)}"];')
    dot_lines.append(f'  {right_path[-1][0]} -> {final_step[0]} [color="{get_edge_color(last_right_key)}"];')
    
    dot_lines.append("}")
    
    dot_string = "\n".join(dot_lines)
    
    # Render in sidebar
    st.sidebar.markdown("**📊 Workflow Progress**")
    st.sidebar.graphviz_chart(dot_string, use_container_width=True)
    
    # Legend
    st.sidebar.markdown(f"""
    <div style='font-size: 0.75rem; color: #666; text-align: center;'>
        <span style='color: {done_color};'>●</span> Completed &nbsp;&nbsp;
        <span style='color: {todo_color};'>●</span> Pending
    </div>
    """, unsafe_allow_html=True)


def render_app_sidebar():
    """
    Render the application sidebar with logo, status, and info
    This function should be called at the top of every page AFTER st.set_page_config()
    """
    
    # Try to get logo as base64
    logo_base64 = get_logo_base64()
    
    # Logo and Title at the very top (always visible)
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
    
    # === WORKFLOW PROGRESS FLOWCHART ===
    render_workflow_flowchart()
    
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
