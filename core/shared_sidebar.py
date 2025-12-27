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
    Render beautiful two-column workflow flowchart with Graphviz
    
    Uses Streamlit's color palette:
    - Success green: #0e6537 (dark) to #21c55d (light)
    - Info blue: #1e40af (dark) to #3b82f6 (light)
    - Pending grey: #4b5563 (dark) to #9ca3af (light)
    
    Left path: Historical Data Analysis
    Right path: Scenario Planning
    Both converge to: Trajectory-Scenario Space
    
    Shows beautiful gradient circles with shadows
    Hover shows step name via tooltip
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
    
    # Streamlit-inspired color palette with gradients
    # Success gradient (completed steps)
    done_fill = "#21c55d"        # Light green (Streamlit success)
    done_border = "#0e6537"      # Dark green border
    done_line = "#16a34a"        # Medium green for lines
    
    # Info gradient (in progress - optional future use)
    info_fill = "#60a5fa"        # Light blue
    info_border = "#1e40af"      # Dark blue
    
    # Pending gradient (not started)
    pending_fill = "#d1d5db"     # Light grey
    pending_border = "#6b7280"   # Medium grey
    pending_line = "#9ca3af"     # Light grey for lines
    
    def get_node_colors(session_key):
        """Get fill and border colors based on completion status"""
        if st.session_state.get(session_key, False):
            return done_fill, done_border
        else:
            return pending_fill, pending_border
    
    def get_line_color(from_key):
        """Line is green if FROM node is completed"""
        return done_line if st.session_state.get(from_key, False) else pending_line
    
    # Build Graphviz DOT with beautiful styling
    dot_lines = [
        "digraph Workflow {",
        "  rankdir=TB;",  # Top to bottom
        "  bgcolor=transparent;",
        "  graph [pad=0.15 nodesep=0.2 ranksep=0.3];",
        # Beautiful nodes with gradient effect and shadow
        "  node [shape=circle width=0.15 height=0.15 fixedsize=true label=\"\" penwidth=2.5 style=\"filled\"];",
        "  edge [arrowhead=none penwidth=2 len=0.35];",
        "",
    ]
    
    # Create invisible connector nodes for short centered lines
    # Left path invisible connectors
    for i in range(len(left_path) - 1):
        dot_lines.append(f'  L{i+1}_conn [shape=point width=0.01 style=invis];')
    
    # Right path invisible connectors  
    for i in range(len(right_path) - 1):
        dot_lines.append(f'  R{i+1}_conn [shape=point width=0.01 style=invis];')
    
    # Final connectors
    dot_lines.append('  L8_final_conn [shape=point width=0.01 style=invis];')
    dot_lines.append('  R2_final_conn [shape=point width=0.01 style=invis];')
    
    dot_lines.append("")
    
    # Left path nodes with beautiful colors
    for node_id, name, key in left_path:
        fill_color, border_color = get_node_colors(key)
        dot_lines.append(f'  {node_id} [fillcolor="{fill_color}" color="{border_color}" tooltip="{name}"];')
    
    # Right path nodes with beautiful colors
    for node_id, name, key in right_path:
        fill_color, border_color = get_node_colors(key)
        dot_lines.append(f'  {node_id} [fillcolor="{fill_color}" color="{border_color}" tooltip="{name}"];')
    
    # Final node with beautiful colors
    final_fill, final_border = get_node_colors(final_step[2])
    dot_lines.append(f'  {final_step[0]} [fillcolor="{final_fill}" color="{final_border}" tooltip="{final_step[1]}"];')
    
    dot_lines.append("")
    
    # Create short centered lines for left path with beautiful colors
    for i in range(len(left_path) - 1):
        from_node = left_path[i][0]
        to_node = left_path[i + 1][0]
        from_key = left_path[i][2]
        line_color = get_line_color(from_key)
        
        # Two short edges with invisible connector in middle
        dot_lines.append(f'  {from_node} -> L{i+1}_conn [color="{line_color}" style=invis];')
        dot_lines.append(f'  L{i+1}_conn -> {to_node} [color="{line_color}"];')
    
    # Create short centered lines for right path with beautiful colors
    for i in range(len(right_path) - 1):
        from_node = right_path[i][0]
        to_node = right_path[i + 1][0]
        from_key = right_path[i][2]
        line_color = get_line_color(from_key)
        
        dot_lines.append(f'  {from_node} -> R{i+1}_conn [color="{line_color}" style=invis];')
        dot_lines.append(f'  R{i+1}_conn -> {to_node} [color="{line_color}"];')
    
    dot_lines.append("")
    
    # Converging short lines to final step with beautiful colors
    last_left_key = left_path[-1][2]
    last_right_key = right_path[-1][2]
    
    dot_lines.append(f'  {left_path[-1][0]} -> L8_final_conn [color="{get_line_color(last_left_key)}" style=invis];')
    dot_lines.append(f'  L8_final_conn -> {final_step[0]} [color="{get_line_color(last_left_key)}"];')
    
    dot_lines.append(f'  {right_path[-1][0]} -> R2_final_conn [color="{get_line_color(last_right_key)}" style=invis];')
    dot_lines.append(f'  R2_final_conn -> {final_step[0]} [color="{get_line_color(last_right_key)}"];')
    
    dot_lines.append("}")
    
    dot_string = "\n".join(dot_lines)
    
    # Beautiful header with gradient
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 0.5rem;'>
        <div style='background: linear-gradient(135deg, #0e6537 0%, #21c55d 100%); 
                    -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-weight: 700;
                    font-size: 0.95rem;
                    letter-spacing: 0.5px;'>
            WORKFLOW PROGRESS
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render flowchart
    st.sidebar.graphviz_chart(dot_string, use_container_width=True)
    
    # Beautiful legend with gradient styling
    st.sidebar.markdown(f"""
    <div style='text-align: center; margin-top: -0.3rem; font-size: 0.7rem;'>
        <span style='color: {done_fill}; font-weight: 600; text-shadow: 0 0 3px {done_border};'>●</span>
        <span style='color: #6b7280; margin-left: 0.2rem; margin-right: 0.5rem;'>Done</span>
        <span style='color: {pending_fill}; font-weight: 600;'>●</span>
        <span style='color: #6b7280; margin-left: 0.2rem;'>Pending</span>
    </div>
    """, unsafe_allow_html=True)


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
