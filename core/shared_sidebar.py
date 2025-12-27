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
    Render ultra-compact workflow flowchart matching logo box height
    
    Strategy: Show only KEY milestones (not all 11 steps)
    - Left: 4 main steps (Data → Clean → Model → Project)
    - Right: 2 steps (Scenarios → Matrix)
    - Final: 1 step (Trajectory)
    
    Total height ~3-4cm on 14" screen (same as logo box)
    All elements scale with sidebar width
    """
    
    # SIMPLIFIED workflow - only key milestones for compact display
    # Format: (node_id, display_name, session_keys_to_check)
    left_path = [
        ("L1", "Data Upload", ["data_loaded"]),
        ("L2", "Data Processing", ["data_cleaned", "data_explored"]),
        ("L3", "Modeling", ["models_trained", "models_evaluated"]),
        ("L4", "Forecasting", ["projections_done"]),
    ]
    
    right_path = [
        ("R1", "Scenarios", ["scenarios_analyzed"]),
        ("R2", "Matrix", ["scenario_matrix_done"]),
    ]
    
    final_step = ("F", "Trajectory Space", ["trajectory_space_done"])
    
    # Streamlit color palette
    done_fill = "#21c55d"
    done_border = "#0e6537"
    done_line = "#16a34a"
    
    pending_fill = "#d1d5db"
    pending_border = "#6b7280"
    pending_line = "#9ca3af"
    
    def is_step_complete(session_keys):
        """Check if ANY of the session keys are True"""
        return any(st.session_state.get(key, False) for key in session_keys)
    
    def get_node_colors(session_keys):
        """Get fill and border colors based on completion"""
        if is_step_complete(session_keys):
            return done_fill, done_border
        else:
            return pending_fill, pending_border
    
    def get_line_color(session_keys):
        """Line color based on source completion"""
        return done_line if is_step_complete(session_keys) else pending_line
    
    # ULTRA-COMPACT Graphviz DOT
    dot_lines = [
        "digraph W {",
        "  rankdir=TB;",
        "  bgcolor=transparent;",
        # EXTREME compression settings
        "  graph [pad=\"0.02\" margin=\"0\" nodesep=\"0.06\" ranksep=\"0.08\" dpi=\"72\"];",
        # TINY nodes - barely visible but still distinguishable
        "  node [shape=circle width=0.06 height=0.06 fixedsize=true label=\"\" penwidth=1.5 style=\"filled\" margin=0];",
        # THIN short lines
        "  edge [arrowhead=none penwidth=1 len=0.1 minlen=1];",
        "",
    ]
    
    # Invisible connector nodes
    for i in range(len(left_path) - 1):
        dot_lines.append(f'  L{i+1}c [shape=point width=0.01 style=invis];')
    
    for i in range(len(right_path) - 1):
        dot_lines.append(f'  R{i+1}c [shape=point width=0.01 style=invis];')
    
    dot_lines.append('  L4c [shape=point width=0.01 style=invis];')
    dot_lines.append('  R2c [shape=point width=0.01 style=invis];')
    dot_lines.append("")
    
    # Left path nodes
    for node_id, name, keys in left_path:
        fill, border = get_node_colors(keys)
        dot_lines.append(f'  {node_id} [fillcolor="{fill}" color="{border}" tooltip="{name}"];')
    
    # Right path nodes
    for node_id, name, keys in right_path:
        fill, border = get_node_colors(keys)
        dot_lines.append(f'  {node_id} [fillcolor="{fill}" color="{border}" tooltip="{name}"];')
    
    # Final node
    final_fill, final_border = get_node_colors(final_step[2])
    dot_lines.append(f'  {final_step[0]} [fillcolor="{final_fill}" color="{final_border}" tooltip="{final_step[1]}"];')
    dot_lines.append("")
    
    # Left path connections
    for i in range(len(left_path) - 1):
        line_color = get_line_color(left_path[i][2])
        dot_lines.append(f'  {left_path[i][0]} -> L{i+1}c [color="{line_color}" style=invis];')
        dot_lines.append(f'  L{i+1}c -> {left_path[i+1][0]} [color="{line_color}"];')
    
    # Right path connections
    for i in range(len(right_path) - 1):
        line_color = get_line_color(right_path[i][2])
        dot_lines.append(f'  {right_path[i][0]} -> R{i+1}c [color="{line_color}" style=invis];')
        dot_lines.append(f'  R{i+1}c -> {right_path[i+1][0]} [color="{line_color}"];')
    
    # Converging to final
    dot_lines.append(f'  L4 -> L4c [color="{get_line_color(left_path[-1][2])}" style=invis];')
    dot_lines.append(f'  L4c -> F [color="{get_line_color(left_path[-1][2])}"];')
    dot_lines.append(f'  R2 -> R2c [color="{get_line_color(right_path[-1][2])}" style=invis];')
    dot_lines.append(f'  R2c -> F [color="{get_line_color(right_path[-1][2])}"];')
    
    dot_lines.append("}")
    dot_string = "\n".join(dot_lines)
    
    # Ultra-compact header
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 0.15rem;'>
        <div style='background: linear-gradient(135deg, #0e6537 0%, #21c55d 100%); 
                    -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-weight: 700;
                    font-size: 0.7rem;
                    letter-spacing: 0.3px;'>
            WORKFLOW
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render flowchart
    st.sidebar.graphviz_chart(dot_string, use_container_width=True)
    
    # Ultra-compact legend
    st.sidebar.markdown(f"""
    <div style='text-align: center; margin-top: -0.4rem; margin-bottom: 0.2rem; font-size: 0.6rem;'>
        <span style='color: {done_fill}; font-weight: 600;'>●</span>
        <span style='color: #6b7280; margin-left: 0.1rem; margin-right: 0.3rem;'>Done</span>
        <span style='color: {pending_fill};'>●</span>
        <span style='color: #6b7280; margin-left: 0.1rem;'>Pending</span>
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
