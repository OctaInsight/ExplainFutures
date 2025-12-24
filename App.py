"""
ExplainFutures - Main Application Entry Point
Enhanced with Login/Logout and Interactive Sidebar
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.config import load_config, initialize_session_state, get_config
from core.utils import setup_page_config

def main():
    """Main application entry point"""
    
    # Load configuration
    config = load_config()
    
    # Setup Streamlit page
    setup_page_config(
        title="ExplainFutures",
        icon="🔮",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize login state if not exists
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    
    # Render interactive sidebar
    render_sidebar()
    
    # Main content
    if not st.session_state.logged_in:
        render_login_page()
    else:
        render_home_page()


def render_sidebar():
    """Render interactive sidebar with status information"""
    
    # Logo and Title at the very top (always visible)
    # Note: For now using emoji, replace with actual logo later
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1.5rem 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔮</div>
            <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
            <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Data-Driven Future Exploration</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Login/Logout section
    if st.session_state.logged_in:
        st.sidebar.success(f"✅ Logged in: **{st.session_state.username}**")
        
        # Show login duration
        if st.session_state.login_time:
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
        st.sidebar.caption("Please login to access features")
    
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
            
            if st.session_state.uploaded_file_name:
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
    
    # Large spacer before extra information
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    
    # Extra Information Section (clearly separated)
    st.sidebar.markdown("""
        <div style='background-color: #e8eaf6; padding: 1rem; border-radius: 8px; margin-top: 2rem;'>
            <h4 style='margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #31333F;'>📚 Additional Information</h4>
        </div>
    """, unsafe_allow_html=True)
    
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
        config_data = get_config()
        version = config_data.get('version', '1.0.0')
    except:
        version = '1.0.0'
    
    st.sidebar.caption(f"ExplainFutures v{version}")
    st.sidebar.caption("© 2024 Octa Insight")


def render_login_page():
    """Render login page"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 Welcome to ExplainFutures")
        st.markdown("*Please login to continue*")
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="login_username_input"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                submit = st.form_submit_button(
                    "🔓 Login",
                    use_container_width=True,
                    type="primary"
                )
            
            if submit:
                # Simple authentication (replace with real auth in production)
                if username and password:
                    # For demo: accept any non-empty credentials
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Please enter both username and password")
        
        st.markdown("---")
        
        # Info box
        st.info("""
        **Demo Mode:** For now, enter any username and password to login.
        
        **Production:** Authentication will be integrated with Supabase in Phase 4.
        """)


def render_home_page():
    """Render home page content"""
    
    # Application header
    st.title("🔮 ExplainFutures")
    st.markdown("### Data-Driven Future Exploration Platform")
    st.markdown("---")
    
    # Welcome message
    st.markdown(f"## Welcome back, {st.session_state.username}! 👋")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.get("data_loaded"):
            st.metric("Data Status", "✅ Loaded")
        else:
            st.metric("Data Status", "⚠️ Not loaded")
    
    with col2:
        if st.session_state.get("df_long") is not None:
            n_vars = len(st.session_state.df_long['variable'].unique())
            st.metric("Variables", n_vars)
        else:
            st.metric("Variables", "0")
    
    with col3:
        if st.session_state.get("preprocessing_applied"):
            st.metric("Preprocessing", "✅ Applied")
        else:
            st.metric("Preprocessing", "Not applied")
    
    with col4:
        # Placeholder for models
        st.metric("Models Trained", "0")
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Getting Started")
        
        st.markdown("""
        ### Current Implementation (Phase 1)
        
        ✅ **Available Features:**
        1. **Upload & Data Diagnostics** - Load and validate your data
        2. **Data Cleaning & Preprocessing** - Handle missing values, outliers
        3. **Data Exploration & Visualization** - Interactive plots and charts
        4. **Dimensionality Reduction** - PCA analysis (basic interface)
        
        🔜 **Coming in Future Phases:**
        5. Variable Relationships (pairwise analysis)
        6. Time-based models
        7. Model evaluation & reliability
        8. Future projections & what-if analysis
        9. Scenario analysis
        
        ### 📍 Recommended Workflow
        
        1. **Start Here:** Upload & Data Diagnostics
        2. **Then:** Clean your data if needed
        3. **Explore:** Create visualizations
        4. **Analyze:** Run PCA and other analyses
        5. **Model:** (Coming in Phase 2+)
        
        👈 **Use the sidebar to navigate between pages!**
        """)
    
    with col2:
        st.markdown("## 💡 Quick Actions")
        
        if not st.session_state.get("data_loaded"):
            st.info("**Next Step:** Upload your data")
            st.page_link(
                "pages/1_Upload_and_Data_Diagnostics.py",
                label="📁 Go to Upload",
                icon="📁"
            )
        else:
            st.success("**Data loaded!**")
            
            if not st.session_state.get("preprocessing_applied"):
                st.info("**Suggested:** Clean your data")
                st.page_link(
                    "pages/2_Data_Cleaning_and_Preprocessing.py",
                    label="🧹 Go to Cleaning",
                    icon="🧹"
                )
            else:
                st.info("**Ready to:** Explore and visualize")
                st.page_link(
                    "pages/3_Data_Exploration_and_Visualization.py",
                    label="📊 Go to Visualization",
                    icon="📊"
                )
        
        st.markdown("---")
        
        # Phase status
        st.markdown("### 📦 Current Phase")
        st.info("""
        **Phase 1: MVP**
        - ✅ Data upload
        - ✅ Data diagnostics
        - ✅ Visualization
        - ✅ Basic cleaning UI
        - ✅ PCA interface
        """)
    
    st.markdown("---")
    
    # Footer - About ExplainFutures ONLY (Octa Insight moved to sidebar)
    st.markdown("## 📚 About the Platform")
    
    with st.expander("📖 Learn More About ExplainFutures"):
        st.markdown("""
        ExplainFutures is a modular, data-driven application designed to help users 
        understand, model, and actively explore the future behavior of complex systems.
        
        **Key Features:**
        - Time-series data analysis
        - Interactive visualizations
        - Interpretable models with equations
        - Future exploration ("what-if" scenarios)
        - Natural language scenario processing
        
        **Built for:** Researchers, analysts, and decision-makers working with 
        time-indexed data in economics, sustainability, climate, and social systems.
        """)


if __name__ == "__main__":
    main()
