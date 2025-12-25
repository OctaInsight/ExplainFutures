"""
ExplainFutures - Main Entry Point
Login/Logout Page
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import base64

# MUST be first Streamlit command
st.set_page_config(
    page_title="ExplainFutures - Login",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Render sidebar
def render_sidebar():
    """Render the sidebar with logo and info"""
    
    # Try to get logo as base64
    logo_base64 = get_logo_base64()
    
    # Logo and Title at the very top
    if logo_base64:
        st.sidebar.markdown(f"""
            <div style='text-align: center; padding: 1.5rem 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
                <img src='data:image/png;base64,{logo_base64}' style='width: 80px; height: 80px; margin-bottom: 0.5rem;' alt='Logo'/>
                <h2 style='margin: 0; color: white; font-weight: 600;'>ExplainFutures</h2>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.9);'>Data-Driven Future Exploration</p>
            </div>
        """, unsafe_allow_html=True)
    else:
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
        
        if st.session_state.get("login_time"):
            duration = datetime.now() - st.session_state.login_time
            minutes = int(duration.total_seconds() / 60)
            st.sidebar.caption(f"Session: {minutes} min")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True, key="sidebar_logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.login_time = None
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Not logged in")
        st.sidebar.caption("Please login below")
    
    st.sidebar.markdown("---")
    
    # About ExplainFutures
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
    
    # About Octa Insight
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
    st.sidebar.caption("ExplainFutures v1.0.0")
    st.sidebar.caption("© 2024 Octa Insight")

# Render sidebar
render_sidebar()

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_time" not in st.session_state:
    st.session_state.login_time = None

# Main content
st.title("🔮 ExplainFutures")
st.markdown("### Data-Driven Future Exploration")
st.markdown("---")

if not st.session_state.logged_in:
    # Login Form
    st.subheader("🔐 Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                login_button = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
            
            with col_btn2:
                guest_button = st.form_submit_button("👤 Continue as Guest", use_container_width=True)
            
            if login_button:
                if username and password:
                    # Simple authentication (you can customize this)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter both username and password")
            
            if guest_button:
                st.session_state.logged_in = True
                st.session_state.username = "Guest"
                st.session_state.login_time = datetime.now()
                st.success("✅ Welcome, Guest!")
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **Demo Credentials:**
        - Username: `demo`
        - Password: `demo`
        
        Or click "Continue as Guest" to explore the app.
        """)

else:
    # Logged In - Show Welcome
    st.success(f"✅ You are logged in as **{st.session_state.username}**")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Path A: Historical Data Analysis
        
        Analyze historical data to understand system behavior and predict future trends:
        
        1. **Upload Data** - Import your dataset
        2. **Clean & Preprocess** - Prepare data for analysis
        3. **Explore** - Visualize patterns and trends
        4. **Analyze Relationships** - Understand variable interactions
        5. **Dimensionality Reduction** - Simplify complex systems
        6. **Build Models** - Train predictive models
        7. **Evaluate** - Compare model performance
        8. **Project Future** - Generate forecasts
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Path B: Scenario Definition
        
        Extract and define scenario parameters from text:
        
        9. **Extract Parameters** - Analyze scenario descriptions
        10. **Scenario Matrix** - Compare and organize scenarios
        
        ### 🎯 Integration
        
        Combine both paths to see which scenarios align with historical trends:
        
        11. **System Trajectories** - Plot scenarios against historical data
        """)
    
    st.markdown("---")
    
    st.info("""
    👈 **Use the sidebar** to navigate between different analysis steps.
    
    **Workflow:**
    - Start with **Path A** if you have historical data to analyze
    - Use **Path B** to define and extract scenario parameters
    - Combine both in **Integration** to validate scenarios against historical trends
    """)
    
    st.markdown("---")
    
    # Logout button in main area
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.login_time = None
        st.rerun()
