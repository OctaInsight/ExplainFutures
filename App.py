"""
App.py - Main Entry Point
Landing page that redirects to login or dashboard
"""

import streamlit as st
import time

# Page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="ExplainFutures",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Check authentication and redirect
if not st.session_state.authenticated:
    # Not authenticated - show welcome and redirect to login
    st.title("🚀 Welcome to ExplainFutures")
    st.markdown("### Scenario Analysis & Futures Forecasting")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ExplainFutures** helps you:
        - 📊 Analyze future scenarios
        - 🔮 Generate forecasts
        - 📈 Compare outcomes
        - 🎯 Make data-driven decisions
        """)
    
    with col2:
        st.info("Please log in to continue")
        if st.button("🔐 Go to Login", type="primary", use_container_width=True):
            st.switch_page("pages/00_Login.py")
        
        st.markdown("---")
        st.caption("Don't have an account? Try the demo!")
        if st.button("🎭 Try Demo", use_container_width=True):
            st.switch_page("pages/00_Login.py")
    
else:
    # Authenticated - redirect to dashboard
    st.success(f"✅ Welcome back, {st.session_state.get('username', 'User')}!")
    st.info("Redirecting to dashboard...")
    time.sleep(1)
    st.switch_page("pages/01_Home.py")
