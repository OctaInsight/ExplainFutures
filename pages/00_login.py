"""
Page 0: Login & Authentication
User login with demo mode support
"""

import streamlit as st
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="ExplainFutures - Login",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Import database manager
try:
    from core.database.supabase_manager import get_db_manager
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database connection not available: {str(e)}")
    st.info("Please ensure supabase_manager.py is in core/database/ and Streamlit secrets are configured.")


def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_demo' not in st.session_state:
        st.session_state.is_demo = False


def login_user(username: str, password: str):
    """
    Authenticate user and initialize session
    
    Args:
        username: Username
        password: Password
    
    Returns:
        bool: True if login successful
    """
    if not DB_AVAILABLE:
        st.error("Database connection not available")
        return False
    
    try:
        db = get_db_manager()
        
        # Get client info
        ip_address = st.context.headers.get('X-Forwarded-For', 'unknown') if hasattr(st, 'context') else None
        user_agent = st.context.headers.get('User-Agent', 'unknown') if hasattr(st, 'context') else None
        
        # Attempt login
        user = db.login_user(username, password, ip_address, user_agent)
        
        if user:
            # Set session state
            st.session_state.authenticated = True
            st.session_state.user_id = user['user_id']
            st.session_state.username = user['username']
            st.session_state.full_name = user.get('full_name', username)
            st.session_state.email = user['email']
            st.session_state.is_demo = db.is_demo_user(user['user_id'])
            st.session_state.subscription_tier = user.get('subscription_tier', 'free')
            
            # If demo user, create demo session
            if st.session_state.is_demo:
                session = db.create_demo_session(
                    user_id=user['user_id'],
                    project_id=db.demo_project_id,
                    duration_seconds=1800  # 30 minutes
                )
                
                if session:
                    st.session_state.demo_session_id = session['session_id']
                    st.session_state.current_project_id = db.demo_project_id
                    st.session_state.demo_expires_at = session['expires_at']
            
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False


def main():
    """Main login page"""
    
    init_session_state()
    
    # If already authenticated, redirect to home
    if st.session_state.authenticated:
        st.success(f"✅ Already logged in as {st.session_state.username}")
        time.sleep(1)
        st.switch_page("01_Home.py")  # Just filename, no "pages/" prefix
        return
    
    # Header
    st.title("🚀 ExplainFutures")
    st.markdown("### Scenario Analysis & Futures Forecasting")
    st.markdown("---")
    
    # Create two columns for login and demo
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Use 'demo' for demo account"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Demo password: 'demo123'"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                login_button = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )
            
            with col_btn2:
                forgot_button = st.form_submit_button(
                    "Forgot Password?",
                    use_container_width=True
                )
            
            if login_button:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        if login_user(username, password):
                            st.success(f"✅ Welcome back, {st.session_state.full_name}!")
                            
                            if st.session_state.is_demo:
                                st.info("🎭 Demo Mode: Your session will expire in 30 minutes. Changes will be reset after logout.")
                            
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials. Please try again.")
            
            if forgot_button:
                st.info("Password reset functionality coming soon. Please contact support.")
    
    with col2:
        st.subheader("🎭 Try Demo")
        
        st.markdown("""
        **Explore ExplainFutures risk-free!**
        
        The demo account lets you:
        - ✅ Explore all features
        - ✅ Use pre-loaded climate scenarios
        - ✅ Train models & generate forecasts
        - ✅ Create scenario analyses
        
        **Demo limitations:**
        - ❌ Cannot upload your own files
        - ❌ Session expires after 30 minutes
        - ❌ Changes reset on logout
        """)
        
        if st.button("🚀 Launch Demo", type="secondary", use_container_width=True):
            with st.spinner("Starting demo session..."):
                if login_user("demo", "demo123"):
                    st.success("✅ Demo session started!")
                    st.info("🎭 Explore with pre-loaded data. Session expires in 30 minutes.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Demo session failed to start. Please try again.")
    
    st.markdown("---")
    
    # Sign up section
    st.markdown("### 📝 New to ExplainFutures?")
    
    col_signup1, col_signup2 = st.columns([2, 1])
    
    with col_signup1:
        st.markdown("""
        Create a free account to:
        - 📁 Upload your own documents and data
        - 💾 Save unlimited projects
        - 🤝 Collaborate with team members
        - 📊 Export all your analyses
        """)
    
    with col_signup2:
        if st.button("Create Account", type="primary", use_container_width=True):
            st.info("Sign-up functionality coming soon!")
    
    # Footer
    st.markdown("---")
    st.caption("© 2024 ExplainFutures | [Privacy Policy](#) | [Terms of Service](#)")


if __name__ == "__main__":
    if not DB_AVAILABLE:
        st.stop()
    
    main()
