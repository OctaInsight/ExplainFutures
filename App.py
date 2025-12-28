"""
App.py - ExplainFutures Main Entry Point
Landing page with login, demo access, and subscription information
"""

import streamlit as st
import time

# Page config FIRST
st.set_page_config(
    page_title="ExplainFutures - Scenario Analysis Platform",
    page_icon= Path("assets/logo_small.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import database manager
try:
    from core.database.supabase_manager import get_db_manager
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database connection not available: {str(e)}")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_user(username: str, password: str):
    """Authenticate user and initialize session"""
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


# ============================================================================
# MAIN PAGE
# ============================================================================

# Check if already authenticated
if st.session_state.authenticated:
    # Redirect to dashboard
    st.success(f"✅ Welcome back, {st.session_state.get('full_name', 'User')}!")
    st.info("Redirecting to dashboard...")
    time.sleep(1)
    st.switch_page("pages/01_Home.py")
    st.stop()

# Not authenticated - show landing page
st.title("🚀 ExplainFutures")
st.markdown("### Scenario Analysis & Futures Forecasting Platform")
st.markdown("---")

# Main content - two columns
col_left, col_right = st.columns([1.2, 1])

with col_left:
    # Login form
    st.markdown("#### 🔐 Login to Your Account")
    
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
            st.info("📧 Password reset: Please contact support at sales@octainsight.com")

    
       if st.button("🚀 Launch Demo Session", type="secondary", use_container_width=True):
        with st.spinner("Starting demo session..."):
            if login_user("demo", "demo123"):
                st.success("✅ Demo session started!")
                st.info("🎭 Explore with pre-loaded data. Session expires in 30 minutes.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Demo session failed to start. Please try again or contact support.")
 
    
    

with col_right:

    st.markdown("#### What is ExplainFutures?")
    st.markdown("""
    **ExplainFutures** helps you:
    - 📊 Analyze future scenarios
    - 🔮 Generate forecasts
    - 📈 Compare outcomes
    - 🎯 Make data-driven decisions
    """)
    
    st.markdown("---")
    
    st.markdown("#### 🎭 Try Demo (No Registration)")
    
    st.markdown("""
    Explore ExplainFutures risk-free with our interactive demo:
    
    **Demo Features:**
    - ✅ Full access to all features
    - ✅ Pre-loaded scenarios
    - ✅ Train models & generate forecasts
    - ✅ Create scenario analyses
    
    **Demo Limitations:**
    - ⏱️ 30-minute session
    - 🔒 Read-only data (cannot upload files)
    - 🔄 Changes reset on logout
    """)
    

st.markdown("---")

# ============================================================================
# SUBSCRIPTION PLANS
# ============================================================================

st.markdown("## 👩‍🔬 Committed to Inclusion & Diversity")

st.info("""
At **Octa Insight**, we believe in making our applications accessible to all. 
We offer flexible pricing based on your location, organization type, and research goals:

- 🌍 **Developing Countries:** Special financial discounts available
- 👩‍🔬 **Underrepresented Groups:** Dedicated support and special offers for women scientists and researchers from underrepresented communities

**All pricing is determined through discussion to ensure accessibility and value.**
""")

st.markdown("### Choose Your Plan")

# Three columns for subscription plans
plan_col1, plan_col2, plan_col3 = st.columns(3)

with plan_col1:
    st.markdown("""
    <div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; height: 100%;'>
        <h3 style='color: #4CAF50;'>👤 Personal License</h3>
        <p><strong>For Individual Researchers & Analysts</strong></p>
        <ul>
            <li>One-year subscription</li>
            <li>Access to all features</li>
            <li>Unlimited projects</li>
            <li>Technical support via email</li>
            <li>Regular updates & improvements</li>
            <li>Data privacy & security</li>
        </ul>
        <p><strong>Ideal for:</strong></p>
        <ul>
            <li>PhD students & researchers</li>
            <li>Independent consultants</li>
            <li>Policy analysts</li>
            <li>Sustainability professionals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with plan_col2:
    st.markdown("""
    <div style='border: 2px solid #2196F3; border-radius: 10px; padding: 20px; height: 100%;'>
        <h3 style='color: #2196F3;'>🏛️ Institutional License</h3>
        <p><strong>For Universities, Ministries & Organizations</strong></p>
        <ul>
            <li>Multiple user licenses</li>
            <li>One year or multi-year plans</li>
            <li>Access to all features</li>
            <li>Priority technical support</li>
            <li>Training & onboarding sessions</li>
            <li>Custom deployment options</li>
            <li>Team collaboration features</li>
        </ul>
        <p><strong>Ideal for:</strong></p>
        <ul>
            <li>Research institutions</li>
            <li>Government ministries</li>
            <li>International organizations</li>
            <li>Large NGOs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with plan_col3:
    st.markdown("""
    <div style='border: 2px solid #FF9800; border-radius: 10px; padding: 20px; height: 100%;'>
        <h3 style='color: #FF9800;'>🤝 Project Partnership</h3>
        <p><strong>Research Collaboration & Co-Development</strong></p>
        <ul>
            <li>Full project duration access</li>
            <li>Scientific collaboration & support</li>
            <li>Co-authorship opportunities</li>
            <li>Custom module development</li>
            <li>Technical & scientific expertise</li>
            <li>Grant proposal support</li>
            <li>Partnership options: Full partner, Associated partner, or Subcontractor</li>
        </ul>
        <p><strong>Octa Insight offers:</strong></p>
        <ul>
            <li>Track record in EU projects</li>
            <li>Experienced scientific team</li>
            <li>Long-term collaboration</li>
            <li>Bespoke solutions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Contact section
st.markdown("### 📧 Request a Custom Quote")

contact_col1, contact_col2 = st.columns([2, 1])

with contact_col1:
    st.markdown("""
    Ready to get started? Contact us to discuss your needs and receive a personalized quote:
    
    - **Email:** [sales@octainsight.com](mailto:sales@octainsight.com)
    - **Subject:** Include "ExplainFutures Subscription Inquiry"
    - **Include:** Your organization type, location, and specific requirements
    
    We typically respond within **1-2 business days** with pricing options tailored to your needs.
    """)

with contact_col2:
    if st.button("📧 Contact Sales", type="primary", use_container_width=True):
        st.info("Please send an email to: **sales@octainsight.com**")
    
    if st.button("💬 Schedule a Demo Call", use_container_width=True):
        st.info("Email sales@octainsight.com to schedule a personalized demo")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>© 2025 Octa Insight | <a href='#'>Privacy Policy</a> | <a href='#'>Terms of Service</a></p>
    <p>Committed to diversity, inclusion, and accessible futures research</p>
</div>
""", unsafe_allow_html=True)
