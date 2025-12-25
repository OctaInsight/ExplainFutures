"""
ExplainFutures - Main Entry Point
Login/Logout Page
"""

import streamlit as st
from datetime import datetime

# MUST be first Streamlit command
st.set_page_config(
    page_title="ExplainFutures - Login",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import sidebar AFTER config
from shared_sidebar import render_app_sidebar

# Render sidebar
render_app_sidebar()

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
                    # Simple authentication (replace with your actual auth logic)
                    if username and password:  # Add your validation here
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.login_time = datetime.now()
                        st.success(f"✅ Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
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
    
    # Logout button
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.login_time = None
        st.rerun()
