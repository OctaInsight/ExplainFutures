"""
App.py Template - Fix for Your Root File

INSTRUCTIONS:
1. If your App.py is in the ROOT (same level as pages/ folder):
   - Use "pages/00_Login.py" (with "pages/" prefix)
   
2. If your App.py is INSIDE pages/ folder:
   - Use "00_Login.py" (no "pages/" prefix)
"""

import streamlit as st
import time

st.set_page_config(
    page_title="ExplainFutures",
    page_icon="🚀",
    layout="wide"
)

def check_authentication():
    """Check authentication - use correct path based on where App.py is located"""
    if not st.session_state.get('authenticated', False):
        st.warning("⚠️ Please log in to continue")
        time.sleep(1)
        
        # ✅ If App.py is in ROOT folder:
        st.switch_page("pages/00_Login.py")
        
        # ✅ If App.py is in pages/ folder:
        # st.switch_page("00_Login.py")
        
        st.stop()

check_authentication()

# Redirect authenticated users to dashboard
st.info("Redirecting to dashboard...")
time.sleep(1)

# ✅ If App.py is in ROOT:
st.switch_page("pages/01_Home.py")

# ✅ If App.py is in pages/:
# st.switch_page("01_Home.py")
