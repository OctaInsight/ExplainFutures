"""
System Analysis Application
Main entry point with sidebar navigation
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="System Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar sections
st.markdown("""
<style>
    /* Sidebar section headers */
    .sidebar-section {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #FF4B4B;
    }
    
    .sidebar-section h3 {
        color: #FF4B4B;
        font-size: 14px;
        margin: 0 0 5px 0;
    }
    
    .sidebar-section p {
        color: #FAFAFA;
        font-size: 12px;
        margin: 0;
        opacity: 0.8;
    }
    
    /* Divider */
    .sidebar-divider {
        border-top: 1px solid #464646;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.title("📊 System Analysis Application")
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

# Instructions
st.info("""
👈 **Use the sidebar** to navigate between different analysis steps.

**Workflow:**
- Start with **Path A** if you have historical data to analyze
- Use **Path B** to define and extract scenario parameters
- Combine both in **Integration** to validate scenarios against historical trends
""")

# Sidebar content
with st.sidebar:
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # PATH A
    st.markdown("""
    <div class="sidebar-section">
        <h3>📊 PATH A: HISTORICAL DATA</h3>
        <p>Analyze historical data and predict future trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Pages 1-8 for historical analysis")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # PATH B
    st.markdown("""
    <div class="sidebar-section">
        <h3>📝 PATH B: SCENARIOS</h3>
        <p>Extract and define scenario parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Pages 9-10 for scenario definition")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # INTEGRATION
    st.markdown("""
    <div class="sidebar-section">
        <h3>🎯 INTEGRATION</h3>
        <p>Compare scenarios with historical trajectories</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Page 11 for combined analysis")
