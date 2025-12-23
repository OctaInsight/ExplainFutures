"""
ExplainFutures - Main Application Entry Point
Phase 1: Data Ingestion, Health Check, and Visualization

Author: Development Team
Version: 1.0.0-phase1
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.config import load_config, initialize_session_state
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
    
    # Application header
    st.title("🔮 ExplainFutures")
    st.markdown("*Data-Driven Future Exploration Platform*")
    st.markdown("---")
    
    # Main content area
    show_home()
    
    # Sidebar info
    render_sidebar_info()


def show_home():
    """Display home page with Phase 1 introduction"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to ExplainFutures! 🚀
        
        ExplainFutures helps you understand and explore the future behavior of complex systems 
        through data-driven analysis and interactive modeling.
        
        ### Phase 1 Features (Current MVP)
        
        This minimal viable product includes:
        
        1. **📁 Data Upload & Parsing**
           - Upload CSV, TXT, or Excel files
           - Automatic time column detection
           - Variable selection and type assignment
           - Conversion to standardized long format
        
        2. **🔍 Data Health Report**
           - Missing values analysis
           - Duplicate detection
           - Time coverage assessment
           - Sampling frequency evaluation
           - Basic outlier detection
        
        3. **📊 Interactive Visualization**
           - Single variable time plots
           - Multi-variable comparison with independent Y-axes
           - Customizable colors, scales, and axis limits
           - Interactive Plotly charts
        
        ### Getting Started
        
        Use the **sidebar navigation** to access:
        - **Upload & Data Health** - Load and assess your data
        - **Explore & Visualize** - Create interactive plots
        
        ---
        
        ### Data Requirements
        
        Your dataset should include:
        - ✅ A time/date column (any standard datetime format)
        - ✅ One or more numeric variables
        - ✅ Consistent time indexing
        
        ### Supported Formats
        - CSV (`.csv`)
        - Text files (`.txt`)
        - Excel (`.xlsx`, `.xls`)
        """)
    
    with col2:
        st.info("""
        **💡 Quick Tips**
        
        1. Start with a clean, time-indexed dataset
        2. Review the data health report before analysis
        3. Use multi-axis plots to compare variables with different scales
        4. All data stays in memory during your session
        """)
        
        # Display session status
        if st.session_state.get("data_loaded", False):
            st.success("✅ **Data Loaded**")
            if "df_long" in st.session_state and st.session_state.df_long is not None:
                df = st.session_state.df_long
                st.metric("Variables", len(df['variable'].unique()))
                st.metric("Data Points", len(df))
                if 'timestamp' in df.columns:
                    time_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                    st.caption(f"📅 {time_range}")
        else:
            st.warning("⚠️ No data loaded yet")
            st.caption("Upload data to get started")


def render_sidebar_info():
    """Render sidebar information"""
    
    st.sidebar.title("📍 Navigation")
    st.sidebar.markdown("Use the pages in the sidebar to navigate")
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📦 Phase 1 Status")
    st.sidebar.markdown("""
    - ✅ Data Upload & Parsing
    - ✅ Data Health Report
    - ✅ Single Variable Plots
    - ✅ Multi-Variable Plots
    """)
    
    st.sidebar.markdown("---")
    
    # Display current session info
    if st.session_state.get("data_loaded", False):
        st.sidebar.success("📊 Data Loaded")
        
        if st.session_state.uploaded_file_name:
            st.sidebar.caption(f"📄 {st.session_state.uploaded_file_name}")
        
        if st.session_state.preprocessing_applied:
            st.sidebar.info("🧹 Preprocessing Applied")
    else:
        st.sidebar.info("📥 Awaiting Data Upload")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ExplainFutures v1.0.0-phase1")


if __name__ == "__main__":
    main()
