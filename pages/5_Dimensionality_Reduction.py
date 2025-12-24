"""
Page 5: Dimensionality Reduction
Principal Component Analysis (PCA) and related techniques
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Dimensionality Reduction", page_icon="🔬", layout="wide")

st.title("🔬 Dimensionality Reduction")
st.markdown("*Reduce data complexity with PCA and other techniques*")
st.markdown("---")


def main():
    """Main page function"""
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please load your data first")
        return
    
    df_long = st.session_state.df_long
    variables = sorted(df_long['variable'].unique().tolist())
    
    st.success("✅ Data loaded and ready for dimensionality reduction")
    
    # Show current data info
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(variables))
    col2.metric("Observations", len(df_long['timestamp'].unique()))
    col3.metric("Potential Components", min(len(variables), len(df_long['timestamp'].unique())))
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 PCA Analysis",
        "📊 Variance Explained",
        "🔍 Component Loadings"
    ])
    
    with tab1:
        perform_pca_analysis(df_long, variables)
    
    with tab2:
        show_variance_explained()
    
    with tab3:
        show_component_loadings()


def perform_pca_analysis(df_long, variables):
    """Perform PCA analysis"""
    
    st.header("Principal Component Analysis (PCA)")
    
    st.markdown("""
    PCA reduces the dimensionality of your data by finding new variables (principal components) 
    that capture the maximum variance in your original variables.
    """)
    
    # Configuration
    st.subheader("Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_vars = st.multiselect(
            "Select variables for PCA",
            variables,
            default=variables,
            help="Choose variables to include in the analysis",
            key="pca_vars"
        )
    
    with col2:
        n_components = st.slider(
            "Number of components",
            min_value=2,
            max_value=min(len(selected_vars), 10) if selected_vars else 10,
            value=min(3, len(selected_vars)) if selected_vars else 3,
            help="Number of principal components to compute",
            key="n_components"
        )
    
    with col3:
        standardize = st.checkbox(
            "Standardize variables",
            value=True,
            help="Recommended when variables have different scales",
            key="standardize"
        )
    
    if len(selected_vars) < 2:
        st.warning("⚠️ Please select at least 2 variables for PCA")
        return
    
    # Run PCA button
    if st.button("🔬 Run PCA Analysis", type="primary", key="run_pca"):
        with st.spinner("Computing PCA..."):
            st.success(f"✅ PCA computed with {n_components} components")
            
            # Placeholder for actual PCA results
            st.info("💡 Note: PCA implementation will be added in core/analysis/pca.py")
            
            # Show placeholder results
            st.subheader("PCA Results Summary")
            
            # Placeholder variance explained
            variance_data = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_components)],
                "Variance Explained (%)": [30, 25, 20, 15, 10][:n_components],
                "Cumulative (%)": [30, 55, 75, 90, 100][:n_components]
            })
            
            st.dataframe(variance_data, use_container_width=True, hide_index=True)


def show_variance_explained():
    """Show variance explained by components"""
    
    st.header("Variance Explained Analysis")
    
    st.info("""
    This section will show:
    - Scree plot (variance per component)
    - Cumulative variance explained
    - Component selection guidance
    
    📊 Visualization will be implemented with Plotly
    """)
    
    # Placeholder
    st.markdown("### Coming Soon:")
    st.markdown("- 📈 Interactive scree plot")
    st.markdown("- 📊 Cumulative variance curve")
    st.markdown("- 🎯 Elbow point detection")
    st.markdown("- 💡 Component selection recommendations")


def show_component_loadings():
    """Show component loadings (variable contributions)"""
    
    st.header("Component Loadings")
    
    st.info("""
    Component loadings show how much each original variable contributes to each principal component.
    High absolute loadings indicate strong relationships.
    """)
    
    st.markdown("### Coming Soon:")
    st.markdown("- 📊 Loadings heatmap")
    st.markdown("- 🎯 Variable contributions per component")
    st.markdown("- 🔍 Interpretation guidance")
    st.markdown("- 📈 Biplot visualization")


if __name__ == "__main__":
    main()
