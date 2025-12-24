"""
Page 2: Data Cleaning and Preprocessing
Handle missing values, outliers, and data transformations
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
st.set_page_config(page_title="Data Cleaning & Preprocessing", page_icon="🧹", layout="wide")

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("*Clean and prepare your data for analysis*")
st.markdown("---")


def main():
    """Main page function"""
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        return
    
    df_long = st.session_state.df_long
    variables = sorted(df_long['variable'].unique().tolist())
    
    st.success("✅ Data loaded and ready for cleaning")
    
    # Show current data info
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(variables))
    col2.metric("Total Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    
    st.markdown("---")
    
    # Create tabs for different cleaning operations
    tab1, tab2, tab3 = st.tabs([
        "🔍 Missing Values", 
        "📊 Outliers", 
        "🔄 Transformations"
    ])
    
    with tab1:
        handle_missing_values(df_long, variables)
    
    with tab2:
        handle_outliers(df_long, variables)
    
    with tab3:
        handle_transformations(df_long, variables)


def handle_missing_values(df_long, variables):
    """Handle missing values in the dataset"""
    
    st.header("Missing Values Treatment")
    
    # Show missing values summary
    st.subheader("Missing Values Summary")
    
    missing_summary = []
    for var in variables:
        var_data = df_long[df_long['variable'] == var]
        missing_count = var_data['value'].isna().sum()
        missing_pct = (missing_count / len(var_data)) * 100 if len(var_data) > 0 else 0
        
        missing_summary.append({
            "Variable": var,
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct:.1f}%",
            "Status": "🔴" if missing_pct > 20 else "🟡" if missing_pct > 5 else "🟢"
        })
    
    summary_df = pd.DataFrame(missing_summary)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Options for handling missing values
    st.markdown("---")
    st.subheader("Select Treatment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        treatment_method = st.selectbox(
            "Missing value treatment",
            [
                "None - Keep as is",
                "Drop - Remove rows with missing values",
                "Forward Fill - Use previous value",
                "Backward Fill - Use next value",
                "Interpolate - Linear interpolation",
                "Mean - Replace with mean",
                "Median - Replace with median"
            ],
            key="missing_treatment"
        )
    
    with col2:
        apply_to = st.multiselect(
            "Apply to variables",
            variables,
            default=variables,
            key="missing_apply_to"
        )
    
    if st.button("✨ Apply Treatment", type="primary", key="apply_missing"):
        if treatment_method != "None - Keep as is" and apply_to:
            with st.spinner("Applying treatment..."):
                # This is a placeholder - you'll implement the actual logic
                st.success(f"✅ Applied {treatment_method} to {len(apply_to)} variable(s)")
                st.info("💡 Note: Treatment logic will be implemented in core/preprocess/cleaning.py")
        else:
            st.warning("⚠️ No treatment selected or no variables selected")


def handle_outliers(df_long, variables):
    """Detect and handle outliers"""
    
    st.header("Outlier Detection and Treatment")
    
    st.subheader("Outlier Detection Summary (IQR Method)")
    
    # Detect outliers
    outlier_summary = []
    for var in variables:
        var_data = df_long[df_long['variable'] == var]['value'].dropna()
        
        if len(var_data) > 4:
            Q1 = var_data.quantile(0.25)
            Q3 = var_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((var_data < lower_bound) | (var_data > upper_bound)).sum()
            outlier_pct = (outliers / len(var_data)) * 100
            
            outlier_summary.append({
                "Variable": var,
                "Outliers": outliers,
                "Outlier %": f"{outlier_pct:.1f}%",
                "Lower Bound": f"{lower_bound:.2f}",
                "Upper Bound": f"{upper_bound:.2f}"
            })
    
    if outlier_summary:
        summary_df = pd.DataFrame(outlier_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Treatment options
    st.markdown("---")
    st.subheader("Select Treatment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outlier_method = st.selectbox(
            "Outlier treatment",
            [
                "None - Keep as is",
                "Remove - Delete outlier rows",
                "Cap - Clip to bounds",
                "Transform - Log transformation",
                "Winsorize - Replace with percentile values"
            ],
            key="outlier_treatment"
        )
    
    with col2:
        apply_to_outliers = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="outlier_apply_to"
        )
    
    if st.button("✨ Apply Outlier Treatment", type="primary", key="apply_outlier"):
        if outlier_method != "None - Keep as is" and apply_to_outliers:
            with st.spinner("Applying treatment..."):
                st.success(f"✅ Applied {outlier_method} to {len(apply_to_outliers)} variable(s)")
                st.info("💡 Note: Treatment logic will be implemented in core/preprocess/cleaning.py")
        else:
            st.warning("⚠️ No treatment selected or no variables selected")


def handle_transformations(df_long, variables):
    """Apply data transformations"""
    
    st.header("Data Transformations")
    
    st.subheader("Available Transformations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transformation = st.selectbox(
            "Select transformation",
            [
                "None",
                "Log - Natural logarithm",
                "Log10 - Base 10 logarithm",
                "Square Root",
                "Standardize - Z-score normalization",
                "Min-Max - Scale to 0-1",
                "Difference - First difference",
                "Percentage Change"
            ],
            key="transformation_type"
        )
    
    with col2:
        transform_vars = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="transform_vars"
        )
    
    if st.button("✨ Apply Transformation", type="primary", key="apply_transform"):
        if transformation != "None" and transform_vars:
            with st.spinner("Applying transformation..."):
                st.success(f"✅ Applied {transformation} to {len(transform_vars)} variable(s)")
                st.info("💡 Note: Transformation logic will be implemented in core/preprocess/cleaning.py")
        else:
            st.warning("⚠️ No transformation selected or no variables selected")
    
    st.markdown("---")
    
    # Resampling
    st.subheader("Time Series Resampling")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        resample_freq = st.selectbox(
            "Resample to frequency",
            ["No resampling", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
            key="resample_freq"
        )
    
    with col2:
        aggregation = st.selectbox(
            "Aggregation method",
            ["mean", "median", "sum", "min", "max", "first", "last"],
            key="resample_agg"
        )
    
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🔄 Resample", key="resample_button"):
            if resample_freq != "No resampling":
                st.success(f"✅ Resampled to {resample_freq} using {aggregation}")
                st.info("💡 Note: Resampling logic will be implemented in core/preprocess/cleaning.py")


if __name__ == "__main__":
    main()
