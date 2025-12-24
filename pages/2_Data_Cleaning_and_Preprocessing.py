"""
Page 2: Data Cleaning and Preprocessing
Handle missing values, outliers, and data transformations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar  

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Data Cleaning & Preprocessing", page_icon="🧹", layout="wide")

# Render shared sidebar
render_app_sidebar()  

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("*Clean and prepare your data for analysis*")
st.markdown("---")


def initialize_cleaning_history():
    """Initialize session state for tracking cleaning operations"""
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = {}
    if "transformed_columns" not in st.session_state:
        st.session_state.transformed_columns = []
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None


def add_to_cleaning_history(operation_type, method, variables, details=None):
    """Add an operation to the cleaning history"""
    operation = {
        "timestamp": datetime.now(),
        "type": operation_type,
        "method": method,
        "variables": variables,
        "details": details or {}
    }
    st.session_state.cleaning_history.append(operation)


def plot_comparison(df_original, df_modified, variable, operation_type):
    """Create a comparison plot between original and modified data"""
    
    try:
        # Get data for the specific variable - try both 'time' and 'timestamp' columns
        time_col = 'timestamp' if 'timestamp' in df_original.columns else 'time'
        
        orig_data = df_original[df_original['variable'] == variable].copy()
        mod_data = df_modified[df_modified['variable'] == variable].copy()
        
        if len(orig_data) == 0 or len(mod_data) == 0:
            st.warning(f"No data found for variable: {variable}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=orig_data[time_col],
            y=orig_data['value'],
            mode='lines+markers',
            name='Original',
            line=dict(color='#3498db', width=2),
            marker=dict(size=4, color='#3498db')
        ))
        
        # Modified data
        fig.add_trace(go.Scatter(
            x=mod_data[time_col],
            y=mod_data['value'],
            mode='lines+markers',
            name='Modified',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4, color='#e74c3c')
        ))
        
        # Layout
        fig.update_layout(
            title=f"Comparison: {variable} - {operation_type}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=450,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison plot: {str(e)}")
        return None


def main():
    """Main page function"""
    
    # Initialize cleaning history
    initialize_cleaning_history()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Diagnostics.py")
        return
    
    df_long = st.session_state.df_long
    variables = sorted(df_long['variable'].unique().tolist())
    
    # Initialize df_clean if not exists
    if st.session_state.df_clean is None:
        st.session_state.df_clean = df_long.copy()
    
    st.success("✅ Data loaded and ready for cleaning")
    
    # Show current data info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Total Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    col4.metric("Operations Applied", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # Create tabs for different cleaning operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Missing Values", 
        "📊 Outliers", 
        "🔄 Transformations",
        "📋 Summary & Save"
    ])
    
    with tab1:
        handle_missing_values(df_long, variables)
    
    with tab2:
        handle_outliers(df_long, variables)
    
    with tab3:
        handle_transformations(df_long, variables)
    
    with tab4:
        show_cleaning_summary()


def handle_missing_values(df_long, variables):
    """Handle missing values in the dataset"""
    
    st.header("Missing Values Treatment")
    
    # Show missing values summary
    st.subheader("📊 Missing Values Summary")
    
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
    st.subheader("⚙️ Select Treatment Method")
    
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
            default=[],
            key="missing_apply_to"
        )
    
    # Suffix for new columns
    if treatment_method != "None - Keep as is":
        suffix = st.text_input(
            "Suffix for cleaned columns",
            value="_cleaned",
            key="missing_suffix",
            help="New columns will be: variable_name + suffix"
        )
    else:
        suffix = "_cleaned"
    
    if st.button("✨ Apply Treatment", type="primary", key="apply_missing"):
        if treatment_method != "None - Keep as is" and apply_to:
            with st.spinner("Applying treatment..."):
                # Apply treatment and create NEW columns
                df_modified, new_columns = apply_missing_value_treatment(
                    st.session_state.df_clean.copy(), 
                    apply_to, 
                    treatment_method,
                    suffix
                )
                
                # Store in session state
                for var in new_columns:
                    st.session_state.cleaned_data[var] = df_modified[df_modified['variable'] == var]
                
                # Add to history
                add_to_cleaning_history(
                    "Missing Values",
                    treatment_method,
                    apply_to,
                    {
                        "before_count": df_long['value'].isna().sum(),
                        "new_columns": new_columns
                    }
                )
                
                st.success(f"✅ Applied {treatment_method} to {len(apply_to)} variable(s)")
                st.info(f"📝 Created new columns: {', '.join(new_columns)}")
                
                # Store modified data
                st.session_state.df_clean = df_modified
                st.session_state.preprocessing_applied = True
                st.session_state.last_missing_treatment = {
                    'method': treatment_method,
                    'variables': apply_to,
                    'new_columns': new_columns
                }
                st.rerun()
        else:
            st.warning("⚠️ No treatment selected or no variables selected")
    
    # Visualization comparison section
    if st.session_state.get("last_missing_treatment"):
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        treatment_info = st.session_state.last_missing_treatment
        new_columns = treatment_info.get('new_columns', [])
        original_vars = treatment_info.get('variables', [])
        
        if new_columns and original_vars:
            # Let user select which variable to compare
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(original_vars)),
                    format_func=lambda i: original_vars[i],
                    key="missing_viz_var"
                )
                selected_orig = original_vars[selected_idx]
                selected_new = new_columns[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Cleaned:** {selected_new}")
            
            with col2:
                # Create comparison plot
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_new,
                    "Missing Value Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Statistics comparison
            st.markdown("#### Statistics Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Original: {selected_orig}**")
                orig_stats = df_long[df_long['variable'] == selected_orig]['value'].describe()
                st.dataframe(orig_stats, use_container_width=True)
            
            with col2:
                st.markdown(f"**Cleaned: {selected_new}**")
                mod_stats = st.session_state.df_clean[
                    st.session_state.df_clean['variable'] == selected_new
                ]['value'].describe()
                st.dataframe(mod_stats, use_container_width=True)
    
    # Save button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💾 **Note:** Creates NEW columns alongside originals (originals are preserved)")
    
    with col2:
        if st.button(
            "💾 Save to Database",
            disabled=True,
            use_container_width=True,
            key="save_missing",
            help="Feature available in Phase 4"
        ):
            pass


def apply_missing_value_treatment(df, variables, method, suffix):
    """Apply missing value treatment and create NEW columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        # Get original data
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        # Create cleaned version (copy of original)
        cleaned_data = original_data.copy()
        cleaned_data['variable'] = new_var_name
        
        # Apply treatment
        if "Drop" in method:
            # Remove rows with missing values
            cleaned_data = cleaned_data.dropna(subset=['value'])
        
        elif "Forward Fill" in method:
            cleaned_data['value'] = cleaned_data['value'].fillna(method='ffill')
        
        elif "Backward Fill" in method:
            cleaned_data['value'] = cleaned_data['value'].fillna(method='bfill')
        
        elif "Interpolate" in method:
            cleaned_data['value'] = cleaned_data['value'].interpolate(method='linear')
        
        elif "Mean" in method:
            mean_val = original_data['value'].mean()
            cleaned_data['value'] = cleaned_data['value'].fillna(mean_val)
        
        elif "Median" in method:
            median_val = original_data['value'].median()
            cleaned_data['value'] = cleaned_data['value'].fillna(median_val)
        
        # Add cleaned data to dataframe (keeps original)
        df_copy = pd.concat([df_copy, cleaned_data], ignore_index=True)
    
    return df_copy, new_columns


def handle_outliers(df_long, variables):
    """Detect and handle outliers"""
    
    st.header("Outlier Detection and Treatment")
    
    st.subheader("📊 Outlier Detection Summary (IQR Method)")
    
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
    st.subheader("⚙️ Select Treatment Method")
    
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
    
    # Suffix for new columns
    if outlier_method != "None - Keep as is":
        outlier_suffix = st.text_input(
            "Suffix for treated columns",
            value="_outlier_treated",
            key="outlier_suffix",
            help="New columns will be: variable_name + suffix"
        )
    else:
        outlier_suffix = "_outlier_treated"
    
    if st.button("✨ Apply Outlier Treatment", type="primary", key="apply_outlier"):
        if outlier_method != "None - Keep as is" and apply_to_outliers:
            with st.spinner("Applying treatment..."):
                # Apply treatment and create NEW columns
                df_modified, new_columns = apply_outlier_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to_outliers,
                    outlier_method,
                    outlier_suffix
                )
                
                # Add to history
                add_to_cleaning_history(
                    "Outliers",
                    outlier_method,
                    apply_to_outliers,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Applied {outlier_method} to {len(apply_to_outliers)} variable(s)")
                st.info(f"📝 Created new columns: {', '.join(new_columns)}")
                
                # Store modified data
                st.session_state.df_clean = df_modified
                st.session_state.preprocessing_applied = True
                st.session_state.last_outlier_treatment = {
                    'method': outlier_method,
                    'variables': apply_to_outliers,
                    'new_columns': new_columns
                }
                st.rerun()
        else:
            st.warning("⚠️ No treatment selected or no variables selected")
    
    # Visualization comparison
    if st.session_state.get("last_outlier_treatment"):
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        treatment_info = st.session_state.last_outlier_treatment
        new_columns = treatment_info.get('new_columns', [])
        original_vars = treatment_info.get('variables', [])
        
        if new_columns and original_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(original_vars)),
                    format_func=lambda i: original_vars[i],
                    key="outlier_viz_var"
                )
                selected_orig = original_vars[selected_idx]
                selected_new = new_columns[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Treated:** {selected_new}")
            
            with col2:
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_new,
                    "Outlier Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show outlier statistics
            st.markdown("#### Outlier Impact")
            col1, col2, col3 = st.columns(3)
            
            orig_data = df_long[df_long['variable'] == selected_orig]['value'].dropna()
            mod_data = st.session_state.df_clean[
                st.session_state.df_clean['variable'] == selected_new
            ]['value'].dropna()
            
            col1.metric("Original Points", len(orig_data))
            col2.metric("Treated Points", len(mod_data))
            col3.metric("Points Removed/Changed", abs(len(orig_data) - len(mod_data)))
    
    # Save button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💾 **Note:** Creates NEW columns alongside originals (originals are preserved)")
    
    with col2:
        if st.button(
            "💾 Save to Database",
            disabled=True,
            use_container_width=True,
            key="save_outliers",
            help="Feature available in Phase 4"
        ):
            pass


def apply_outlier_treatment(df, variables, method, suffix):
    """Apply outlier treatment and create NEW columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        # Get original data
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        # Create treated version
        treated_data = original_data.copy()
        treated_data['variable'] = new_var_name
        
        var_data = treated_data['value'].dropna()
        
        if len(var_data) > 4:
            Q1 = var_data.quantile(0.25)
            Q3 = var_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            if "Remove" in method:
                # Remove rows with outliers
                treated_data = treated_data[
                    ~((treated_data['value'] < lower_bound) | (treated_data['value'] > upper_bound))
                ]
            
            elif "Cap" in method:
                # Clip values to bounds
                treated_data['value'] = treated_data['value'].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )
            
            elif "Transform" in method:
                # Log transformation (shift to positive if needed)
                min_val = treated_data['value'].min()
                if min_val <= 0:
                    treated_data['value'] = np.log1p(
                        treated_data['value'] - min_val + 1
                    )
                else:
                    treated_data['value'] = np.log(treated_data['value'])
            
            elif "Winsorize" in method:
                # Replace with 5th and 95th percentiles
                p05 = var_data.quantile(0.05)
                p95 = var_data.quantile(0.95)
                treated_data['value'] = treated_data['value'].clip(
                    lower=p05,
                    upper=p95
                )
        
        # Add treated data to dataframe (keeps original)
        df_copy = pd.concat([df_copy, treated_data], ignore_index=True)
    
    return df_copy, new_columns


def handle_transformations(df_long, variables):
    """Apply data transformations"""
    
    st.header("Data Transformations")
    
    st.subheader("⚙️ Available Transformations")
    
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
    
    # Suffix for transformed columns
    if transformation != "None":
        suffix = st.text_input(
            "Suffix for new columns",
            value=f"_{transformation.split('-')[0].strip().lower()}",
            key="transform_suffix",
            help="New transformed columns will be: variable_name + suffix"
        )
    else:
        suffix = "_transformed"
    
    if st.button("✨ Apply Transformation", type="primary", key="apply_transform"):
        if transformation != "None" and transform_vars:
            with st.spinner("Applying transformation..."):
                # Apply transformation
                df_modified, new_columns = apply_transformation(
                    st.session_state.df_clean.copy(),
                    transform_vars,
                    transformation,
                    suffix
                )
                
                # Track transformed columns
                st.session_state.transformed_columns.extend(new_columns)
                
                # Add to history
                add_to_cleaning_history(
                    "Transformation",
                    transformation,
                    transform_vars,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Created {len(new_columns)} new transformed column(s)")
                st.info(f"📝 New columns: {', '.join(new_columns)}")
                
                # Store modified data
                st.session_state.df_clean = df_modified
                st.session_state.preprocessing_applied = True
                st.session_state.last_transformation = {
                    'method': transformation,
                    'variables': transform_vars,
                    'new_columns': new_columns
                }
                st.rerun()
        else:
            st.warning("⚠️ No transformation selected or no variables selected")
    
    # Visualization comparison
    if st.session_state.get("last_transformation"):
        st.markdown("---")
        st.subheader("📈 Original vs Transformed")
        
        transform_info = st.session_state.last_transformation
        new_columns = transform_info.get('new_columns', [])
        original_vars = transform_info.get('variables', [])
        
        if new_columns and original_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(original_vars)),
                    format_func=lambda i: original_vars[i],
                    key="transform_viz_var"
                )
                selected_orig = original_vars[selected_idx]
                selected_new = new_columns[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Transformed:** {selected_new}")
            
            with col2:
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_new,
                    "Transformation"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Distribution comparison
            st.markdown("#### Distribution Comparison")
            col1, col2 = st.columns(2)
            
            orig_vals = df_long[df_long['variable'] == selected_orig]['value'].dropna()
            trans_vals = st.session_state.df_clean[
                st.session_state.df_clean['variable'] == selected_new
            ]['value'].dropna()
            
            with col1:
                st.markdown(f"**Original: {selected_orig}**")
                st.metric("Mean", f"{orig_vals.mean():.2f}")
                st.metric("Std Dev", f"{orig_vals.std():.2f}")
                st.metric("Min/Max", f"{orig_vals.min():.2f} / {orig_vals.max():.2f}")
            
            with col2:
                st.markdown(f"**Transformed: {selected_new}**")
                st.metric("Mean", f"{trans_vals.mean():.2f}")
                st.metric("Std Dev", f"{trans_vals.std():.2f}")
                st.metric("Min/Max", f"{trans_vals.min():.2f} / {trans_vals.max():.2f}")
    
    # Save button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💾 **Note:** Transformations create NEW columns alongside originals")
    
    with col2:
        if st.button(
            "💾 Save to Database",
            disabled=True,
            use_container_width=True,
            key="save_transforms",
            help="Feature available in Phase 4"
        ):
            pass


def apply_transformation(df, variables, transformation, suffix):
    """Apply transformation and create new columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        # Get original data
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        # Create transformed version
        transformed_data = original_data.copy()
        transformed_data['variable'] = new_var_name
        
        values = transformed_data['value'].dropna()
        
        if "Log -" in transformation:
            # Natural log
            min_val = values.min()
            if min_val <= 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log1p(transformed_data.loc[transformed_data['value'].notna(), 'value'] - min_val + 1)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Log10" in transformation:
            min_val = values.min()
            if min_val <= 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'] - min_val + 1)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Square Root" in transformation:
            min_val = values.min()
            if min_val < 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.sqrt(transformed_data.loc[transformed_data['value'].notna(), 'value'] - min_val)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.sqrt(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Standardize" in transformation:
            mean = values.mean()
            std = values.std()
            if std > 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    (transformed_data.loc[transformed_data['value'].notna(), 'value'] - mean) / std
        
        elif "Min-Max" in transformation:
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    (transformed_data.loc[transformed_data['value'].notna(), 'value'] - min_val) / (max_val - min_val)
        
        elif "Difference" in transformation:
            transformed_data['value'] = transformed_data['value'].diff()
        
        elif "Percentage Change" in transformation:
            transformed_data['value'] = transformed_data['value'].pct_change() * 100
        
        # Add transformed data to dataframe
        df_copy = pd.concat([df_copy, transformed_data], ignore_index=True)
    
    return df_copy, new_columns


def show_cleaning_summary():
    """Show summary of all cleaning operations"""
    
    st.header("📋 Cleaning Summary & Data Save")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations performed yet")
        st.markdown("Apply some cleaning operations in the tabs above, then return here to review and save.")
        return
    
    # Summary statistics
    st.subheader("📊 Operation Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Operations", len(st.session_state.cleaning_history))
    
    # Count unique original variables that were modified
    modified_vars = set()
    for op in st.session_state.cleaning_history:
        modified_vars.update(op['variables'])
    col2.metric("Variables Modified", len(modified_vars))
    
    # Count total new columns created
    total_new_columns = 0
    for op in st.session_state.cleaning_history:
        if 'new_columns' in op.get('details', {}):
            total_new_columns += len(op['details']['new_columns'])
    col3.metric("New Columns Created", total_new_columns)
    
    st.markdown("---")
    
    # Detailed operation log
    st.subheader("📜 Operation Log")
    
    operations_data = []
    for i, op in enumerate(st.session_state.cleaning_history, 1):
        new_cols = op.get('details', {}).get('new_columns', [])
        operations_data.append({
            "#": i,
            "Timestamp": op['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "Type": op['type'],
            "Method": op['method'],
            "Original Variables": ", ".join(op['variables'][:3]) + 
                        (f" (+{len(op['variables'])-3} more)" if len(op['variables']) > 3 else ""),
            "New Columns": ", ".join(new_cols[:2]) + 
                        (f" (+{len(new_cols)-2} more)" if len(new_cols) > 2 else "") if new_cols else "N/A"
        })
    
    operations_df = pd.DataFrame(operations_data)
    st.dataframe(operations_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Data comparison
    st.subheader("📈 Overall Data Comparison")
    
    if st.session_state.df_clean is not None:
        original_df = st.session_state.df_long
        cleaned_df = st.session_state.df_clean
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Data**")
            st.metric("Total Points", len(original_df))
            st.metric("Missing Values", original_df['value'].isna().sum())
            st.metric("Variables", len(original_df['variable'].unique()))
        
        with col2:
            st.markdown("**Enhanced Data (with new columns)**")
            st.metric("Total Points", len(cleaned_df))
            st.metric("Missing Values", cleaned_df['value'].isna().sum())
            st.metric("Variables", len(cleaned_df['variable'].unique()))
            
            new_var_count = len(cleaned_df['variable'].unique()) - len(original_df['variable'].unique())
            if new_var_count > 0:
                st.success(f"🆕 +{new_var_count} new columns added")
    
    st.markdown("---")
    
    # Show all variables (original + new)
    st.subheader("📊 All Available Variables")
    
    if st.session_state.df_clean is not None:
        all_vars = sorted(st.session_state.df_clean['variable'].unique())
        original_vars = sorted(st.session_state.df_long['variable'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Variables**")
            for var in original_vars:
                st.text(f"• {var}")
        
        with col2:
            st.markdown("**New/Processed Variables**")
            new_vars = [v for v in all_vars if v not in original_vars]
            if new_vars:
                for var in new_vars:
                    st.text(f"• {var}")
            else:
                st.caption("No new variables yet")
    
    st.markdown("---")
    
    # Save options
    st.subheader("💾 Save Enhanced Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Save Options:**
        - ✅ Original variables: **Preserved unchanged**
        - ✅ Cleaned variables: **Saved as new columns** (e.g., `Temperature_cleaned`)
        - ✅ Transformed variables: **Saved as new columns** (e.g., `GDP_log`)
        - ✅ Metadata: Operation log with timestamps
        
        **Important:** All original data is preserved. You have both original and processed versions.
        """)
        
        save_notes = st.text_area(
            "Add notes (optional)",
            placeholder="e.g., Cleaned for Q1 2024 analysis, removed temperature outliers...",
            key="save_notes"
        )
    
    with col2:
        st.write("")
        st.write("")
        
        if st.button(
            "💾 Save All to Database",
            type="primary",
            use_container_width=True,
            disabled=True,
            help="Feature available in Phase 4 - will save to Supabase"
        ):
            # Placeholder for database save
            pass
        
        st.markdown("---")
        
        if st.button(
            "📥 Download Enhanced Data",
            use_container_width=True,
            key="download_cleaned"
        ):
            if st.session_state.df_clean is not None:
                # Convert to CSV
                csv = st.session_state.df_clean.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"enhanced_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_btn"
                )
    
    st.markdown("---")
    
    # Next steps
    st.subheader("🎯 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Visualize**")
        st.markdown("Explore your enhanced data")
        if st.button("Go to Visualization", key="goto_viz", use_container_width=True):
            st.switch_page("pages/3_Data_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("**🔬 Analyze**")
        st.markdown("Run PCA and analysis")
        if st.button("Go to Analysis", key="goto_analysis", use_container_width=True):
            st.switch_page("pages/5_Dimensionality_Reduction.py")
    
    with col3:
        st.markdown("**🔄 Clear History**")
        st.markdown("Start cleaning over")
        if st.button("Clear Operations", key="clear_history", use_container_width=True):
            st.session_state.cleaning_history = []
            st.session_state.cleaned_data = {}
            st.session_state.transformed_columns = []
            st.session_state.df_clean = st.session_state.df_long.copy()
            st.session_state.preprocessing_applied = False
            # Clear treatment info
            if 'last_missing_treatment' in st.session_state:
                del st.session_state.last_missing_treatment
            if 'last_outlier_treatment' in st.session_state:
                del st.session_state.last_outlier_treatment
            if 'last_transformation' in st.session_state:
                del st.session_state.last_transformation
            st.success("✅ Cleaning history cleared!")
            st.rerun()


if __name__ == "__main__":
    main()
