"""
Page 3: Data Cleaning and Preprocessing (COMPLETE WITH DATABASE SAVE)
Handle missing values, outliers, and data transformations
✅ Loads data from database automatically
✅ Saves cleaned data to database (appends, not overwrites)
✅ Updates all progress indicators (database, session state, sidebar, home)
✅ Turns 2nd workflow dot green
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Data Cleaning and Preprocessing",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

# 🆕 Import database manager
try:
    from core.database.supabase_manager import get_db_manager
    DB_AVAILABLE = True
    db = get_db_manager()
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database import error: {str(e)}")

# Initialize
initialize_session_state()
config = get_config()

# Render shared sidebar
render_app_sidebar()

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("*Clean and prepare your data for analysis*")
st.markdown("---")

# Authentication check
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()


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


def convert_to_wide_format(df_long):
    """Convert long format dataframe back to wide format for download"""
    try:
        time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
        
        df_wide = df_long.pivot(
            index=time_col,
            columns='variable',
            values='value'
        ).reset_index()
        
        time_cols = [time_col]
        other_cols = sorted([col for col in df_wide.columns if col != time_col])
        df_wide = df_wide[time_cols + other_cols]
        
        return df_wide
        
    except Exception as e:
        st.error(f"Error converting to wide format: {str(e)}")
        return None


def plot_comparison(df_original, df_modified, original_variable, new_variable, operation_type):
    """Create a comparison plot between original and modified data"""
    
    try:
        time_col = 'timestamp' if 'timestamp' in df_original.columns else 'time'
        
        orig_data = df_original[df_original['variable'] == original_variable].copy()
        mod_data = df_modified[df_modified['variable'] == new_variable].copy()
        
        if len(orig_data) == 0:
            st.error(f"❌ No original data found for: {original_variable}")
            return None
            
        if len(mod_data) == 0:
            st.error(f"❌ No cleaned data found for: {new_variable}")
            return None
        
        orig_data = orig_data.sort_values(time_col)
        mod_data = mod_data.sort_values(time_col)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=orig_data[time_col],
            y=orig_data['value'],
            mode='lines+markers',
            name=f'Original ({original_variable})',
            line=dict(color='#3498db', width=2),
            marker=dict(size=4, color='#3498db'),
            hovertemplate='<b>Original</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=mod_data[time_col],
            y=mod_data['value'],
            mode='lines+markers',
            name=f'Cleaned ({new_variable})',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4, color='#e74c3c'),
            hovertemplate='<b>Cleaned</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Before vs After: {original_variable} → {new_variable}<br><sub>{operation_type}</sub>",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            template='plotly_white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating comparison plot: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def main():
    """Main page function"""
    
    initialize_cleaning_history()
    
    # ============================================================
    # 🆕 NEW: Load data from database if not in session state
    # ============================================================
    
    needs_data_load = (
        not st.session_state.get('data_loaded') or 
        st.session_state.get('df_long') is None
    )
    
    if needs_data_load:
        if not DB_AVAILABLE:
            st.error("❌ Database not available")
            st.info("👈 Please go to **Upload & Data Diagnostics** to load your data")
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            st.stop()
        
        if not st.session_state.get('current_project_id'):
            st.warning("⚠️ No project selected")
            if st.button("← Go to Home"):
                st.switch_page("pages/01_Home.py")
            st.stop()
        
        # Load data from database
        with st.spinner("📊 Loading data from database..."):
            try:
                df_long = db.load_timeseries_data(
                    project_id=st.session_state.current_project_id,
                    data_source='original'
                )
                
                if df_long is None or len(df_long) == 0:
                    st.warning("⚠️ No data found in database")
                    st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
                    
                    with st.expander("🔍 Debug Info"):
                        st.write(f"**Project ID:** {st.session_state.current_project_id}")
                        st.write(f"**Data Source:** original")
                        
                        if DB_AVAILABLE:
                            summary = db.get_timeseries_summary(
                                st.session_state.current_project_id,
                                data_source='original'
                            )
                            st.json(summary)
                    
                    if st.button("📁 Go to Upload Page"):
                        st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                    st.stop()
                
                # Success! Store in session state
                st.session_state.df_long = df_long
                st.session_state.data_loaded = True
                
                variables = sorted(df_long['variable'].unique().tolist())
                st.session_state.value_columns = variables
                
                time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
                st.session_state.time_column = time_col
                
                st.success(f"✅ Loaded {len(df_long):,} data points from database")
                st.info(f"📊 {len(variables)} variables: {', '.join(variables[:5])}{'...' if len(variables) > 5 else ''}")
                
            except Exception as e:
                st.error(f"❌ Error loading data from database: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                
                if st.button("📁 Go to Upload Page"):
                    st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                st.stop()
    
    # ============================================================
    # END NEW CODE - Data is now guaranteed to be loaded
    # ============================================================
    
    df_long = st.session_state.df_long
    variables = sorted(df_long['variable'].unique().tolist())
    
    if st.session_state.df_clean is None:
        st.session_state.df_clean = df_long.copy()
    
    st.success("✅ Data loaded and ready for cleaning")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Total Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    col4.metric("Operations Applied", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
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
                df_modified, new_columns = apply_missing_value_treatment(
                    st.session_state.df_clean.copy(), 
                    apply_to, 
                    treatment_method,
                    suffix
                )
                
                for var in new_columns:
                    st.session_state.cleaned_data[var] = df_modified[df_modified['variable'] == var]
                
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
    
    # Visualization comparison
    if st.session_state.get("last_missing_treatment") or any(
        op['type'] == 'Missing Values' for op in st.session_state.cleaning_history
    ):
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        all_cleaned_vars = []
        all_original_vars = []
        
        for op in st.session_state.cleaning_history:
            if op['type'] == 'Missing Values':
                new_cols = op.get('details', {}).get('new_columns', [])
                orig_vars = op.get('variables', [])
                all_cleaned_vars.extend(new_cols)
                all_original_vars.extend(orig_vars)
        
        if all_cleaned_vars and all_original_vars:
            st.info(f"📊 {len(all_cleaned_vars)} variable(s) cleaned for missing values in this session")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                display_options = [f"{orig} → {clean}" for orig, clean in zip(all_original_vars, all_cleaned_vars)]
                
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(display_options)),
                    format_func=lambda i: display_options[i],
                    key="missing_viz_var_all"
                )
                
                selected_orig = all_original_vars[selected_idx]
                selected_new = all_cleaned_vars[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Cleaned:** {selected_new}")
            
            with col2:
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_orig,
                    selected_new,
                    "Missing Value Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Statistics Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Original: {selected_orig}**")
                orig_stats = df_long[df_long['variable'] == selected_orig]['value'].describe()
                st.dataframe(orig_stats, use_container_width=True)
            
            with col2:
                st.markdown(f"**Cleaned: {selected_new}**")
                if selected_new in st.session_state.df_clean['variable'].values:
                    mod_stats = st.session_state.df_clean[
                        st.session_state.df_clean['variable'] == selected_new
                    ]['value'].describe()
                    st.dataframe(mod_stats, use_container_width=True)


def apply_missing_value_treatment(df, variables, method, suffix):
    """Apply missing value treatment and create NEW columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        cleaned_data = original_data.copy(deep=True)
        cleaned_data['variable'] = new_var_name
        
        if "Drop" in method:
            cleaned_data = cleaned_data.dropna(subset=['value'])
        
        elif "Forward Fill" in method:
            time_col = 'timestamp' if 'timestamp' in cleaned_data.columns else 'time'
            cleaned_data = cleaned_data.sort_values(time_col)
            cleaned_data['value'] = cleaned_data['value'].fillna(method='ffill')
        
        elif "Backward Fill" in method:
            time_col = 'timestamp' if 'timestamp' in cleaned_data.columns else 'time'
            cleaned_data = cleaned_data.sort_values(time_col)
            cleaned_data['value'] = cleaned_data['value'].fillna(method='bfill')
        
        elif "Interpolate" in method:
            time_col = 'timestamp' if 'timestamp' in cleaned_data.columns else 'time'
            cleaned_data = cleaned_data.sort_values(time_col)
            cleaned_data['value'] = cleaned_data['value'].interpolate(method='linear')
        
        elif "Mean" in method:
            mean_val = original_data['value'].mean()
            cleaned_data['value'] = cleaned_data['value'].fillna(mean_val)
        
        elif "Median" in method:
            median_val = original_data['value'].median()
            cleaned_data['value'] = cleaned_data['value'].fillna(median_val)
        
        df_copy = pd.concat([df_copy, cleaned_data], ignore_index=True)
    
    return df_copy, new_columns


def handle_outliers(df_long, variables):
    """Detect and handle outliers"""
    
    st.header("Outlier Detection and Treatment")
    
    st.subheader("📊 Outlier Detection Summary (IQR Method)")
    
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
                df_modified, new_columns = apply_outlier_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to_outliers,
                    outlier_method,
                    outlier_suffix
                )
                
                add_to_cleaning_history(
                    "Outliers",
                    outlier_method,
                    apply_to_outliers,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Applied {outlier_method} to {len(apply_to_outliers)} variable(s)")
                st.info(f"📝 Created new columns: {', '.join(new_columns)}")
                
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
    
    # Visualization comparison (similar to missing values)
    if st.session_state.get("last_outlier_treatment") or any(
        op['type'] == 'Outliers' for op in st.session_state.cleaning_history
    ):
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        all_treated_vars = []
        all_original_vars = []
        
        for op in st.session_state.cleaning_history:
            if op['type'] == 'Outliers':
                new_cols = op.get('details', {}).get('new_columns', [])
                orig_vars = op.get('variables', [])
                all_treated_vars.extend(new_cols)
                all_original_vars.extend(orig_vars)
        
        if all_treated_vars and all_original_vars:
            st.info(f"📊 {len(all_treated_vars)} variable(s) treated for outliers in this session")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                display_options = [f"{orig} → {treat}" for orig, treat in zip(all_original_vars, all_treated_vars)]
                
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(display_options)),
                    format_func=lambda i: display_options[i],
                    key="outlier_viz_var_all"
                )
                
                selected_orig = all_original_vars[selected_idx]
                selected_new = all_treated_vars[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Treated:** {selected_new}")
            
            with col2:
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_orig,
                    selected_new,
                    "Outlier Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_outlier_treatment(df, variables, method, suffix):
    """Apply outlier treatment and create NEW columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        treated_data = original_data.copy(deep=True)
        treated_data['variable'] = new_var_name
        
        var_data = treated_data['value'].dropna()
        
        if len(var_data) > 4:
            Q1 = var_data.quantile(0.25)
            Q3 = var_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            is_outlier = (treated_data['value'] < lower_bound) | (treated_data['value'] > upper_bound)
            
            if "Remove" in method:
                treated_data = treated_data[~is_outlier]
            
            elif "Cap" in method:
                treated_data.loc[treated_data['value'] < lower_bound, 'value'] = lower_bound
                treated_data.loc[treated_data['value'] > upper_bound, 'value'] = upper_bound
            
            elif "Transform" in method:
                min_val = treated_data['value'].min()
                if pd.notna(min_val):
                    if min_val <= 0:
                        treated_data['value'] = np.log1p(
                            treated_data['value'] - min_val + 1
                        )
                    else:
                        treated_data['value'] = np.log(treated_data['value'])
            
            elif "Winsorize" in method:
                p05 = var_data.quantile(0.05)
                p95 = var_data.quantile(0.95)
                treated_data.loc[treated_data['value'] < p05, 'value'] = p05
                treated_data.loc[treated_data['value'] > p95, 'value'] = p95
        
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
                df_modified, new_columns = apply_transformation(
                    st.session_state.df_clean.copy(),
                    transform_vars,
                    transformation,
                    suffix
                )
                
                st.session_state.transformed_columns.extend(new_columns)
                
                add_to_cleaning_history(
                    "Transformation",
                    transformation,
                    transform_vars,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Created {len(new_columns)} new transformed column(s)")
                st.info(f"📝 New columns: {', '.join(new_columns)}")
                
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
    
    # Visualization comparison (similar pattern)
    if st.session_state.get("last_transformation") or any(
        op['type'] == 'Transformation' for op in st.session_state.cleaning_history
    ):
        st.markdown("---")
        st.subheader("📈 Original vs Transformed")
        
        all_transformed_vars = []
        all_original_vars = []
        
        for op in st.session_state.cleaning_history:
            if op['type'] == 'Transformation':
                new_cols = op.get('details', {}).get('new_columns', [])
                orig_vars = op.get('variables', [])
                all_transformed_vars.extend(new_cols)
                all_original_vars.extend(orig_vars)
        
        if all_transformed_vars and all_original_vars:
            st.info(f"📊 {len(all_transformed_vars)} variable(s) transformed in this session")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                display_options = [f"{orig} → {trans}" for orig, trans in zip(all_original_vars, all_transformed_vars)]
                
                selected_idx = st.selectbox(
                    "Select variable to visualize",
                    range(len(display_options)),
                    format_func=lambda i: display_options[i],
                    key="transform_viz_var_all"
                )
                
                selected_orig = all_original_vars[selected_idx]
                selected_new = all_transformed_vars[selected_idx]
                
                st.caption(f"**Original:** {selected_orig}")
                st.caption(f"**Transformed:** {selected_new}")
            
            with col2:
                fig = plot_comparison(
                    df_long,
                    st.session_state.df_clean,
                    selected_orig,
                    selected_new,
                    "Transformation"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_transformation(df, variables, transformation, suffix):
    """Apply transformation and create new columns"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        transformed_data = original_data.copy(deep=True)
        transformed_data['variable'] = new_var_name
        
        values = transformed_data['value'].dropna()
        
        if "Log -" in transformation:
            min_val = values.min()
            if pd.notna(min_val):
                if min_val <= 0:
                    shift_amount = abs(min_val) + 1
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.log1p(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
                else:
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.log(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Log10" in transformation:
            min_val = values.min()
            if pd.notna(min_val):
                if min_val <= 0:
                    shift_amount = abs(min_val) + 1
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
                else:
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Square Root" in transformation:
            min_val = values.min()
            if pd.notna(min_val):
                if min_val < 0:
                    shift_amount = abs(min_val)
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.sqrt(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
                else:
                    transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                        np.sqrt(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        
        elif "Standardize" in transformation:
            mean = values.mean()
            std = values.std()
            if pd.notna(mean) and pd.notna(std) and std > 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    (transformed_data.loc[transformed_data['value'].notna(), 'value'] - mean) / std
        
        elif "Min-Max" in transformation:
            min_val = values.min()
            max_val = values.max()
            if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    (transformed_data.loc[transformed_data['value'].notna(), 'value'] - min_val) / (max_val - min_val)
        
        elif "Difference" in transformation:
            time_col = 'timestamp' if 'timestamp' in transformed_data.columns else 'time'
            transformed_data = transformed_data.sort_values(time_col)
            transformed_data['value'] = transformed_data['value'].diff()
        
        elif "Percentage Change" in transformation:
            time_col = 'timestamp' if 'timestamp' in transformed_data.columns else 'time'
            transformed_data = transformed_data.sort_values(time_col)
            transformed_data['value'] = transformed_data['value'].pct_change() * 100
        
        df_copy = pd.concat([df_copy, transformed_data], ignore_index=True)
    
    return df_copy, new_columns


def save_cleaned_data_to_database():
    """
    🆕 NEW FUNCTION: Save cleaned data and update ALL progress indicators
    - Saves cleaned data to database (appends, doesn't overwrite)
    - Updates parameters table with cleaned variables
    - Updates workflow progress in database
    - Updates session state for immediate UI refresh
    - Turns 2nd workflow dot green
    """
    
    with st.spinner("💾 Saving cleaned data to database..."):
        try:
            project_id = st.session_state.current_project_id
            
            # ========================================================
            # STEP 1: Save cleaned data to timeseries_data (APPEND)
            # ========================================================
            st.info("📊 Step 1/4: Saving cleaned data to database...")
            
            success = db.save_timeseries_data(
                project_id=project_id,
                df_long=st.session_state.df_clean,
                data_source='cleaned',  # Separate from 'original'
                batch_size=1000
            )
            
            if not success:
                st.error("❌ Failed to save cleaned data")
                return
            
            summary = db.get_timeseries_summary(project_id, data_source='cleaned')
            st.success(f"✅ Saved {summary['total_records']:,} cleaned data points!")
            
            # ========================================================
            # STEP 2: Update parameters table with cleaned variables
            # ========================================================
            st.info("📊 Step 2/4: Updating parameter statistics...")
            
            all_vars = st.session_state.df_clean['variable'].unique()
            original_vars = st.session_state.df_long['variable'].unique()
            new_vars = [v for v in all_vars if v not in original_vars]
            
            cleaned_params = []
            for var in new_vars:
                var_data = st.session_state.df_clean[
                    st.session_state.df_clean['variable'] == var
                ]['value'].dropna()
                
                if len(var_data) > 0:
                    cleaned_params.append({
                        'name': var,
                        'data_type': 'numeric',
                        'min_value': float(var_data.min()),
                        'max_value': float(var_data.max()),
                        'mean_value': float(var_data.mean()),
                        'std_value': float(var_data.std()),
                        'missing_count': int(st.session_state.df_clean[
                            st.session_state.df_clean['variable'] == var
                        ]['value'].isna().sum()),
                        'total_count': len(st.session_state.df_clean[
                            st.session_state.df_clean['variable'] == var
                        ])
                    })
            
            if cleaned_params:
                db.save_parameters(project_id, cleaned_params)
                st.success(f"✅ Updated {len(cleaned_params)} cleaned parameters")
            
            # ========================================================
            # STEP 3: Mark preprocessing step complete in database
            # ========================================================
            st.info("📊 Step 3/4: Updating workflow progress...")
            
            # Update step completion (turns 2nd dot green!)
            db.update_step_completion(
                project_id=project_id,
                step_key='data_cleaned',
                completed=True
            )
            
            # Update project progress
            db.update_project_progress(
                project_id=project_id,
                workflow_state="preprocessing_complete",
                current_page=3,
                completion_percentage=15
            )
            
            st.success("✅ Workflow progress updated in database")
            
            # ========================================================
            # STEP 4: Update session state for immediate UI update
            # ========================================================
            st.info("📊 Step 4/4: Updating session state...")
            
            st.session_state.preprocessing_applied = True
            st.session_state.data_cleaned = True  # This triggers green dot!
            
            st.success("✅ Session state updated")
            
            # ========================================================
            # FINAL: Show success and balloons!
            # ========================================================
            st.markdown("---")
            st.success("🎉 **All done! Cleaned data saved successfully!**")
            st.balloons()
            
            st.info(f"""
            **✅ Completed:**
            - Saved {summary['total_records']:,} data points (source='cleaned')
            - Updated {len(cleaned_params)} parameter statistics
            - Marked preprocessing step as complete
            - Updated workflow progress to 15%
            - 2nd workflow dot turned green ✨
            """)
            
            # Offer to continue to next page
            st.markdown("---")
            st.markdown("### 🎯 Ready for Next Step!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Visualize Data", use_container_width=True, type="primary"):
                    st.switch_page("pages/04_Exploration_and_Visualization.py")
            
            with col2:
                if st.button("🔬 Analyze Relationships", use_container_width=True):
                    st.switch_page("pages/05_Understand_The_System_(Dimensionality_Reduction).py")
            
            with col3:
                if st.button("🏠 Back to Home", use_container_width=True):
                    st.switch_page("pages/01_Home.py")
            
            # Force rerun to update sidebar
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error saving cleaned data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


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
    
    modified_vars = set()
    for op in st.session_state.cleaning_history:
        modified_vars.update(op['variables'])
    col2.metric("Variables Modified", len(modified_vars))
    
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
    
    # ============================================================
    # 🆕 NEW: SAVE CLEANED DATA TO DATABASE & UPDATE PROGRESS
    # ============================================================
    st.subheader("💾 Save Cleaned Data to Database")
    
    if st.session_state.df_clean is not None and DB_AVAILABLE:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **What gets saved:**
            - ✅ Cleaned data APPENDED to database (original preserved)
            - ✅ Data source: 'cleaned'
            - ✅ All new columns with suffixes (_cleaned, _outlier_treated, etc.)
            - ✅ Progress updated in database & sidebar
            - ✅ Workflow step marked complete (2nd dot turns green) ✨
            
            **Important:** Original data remains unchanged (source='original')
            """)
            
            save_notes = st.text_area(
                "Add cleaning notes (optional)",
                placeholder="e.g., Removed outliers from Temperature, filled missing values...",
                key="cleaned_data_notes"
            )
        
        with col2:
            st.write("")
            st.write("")
            
            # Check if cleaned data already exists
            try:
                summary = db.get_timeseries_summary(
                    st.session_state.current_project_id,
                    data_source='cleaned'
                )
                
                if summary['total_records'] > 0:
                    st.warning(f"⚠️ Cleaned data exists ({summary['total_records']:,} records)")
                    st.caption("Saving will replace existing cleaned data")
            except:
                pass
            
            if st.button("💾 Save Cleaned Data", 
                        type="primary", 
                        use_container_width=True,
                        key="save_cleaned_to_db"):
                # Call our new save function
                save_cleaned_data_to_database()
    
    # ============================================================
    # END NEW CODE
    # ============================================================
    
    st.markdown("---")
    
    # Download option
    st.subheader("📥 Download Enhanced Data")
    
    if st.session_state.df_clean is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("💡 Download includes all variables (original + cleaned)")
        
        with col2:
            if st.button("📥 Download CSV", use_container_width=True, key="download_cleaned"):
                with st.spinner("Preparing download..."):
                    df_wide = convert_to_wide_format(st.session_state.df_clean)
                    
                    if df_wide is not None:
                        csv = df_wide.to_csv(index=False)
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
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("**🔬 Analyze**")
        st.markdown("Run PCA and analysis")
        if st.button("Go to Analysis", key="goto_analysis", use_container_width=True):
            st.switch_page("pages/05_Understand_The_System_(Dimensionality_Reduction).py")
    
    with col3:
        st.markdown("**🔄 Clear History**")
        st.markdown("Start cleaning over")
        if st.button("Clear Operations", key="clear_history", use_container_width=True):
            st.session_state.cleaning_history = []
            st.session_state.cleaned_data = {}
            st.session_state.transformed_columns = []
            st.session_state.df_clean = st.session_state.df_long.copy()
            st.session_state.preprocessing_applied = False
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
