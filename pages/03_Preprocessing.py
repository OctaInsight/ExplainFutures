"""
Page 3: Data Cleaning - COMPLETE WITH ALL FEATURES
✅ Missing Values Treatment
✅ Outlier Detection & Treatment
✅ Data Transformations
✅ Before/After Comparisons
✅ Save to Database
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import time
import hashlib

st.set_page_config(
    page_title="Data Cleaning and Preprocessing",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

try:
    from core.database.supabase_manager import get_db_manager
    DB_AVAILABLE = True
    db = get_db_manager()
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database import error: {str(e)}")

initialize_session_state()
config = get_config()
render_app_sidebar()

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("*Clean and prepare your data for analysis*")
st.markdown("---")

if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()


def calculate_data_hash(df):
    """Calculate a hash of the dataframe to detect changes"""
    if df is None or len(df) == 0:
        return None
    try:
        data_str = df.to_json()
        return hashlib.md5(data_str.encode()).hexdigest()
    except:
        return None


def initialize_cleaning_history():
    """Initialize session state"""
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = {}
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    if "initial_data_hash" not in st.session_state:
        st.session_state.initial_data_hash = None
    if "has_unsaved_changes" not in st.session_state:
        st.session_state.has_unsaved_changes = False


def add_to_cleaning_history(operation_type, method, variables, details=None):
    """Add operation to history"""
    operation = {
        "timestamp": datetime.now(),
        "type": operation_type,
        "method": method,
        "variables": variables,
        "details": details or {}
    }
    st.session_state.cleaning_history.append(operation)
    st.session_state.has_unsaved_changes = True


def load_data_from_database():
    """Load data from database"""
    
    if not DB_AVAILABLE:
        return False
    
    project_id = st.session_state.get('current_project_id')
    
    if not project_id:
        return False
    
    try:
        df_long = db.load_timeseries_data(
            project_id=project_id,
            data_source='raw'
        )
        
        if df_long is None or len(df_long) == 0:
            return False
        
        st.session_state.df_long = df_long
        st.session_state.df_clean = df_long.copy()
        st.session_state.data_loaded = True
        
        variables = list(df_long['variable'].unique())
        variables.sort()
        st.session_state.value_columns = variables
        
        time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
        st.session_state.time_column = time_col
        
        st.session_state.initial_data_hash = calculate_data_hash(df_long)
        st.session_state.has_unsaved_changes = False
        
        return True
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


def plot_comparison(df_original, df_modified, original_variable, new_variable, operation_type):
    """Create comparison plot"""
    try:
        time_col = 'timestamp' if 'timestamp' in df_original.columns else 'time'
        
        orig_data = df_original[df_original['variable'] == original_variable].copy()
        mod_data = df_modified[df_modified['variable'] == new_variable].copy()
        
        if len(orig_data) == 0 or len(mod_data) == 0:
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
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=mod_data[time_col],
            y=mod_data['value'],
            mode='lines+markers',
            name=f'Cleaned ({new_variable})',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"Before vs After: {original_variable} → {new_variable}<br><sub>{operation_type}</sub>",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
        
    except:
        return None


def main():
    """Main function"""
    
    initialize_cleaning_history()
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Check if df_long actually exists AND is not None
    needs_loading = (
        not st.session_state.get('data_loaded') or 
        st.session_state.get('df_long') is None
    )
    
    if needs_loading:
        if not DB_AVAILABLE:
            st.error("❌ Database not available")
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            st.stop()
        
        with st.spinner("📊 Loading data from database..."):
            success = load_data_from_database()
            
            if not success:
                st.warning("⚠️ No data found in database")
                st.info("Please upload data first in **Upload & Data Diagnostics** page")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📁 Go to Upload Page", use_container_width=True):
                        st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                
                with col2:
                    if st.button("🔄 Retry", use_container_width=True):
                        st.session_state.data_loaded = False
                        st.session_state.df_long = None
                        st.rerun()
                
                st.stop()
            
            st.success(f"✅ Loaded {len(st.session_state.df_long):,} data points")
    
    df_long = st.session_state.df_long
    variables = st.session_state.get('value_columns', [])
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    col4.metric("Operations", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # Tabs
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
        show_summary()


def handle_missing_values(df_long, variables):
    """Handle missing values"""
    
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
    
    st.dataframe(pd.DataFrame(missing_summary), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("⚙️ Select Treatment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Treatment method",
            [
                "None - Keep as is",
                "Drop - Remove rows",
                "Forward Fill - Use previous value",
                "Backward Fill - Use next value",
                "Interpolate - Linear",
                "Mean - Replace with mean",
                "Median - Replace with median"
            ],
            key="missing_method"
        )
    
    with col2:
        apply_to = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="missing_vars"
        )
    
    if method != "None - Keep as is":
        suffix = st.text_input("Suffix for cleaned columns", value="_cleaned", key="missing_suffix")
    else:
        suffix = "_cleaned"
    
    if st.button("✨ Apply Treatment", type="primary", key="apply_missing_btn"):
        if method != "None - Keep as is" and apply_to:
            with st.spinner("Applying treatment..."):
                df_modified, new_cols = apply_missing_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to,
                    method,
                    suffix
                )
                
                add_to_cleaning_history("Missing Values", method, apply_to, {"new_columns": new_cols})
                
                st.session_state.df_clean = df_modified
                st.session_state.last_treatment = {
                    'type': 'missing',
                    'method': method,
                    'variables': apply_to,
                    'new_columns': new_cols
                }
                
                st.success(f"✅ Applied {method} to {len(apply_to)} variable(s)")
                st.info(f"📝 Created: {', '.join(new_cols)}")
                st.rerun()
        else:
            st.warning("⚠️ Please select a method and variables")
    
    # Show comparison
    if st.session_state.get('last_treatment', {}).get('type') == 'missing':
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        treatment = st.session_state.last_treatment
        new_cols = treatment.get('new_columns', [])
        orig_vars = treatment.get('variables', [])
        
        if new_cols and orig_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                options = [f"{o} → {n}" for o, n in zip(orig_vars, new_cols)]
                idx = st.selectbox("Select", range(len(options)), format_func=lambda i: options[i], key="missing_compare")
                
                orig = orig_vars[idx]
                cleaned = new_cols[idx]
            
            with col2:
                fig = plot_comparison(
                    st.session_state.df_long,
                    st.session_state.df_clean,
                    orig,
                    cleaned,
                    "Missing Value Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_missing_treatment(df, variables, method, suffix):
    """Apply missing value treatment"""
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
        st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("⚙️ Select Treatment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Outlier treatment",
            [
                "None - Keep as is",
                "Remove - Delete outlier rows",
                "Cap - Clip to bounds",
                "Transform - Log transformation",
                "Winsorize - Replace with percentile values"
            ],
            key="outlier_method"
        )
    
    with col2:
        apply_to = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="outlier_vars"
        )
    
    if method != "None - Keep as is":
        suffix = st.text_input("Suffix for treated columns", value="_outlier_treated", key="outlier_suffix")
    else:
        suffix = "_outlier_treated"
    
    if st.button("✨ Apply Outlier Treatment", type="primary", key="apply_outlier_btn"):
        if method != "None - Keep as is" and apply_to:
            with st.spinner("Applying treatment..."):
                df_modified, new_cols = apply_outlier_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to,
                    method,
                    suffix
                )
                
                add_to_cleaning_history("Outliers", method, apply_to, {"new_columns": new_cols})
                
                st.session_state.df_clean = df_modified
                st.session_state.last_treatment = {
                    'type': 'outlier',
                    'method': method,
                    'variables': apply_to,
                    'new_columns': new_cols
                }
                
                st.success(f"✅ Applied {method} to {len(apply_to)} variable(s)")
                st.info(f"📝 Created: {', '.join(new_cols)}")
                st.rerun()
    
    # Show comparison
    if st.session_state.get('last_treatment', {}).get('type') == 'outlier':
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        treatment = st.session_state.last_treatment
        new_cols = treatment.get('new_columns', [])
        orig_vars = treatment.get('variables', [])
        
        if new_cols and orig_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                options = [f"{o} → {n}" for o, n in zip(orig_vars, new_cols)]
                idx = st.selectbox("Select", range(len(options)), format_func=lambda i: options[i], key="outlier_compare")
                
                orig = orig_vars[idx]
                cleaned = new_cols[idx]
            
            with col2:
                fig = plot_comparison(
                    st.session_state.df_long,
                    st.session_state.df_clean,
                    orig,
                    cleaned,
                    "Outlier Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_outlier_treatment(df, variables, method, suffix):
    """Apply outlier treatment"""
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
                        treated_data['value'] = np.log1p(treated_data['value'] - min_val + 1)
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
                "Min-Max - Scale to 0-1"
            ],
            key="transform_type"
        )
    
    with col2:
        apply_to = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="transform_vars"
        )
    
    if transformation != "None":
        suffix = st.text_input(
            "Suffix for transformed columns",
            value=f"_{transformation.split('-')[0].strip().lower()}",
            key="transform_suffix"
        )
    else:
        suffix = "_transformed"
    
    if st.button("✨ Apply Transformation", type="primary", key="apply_transform_btn"):
        if transformation != "None" and apply_to:
            with st.spinner("Applying transformation..."):
                df_modified, new_cols = apply_transformation(
                    st.session_state.df_clean.copy(),
                    apply_to,
                    transformation,
                    suffix
                )
                
                add_to_cleaning_history("Transformation", transformation, apply_to, {"new_columns": new_cols})
                
                st.session_state.df_clean = df_modified
                st.session_state.last_treatment = {
                    'type': 'transform',
                    'method': transformation,
                    'variables': apply_to,
                    'new_columns': new_cols
                }
                
                st.success(f"✅ Created {len(new_cols)} transformed column(s)")
                st.info(f"📝 New columns: {', '.join(new_cols)}")
                st.rerun()
    
    # Show comparison
    if st.session_state.get('last_treatment', {}).get('type') == 'transform':
        st.markdown("---")
        st.subheader("📈 Original vs Transformed")
        
        treatment = st.session_state.last_treatment
        new_cols = treatment.get('new_columns', [])
        orig_vars = treatment.get('variables', [])
        
        if new_cols and orig_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                options = [f"{o} → {n}" for o, n in zip(orig_vars, new_cols)]
                idx = st.selectbox("Select", range(len(options)), format_func=lambda i: options[i], key="transform_compare")
                
                orig = orig_vars[idx]
                transformed = new_cols[idx]
            
            with col2:
                fig = plot_comparison(
                    st.session_state.df_long,
                    st.session_state.df_clean,
                    orig,
                    transformed,
                    "Data Transformation"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_transformation(df, variables, transformation, suffix):
    """Apply transformation"""
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
            if pd.notna(min_val) and min_val <= 0:
                shift_amount = abs(min_val) + 1
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log1p(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        elif "Log10" in transformation:
            min_val = values.min()
            if pd.notna(min_val) and min_val <= 0:
                shift_amount = abs(min_val) + 1
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log10(transformed_data.loc[transformed_data['value'].notna(), 'value'])
        elif "Square Root" in transformation:
            min_val = values.min()
            if pd.notna(min_val) and min_val < 0:
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
        
        df_copy = pd.concat([df_copy, transformed_data], ignore_index=True)
    
    return df_copy, new_columns


def show_summary():
    """Show summary and save"""
    
    st.header("📋 Cleaning Summary & Save")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations performed yet")
        st.markdown("Apply cleaning operations in the tabs above, then return here to save.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Operations", len(st.session_state.cleaning_history))
    
    modified_vars = set()
    for op in st.session_state.cleaning_history:
        modified_vars.update(op['variables'])
    col2.metric("Variables Modified", len(modified_vars))
    
    total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                    for op in st.session_state.cleaning_history)
    col3.metric("New Columns Created", total_new)
    
    st.markdown("---")
    
    # Operation log
    st.subheader("📜 Operation Log")
    
    ops_data = []
    for i, op in enumerate(st.session_state.cleaning_history, 1):
        new_cols = op.get('details', {}).get('new_columns', [])
        ops_data.append({
            "#": i,
            "Type": op['type'],
            "Method": op['method'],
            "Variables": ", ".join(op['variables'][:2]) + ("..." if len(op['variables']) > 2 else ""),
            "New Columns": ", ".join(new_cols[:2]) + ("..." if len(new_cols) > 2 else "")
        })
    
    st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Save button
    st.subheader("💾 Save to Database")
    
    st.info("""
    **What will be saved:**
    - ✅ Only NEW cleaned variables (with suffixes)
    - ✅ Appended to database (source='cleaned')
    - ✅ Raw data unchanged (source='raw')
    - ✅ Progress updated (2nd dot turns green)
    """)
    
    if DB_AVAILABLE:
        if st.button("💾 Save Cleaned Data", type="primary", use_container_width=True):
            save_to_database()


def save_to_database():
    """Save cleaned data to database"""
    
    with st.spinner("💾 Saving to database..."):
        try:
            project_id = st.session_state.current_project_id
            
            # Get only new cleaned variables
            all_vars = st.session_state.df_clean['variable'].unique()
            original_vars = st.session_state.df_long['variable'].unique()
            new_cleaned_vars = [v for v in all_vars if v not in original_vars]
            
            if not new_cleaned_vars:
                st.warning("⚠️ No cleaned variables to save")
                return
            
            # Extract only cleaned data
            cleaned_df = st.session_state.df_clean[
                st.session_state.df_clean['variable'].isin(new_cleaned_vars)
            ].copy()
            
            st.info(f"Saving {len(cleaned_df):,} records for {len(new_cleaned_vars)} variables...")
            
            # Save to database
            success = db.save_timeseries_data(
                project_id=project_id,
                df_long=cleaned_df,
                data_source='cleaned',
                batch_size=1000
            )
            
            if not success:
                st.error("❌ Failed to save")
                return
            
            # Update parameters
            cleaned_params = []
            for var in new_cleaned_vars:
                var_data = cleaned_df[cleaned_df['variable'] == var]['value'].dropna()
                
                if len(var_data) > 0:
                    cleaned_params.append({
                        'name': var,
                        'data_type': 'numeric',
                        'min_value': float(var_data.min()),
                        'max_value': float(var_data.max()),
                        'mean_value': float(var_data.mean()),
                        'std_value': float(var_data.std()),
                        'missing_count': int(cleaned_df[cleaned_df['variable'] == var]['value'].isna().sum()),
                        'total_count': len(cleaned_df[cleaned_df['variable'] == var])
                    })
            
            if cleaned_params:
                db.save_parameters(project_id, cleaned_params)
            
            # Update progress
            db.update_step_completion(project_id, 'data_cleaned', True)
            db.update_project_progress(
                project_id=project_id,
                workflow_state="preprocessing_complete",
                current_page=3,
                completion_percentage=15
            )
            
            # Update session
            st.session_state.data_cleaned = True
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
            
            st.success("🎉 Successfully saved to database!")
            st.balloons()
            
            st.info(f"""
            **✅ Saved:**
            - {len(cleaned_df):,} cleaned data points
            - {len(cleaned_params)} parameters updated
            - Progress: 15%
            - 2nd workflow dot: GREEN ✨
            """)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
