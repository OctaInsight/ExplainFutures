"""
Page 3: Data Cleaning and Preprocessing - FIXED
Fixed: None check before accessing df_long
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
    """Initialize session state for tracking cleaning operations"""
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = {}
    if "transformed_columns" not in st.session_state:
        st.session_state.transformed_columns = []
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    if "initial_data_hash" not in st.session_state:
        st.session_state.initial_data_hash = None
    if "has_unsaved_changes" not in st.session_state:
        st.session_state.has_unsaved_changes = False


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
    st.session_state.has_unsaved_changes = True


def load_data_from_database():
    """Load ALL raw data from database into session state"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return False
    
    try:
        df_long = db.load_timeseries_data(
            project_id=st.session_state.current_project_id,
            data_source='raw'
        )
        
        if df_long is None or len(df_long) == 0:
            return False
        
        st.session_state.df_long = df_long
        st.session_state.df_clean = df_long.copy()
        st.session_state.data_loaded = True
        
        # Convert to list properly
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


def check_for_unsaved_changes():
    """Check if there are unsaved changes"""
    if not st.session_state.get('has_unsaved_changes', False):
        return False
    
    current_hash = calculate_data_hash(st.session_state.get('df_clean'))
    initial_hash = st.session_state.get('initial_data_hash')
    
    return current_hash != initial_hash


def show_unsaved_changes_warning():
    """Show warning about unsaved changes"""
    st.warning("⚠️ **You have unsaved changes!**")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            save_cleaned_data_to_database()
            st.session_state.has_unsaved_changes = False
            return True
    
    with col2:
        if st.button("❌ Discard Changes", use_container_width=True):
            st.session_state.has_unsaved_changes = False
            return True
    
    with col3:
        if st.button("↩️ Keep Working", use_container_width=True):
            return False
    
    return None


def plot_comparison(df_original, df_modified, original_variable, new_variable, operation_type):
    """Create a comparison plot between original and modified data"""
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
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        return None


def main():
    """Main page function"""
    
    initialize_cleaning_history()
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # ============================================================
    # CRITICAL: Load data first, check success before continuing
    # ============================================================
    if not st.session_state.get('data_loaded'):
        if not DB_AVAILABLE:
            st.error("❌ Database not available")
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            st.stop()
        
        with st.spinner("📊 Loading data from database..."):
            success = load_data_from_database()
            
            if not success:
                st.warning("⚠️ No data found in database")
                st.info("👈 Please go to **Upload & Data Diagnostics** to upload data first")
                
                if st.button("📁 Go to Upload Page"):
                    st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                st.stop()
            
            st.success(f"✅ Loaded {len(st.session_state.df_long):,} data points")
            st.info(f"📊 {len(st.session_state.value_columns)} variables")
    
    # ============================================================
    # NOW we know data is loaded, safe to access
    # ============================================================
    
    # FIXED: Check if df_long exists before accessing
    if st.session_state.df_long is None:
        st.error("❌ Data not loaded properly")
        if st.button("🔄 Reload Page"):
            st.session_state.data_loaded = False
            st.rerun()
        st.stop()
    
    df_long = st.session_state.df_long
    
    # Get variables from session state (already loaded)
    variables = st.session_state.get('value_columns', [])
    
    if not variables:
        # Fallback: extract from df_long
        variables = list(df_long['variable'].unique())
        variables.sort()
        st.session_state.value_columns = variables
    
    # Show unsaved changes warning
    if check_for_unsaved_changes():
        st.markdown("---")
        result = show_unsaved_changes_warning()
        if result is True:
            st.rerun()
        st.markdown("---")
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Total Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    col4.metric("Operations Applied", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # Create tabs
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
            key="missing_suffix"
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
                
                add_to_cleaning_history(
                    "Missing Values",
                    treatment_method,
                    apply_to,
                    {"new_columns": new_columns}
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
    
    # Show comparison
    if st.session_state.get("last_missing_treatment"):
        st.markdown("---")
        st.subheader("📈 Before/After Comparison")
        
        treatment_info = st.session_state.last_missing_treatment
        new_cols = treatment_info.get('new_columns', [])
        orig_vars = treatment_info.get('variables', [])
        
        if new_cols and orig_vars:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                display_options = [f"{o} → {n}" for o, n in zip(orig_vars, new_cols)]
                selected_idx = st.selectbox(
                    "Select variable",
                    range(len(display_options)),
                    format_func=lambda i: display_options[i],
                    key="missing_viz_var"
                )
                
                selected_orig = orig_vars[selected_idx]
                selected_new = new_cols[selected_idx]
            
            with col2:
                fig = plot_comparison(
                    st.session_state.df_long,
                    st.session_state.df_clean,
                    selected_orig,
                    selected_new,
                    "Missing Value Treatment"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


def apply_missing_value_treatment(df, variables, method, suffix):
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
    st.info("Outlier detection using IQR method")


def handle_transformations(df_long, variables):
    """Apply data transformations"""
    st.header("Data Transformations")
    st.info("Apply mathematical transformations to your data")


def save_cleaned_data_to_database():
    """Save cleaned data to database"""
    
    with st.spinner("💾 Saving..."):
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
            st.session_state.preprocessing_applied = True
            st.session_state.data_cleaned = True
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
            
            st.success("🎉 Saved successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_cleaning_summary():
    """Show summary and save"""
    
    st.header("📋 Cleaning Summary & Save")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations yet")
        return
    
    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Operations", len(st.session_state.cleaning_history))
    
    modified_vars = set()
    for op in st.session_state.cleaning_history:
        modified_vars.update(op['variables'])
    col2.metric("Variables Modified", len(modified_vars))
    
    total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                    for op in st.session_state.cleaning_history)
    col3.metric("New Columns", total_new)
    
    st.markdown("---")
    
    # Save button
    if DB_AVAILABLE:
        if st.button("💾 Save to Database", type="primary", use_container_width=True):
            save_cleaned_data_to_database()


if __name__ == "__main__":
    main()
