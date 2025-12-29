"""
Page 3: Data Cleaning - WITH DEBUG INFO
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
    """Load data from database with debug info"""
    
    st.info("🔍 DEBUG: Starting data load...")
    
    if not DB_AVAILABLE:
        st.error("❌ DEBUG: Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    st.info(f"🔍 DEBUG: Project ID: {project_id}")
    
    if not project_id:
        st.error("❌ DEBUG: No project ID")
        return False
    
    try:
        st.info("🔍 DEBUG: Calling db.load_timeseries_data...")
        
        df_long = db.load_timeseries_data(
            project_id=project_id,
            data_source='raw'
        )
        
        st.info(f"🔍 DEBUG: Returned data type: {type(df_long)}")
        
        if df_long is None:
            st.error("❌ DEBUG: df_long is None")
            return False
        
        st.info(f"🔍 DEBUG: Data length: {len(df_long)}")
        
        if len(df_long) == 0:
            st.error("❌ DEBUG: df_long is empty")
            return False
        
        st.info(f"🔍 DEBUG: Columns: {df_long.columns.tolist()}")
        
        # Store in session
        st.session_state.df_long = df_long
        st.session_state.df_clean = df_long.copy()
        st.session_state.data_loaded = True
        
        # Get variables
        variables = list(df_long['variable'].unique())
        variables.sort()
        st.session_state.value_columns = variables
        
        st.info(f"🔍 DEBUG: Variables: {variables}")
        
        time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
        st.session_state.time_column = time_col
        
        st.session_state.initial_data_hash = calculate_data_hash(df_long)
        st.session_state.has_unsaved_changes = False
        
        st.success("✅ DEBUG: Data loaded successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ DEBUG: Exception: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def main():
    """Main function"""
    
    initialize_cleaning_history()
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Show current state
    with st.expander("🔍 Debug Info - Current State"):
        st.write("**Session State:**")
        st.write(f"- data_loaded: {st.session_state.get('data_loaded', False)}")
        st.write(f"- current_project_id: {st.session_state.get('current_project_id', 'None')}")
        st.write(f"- df_long exists: {'df_long' in st.session_state}")
        if 'df_long' in st.session_state:
            st.write(f"- df_long type: {type(st.session_state.df_long)}")
            if st.session_state.df_long is not None:
                st.write(f"- df_long length: {len(st.session_state.df_long)}")
    
    # Load data
    if not st.session_state.get('data_loaded'):
        st.info("📊 Loading data from database...")
        
        success = load_data_from_database()
        
        if not success:
            st.error("❌ Failed to load data")
            st.info("Please check:")
            st.write("1. Have you uploaded data in Page 02?")
            st.write("2. Is the data saved with data_source='raw'?")
            st.write("3. Is the project ID correct?")
            
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            
            if st.button("🔄 Try Again"):
                st.session_state.data_loaded = False
                st.rerun()
            
            st.stop()
    
    # Check data loaded properly
    if st.session_state.df_long is None:
        st.error("❌ Data is None after loading")
        
        if st.button("🔄 Reload"):
            st.session_state.data_loaded = False
            st.rerun()
        
        st.stop()
    
    df_long = st.session_state.df_long
    variables = st.session_state.get('value_columns', [])
    
    st.success(f"✅ Data loaded: {len(df_long):,} points, {len(variables)} variables")
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables", len(variables))
    col2.metric("Data Points", len(df_long))
    col3.metric("Missing Values", df_long['value'].isna().sum())
    col4.metric("Operations", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Missing Values", "📋 Summary"])
    
    with tab1:
        handle_missing_values(df_long, variables)
    
    with tab2:
        show_summary()


def handle_missing_values(df_long, variables):
    """Handle missing values"""
    
    st.header("Missing Values Treatment")
    st.subheader("📊 Summary")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Treatment method",
            [
                "None",
                "Forward Fill",
                "Interpolate",
                "Mean",
                "Median"
            ]
        )
    
    with col2:
        apply_to = st.multiselect(
            "Apply to variables",
            variables,
            default=[]
        )
    
    if method != "None":
        suffix = st.text_input("Suffix", value="_cleaned")
    else:
        suffix = "_cleaned"
    
    if st.button("✨ Apply", type="primary"):
        if method != "None" and apply_to:
            with st.spinner("Applying..."):
                df_modified, new_cols = apply_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to,
                    method,
                    suffix
                )
                
                add_to_cleaning_history("Missing Values", method, apply_to, {"new_columns": new_cols})
                
                st.session_state.df_clean = df_modified
                st.success(f"✅ Created: {', '.join(new_cols)}")
                st.rerun()


def apply_treatment(df, variables, method, suffix):
    """Apply treatment"""
    df_copy = df.copy()
    new_columns = []
    
    for var in variables:
        new_var_name = f"{var}{suffix}"
        new_columns.append(new_var_name)
        
        mask = df_copy['variable'] == var
        original_data = df_copy[mask].copy()
        
        cleaned_data = original_data.copy(deep=True)
        cleaned_data['variable'] = new_var_name
        
        if "Forward Fill" in method:
            time_col = 'timestamp' if 'timestamp' in cleaned_data.columns else 'time'
            cleaned_data = cleaned_data.sort_values(time_col)
            cleaned_data['value'] = cleaned_data['value'].fillna(method='ffill')
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


def show_summary():
    """Show summary"""
    
    st.header("📋 Summary")
    
    if not st.session_state.cleaning_history:
        st.info("No operations yet")
        return
    
    col1, col2 = st.columns(2)
    col1.metric("Operations", len(st.session_state.cleaning_history))
    
    total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                    for op in st.session_state.cleaning_history)
    col2.metric("New Columns", total_new)
    
    st.markdown("---")
    
    if DB_AVAILABLE and st.button("💾 Save to Database", type="primary"):
        save_to_database()


def save_to_database():
    """Save to database"""
    try:
        project_id = st.session_state.current_project_id
        
        all_vars = st.session_state.df_clean['variable'].unique()
        original_vars = st.session_state.df_long['variable'].unique()
        new_cleaned_vars = [v for v in all_vars if v not in original_vars]
        
        if not new_cleaned_vars:
            st.warning("No cleaned variables")
            return
        
        cleaned_df = st.session_state.df_clean[
            st.session_state.df_clean['variable'].isin(new_cleaned_vars)
        ].copy()
        
        success = db.save_timeseries_data(
            project_id=project_id,
            df_long=cleaned_df,
            data_source='cleaned',
            batch_size=1000
        )
        
        if success:
            db.update_step_completion(project_id, 'data_cleaned', True)
            db.update_project_progress(
                project_id=project_id,
                workflow_state="preprocessing_complete",
                current_page=3,
                completion_percentage=15
            )
            
            st.session_state.data_cleaned = True
            st.session_state.has_unsaved_changes = False
            
            st.success("✅ Saved!")
            st.balloons()
        else:
            st.error("❌ Failed")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
