"""
Page 3: Data Cleaning - FIXED
Fixed: Properly checks if df_long is actually loaded (not just the flag)
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
        
        # Store in session
        st.session_state.df_long = df_long
        st.session_state.df_clean = df_long.copy()
        st.session_state.data_loaded = True
        
        # Get variables
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
            title=f"Before vs After: {original_variable} → {new_variable}",
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
    
    # FIXED: Check if df_long actually exists AND is not None
    # Don't just trust the data_loaded flag
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
    
    # Now we know data is loaded
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
    tab1, tab2, tab3 = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers", 
        "📋 Summary & Save"
    ])
    
    with tab1:
        handle_missing_values(df_long, variables)
    
    with tab2:
        st.header("📊 Outlier Detection")
        st.info("Coming soon - outlier detection and treatment")
    
    with tab3:
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
                idx = st.selectbox("Select", range(len(options)), format_func=lambda i: options[i])
                
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
