"""
Page 3: Data Cleaning & Preprocessing

FIXED:
1. Session state reset on page load (keeps only project/auth/progress keys)
2. Loads ALL data from database on page load (raw + cleaned + parameters + health)
3. Append-only save with duplicate checking
4. Progress tracking with upsert to project_progress_steps
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
import plotly.io as pio

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


# =============================================================================
# SECTION 1: SESSION STATE RESET ON PAGE LOAD
# =============================================================================

def reset_page_session_state():
    """
    Reset session state for this page, keeping only:
    - Project identifiers (current_project_id, current_project, etc.)
    - User/auth keys (authenticated, user_id, profile, etc.)
    - Progress keys (workflow_state, completion_percentage, etc.)
    """
    # Define keys to preserve
    keys_to_keep = {
        # Project identifiers
        'current_project_id', 'current_project', 'selected_project',
        'project_name', 'project_description', 'project_created_at',
        
        # User/auth
        'authenticated', 'user_id', 'user_email', 'user_name',
        'user_profile', 'session_id',
        
        # Progress tracking
        'workflow_state', 'completion_percentage', 'current_page',
        'project_progress', 'step_completion',
        
        # Sidebar state
        'sidebar_state', 'page_config'
    }
    
    # Get all current keys
    all_keys = list(st.session_state.keys())
    
    # Remove keys not in preserve list
    for key in all_keys:
        if key not in keys_to_keep:
            del st.session_state[key]


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
    """Initialize session state for cleaning operations"""
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []
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


# =============================================================================
# SECTION 2: LOAD DATA FROM DATABASE ON PAGE LOAD
# =============================================================================

def load_project_data_from_database():
    """
    Load ALL project data from database on page load:
    1. timeseries_data where data_source IN ('raw', 'cleaned')
    2. parameters table (all rows for project)
    3. health_reports (latest report)
    
    Stores in session_state as:
    - df_raw: Raw timeseries data
    - df_cleaned_db: Cleaned timeseries data from database
    - df_working: Combined raw + cleaned for working/editing
    - parameters_df: DataFrame of parameters
    - health_report: Dict of health report
    """
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        st.error("❌ No project selected")
        return False
    
    try:
        # 1. Load RAW timeseries data
        df_raw = None
        for src_label in ('raw', 'original'):
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_temp is not None and len(df_temp) > 0:
                    df_raw = df_temp
                    break
            except Exception:
                continue
        
        if df_raw is None or len(df_raw) == 0:
            st.error("❌ No raw data found in database. Please upload data on Page 2.")
            return False
        
        # Store raw data
        st.session_state.df_raw = df_raw.copy()
        
        # 2. Load CLEANED timeseries data
        df_cleaned_db = None
        try:
            df_cleaned_db = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            if df_cleaned_db is not None and len(df_cleaned_db) > 0:
                st.session_state.df_cleaned_db = df_cleaned_db.copy()
            else:
                st.session_state.df_cleaned_db = pd.DataFrame(columns=df_raw.columns)
        except Exception:
            st.session_state.df_cleaned_db = pd.DataFrame(columns=df_raw.columns)
        
        # 3. Create working dataframe (raw + cleaned)
        if df_cleaned_db is not None and len(df_cleaned_db) > 0:
            st.session_state.df_working = pd.concat([df_raw, df_cleaned_db], ignore_index=True)
        else:
            st.session_state.df_working = df_raw.copy()
        
        # 4. Load parameters
        try:
            parameters = db.get_project_parameters(project_id)
            if parameters:
                st.session_state.parameters_df = pd.DataFrame(parameters)
            else:
                st.session_state.parameters_df = pd.DataFrame()
        except Exception:
            st.session_state.parameters_df = pd.DataFrame()
        
        # 5. Load health report
        try:
            health_report = db.get_health_report(project_id)
            st.session_state.health_report = health_report if health_report else {}
        except Exception:
            st.session_state.health_report = {}
        
        # 6. Extract variable lists
        st.session_state.raw_variables = sorted(df_raw['variable'].unique().tolist())
        
        if df_cleaned_db is not None and len(df_cleaned_db) > 0:
            cleaned_vars = df_cleaned_db['variable'].unique().tolist()
            st.session_state.cleaned_variables = sorted([v for v in cleaned_vars if v not in st.session_state.raw_variables])
        else:
            st.session_state.cleaned_variables = []
        
        st.session_state.all_variables = sorted(st.session_state.df_working['variable'].unique().tolist())
        
        # 7. Detect time column
        time_col = 'timestamp' if 'timestamp' in df_raw.columns else 'time'
        st.session_state.time_column = time_col
        
        # 8. Initialize hash for change tracking
        st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_working)
        
        # 9. Mark as loaded
        st.session_state.page3_data_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


# =============================================================================
# SECTION 3: EXPORT AND PLOTTING FUNCTIONS
# =============================================================================

def export_figure(fig, filename_prefix):
    """Export figure as PNG, PDF, HTML"""
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        png_bytes = pio.to_image(fig, format='png', width=1200, height=600)
        st.download_button(
            label="📊 Download PNG",
            data=png_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        pdf_bytes = pio.to_image(fig, format='pdf', width=1200, height=600)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col3:
        html_bytes = pio.to_html(fig, include_plotlyjs='cdn').encode()
        st.download_button(
            label="🌐 Download HTML",
            data=html_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )


def plot_comparison_before_after(df_all, var_original, var_cleaned, title="Before vs After"):
    """Plot comparison between original and cleaned variable"""
    try:
        time_col = st.session_state.get('time_column', 'timestamp')
        
        data_original = df_all[df_all['variable'] == var_original].copy().sort_values(time_col)
        data_cleaned = df_all[df_all['variable'] == var_cleaned].copy().sort_values(time_col)
        
        if len(data_original) == 0 and len(data_cleaned) == 0:
            return None
        
        fig = go.Figure()
        
        if len(data_original) > 0:
            fig.add_trace(go.Scatter(
                x=data_original[time_col],
                y=data_original['value'],
                mode='lines+markers',
                name=f'{var_original} (Original)',
                line=dict(color='#95a5a6', width=2, dash='dot'),
                marker=dict(size=4, opacity=0.6)
            ))
        
        if len(data_cleaned) > 0:
            fig.add_trace(go.Scatter(
                x=data_cleaned[time_col],
                y=data_cleaned['value'],
                mode='lines+markers',
                name=f'{var_cleaned} (Cleaned)',
                line=dict(color='#27ae60', width=2),
                marker=dict(size=5)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None


# =============================================================================
# SECTION 4: CLEANING FUNCTIONS
# =============================================================================

def handle_missing_values(df, variables, method, **kwargs):
    """Handle missing values"""
    df_copy = df.copy()
    
    for var in variables:
        mask = df_copy['variable'] == var
        values = df_copy.loc[mask, 'value'].copy()
        
        if method == "Drop Rows":
            df_copy = df_copy[~(mask & df_copy['value'].isna())]
        elif method == "Forward Fill":
            df_copy.loc[mask, 'value'] = values.fillna(method='ffill')
        elif method == "Backward Fill":
            df_copy.loc[mask, 'value'] = values.fillna(method='bfill')
        elif method == "Interpolate":
            df_copy.loc[mask, 'value'] = values.interpolate()
        elif method == "Fill with Mean":
            df_copy.loc[mask, 'value'] = values.fillna(values.mean())
        elif method == "Fill with Median":
            df_copy.loc[mask, 'value'] = values.fillna(values.median())
        elif method == "Fill with Constant":
            fill_val = kwargs.get('fill_value', 0)
            df_copy.loc[mask, 'value'] = values.fillna(fill_val)
    
    return df_copy


def handle_outliers(df, variables, method, **kwargs):
    """Handle outliers"""
    df_copy = df.copy()
    
    for var in variables:
        mask = df_copy['variable'] == var
        values = df_copy.loc[mask, 'value'].dropna()
        
        if len(values) == 0:
            continue
        
        if method == "IQR Method":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df_copy.loc[mask, 'value'] < lower) | (df_copy.loc[mask, 'value'] > upper)
            
            action = kwargs.get('action', 'remove')
            if action == 'remove':
                df_copy = df_copy[~(mask & outlier_mask)]
            elif action == 'cap':
                df_copy.loc[mask & outlier_mask & (df_copy['value'] < lower), 'value'] = lower
                df_copy.loc[mask & outlier_mask & (df_copy['value'] > upper), 'value'] = upper
        
        elif method == "Z-Score":
            mean = values.mean()
            std = values.std()
            if std > 0:
                z_scores = np.abs((df_copy.loc[mask, 'value'] - mean) / std)
                outlier_mask = z_scores > 3
                
                action = kwargs.get('action', 'remove')
                if action == 'remove':
                    df_copy = df_copy[~(mask & outlier_mask)]
    
    return df_copy


def apply_smoothing(df, variables, method, **kwargs):
    """Apply smoothing"""
    df_copy = df.copy()
    
    for var in variables:
        mask = df_copy['variable'] == var
        values = df_copy.loc[mask, 'value'].copy()
        
        if method == "Moving Average":
            window = kwargs.get('window', 3)
            smoothed = values.rolling(window=window, center=True, min_periods=1).mean()
            df_copy.loc[mask, 'value'] = smoothed
        elif method == "Exponential Smoothing":
            alpha = kwargs.get('alpha', 0.3)
            smoothed = values.ewm(alpha=alpha, adjust=False).mean()
            df_copy.loc[mask, 'value'] = smoothed
    
    return df_copy


def apply_transformations(df, variables, transformation):
    """Apply mathematical transformations"""
    df_copy = df.copy()
    new_columns = []
    
    time_col = st.session_state.get('time_column', 'timestamp')
    
    for var in variables:
        new_var_name = f"{var}_{transformation.lower().replace(' ', '_')}"
        new_columns.append(new_var_name)
        
        var_data = df[df['variable'] == var].copy()
        values = var_data['value']
        
        transformed_data = var_data.copy()
        transformed_data['variable'] = new_var_name
        
        if "Log" in transformation:
            min_val = values.min()
            if pd.notna(min_val) and min_val <= 0:
                shift_amount = abs(min_val) + 1
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log1p(transformed_data.loc[transformed_data['value'].notna(), 'value'] + shift_amount)
            else:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    np.log(transformed_data.loc[transformed_data['value'].notna(), 'value'])
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


# =============================================================================
# SECTION 5: SAVE TO DATABASE (APPEND-ONLY WITH DUPLICATE CHECKING)
# =============================================================================

def save_to_database():
    """
    Save cleaned data to database with append-only logic:
    1. Identify newly created cleaned variables (not in raw data)
    2. Check for existing rows to avoid duplicates
    3. Save only new rows with data_source='cleaned'
    4. Update parameters table
    5. Update progress: upsert to project_progress_steps
    """
    project_id = st.session_state.get('current_project_id')
    
    if not project_id:
        st.error("❌ No project selected")
        return
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        return
    
    with st.spinner("💾 Saving to database..."):
        try:
            # 1. Identify newly created cleaned variables
            all_vars_in_working = st.session_state.df_working['variable'].unique()
            raw_vars = st.session_state.raw_variables
            new_cleaned_vars = [v for v in all_vars_in_working if v not in raw_vars]
            
            if not new_cleaned_vars:
                st.warning("⚠️ No new cleaned variables to save")
                return
            
            # 2. Extract data for new cleaned variables
            time_col = st.session_state.get('time_column', 'timestamp')
            cleaned_df_new = st.session_state.df_working[
                st.session_state.df_working['variable'].isin(new_cleaned_vars)
            ].copy()
            
            # 3. Check for existing cleaned data to avoid duplicates
            existing_cleaned = st.session_state.get('df_cleaned_db')
            if existing_cleaned is not None and len(existing_cleaned) > 0:
                # Remove rows that already exist in database
                for var in new_cleaned_vars:
                    var_existing = existing_cleaned[existing_cleaned['variable'] == var]
                    if len(var_existing) > 0:
                        existing_timestamps = set(var_existing[time_col].values)
                        var_new = cleaned_df_new[cleaned_df_new['variable'] == var]
                        
                        # Keep only rows with timestamps NOT in existing data
                        mask_to_keep = ~var_new[time_col].isin(existing_timestamps)
                        rows_to_remove = var_new[~mask_to_keep]
                        
                        if len(rows_to_remove) > 0:
                            # Remove duplicate rows
                            cleaned_df_new = cleaned_df_new[
                                ~((cleaned_df_new['variable'] == var) & 
                                  (cleaned_df_new[time_col].isin(existing_timestamps)))
                            ]
            
            if len(cleaned_df_new) == 0:
                st.info("ℹ️ All cleaned data already exists in database")
                return
            
            st.info(f"Saving {len(cleaned_df_new):,} new records for {len(new_cleaned_vars)} variables...")
            
            # 4. Save to timeseries_data with data_source='cleaned'
            success = False
            
            if hasattr(db, "upsert_timeseries_data"):
                success = db.upsert_timeseries_data(
                    project_id=project_id,
                    df_long=cleaned_df_new,
                    data_source="cleaned",
                    batch_size=1000,
                    ignore_duplicates=True
                )
            elif hasattr(db, "save_timeseries_data"):
                success = db.save_timeseries_data(
                    project_id=project_id,
                    df_long=cleaned_df_new,
                    data_source="cleaned",
                    batch_size=1000
                )
            else:
                st.error("❌ Database manager missing save method")
                return
            
            if not success:
                st.error("❌ Failed to save data")
                return
            
            # 5. Update parameters table
            cleaned_params = []
            for var in new_cleaned_vars:
                var_subset = cleaned_df_new[cleaned_df_new['variable'] == var]
                var_data = var_subset['value'].dropna()
                
                if len(var_data) > 0:
                    cleaned_params.append({
                        'name': var,
                        'data_type': 'numeric',
                        'min_value': float(var_data.min()),
                        'max_value': float(var_data.max()),
                        'mean_value': float(var_data.mean()),
                        'std_value': float(var_data.std()),
                        'missing_count': int(var_subset['value'].isna().sum()),
                        'total_count': int(len(var_subset))
                    })
            
            if cleaned_params:
                db.save_parameters(project_id, cleaned_params)
            
            # 6. Update progress: upsert to project_progress_steps
            try:
                # Upsert step_key='data_cleaned', step_percent=7
                if hasattr(db, "upsert_progress_step"):
                    db.upsert_progress_step(
                        project_id=project_id,
                        step_key="data_cleaned",
                        step_percent=7
                    )
                else:
                    # Fallback: direct upsert
                    db.client.table("project_progress_steps").upsert({
                        "project_id": project_id,
                        "step_key": "data_cleaned",
                        "step_percent": 7,
                        "updated_at": datetime.now().isoformat()
                    }, on_conflict="project_id,step_key").execute()
                
                # Recompute total progress
                if hasattr(db, "recompute_and_update_project_progress"):
                    db.recompute_and_update_project_progress(
                        project_id=project_id,
                        workflow_state="preprocessing",
                        current_page=3
                    )
            except Exception as e:
                st.warning(f"Progress update failed: {e}")
            
            # 7. Update session state
            st.session_state.has_unsaved_changes = False
            st.session_state.cleaned_variables = list(
                set(st.session_state.get('cleaned_variables', []) + new_cleaned_vars)
            )
            
            # Success message
            st.success(f"""
            🎉 **Successfully saved!**
            - {len(cleaned_df_new):,} new data points
            - {len(cleaned_params)} parameters updated
            - Progress: Step 'data_cleaned' = 7%
            """)
            
            st.balloons()
            
            time.sleep(2)
            
            # Reload data from database
            st.session_state.page3_data_loaded = False
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Save error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


# =============================================================================
# SECTION 6: MAIN FUNCTION
# =============================================================================

def main():
    """Main function with session reset and data loading"""
    
    # Check project
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # STEP 1: Reset session state on page load (if not already loaded)
    if not st.session_state.get('page3_data_loaded', False):
        reset_page_session_state()
        initialize_cleaning_history()
    
    # STEP 2: Load data from database
    if not st.session_state.get('page3_data_loaded', False):
        with st.spinner("📊 Loading project data from database..."):
            success = load_project_data_from_database()
            
            if not success:
                st.error("❌ Failed to load project data")
                if st.button("🔄 Retry"):
                    st.rerun()
                st.stop()
    
    # Get data from session state
    df_working = st.session_state.get('df_working')
    raw_vars = st.session_state.get('raw_variables', [])
    cleaned_vars = st.session_state.get('cleaned_variables', [])
    all_vars = st.session_state.get('all_variables', [])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Variables", len(all_vars))
    col2.metric("Raw Variables", len(raw_vars))
    col3.metric("Cleaned Variables", len(cleaned_vars))
    col4.metric("Operations", len(st.session_state.get('cleaning_history', [])))
    
    st.markdown("---")
    
    # Save button (always visible)
    if st.session_state.get('cleaning_history'):
        st.markdown("### 💾 Save Your Work")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                           for op in st.session_state.cleaning_history)
            st.info(f"📝 {len(st.session_state.cleaning_history)} operations creating {total_new} new columns")
        
        with col2:
            if st.button("💾 Save", type="primary", use_container_width=True, key="save_top"):
                save_to_database()
        
        st.markdown("---")
    
    # Comparison section
    if cleaned_vars and raw_vars:
        st.markdown("### 📊 Compare Original vs Cleaned Data")
        
        col1, col2, col3 = st.columns([2, 2, 4])
        
        with col1:
            selected_raw = st.selectbox("Original Variable", raw_vars, key="cmp_raw")
        
        with col2:
            selected_cleaned = st.selectbox("Cleaned Variable", cleaned_vars, key="cmp_clean")
        
        with col3:
            if st.button("🔍 Compare", use_container_width=True, type="primary", key="cmp_btn"):
                st.session_state.show_comparison = True
        
        if st.session_state.get('show_comparison', False):
            fig = plot_comparison_before_after(
                df_working,
                selected_raw,
                selected_cleaned,
                f"Before vs After: {selected_raw} → {selected_cleaned}"
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                export_figure(fig, f"comparison_{selected_raw}_vs_{selected_cleaned}")
            else:
                st.warning("No data to compare")
        
        st.markdown("---")
    
    # Tabs for cleaning operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers",
        "📈 Smoothing",
        "🔄 Transformations"
    ])
    
    with tab1:
        st.header("🔍 Missing Values")
        
        if not raw_vars:
            st.info("No variables available")
        else:
            selected = st.multiselect("Select variables", raw_vars, key="miss_vars")
            
            if selected:
                method = st.selectbox(
                    "Method",
                    ["Drop Rows", "Forward Fill", "Backward Fill", "Interpolate", 
                     "Fill with Mean", "Fill with Median", "Fill with Constant"],
                    key="miss_method"
                )
                
                kwargs = {}
                if method == "Fill with Constant":
                    kwargs['fill_value'] = st.number_input("Fill value", value=0.0)
                
                if st.button("✅ Apply", key="apply_miss"):
                    st.session_state.df_working = handle_missing_values(
                        st.session_state.df_working, selected, method, **kwargs
                    )
                    add_to_cleaning_history("missing_values", method, selected, kwargs)
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab2:
        st.header("📊 Outliers")
        
        if not raw_vars:
            st.info("No variables available")
        else:
            selected = st.multiselect("Select variables", raw_vars, key="out_vars")
            
            if selected:
                method = st.selectbox("Method", ["IQR Method", "Z-Score"], key="out_method")
                action = st.selectbox("Action", ["remove", "cap"], key="out_action")
                
                if st.button("✅ Apply", key="apply_out"):
                    st.session_state.df_working = handle_outliers(
                        st.session_state.df_working, selected, method, action=action
                    )
                    add_to_cleaning_history("outliers", method, selected, {'action': action})
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab3:
        st.header("📈 Smoothing")
        
        if not raw_vars:
            st.info("No variables available")
        else:
            selected = st.multiselect("Select variables", raw_vars, key="smooth_vars")
            
            if selected:
                method = st.selectbox("Method", ["Moving Average", "Exponential Smoothing"], key="smooth_method")
                
                kwargs = {}
                if method == "Moving Average":
                    kwargs['window'] = st.slider("Window", 3, 21, 5)
                elif method == "Exponential Smoothing":
                    kwargs['alpha'] = st.slider("Alpha", 0.1, 0.9, 0.3)
                
                if st.button("✅ Apply", key="apply_smooth"):
                    st.session_state.df_working = apply_smoothing(
                        st.session_state.df_working, selected, method, **kwargs
                    )
                    add_to_cleaning_history("smoothing", method, selected, kwargs)
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab4:
        st.header("🔄 Transformations")
        
        if not raw_vars:
            st.info("No variables available")
        else:
            selected = st.multiselect("Select variables", raw_vars, key="trans_vars")
            
            if selected:
                transformation = st.selectbox(
                    "Transformation",
                    ["Log", "Square Root", "Standardize (Z-Score)", "Min-Max Normalize"],
                    key="trans_method"
                )
                
                if st.button("✅ Apply", key="apply_trans"):
                    df_result, new_cols = apply_transformations(
                        st.session_state.df_working, selected, transformation
                    )
                    st.session_state.df_working = df_result
                    add_to_cleaning_history(
                        "transformation", transformation, selected, {'new_columns': new_cols}
                    )
                    st.success(f"✅ Created {len(new_cols)} new variables")
                    st.rerun()


if __name__ == "__main__":
    main()
