"""
Page 3: Data Cleaning & Preprocessing - COMPLETE FIX

FIXES:
1. Shows ALL parameters from database (not just variables with data)
2. Comparison plots with export (PNG, PDF, HTML)
3. Auto-disappearing success messages
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


def calculate_data_hash(df):
    """Calculate a hash of the dataframe to detect changes"""
    if df is None or len(df) == 0:
        return None
    try:
        data_str = df.to_json()
        return hashlib.md5(data_str.encode()).hexdigest()
    except:
        return None


def show_temporary_message(message, message_type="success", duration=3):
    """Show a message that disappears after duration seconds"""
    placeholder = st.empty()
    
    if message_type == "success":
        placeholder.success(message)
    elif message_type == "info":
        placeholder.info(message)
    elif message_type == "warning":
        placeholder.warning(message)
    elif message_type == "error":
        placeholder.error(message)
    
    time.sleep(duration)
    placeholder.empty()


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
    """
    Load ALL data from database - FIXED to show all parameters
    
    Key differences:
    - Parameters: ALL from parameters table (may not have data)
    - Variables: Only those with actual data in timeseries_data
    """
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    
    if not project_id:
        st.error("❌ No project selected")
        return False
    
    try:
        # Create placeholder for temporary messages
        msg_placeholder = st.empty()
        
        msg_placeholder.info("🔄 Loading ALL project data from database...")
        
        # 1) Load ALL RAW/ORIGINAL timeseries from database
        df_raw = None
        data_source_used = None
        
        for src_label in ("raw", "original"):
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_temp is not None and len(df_temp) > 0:
                    df_raw = df_temp
                    data_source_used = src_label
                    break
            except Exception:
                continue
        
        if df_raw is None or len(df_raw) == 0:
            msg_placeholder.error("❌ No raw/original data found in database")
            return False
        
        # 2) Load ALL CLEANED timeseries from database
        df_cleaned = None
        try:
            df_cleaned = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
        except Exception:
            df_cleaned = None
        
        # 3) Load ALL parameters from parameters table (CRITICAL - these might not have data yet)
        parameters = []
        param_names = []
        try:
            parameters = db.get_project_parameters(project_id)
            if parameters:
                param_names = [p['parameter_name'] for p in parameters]
        except Exception:
            parameters = []
            param_names = []
        
        # 4) Load health report
        health_report = None
        try:
            health_report = db.get_health_report(project_id)
        except Exception:
            health_report = None
        
        # ============================================================
        # POPULATE SESSION STATE
        # ============================================================
        
        # Store RAW data
        st.session_state.df_long = df_raw
        
        # Combine raw + cleaned for working data
        if df_cleaned is not None and len(df_cleaned) > 0:
            st.session_state.df_clean = pd.concat([df_raw, df_cleaned], ignore_index=True)
        else:
            st.session_state.df_clean = df_raw.copy()
        
        # CRITICAL FIX: Use ALL parameter names from parameters table, not just variables with data
        # This ensures we show all 15 parameters even if only 6 have data
        if param_names:
            # Parameters are the authoritative source
            st.session_state.all_parameters = sorted(param_names)
            st.session_state.value_columns = sorted(param_names)
        else:
            # Fallback to variables in data if no parameters table
            all_variables = list(st.session_state.df_clean['variable'].unique())
            all_variables.sort()
            st.session_state.all_parameters = all_variables
            st.session_state.value_columns = all_variables
        
        # Get variables that actually have data
        raw_variables = list(df_raw['variable'].unique())
        raw_variables.sort()
        st.session_state.raw_variables = raw_variables
        
        # Get cleaned variables
        if df_cleaned is not None and len(df_cleaned) > 0:
            cleaned_vars = list(df_cleaned['variable'].unique())
            cleaned_variables = [v for v in cleaned_vars if v not in raw_variables]
            cleaned_variables.sort()
        else:
            cleaned_variables = []
        
        st.session_state.cleaned_variables = cleaned_variables
        st.session_state.project_parameters = parameters
        st.session_state.health_report = health_report
        
        # Time column
        time_col = 'timestamp' if 'timestamp' in df_raw.columns else 'time'
        st.session_state.time_column = time_col
        
        # Hash for dirty tracking
        st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
        st.session_state.has_unsaved_changes = False
        st.session_state.data_loaded = True
        
        # Show summary message (disappears after 3 seconds)
        total_params = len(param_names) if param_names else len(raw_variables)
        variables_with_data = len(raw_variables)
        
        msg_placeholder.success(f"""
        **✅ Data Loaded Successfully:**
        - **Total Parameters:** {total_params} (from parameters table)
        - **Variables with Data:** {variables_with_data} (from timeseries_data)
        - **Raw Data Points:** {len(df_raw):,}
        - **Cleaned Variables:** {len(cleaned_variables)}
        """)
        
        # Auto-clear message after 3 seconds
        time.sleep(3)
        msg_placeholder.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


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


def plot_comparison_before_after(df_all, var_original, var_cleaned, title="Before vs After Cleaning"):
    """
    Plot comparison between original and cleaned version of the same variable
    """
    try:
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'time'
        
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
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison plot: {str(e)}")
        return None


def plot_comparison_advanced(df_all, var1, var2, title="Data Comparison"):
    """Advanced comparison plot with two separate variables"""
    try:
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'time'
        
        data1 = df_all[df_all['variable'] == var1].copy()
        data2 = df_all[df_all['variable'] == var2].copy()
        
        if len(data1) == 0 and len(data2) == 0:
            return None
        
        data1 = data1.sort_values(time_col) if len(data1) > 0 else data1
        data2 = data2.sort_values(time_col) if len(data2) > 0 else data2
        
        fig = go.Figure()
        
        if len(data1) > 0:
            fig.add_trace(go.Scatter(
                x=data1[time_col],
                y=data1['value'],
                mode='lines+markers',
                name=var1,
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))
        
        if len(data2) > 0:
            fig.add_trace(go.Scatter(
                x=data2[time_col],
                y=data2['value'],
                mode='lines+markers',
                name=var2,
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None


# ============================================================
# CLEANING FUNCTIONS (Core - Preserved from original)
# ============================================================

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
    
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    
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


def show_summary():
    """Show summary"""
    st.header("📋 Cleaning Summary")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations performed yet")
        return
    
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


def save_to_database():
    """Save cleaned data with append-only logic"""
    msg_placeholder = st.empty()
    
    with msg_placeholder.container():
        st.info("💾 Saving to database...")
    
    try:
        project_id = st.session_state.current_project_id

        all_vars = st.session_state.df_clean['variable'].unique()
        original_vars = st.session_state.df_long['variable'].unique()
        new_cleaned_vars = [v for v in all_vars if v not in original_vars]

        if not new_cleaned_vars:
            msg_placeholder.warning("⚠️ No new cleaned variables to save")
            time.sleep(2)
            msg_placeholder.empty()
            return

        time_col = 'timestamp' if 'timestamp' in st.session_state.df_clean.columns else 'time'
        cleaned_df_new = st.session_state.df_clean[
            st.session_state.df_clean['variable'].isin(new_cleaned_vars)
        ].copy()

        required_cols = {time_col, "variable", "value"}
        missing_cols = [c for c in required_cols if c not in cleaned_df_new.columns]
        if missing_cols:
            msg_placeholder.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            time.sleep(2)
            msg_placeholder.empty()
            return

        # Save using append-only method
        if hasattr(db, "save_cleaned_timeseries_append"):
            success = db.save_cleaned_timeseries_append(project_id, cleaned_df_new, batch_size=1000)
        elif hasattr(db, "upsert_timeseries_data"):
            success = db.upsert_timeseries_data(
                project_id=project_id,
                df_long=cleaned_df_new,
                data_source="cleaned",
                batch_size=1000,
                ignore_duplicates=True
            )
        else:
            msg_placeholder.error("❌ Database manager missing append method")
            time.sleep(2)
            msg_placeholder.empty()
            return

        if not success:
            msg_placeholder.error("❌ Failed to save")
            time.sleep(2)
            msg_placeholder.empty()
            return

        # Update parameters
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

        # Update progress
        db.update_step_completion(project_id, 'data_cleaned', True)
        
        if DB_AVAILABLE:
            db.upsert_progress_step(project_id, "data_cleaned", 7)
            db.recompute_and_update_project_progress(
                project_id,
                workflow_state="preprocessing",
                current_page=3
            )

        # Update session
        st.session_state.data_cleaned = True
        st.session_state.has_unsaved_changes = False
        st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
        st.session_state.cleaned_variables = list(
            set(st.session_state.get('cleaned_variables', []) + new_cleaned_vars)
        )

        msg_placeholder.success(f"""
        🎉 **Successfully saved!**
        - {len(cleaned_df_new):,} cleaned data points
        - {len(cleaned_params)} parameters updated
        - Progress updated ✨
        """)
        
        # Auto-clear after 3 seconds
        time.sleep(3)
        msg_placeholder.empty()
        
        st.session_state.data_loaded = False
        st.rerun()

    except Exception as e:
        msg_placeholder.error(f"❌ Error: {str(e)}")
        time.sleep(3)
        msg_placeholder.empty()


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function"""
    
    initialize_cleaning_history()
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Force reload from database
    st.session_state.data_loaded = False
    st.session_state.df_long = None
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        st.stop()
    
    success = load_data_from_database()
    
    if not success:
        st.warning("⚠️ No data found")
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        st.stop()
    
    # Get all variables
    all_params = st.session_state.get('all_parameters', [])
    raw_vars = st.session_state.get('raw_variables', [])
    cleaned_vars = st.session_state.get('cleaned_variables', [])
    
    # Show metrics - FIXED to show all parameters
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Parameters", len(all_params))
    col2.metric("Raw Variables", len(raw_vars))
    col3.metric("Cleaned Variables", len(cleaned_vars))
    col4.metric("Operations", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # SAVE BUTTON (always visible)
    if DB_AVAILABLE and st.session_state.cleaning_history:
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
    
    # COMPARISON SECTION - Before vs After with Export
    if cleaned_vars and raw_vars:
        st.markdown("### 📊 Compare Original vs Cleaned Data")
        
        col1, col2, col3 = st.columns([2, 2, 4])
        
        with col1:
            selected_raw = st.selectbox(
                "Original Variable",
                raw_vars,
                key="cmp_raw"
            )
        
        with col2:
            selected_cleaned = st.selectbox(
                "Cleaned Variable",
                cleaned_vars,
                key="cmp_clean"
            )
        
        with col3:
            if st.button("🔍 Compare", use_container_width=True, type="primary", key="cmp_btn"):
                st.session_state.show_comparison = True
        
        if st.session_state.get('show_comparison', False):
            fig = plot_comparison_before_after(
                st.session_state.df_clean,
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
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers",
        "📈 Smoothing",
        "🔄 Transformations"
    ])
    
    with tab1:
        st.header("🔍 Missing Values")
        
        # Only show variables that have actual data
        available_vars = [v for v in all_params if v in raw_vars]
        
        if not available_vars:
            st.info("No variables with data available")
        else:
            selected = st.multiselect("Select variables", available_vars, key="miss_vars")
            
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
                    st.session_state.df_clean = handle_missing_values(
                        st.session_state.df_clean, selected, method, **kwargs
                    )
                    add_to_cleaning_history("missing_values", method, selected, kwargs)
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab2:
        st.header("📊 Outliers")
        
        available_vars = [v for v in all_params if v in raw_vars]
        
        if not available_vars:
            st.info("No variables with data available")
        else:
            selected = st.multiselect("Select variables", available_vars, key="out_vars")
            
            if selected:
                method = st.selectbox("Method", ["IQR Method", "Z-Score"], key="out_method")
                action = st.selectbox("Action", ["remove", "cap"], key="out_action")
                
                if st.button("✅ Apply", key="apply_out"):
                    st.session_state.df_clean = handle_outliers(
                        st.session_state.df_clean, selected, method, action=action
                    )
                    add_to_cleaning_history("outliers", method, selected, {'action': action})
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab3:
        st.header("📈 Smoothing")
        
        available_vars = [v for v in all_params if v in raw_vars]
        
        if not available_vars:
            st.info("No variables with data available")
        else:
            selected = st.multiselect("Select variables", available_vars, key="smooth_vars")
            
            if selected:
                method = st.selectbox("Method", ["Moving Average", "Exponential Smoothing"], key="smooth_method")
                
                kwargs = {}
                if method == "Moving Average":
                    kwargs['window'] = st.slider("Window", 3, 21, 5)
                elif method == "Exponential Smoothing":
                    kwargs['alpha'] = st.slider("Alpha", 0.1, 0.9, 0.3)
                
                if st.button("✅ Apply", key="apply_smooth"):
                    st.session_state.df_clean = apply_smoothing(
                        st.session_state.df_clean, selected, method, **kwargs
                    )
                    add_to_cleaning_history("smoothing", method, selected, kwargs)
                    st.success(f"✅ Applied to {len(selected)} variables")
                    st.rerun()
    
    with tab4:
        st.header("🔄 Transformations")
        
        available_vars = [v for v in all_params if v in raw_vars]
        
        if not available_vars:
            st.info("No variables with data available")
        else:
            selected = st.multiselect("Select variables", available_vars, key="trans_vars")
            
            if selected:
                transformation = st.selectbox(
                    "Transformation",
                    ["Log", "Square Root", "Standardize (Z-Score)", "Min-Max Normalize"],
                    key="trans_method"
                )
                
                if st.button("✅ Apply", key="apply_trans"):
                    df_result, new_cols = apply_transformations(
                        st.session_state.df_clean, selected, transformation
                    )
                    st.session_state.df_clean = df_result
                    add_to_cleaning_history(
                        "transformation", transformation, selected, {'new_columns': new_cols}
                    )
                    st.success(f"✅ Created {len(new_cols)} new variables")
                    st.rerun()


if __name__ == "__main__":
    main()
