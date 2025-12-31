"""
Page 3: Data Cleaning - COMPLETE WITH 3 FIXES
✅ FIXED 1: Loads ALL data (raw + cleaned) from database
✅ FIXED 2: Comparison plots with separate raw/cleaned dropdowns + export options
✅ FIXED 3: Save button visible outside tabs (always visible)
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
    Load ALL data (raw + cleaned + parameters + health_report) from database
    This function IGNORES session state and always loads fresh from DB
    """
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    
    if not project_id:
        st.error("❌ No project selected")
        return False
    
    try:
        st.info("🔄 Loading ALL project data from database...")
        
        # 1) Load ALL RAW/ORIGINAL timeseries from database
        df_raw = None
        data_source_used = None
        
        for src_label in ("raw", "original"):
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_temp is not None and len(df_temp) > 0:
                    df_raw = df_temp
                    data_source_used = src_label
                    st.success(f"✅ Found {len(df_raw):,} data points with data_source='{src_label}'")
                    break
            except Exception as e:
                continue
        
        if df_raw is None or len(df_raw) == 0:
            st.error("❌ No raw/original data found in database")
            return False
        
        # 2) Load ALL CLEANED timeseries from database
        df_cleaned = None
        try:
            df_cleaned = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            if df_cleaned is not None and len(df_cleaned) > 0:
                st.success(f"✅ Found {len(df_cleaned):,} cleaned data points")
        except Exception as e:
            st.info(f"ℹ️ No cleaned data yet (normal for new projects)")
            df_cleaned = None
        
        # 3) Load ALL parameters from parameters table
        parameters = []
        try:
            parameters = db.get_project_parameters(project_id)
            if parameters:
                st.success(f"✅ Loaded {len(parameters)} parameters from database")
        except Exception as e:
            st.warning(f"⚠️ Could not load parameters: {e}")
            parameters = []
        
        # 4) Load latest health report from health_reports table
        health_report = None
        try:
            health_report = db.get_health_report(project_id)
            if health_report:
                st.success(f"✅ Loaded health report (score: {health_report.get('health_score', 'N/A')})")
        except Exception as e:
            st.info(f"ℹ️ No health report yet")
            health_report = None
        
        # ============================================================
        # POPULATE SESSION STATE FROM DATABASE DATA
        # ============================================================
        
        # Store RAW data
        st.session_state.df_long = df_raw
        
        # Combine raw + cleaned for working data
        if df_cleaned is not None and len(df_cleaned) > 0:
            st.session_state.df_clean = pd.concat([df_raw, df_cleaned], ignore_index=True)
        else:
            st.session_state.df_clean = df_raw.copy()
        
        # Get ALL unique variables from combined data
        all_variables = list(st.session_state.df_clean['variable'].unique())
        all_variables.sort()
        st.session_state.value_columns = all_variables
        
        # Get ONLY raw variables
        raw_variables = list(df_raw['variable'].unique())
        raw_variables.sort()
        st.session_state.raw_variables = raw_variables
        
        # Get ONLY cleaned variables (only in df_cleaned, NOT in df_raw)
        if df_cleaned is not None and len(df_cleaned) > 0:
            cleaned_vars = list(df_cleaned['variable'].unique())
            # Only keep variables that are NOT in raw data
            cleaned_variables = [v for v in cleaned_vars if v not in raw_variables]
            cleaned_variables.sort()
        else:
            cleaned_variables = []
        
        st.session_state.cleaned_variables = cleaned_variables
        st.session_state.project_parameters = parameters
        st.session_state.health_report = health_report
        
        # Time column detection
        time_col = 'timestamp' if 'timestamp' in df_raw.columns else 'time'
        st.session_state.time_column = time_col
        
        # Snapshot hash for dirty tracking
        st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
        st.session_state.has_unsaved_changes = False
        
        # Mark as loaded
        st.session_state.data_loaded = True
        
        # Summary
        st.success(f"""
        **📊 Data Loaded Successfully:**
        - **Total Variables:** {len(all_variables)}
        - **Raw Variables:** {len(raw_variables)}
        - **Cleaned Variables:** {len(cleaned_variables)}
        - **Data Points:** {len(st.session_state.df_clean):,}
        - **Parameters in DB:** {len(parameters)}
        """)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data from database: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def export_figure(fig, filename_prefix):
    """
    FIXED 2: Export figure as PNG, PDF, HTML
    """
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # PNG export
        png_bytes = pio.to_image(fig, format='png', width=1200, height=600)
        st.download_button(
            label="📊 Download PNG",
            data=png_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        # PDF export
        pdf_bytes = pio.to_image(fig, format='pdf', width=1200, height=600)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col3:
        # HTML export
        html_bytes = pio.to_html(fig, include_plotlyjs='cdn').encode()
        st.download_button(
            label="🌐 Download HTML",
            data=html_bytes,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )


def plot_comparison_advanced(df_all, var1, var2, title="Data Comparison"):
    """
    FIXED 2: Advanced comparison plot with two separate variable selections
    """
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
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None


def plot_comparison(df_original, df_modified, original_variable, new_variable, operation_type):
    """Create comparison plot (original version for backward compatibility)"""
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
    """Main function - ALWAYS loads fresh from database"""
    
    initialize_cleaning_history()
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # ALWAYS FORCE RELOAD FROM DATABASE (ignore cached data_loaded flag)
    # This ensures we see ALL data from ALL uploads
    st.session_state.data_loaded = False  # Force reload
    st.session_state.df_long = None  # Clear cached data
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        st.stop()
    
    with st.spinner("📊 Loading ALL data from database (raw + cleaned + parameters)..."):
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
                    st.rerun()
            
            st.stop()
    
    df_long = st.session_state.df_long
    variables = st.session_state.get('value_columns', [])
    raw_vars = st.session_state.get('raw_variables', [])
    cleaned_vars = st.session_state.get('cleaned_variables', [])
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Variables", len(variables))
    col2.metric("Raw Variables", len(raw_vars))
    col3.metric("Cleaned Variables", len(cleaned_vars))
    col4.metric("Operations", len(st.session_state.cleaning_history))
    
    st.markdown("---")
    
    # ============================================================
    # FIXED 3: SAVE BUTTON OUTSIDE TABS (Always Visible)
    # ============================================================
    if DB_AVAILABLE and st.session_state.cleaning_history:
        st.markdown("### 💾 Save Your Work")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                           for op in st.session_state.cleaning_history)
            st.info(f"📝 You have {len(st.session_state.cleaning_history)} operations creating {total_new} new columns ready to save")
        
        with col2:
            if st.button("💾 Save to Database", type="primary", use_container_width=True, key="save_btn_top"):
                save_to_database()
        
        st.markdown("---")
    
    # ============================================================
    # FIXED 2: COMPARISON SECTION (Raw vs Cleaned from Database)
    # ============================================================
    if cleaned_vars:
        st.markdown("### 📊 Compare Raw vs Cleaned Data")
        
        col1, col2, col3 = st.columns([2, 2, 4])
        
        with col1:
            selected_raw = st.selectbox(
                "Select Raw Variable",
                raw_vars,
                key="compare_raw_var"
            )
        
        with col2:
            selected_cleaned = st.selectbox(
                "Select Cleaned Variable",
                cleaned_vars,
                key="compare_cleaned_var"
            )
        
        with col3:
            if st.button("🔍 Compare", use_container_width=True, type="primary"):
                st.session_state.show_comparison = True
        
        # Show comparison plot
        if st.session_state.get('show_comparison', False):
            fig = plot_comparison_advanced(
                st.session_state.df_clean,
                selected_raw,
                selected_cleaned,
                f"Comparison: {selected_raw} vs {selected_cleaned}"
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                export_figure(fig, f"comparison_{selected_raw}_vs_{selected_cleaned}")
            else:
                st.warning("No data to compare")
        
        st.markdown("---")
    
    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers",
        "🔄 Transformations",
        "📋 Summary"
    ])
    
    with tab1:
        handle_missing_values(df_long, raw_vars)
    
    with tab2:
        handle_outliers(df_long, raw_vars)
    
    with tab3:
        handle_transformations(df_long, raw_vars)
    
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
                    export_figure(fig, f"missing_treatment_{orig}_to_{cleaned}")


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
                    export_figure(fig, f"outlier_treatment_{orig}_to_{cleaned}")


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
                    export_figure(fig, f"transformation_{orig}_to_{transformed}")


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
    """Show summary"""
    
    st.header("📋 Cleaning Summary")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations performed yet")
        st.markdown("Apply cleaning operations in the tabs above.")
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


def save_to_database():
    """
    Save cleaned data to database using APPEND-ONLY logic
    - Only saves NEW cleaned variables (not in raw data)
    - Uses upsert to avoid duplicates
    - Proper timestamp checking
    """
    with st.spinner("💾 Saving to database..."):
        try:
            project_id = st.session_state.current_project_id
    
            # Identify newly created cleaned variables (do NOT overwrite/delete existing DB data)
            all_vars = st.session_state.df_clean['variable'].unique()
            original_vars = st.session_state.df_long['variable'].unique()
            new_cleaned_vars = [v for v in all_vars if v not in original_vars]
    
            if not new_cleaned_vars:
                st.warning("⚠️ No new cleaned variables to save")
                return
    
            # Build dataframe containing ONLY the new cleaned variables
            time_col = 'timestamp' if 'timestamp' in st.session_state.df_clean.columns else 'time'
            cleaned_df_new = st.session_state.df_clean[
                st.session_state.df_clean['variable'].isin(new_cleaned_vars)
            ].copy()
    
            # Ensure required columns exist for DB write
            required_cols = {time_col, "variable", "value"}
            missing_cols = [c for c in required_cols if c not in cleaned_df_new.columns]
            if missing_cols:
                st.error(f"❌ Cleaned data missing required column(s): {', '.join(missing_cols)}")
                return
    
            st.info(
                f"Appending cleaned dataset (no delete/overwrite): "
                f"{len(cleaned_df_new):,} new records across {len(new_cleaned_vars)} new variable(s)."
            )
    
            # APPEND-ONLY SAVE (NO DELETE, NO DUPLICATION)
            # - Requires DB uniqueness: (project_id, data_source, timestamp, variable)
            # - Uses UPSERT/ignore duplicates in supabase_manager
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
                st.error("❌ Database manager is missing append/UPSERT method for cleaned data.")
                st.info("Please ensure supabase_manager includes save_cleaned_timeseries_append() or upsert_timeseries_data().")
                return
    
            if not success:
                st.error("❌ Failed to append cleaned data")
                return
    
            # Update parameters for ONLY newly created cleaned variables
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
    
            # State transition (existing behavior kept)
            db.update_step_completion(project_id, 'data_cleaned', True)
    
            # Progress subsystem (agreed approach)
            if DB_AVAILABLE and st.session_state.get("current_project_id"):
                pid = st.session_state.current_project_id
    
                # Step contribution for page 3
                db.upsert_progress_step(pid, "data_cleaned", 7)
    
                # Recompute total and write back to projects
                db.recompute_and_update_project_progress(
                    pid,
                    workflow_state="preprocessing",
                    current_page=3
                )
    
            # Reset dirty tracking AFTER successful save
            st.session_state.data_cleaned = True
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_clean)
    
            st.session_state.cleaned_variables = list(
                set(st.session_state.get('cleaned_variables', []) + new_cleaned_vars)
            )
    
            st.success("🎉 Successfully appended cleaned data to database!")
            st.balloons()
    
            st.info(f"""
            **✅ Saved (append-only):**
            - {len(cleaned_df_new):,} new cleaned data points
            - {len(cleaned_params)} parameters updated for new cleaned variables
            - Progress: Updated with workflow dot GREEN ✨
            """)
    
            time.sleep(1)
            # Force reload from DB to reflect persisted state
            st.session_state.data_loaded = False
            st.rerun()
    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
