"""
Page 3: Data Cleaning - COMPLETE WITH 3 FIXES + DIRTY-CHECK BANNER
✅ FIXED 1: Loads ALL data (raw + cleaned) from database
✅ FIXED 2: Comparison plots with separate raw/cleaned dropdowns + export options
✅ FIXED 3: Save button visible outside tabs (always visible)
✅ NEW: Always-visible "Unsaved changes" banner (works across all tabs)
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
# HASHING + DIRTY-STATE TRACKING (NEW)
# =============================================================================

def calculate_df_hash_fast(df: pd.DataFrame) -> str | None:
    """
    Fast, stable hash for change detection.
    - Sort rows and columns deterministically
    - Hash using pandas internal hashing (avoid to_json instability)
    """
    if df is None or len(df) == 0:
        return None

    try:
        # Normalize column order
        cols = list(df.columns)
        df2 = df[sorted(cols)].copy()

        # Normalize row order if typical long-format columns exist
        sort_cols = []
        for c in ["variable", "timestamp", "time", "value"]:
            if c in df2.columns:
                sort_cols.append(c)

        if sort_cols:
            df2 = df2.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            df2 = df2.reset_index(drop=True)

        # Make datetimes consistent
        for c in df2.columns:
            if pd.api.types.is_datetime64_any_dtype(df2[c]):
                # Convert to ISO-like string to avoid timezone/precision drift
                df2[c] = df2[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        # Hash all values (including NaN consistently)
        hv = pd.util.hash_pandas_object(df2, index=True).values
        return hashlib.md5(hv.tobytes()).hexdigest()

    except Exception:
        # Fallback
        try:
            data_str = df.to_csv(index=False)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return None


def track_dirty_state():
    """
    Compare the initial DB-loaded snapshot hash vs current working df_clean hash.
    Sets:
      - st.session_state.initial_data_hash
      - st.session_state.current_data_hash
      - st.session_state.has_unsaved_changes
    """
    current_hash = calculate_df_hash_fast(st.session_state.get("df_clean"))
    st.session_state.current_data_hash = current_hash

    initial_hash = st.session_state.get("initial_data_hash")
    if initial_hash is None:
        # If not set yet, initialize it (first time)
        st.session_state.initial_data_hash = current_hash
        st.session_state.has_unsaved_changes = False
        return

    # Dirty if hash differs
    st.session_state.has_unsaved_changes = (current_hash != initial_hash)


def render_unsaved_changes_banner():
    """
    Always-visible banner (outside tabs).
    Shows:
      - Safe-to-leave indicator OR unsaved changes warning
      - Save / Discard controls (visible regardless of tab)
    """
    track_dirty_state()

    # Only show banner if database is available and a project is active
    if not DB_AVAILABLE or not st.session_state.get("current_project_id"):
        return

    has_ops = bool(st.session_state.get("cleaning_history"))
    dirty = st.session_state.get("has_unsaved_changes", False)

    # Banner
    st.markdown("### 🧭 Page Status")

    colA, colB, colC = st.columns([3, 1, 1])

    with colA:
        if dirty:
            st.warning(
                "Unsaved changes detected in the working dataset. "
                "Please save before leaving this page, or discard changes if you want to revert to the database version."
            )
        else:
            st.success("No changes detected since the last database load/save. It is safe to move to another page.")

        # Optional: show quick diagnostics
        st.caption(
            f"Snapshot hash: {st.session_state.get('initial_data_hash')} | "
            f"Current hash: {st.session_state.get('current_data_hash')}"
        )

    with colB:
        # Save button should be available when there are operations OR dirty data
        save_disabled = (not dirty and not has_ops)
        if st.button("💾 Save", type="primary", use_container_width=True, disabled=save_disabled, key="save_btn_banner"):
            save_to_database()
            st.rerun()

    with colC:
        # Discard changes: reload from database
        if st.button("↩️ Discard & Reload", use_container_width=True, key="discard_btn_banner"):
            # Force reload
            st.session_state.data_loaded = False
            st.session_state.df_long = None
            st.session_state.df_clean = None
            st.session_state.cleaning_history = []
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = None
            st.session_state.current_data_hash = None
            st.rerun()

    st.markdown("---")


# =============================================================================
# SESSION INIT
# =============================================================================

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
    if "current_data_hash" not in st.session_state:
        st.session_state.current_data_hash = None
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
    # Mark dirty (hash will be re-evaluated on rerun)
    st.session_state.has_unsaved_changes = True


# =============================================================================
# DATABASE LOAD
# =============================================================================

def load_data_from_database():
    """
    FIXED 1: Load ALL data (raw + cleaned) from database
    """
    if not DB_AVAILABLE:
        return False

    project_id = st.session_state.get('current_project_id')
    if not project_id:
        return False

    try:
        # Load RAW data
        df_raw = db.load_timeseries_data(
            project_id=project_id,
            data_source='raw'
        )

        if df_raw is None or len(df_raw) == 0:
            return False

        # Load CLEANED data (if exists)
        df_cleaned = db.load_timeseries_data(
            project_id=project_id,
            data_source='cleaned'
        )

        # Store RAW data
        st.session_state.df_long = df_raw

        # Combine raw + cleaned for working data
        if df_cleaned is not None and len(df_cleaned) > 0:
            st.session_state.df_clean = pd.concat([df_raw, df_cleaned], ignore_index=True)
        else:
            st.session_state.df_clean = df_raw.copy()

        st.session_state.data_loaded = True

        # Get ALL variables (raw + cleaned)
        all_variables = list(st.session_state.df_clean['variable'].unique())
        all_variables.sort()
        st.session_state.value_columns = all_variables

        # Get ONLY raw variables
        raw_variables = list(df_raw['variable'].unique())
        raw_variables.sort()
        st.session_state.raw_variables = raw_variables

        # Get ONLY cleaned variables
        if df_cleaned is not None and len(df_cleaned) > 0:
            cleaned_variables = list(df_cleaned['variable'].unique())
            cleaned_variables.sort()
            st.session_state.cleaned_variables = cleaned_variables
        else:
            st.session_state.cleaned_variables = []

        time_col = 'timestamp' if 'timestamp' in df_raw.columns else 'time'
        st.session_state.time_column = time_col

        # Snapshot hash of what we loaded from DB
        st.session_state.initial_data_hash = calculate_df_hash_fast(st.session_state.df_clean)
        st.session_state.current_data_hash = st.session_state.initial_data_hash
        st.session_state.has_unsaved_changes = False

        return True

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


# =============================================================================
# EXPORT / PLOTS
# =============================================================================

def export_figure(fig, filename_prefix):
    """
    FIXED 2: Export figure as PNG, PDF, HTML
    """
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


# =============================================================================
# MAIN
# =============================================================================

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

        with st.spinner("📊 Loading ALL data from database (raw + cleaned)..."):
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

            raw_count = len(st.session_state.df_long)
            cleaned_count = len(st.session_state.cleaned_variables) if st.session_state.get('cleaned_variables') else 0
            st.success(f"✅ Loaded: {raw_count:,} raw data points, {cleaned_count} cleaned variables")

    df_long = st.session_state.df_long
    variables = st.session_state.get('value_columns', [])
    raw_vars = st.session_state.get('raw_variables', [])
    cleaned_vars = st.session_state.get('cleaned_variables', [])

    # ALWAYS-visible banner (works across tabs)
    render_unsaved_changes_banner()

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Variables", len(variables))
    col2.metric("Raw Variables", len(raw_vars))
    col3.metric("Cleaned Variables", len(cleaned_vars))
    col4.metric("Operations", len(st.session_state.cleaning_history))

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

        if st.session_state.get('show_comparison', False):
            fig = plot_comparison_advanced(
                st.session_state.df_clean,
                selected_raw,
                selected_cleaned,
                f"Comparison: {selected_raw} vs {selected_cleaned}"
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
                export_figure(fig, f"comparison_{selected_raw}_vs_{selected_cleaned}")
            else:
                st.warning("No data to compare")

        st.markdown("---")


# =============================================================================
# TAB HANDLERS (unchanged except they rely on df_clean and history)
# =============================================================================

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


# =============================================================================
# SAVE
# =============================================================================

def save_to_database():
    """Save cleaned data to database"""

    with st.spinner("💾 Saving to database..."):
        try:
            project_id = st.session_state.current_project_id

            all_vars = st.session_state.df_clean['variable'].unique()
            original_vars = st.session_state.df_long['variable'].unique()
            new_cleaned_vars = [v for v in all_vars if v not in original_vars]

            if not new_cleaned_vars:
                st.warning("⚠️ No new cleaned variables to save")
                return

            cleaned_df = st.session_state.df_clean[
                st.session_state.df_clean['variable'].isin(new_cleaned_vars)
            ].copy()

            st.info(f"Saving {len(cleaned_df):,} records for {len(new_cleaned_vars)} variables...")

            success = db.save_timeseries_data(
                project_id=project_id,
                df_long=cleaned_df,
                data_source='cleaned',
                batch_size=1000
            )

            if not success:
                st.error("❌ Failed to save")
                return

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

            db.update_step_completion(project_id, 'data_cleaned', True)
            db.update_project_progress(
                project_id=project_id,
                workflow_state="preprocessing_complete",
                current_page=3,
                completion_percentage=15
            )

            # Reset dirty tracking AFTER successful save
            st.session_state.data_cleaned = True
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = calculate_df_hash_fast(st.session_state.df_clean)
            st.session_state.current_data_hash = st.session_state.initial_data_hash

            st.session_state.cleaned_variables = list(set(st.session_state.get('cleaned_variables', []) + new_cleaned_vars))

            st.success("🎉 Successfully saved to database!")
            st.balloons()

            st.info(f"""
            **✅ Saved:**
            - {len(cleaned_df):,} cleaned data points
            - {len(cleaned_params)} parameters updated
            - Progress: 15%
            - 2nd workflow dot: GREEN ✨
            """)

            time.sleep(1)
            st.session_state.data_loaded = False
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
