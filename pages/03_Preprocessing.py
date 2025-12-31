"""
Page 3: Data Cleaning & Preprocessing (FIXED - Loads ALL data from database)

FIXES:
- Now loads ALL timeseries data (not just last upload)
- Loads parameters from parameters table
- Loads health_report from health_reports table
- Shows correct metrics for Total Variables, Raw Variables, Cleaned Variables
- Properly tracks cleaned data saves with timestamp checking
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


#---------------------
# Section 1: Hashing + Dirty-State Tracking
#---------------------

def calculate_df_hash_fast(df: pd.DataFrame) -> str | None:
    """
    Fast, stable hash for change detection.
    - Sort rows and columns deterministically
    - Hash using pandas internal hashing (avoid to_json instability)
    """
    if df is None or len(df) == 0:
        return None

    try:
        cols = list(df.columns)
        df2 = df[sorted(cols)].copy()

        sort_cols = []
        for c in ["variable", "timestamp", "time", "value"]:
            if c in df2.columns:
                sort_cols.append(c)

        if sort_cols:
            df2 = df2.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            df2 = df2.reset_index(drop=True)

        for c in df2.columns:
            if pd.api.types.is_datetime64_any_dtype(df2[c]):
                df2[c] = df2[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        hv = pd.util.hash_pandas_object(df2, index=True).values
        return hashlib.md5(hv.tobytes()).hexdigest()

    except Exception:
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
        st.session_state.initial_data_hash = current_hash
        st.session_state.has_unsaved_changes = False
        return

    st.session_state.has_unsaved_changes = (current_hash != initial_hash)


def render_unsaved_changes_banner():
    """
    Always-visible banner (outside tabs).
    Shows:
      - Safe-to-leave indicator OR unsaved changes warning
      - Save / Discard controls (visible regardless of tab)
    """
    track_dirty_state()

    if not DB_AVAILABLE or not st.session_state.get("current_project_id"):
        return

    has_ops = bool(st.session_state.get("cleaning_history"))
    dirty = st.session_state.get("has_unsaved_changes", False)

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

        st.caption(
            f"Snapshot hash: {st.session_state.get('initial_data_hash')} | "
            f"Current hash: {st.session_state.get('current_data_hash')}"
        )

    with colB:
        save_disabled = (not dirty and not has_ops)
        if st.button("💾 Save", type="primary", use_container_width=True, disabled=save_disabled, key="save_btn_banner"):
            save_to_database()
            st.rerun()

    with colC:
        if st.button("↩️ Discard & Reload", use_container_width=True, key="discard_btn_banner"):
            st.session_state.data_loaded = False
            st.session_state.df_long = None
            st.session_state.df_clean = None
            st.session_state.cleaning_history = []
            st.session_state.has_unsaved_changes = False
            st.session_state.initial_data_hash = None
            st.session_state.current_data_hash = None
            st.session_state.project_parameters = None
            st.session_state.health_report = None
            st.rerun()

    st.markdown("---")


#---------------------
# Section 2: Session Initialization
#---------------------

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
    if "project_parameters" not in st.session_state:
        st.session_state.project_parameters = None
    if "health_report" not in st.session_state:
        st.session_state.health_report = None



# ---------------------
# Section A: Page-3 I/O on load (DB → session_state) - FIXED TO LOAD ALL DATA
# ---------------------
def load_page3_data_on_page_load() -> bool:
    """
    Page 3 load (DB → session_state). FIXED to load ALL data from database.

    Loads:
      1) ALL timeseries_data with data_source='original' or 'raw' -> st.session_state.df_long
      2) ALL timeseries_data with data_source='cleaned'          -> merged into st.session_state.df_clean
      3) ALL parameters from parameters table                     -> st.session_state.project_parameters
      4) Latest health_report from health_reports table          -> st.session_state.health_report
      5) Derived helper lists -> value_columns, raw_variables, cleaned_variables, time_column
      6) Snapshot hashes      -> initial_data_hash, current_data_hash, has_unsaved_changes

    This ensures the page shows ALL data from ALL uploads, not just the last one.
    """
    if not DB_AVAILABLE:
        return False

    project_id = st.session_state.get("current_project_id")
    if not project_id:
        return False

    try:
        # 1) Load ALL ORIGINAL/RAW timeseries from database
        df_raw = None
        for src_label in ("raw", "original"):
            try:
                df_raw = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_raw is not None and len(df_raw) > 0:
                    break
            except Exception:
                df_raw = None

        if df_raw is None or len(df_raw) == 0:
            st.warning("⚠️ No raw/original data found in database. Please upload data first.")
            return False

        # 2) Load ALL CLEANED timeseries from database
        df_cleaned = None
        try:
            df_cleaned = db.load_timeseries_data(project_id=project_id, data_source="cleaned")
        except Exception:
            df_cleaned = None

        # 3) Load ALL parameters from parameters table
        try:
            parameters = db.get_project_parameters(project_id)
            st.session_state.project_parameters = parameters if parameters else []
        except Exception as e:
            st.warning(f"Could not load parameters: {e}")
            st.session_state.project_parameters = []

        # 4) Load latest health report from health_reports table
        try:
            health_report = db.get_health_report(project_id)
            st.session_state.health_report = health_report
        except Exception as e:
            st.warning(f"Could not load health report: {e}")
            st.session_state.health_report = None

        # Session state wiring (expected by the rest of the page)
        st.session_state.df_long = df_raw

        if df_cleaned is not None and len(df_cleaned) > 0:
            # Merge raw and cleaned data
            st.session_state.df_clean = pd.concat([df_raw, df_cleaned], ignore_index=True)
        else:
            st.session_state.df_clean = df_raw.copy()

        # Derived lists
        all_variables = list(st.session_state.df_clean["variable"].unique())
        all_variables.sort()
        st.session_state.value_columns = all_variables

        raw_variables = list(df_raw["variable"].unique())
        raw_variables.sort()
        st.session_state.raw_variables = raw_variables

        if df_cleaned is not None and len(df_cleaned) > 0:
            cleaned_variables = list(df_cleaned["variable"].unique())
            cleaned_variables.sort()
            st.session_state.cleaned_variables = cleaned_variables
        else:
            st.session_state.cleaned_variables = []

        time_col = "timestamp" if "timestamp" in df_raw.columns else "time"
        st.session_state.time_column = time_col

        # Snapshot hashes
        st.session_state.initial_data_hash = calculate_df_hash_fast(st.session_state.df_clean)
        st.session_state.current_data_hash = st.session_state.initial_data_hash
        st.session_state.has_unsaved_changes = False

        # Mark loaded
        st.session_state.data_loaded = True

        # Log what was loaded
        st.success(f"✅ Loaded {len(df_raw):,} raw data points across {len(raw_variables)} variables")
        if df_cleaned is not None and len(df_cleaned) > 0:
            st.success(f"✅ Loaded {len(df_cleaned):,} cleaned data points across {len(cleaned_variables)} variables")
        if st.session_state.project_parameters:
            st.success(f"✅ Loaded {len(st.session_state.project_parameters)} parameters from database")

        return True

    except Exception as e:
        st.error(f"Error during page-load data fetch: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


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


#---------------------
# Section 3: Core Cleaning Functions (FROZEN - Do not modify)
#---------------------

def apply_missing_values_handling(df, variables, method, **kwargs):
    """Handle missing values"""
    df_result = df.copy()
    
    for var in variables:
        mask = df_result['variable'] == var
        values = df_result.loc[mask, 'value']
        
        if method == 'drop':
            df_result = df_result[~(mask & df_result['value'].isna())]
            
        elif method == 'forward_fill':
            df_result.loc[mask, 'value'] = values.fillna(method='ffill')
            
        elif method == 'backward_fill':
            df_result.loc[mask, 'value'] = values.fillna(method='bfill')
            
        elif method == 'interpolate':
            interp_method = kwargs.get('interpolation_method', 'linear')
            df_result.loc[mask, 'value'] = values.interpolate(method=interp_method)
            
        elif method == 'constant':
            fill_value = kwargs.get('constant_value', 0)
            df_result.loc[mask, 'value'] = values.fillna(fill_value)
            
        elif method == 'mean':
            df_result.loc[mask, 'value'] = values.fillna(values.mean())
            
        elif method == 'median':
            df_result.loc[mask, 'value'] = values.fillna(values.median())
    
    return df_result


def apply_outlier_detection(df, variables, method, **kwargs):
    """Detect outliers"""
    df_result = df.copy()
    outlier_info = {}
    
    for var in variables:
        mask = df_result['variable'] == var
        values = df_result.loc[mask, 'value'].dropna()
        
        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            factor = kwargs.get('iqr_factor', 1.5)
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outliers = (df_result.loc[mask, 'value'] < lower) | (df_result.loc[mask, 'value'] > upper)
            
        elif method == 'zscore':
            threshold = kwargs.get('zscore_threshold', 3)
            mean = values.mean()
            std = values.std()
            z_scores = np.abs((df_result.loc[mask, 'value'] - mean) / std)
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            threshold = kwargs.get('modified_zscore_threshold', 3.5)
            median = values.median()
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (df_result.loc[mask, 'value'] - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        outlier_count = outliers.sum() if hasattr(outliers, 'sum') else 0
        outlier_info[var] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(values) * 100) if len(values) > 0 else 0
        }
        
        if kwargs.get('remove_outliers', False):
            df_result = df_result[~(mask & outliers)]
    
    return df_result, outlier_info


def apply_smoothing(df, variables, method, **kwargs):
    """Apply smoothing"""
    df_result = df.copy()
    
    for var in variables:
        mask = df_result['variable'] == var
        values = df_result.loc[mask, 'value'].copy()
        
        if method == 'moving_average':
            window = kwargs.get('window_size', 3)
            smoothed = values.rolling(window=window, center=True, min_periods=1).mean()
            df_result.loc[mask, 'value'] = smoothed
            
        elif method == 'exponential':
            alpha = kwargs.get('alpha', 0.3)
            smoothed = values.ewm(alpha=alpha, adjust=False).mean()
            df_result.loc[mask, 'value'] = smoothed
            
        elif method == 'savitzky_golay':
            from scipy.signal import savgol_filter
            window = kwargs.get('window_size', 5)
            polyorder = kwargs.get('polyorder', 2)
            smoothed = savgol_filter(values.dropna(), window_length=window, polyorder=polyorder)
            df_result.loc[mask & df_result['value'].notna(), 'value'] = smoothed
    
    return df_result


def apply_normalization(df, variables, method, **kwargs):
    """Apply normalization/standardization"""
    df_result = df.copy()
    transform_info = {}
    
    for var in variables:
        mask = df_result['variable'] == var
        values = df_result.loc[mask, 'value'].copy()
        
        if method == 'min_max':
            min_val = kwargs.get('min_value', 0)
            max_val = kwargs.get('max_value', 1)
            original_min = values.min()
            original_max = values.max()
            
            if original_max > original_min:
                normalized = min_val + (values - original_min) * (max_val - min_val) / (original_max - original_min)
                df_result.loc[mask, 'value'] = normalized
            
            transform_info[var] = {
                'method': 'min_max',
                'original_min': original_min,
                'original_max': original_max,
                'target_min': min_val,
                'target_max': max_val
            }
            
        elif method == 'z_score':
            mean = values.mean()
            std = values.std()
            
            if std > 0:
                standardized = (values - mean) / std
                df_result.loc[mask, 'value'] = standardized
            
            transform_info[var] = {
                'method': 'z_score',
                'mean': mean,
                'std': std
            }
            
        elif method == 'robust':
            median = values.median()
            iqr = values.quantile(0.75) - values.quantile(0.25)
            
            if iqr > 0:
                robust_scaled = (values - median) / iqr
                df_result.loc[mask, 'value'] = robust_scaled
            
            transform_info[var] = {
                'method': 'robust',
                'median': median,
                'iqr': iqr
            }
    
    return df_result, transform_info


def apply_transformation(df, variables, method, **kwargs):
    """Apply mathematical transformation"""
    df_result = df.copy()
    
    for var in variables:
        mask = df_result['variable'] == var
        values = df_result.loc[mask, 'value'].copy()
        
        if method == 'log':
            base = kwargs.get('log_base', 'natural')
            if base == 'natural':
                transformed = np.log(values + 1e-10)
            elif base == 10:
                transformed = np.log10(values + 1e-10)
            else:
                transformed = np.log(values + 1e-10) / np.log(base)
            df_result.loc[mask, 'value'] = transformed
            
        elif method == 'sqrt':
            df_result.loc[mask, 'value'] = np.sqrt(np.maximum(values, 0))
            
        elif method == 'square':
            df_result.loc[mask, 'value'] = values ** 2
            
        elif method == 'cube_root':
            df_result.loc[mask, 'value'] = np.cbrt(values)
            
        elif method == 'box_cox':
            from scipy import stats
            try:
                transformed, lambda_param = stats.boxcox(values + 1 - values.min())
                df_result.loc[mask, 'value'] = transformed
            except:
                st.warning(f"Box-Cox failed for {var}, skipping")
    
    return df_result


def create_derived_variable(df, source_vars, operation, new_var_name):
    """Create derived variable from source variables"""
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    
    # Get all timestamps from source variables
    source_dfs = []
    for var in source_vars:
        var_df = df[df['variable'] == var][[time_col, 'value']].copy()
        var_df = var_df.rename(columns={'value': var})
        source_dfs.append(var_df)
    
    # Merge on timestamp
    merged = source_dfs[0]
    for var_df in source_dfs[1:]:
        merged = pd.merge(merged, var_df, on=time_col, how='outer')
    
    # Apply operation
    if operation == 'sum':
        merged[new_var_name] = merged[source_vars].sum(axis=1)
    elif operation == 'mean':
        merged[new_var_name] = merged[source_vars].mean(axis=1)
    elif operation == 'product':
        merged[new_var_name] = merged[source_vars].prod(axis=1)
    elif operation == 'difference':
        if len(source_vars) == 2:
            merged[new_var_name] = merged[source_vars[0]] - merged[source_vars[1]]
    elif operation == 'ratio':
        if len(source_vars) == 2:
            merged[new_var_name] = merged[source_vars[0]] / (merged[source_vars[1]] + 1e-10)
    
    # Convert back to long format
    new_df = pd.DataFrame({
        time_col: merged[time_col],
        'variable': new_var_name,
        'value': merged[new_var_name]
    })
    
    return pd.concat([df, new_df], ignore_index=True)


#---------------------
# Section 4: Preview Data
#---------------------

def preview_cleaned_data():
    """Show preview of cleaned data"""
    if st.session_state.df_clean is None or len(st.session_state.df_clean) == 0:
        st.info("No cleaned data available")
        return
    
    st.subheader("📊 Data Preview")
    
    # Variable selector
    available_vars = sorted(st.session_state.df_clean['variable'].unique())
    if not available_vars:
        st.info("No variables available")
        return
    
    selected_var = st.selectbox("Select variable to preview", available_vars, key="preview_var")
    
    if selected_var:
        time_col = st.session_state.get('time_column', 'timestamp')
        var_data = st.session_state.df_clean[st.session_state.df_clean['variable'] == selected_var].copy()
        var_data = var_data.sort_values(time_col)
        
        # Show stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", len(var_data))
        with col2:
            st.metric("Missing Values", var_data['value'].isna().sum())
        with col3:
            if var_data['value'].notna().any():
                st.metric("Mean", f"{var_data['value'].mean():.2f}")
        with col4:
            if var_data['value'].notna().any():
                st.metric("Std Dev", f"{var_data['value'].std():.2f}")
        
        # Show plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=var_data[time_col],
            y=var_data['value'],
            mode='lines+markers',
            name=selected_var,
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"Time Series: {selected_var}",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📋 View Data Table"):
            st.dataframe(var_data, use_container_width=True)


#---------------------
# Section 5: Tab 1 - Missing Values
#---------------------

def render_missing_values_tab():
    """Render missing values handling tab"""
    st.header("🔍 Missing Values Handling")
    
    if st.session_state.df_clean is None:
        st.info("Please load data first")
        return
    
    # Variable selection
    available_vars = sorted(st.session_state.df_clean['variable'].unique())
    selected_vars = st.multiselect("Select variables", available_vars, key="missing_vars")
    
    if not selected_vars:
        st.info("Select variables to process")
        return
    
    # Method selection
    method = st.selectbox(
        "Handling method",
        ["drop", "forward_fill", "backward_fill", "interpolate", "constant", "mean", "median"],
        key="missing_method"
    )
    
    # Additional parameters
    kwargs = {}
    if method == "interpolate":
        kwargs['interpolation_method'] = st.selectbox(
            "Interpolation method",
            ["linear", "polynomial", "spline"],
            key="interp_method"
        )
    elif method == "constant":
        kwargs['constant_value'] = st.number_input("Fill value", value=0.0, key="constant_val")
    
    # Preview button
    if st.button("🔍 Preview", key="preview_missing"):
        with st.spinner("Processing..."):
            df_preview = apply_missing_values_handling(
                st.session_state.df_clean,
                selected_vars,
                method,
                **kwargs
            )
            
            st.subheader("Before vs After")
            for var in selected_vars[:2]:  # Show first 2 variables
                col1, col2 = st.columns(2)
                
                time_col = st.session_state.get('time_column', 'timestamp')
                original = st.session_state.df_clean[st.session_state.df_clean['variable'] == var]
                processed = df_preview[df_preview['variable'] == var]
                
                with col1:
                    st.caption(f"Original: {var}")
                    orig_missing = original['value'].isna().sum()
                    st.metric("Missing", orig_missing)
                
                with col2:
                    st.caption(f"After: {var}")
                    new_missing = processed['value'].isna().sum()
                    st.metric("Missing", new_missing, delta=new_missing - orig_missing)
    
    # Apply button
    if st.button("✅ Apply", type="primary", key="apply_missing"):
        with st.spinner("Applying..."):
            st.session_state.df_clean = apply_missing_values_handling(
                st.session_state.df_clean,
                selected_vars,
                method,
                **kwargs
            )
            
            add_to_cleaning_history(
                "missing_values",
                method,
                selected_vars,
                kwargs
            )
            
            st.success(f"✅ Applied {method} to {len(selected_vars)} variables")
            st.rerun()


#---------------------
# Section 6: Tab 2 - Outliers
#---------------------

def render_outliers_tab():
    """Render outlier detection tab"""
    st.header("📊 Outlier Detection & Handling")
    
    if st.session_state.df_clean is None:
        st.info("Please load data first")
        return
    
    available_vars = sorted(st.session_state.df_clean['variable'].unique())
    selected_vars = st.multiselect("Select variables", available_vars, key="outlier_vars")
    
    if not selected_vars:
        st.info("Select variables to analyze")
        return
    
    method = st.selectbox(
        "Detection method",
        ["iqr", "zscore", "modified_zscore"],
        key="outlier_method"
    )
    
    kwargs = {}
    if method == "iqr":
        kwargs['iqr_factor'] = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1, key="iqr_factor")
    elif method == "zscore":
        kwargs['zscore_threshold'] = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1, key="z_thresh")
    elif method == "modified_zscore":
        kwargs['modified_zscore_threshold'] = st.slider("Modified Z-score threshold", 2.5, 5.0, 3.5, 0.1, key="mod_z_thresh")
    
    kwargs['remove_outliers'] = st.checkbox("Remove outliers", value=False, key="remove_outliers")
    
    if st.button("🔍 Detect", key="detect_outliers"):
        with st.spinner("Detecting outliers..."):
            df_result, outlier_info = apply_outlier_detection(
                st.session_state.df_clean,
                selected_vars,
                method,
                **kwargs
            )
            
            st.subheader("Outlier Statistics")
            outlier_data = []
            for var, info in outlier_info.items():
                outlier_data.append({
                    "Variable": var,
                    "Outliers": info['count'],
                    "Percentage": f"{info['percentage']:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(outlier_data), use_container_width=True, hide_index=True)
            
            if kwargs['remove_outliers']:
                if st.button("✅ Apply Removal", type="primary", key="apply_outlier_removal"):
                    st.session_state.df_clean = df_result
                    add_to_cleaning_history("outliers", method, selected_vars, kwargs)
                    st.success(f"✅ Removed outliers from {len(selected_vars)} variables")
                    st.rerun()


#---------------------
# Section 7: Tab 3 - Smoothing
#---------------------

def render_smoothing_tab():
    """Render smoothing tab"""
    st.header("📈 Smoothing")
    
    if st.session_state.df_clean is None:
        st.info("Please load data first")
        return
    
    available_vars = sorted(st.session_state.df_clean['variable'].unique())
    selected_vars = st.multiselect("Select variables", available_vars, key="smooth_vars")
    
    if not selected_vars:
        st.info("Select variables to smooth")
        return
    
    method = st.selectbox(
        "Smoothing method",
        ["moving_average", "exponential", "savitzky_golay"],
        key="smooth_method"
    )
    
    kwargs = {}
    if method == "moving_average":
        kwargs['window_size'] = st.slider("Window size", 3, 21, 5, 2, key="ma_window")
    elif method == "exponential":
        kwargs['alpha'] = st.slider("Alpha", 0.1, 0.9, 0.3, 0.05, key="exp_alpha")
    elif method == "savitzky_golay":
        kwargs['window_size'] = st.slider("Window size", 5, 21, 7, 2, key="sg_window")
        kwargs['polyorder'] = st.slider("Polynomial order", 2, 5, 2, 1, key="sg_poly")
    
    if st.button("✅ Apply", type="primary", key="apply_smoothing"):
        with st.spinner("Applying smoothing..."):
            st.session_state.df_clean = apply_smoothing(
                st.session_state.df_clean,
                selected_vars,
                method,
                **kwargs
            )
            
            add_to_cleaning_history("smoothing", method, selected_vars, kwargs)
            st.success(f"✅ Applied {method} to {len(selected_vars)} variables")
            st.rerun()


#---------------------
# Section 8: Tab 4 - Normalization
#---------------------

def render_normalization_tab():
    """Render normalization tab"""
    st.header("📏 Normalization & Standardization")
    
    if st.session_state.df_clean is None:
        st.info("Please load data first")
        return
    
    available_vars = sorted(st.session_state.df_clean['variable'].unique())
    selected_vars = st.multiselect("Select variables", available_vars, key="norm_vars")
    
    if not selected_vars:
        st.info("Select variables to normalize")
        return
    
    method = st.selectbox(
        "Normalization method",
        ["min_max", "z_score", "robust"],
        key="norm_method"
    )
    
    kwargs = {}
    if method == "min_max":
        kwargs['min_value'] = st.number_input("Min value", value=0.0, key="norm_min")
        kwargs['max_value'] = st.number_input("Max value", value=1.0, key="norm_max")
    
    if st.button("✅ Apply", type="primary", key="apply_norm"):
        with st.spinner("Applying normalization..."):
            df_result, transform_info = apply_normalization(
                st.session_state.df_clean,
                selected_vars,
                method,
                **kwargs
            )
            
            st.session_state.df_clean = df_result
            add_to_cleaning_history("normalization", method, selected_vars, {**kwargs, 'transform_info': transform_info})
            st.success(f"✅ Applied {method} to {len(selected_vars)} variables")
            st.rerun()


#---------------------
# Section 9: Tab 5 - Transformations & Derived Variables
#---------------------

def render_transformations_tab():
    """Render transformations tab"""
    st.header("🔄 Transformations & Derived Variables")
    
    if st.session_state.df_clean is None:
        st.info("Please load data first")
        return
    
    tab_a, tab_b = st.tabs(["Mathematical Transforms", "Derived Variables"])
    
    with tab_a:
        st.subheader("Mathematical Transformations")
        
        available_vars = sorted(st.session_state.df_clean['variable'].unique())
        selected_vars = st.multiselect("Select variables", available_vars, key="transform_vars")
        
        if selected_vars:
            method = st.selectbox(
                "Transformation",
                ["log", "sqrt", "square", "cube_root", "box_cox"],
                key="transform_method"
            )
            
            kwargs = {}
            if method == "log":
                kwargs['log_base'] = st.selectbox("Base", ["natural", 10, 2], key="log_base")
            
            if st.button("✅ Apply Transform", type="primary", key="apply_transform"):
                with st.spinner("Applying transformation..."):
                    st.session_state.df_clean = apply_transformation(
                        st.session_state.df_clean,
                        selected_vars,
                        method,
                        **kwargs
                    )
                    
                    add_to_cleaning_history("transformation", method, selected_vars, kwargs)
                    st.success(f"✅ Applied {method} to {len(selected_vars)} variables")
                    st.rerun()
    
    with tab_b:
        st.subheader("Create Derived Variables")
        
        available_vars = sorted(st.session_state.df_clean['variable'].unique())
        source_vars = st.multiselect("Select source variables", available_vars, key="derived_sources")
        
        if source_vars:
            operation = st.selectbox(
                "Operation",
                ["sum", "mean", "product", "difference", "ratio"],
                key="derived_op"
            )
            
            new_var_name = st.text_input("New variable name", key="derived_name")
            
            if st.button("✅ Create Variable", type="primary", key="create_derived") and new_var_name:
                with st.spinner("Creating derived variable..."):
                    st.session_state.df_clean = create_derived_variable(
                        st.session_state.df_clean,
                        source_vars,
                        operation,
                        new_var_name
                    )
                    
                    add_to_cleaning_history(
                        "derived_variable",
                        operation,
                        source_vars,
                        {'new_variable': new_var_name}
                    )
                    
                    st.success(f"✅ Created {new_var_name}")
                    st.rerun()


#---------------------
# Section 9B: Cleaning History Tab
#---------------------

def render_cleaning_history_tab():
    """Show cleaning operations history"""
    st.header("📜 Cleaning History")
    
    if not st.session_state.get("cleaning_history"):
        st.info("No operations performed yet")
        return
    
    st.subheader(f"Operations: {len(st.session_state.cleaning_history)}")
    
    ops_data = []
    for i, op in enumerate(reversed(st.session_state.cleaning_history)):
        new_cols = []
        if op['type'] == 'derived_variable':
            new_cols = [op['details'].get('new_variable', '')]
        
        ops_data.append({
            "#": len(st.session_state.cleaning_history) - i,
            "Time": op['timestamp'].strftime("%H:%M:%S"),
            "Type": op['type'].replace("_", " ").title(),
            "Method": op['method'],
            "Variables": ", ".join(op['variables'][:2]) + ("..." if len(op['variables']) > 2 else ""),
            "New Columns": ", ".join(new_cols[:2]) + ("..." if len(new_cols) > 2 else "")
        })
    
    st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)


#---------------------
# Section 10: Save to Database (FIXED - Proper timestamp checking and cleaned data save)
#---------------------

def save_to_database():
    """
    Save cleaned data to database with proper timestamp checking.
    
    This function:
    1. Identifies newly created cleaned variables (not in original data)
    2. Checks timestamps to avoid duplicates
    3. Saves ONLY new data points to database with data_source='cleaned'
    4. Updates parameters table for new cleaned variables
    5. Updates progress tracking
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

            # Check for existing cleaned data in database to avoid duplicates
            try:
                existing_cleaned = db.load_timeseries_data(project_id=project_id, data_source="cleaned")
                if existing_cleaned is not None and len(existing_cleaned) > 0:
                    # Get existing timestamps for each variable
                    for var in new_cleaned_vars:
                        var_existing = existing_cleaned[existing_cleaned['variable'] == var]
                        if len(var_existing) > 0:
                            existing_timestamps = set(var_existing['timestamp'])
                            var_new = cleaned_df_new[cleaned_df_new['variable'] == var]
                            # Keep only new timestamps
                            mask = ~var_new[time_col].isin(existing_timestamps)
                            cleaned_df_new = cleaned_df_new[
                                (cleaned_df_new['variable'] != var) | 
                                (cleaned_df_new[time_col].isin(var_new[mask][time_col]))
                            ]
            except Exception:
                # If no existing cleaned data, continue with all new data
                pass

            if len(cleaned_df_new) == 0:
                st.info("ℹ️ All cleaned data already exists in database")
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
            st.session_state.initial_data_hash = calculate_df_hash_fast(st.session_state.df_clean)
            st.session_state.current_data_hash = st.session_state.initial_data_hash

            st.session_state.cleaned_variables = list(
                set(st.session_state.get('cleaned_variables', []) + new_cleaned_vars)
            )

            st.success("🎉 Successfully appended cleaned data to database!")
            st.balloons()

            st.info(f"""
            **✅ Saved (append-only):**
            - {len(cleaned_df_new):,} new cleaned data points
            - {len(cleaned_params)} parameters updated for new cleaned variables
            """)

            time.sleep(1)
            # Force reload from DB to reflect persisted state
            st.session_state.data_loaded = False
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


#---------------------
# Main Function
#---------------------

def main():
    """Main page function"""
    
    # Check project
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project from the Home page")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Initialize
    initialize_cleaning_history()
    
    # Load data if not already loaded
    if not st.session_state.get('data_loaded'):
        with st.spinner("Loading project data from database..."):
            success = load_page3_data_on_page_load()
            if not success:
                st.error("Failed to load project data")
                st.stop()
    
    # Render unsaved changes banner (always visible)
    render_unsaved_changes_banner()
    
    # Metrics row - FIXED to show correct counts
    col1, col2, col3, col4 = st.columns(4)
    
    total_vars = len(st.session_state.get('value_columns', []))
    raw_vars = len(st.session_state.get('raw_variables', []))
    cleaned_vars = len(st.session_state.get('cleaned_variables', []))
    operations = len(st.session_state.get('cleaning_history', []))
    
    with col1:
        st.metric("Total Variables", total_vars)
    with col2:
        st.metric("Raw Variables", raw_vars)
    with col3:
        st.metric("Cleaned Variables", cleaned_vars)
    with col4:
        st.metric("Operations", operations)
    
    st.markdown("---")
    
    # Main tabs
    tabs = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers",
        "📈 Smoothing",
        "📏 Normalization",
        "🔄 Transformations",
        "📊 Preview",
        "📜 History"
    ])
    
    with tabs[0]:
        render_missing_values_tab()
    
    with tabs[1]:
        render_outliers_tab()
    
    with tabs[2]:
        render_smoothing_tab()
    
    with tabs[3]:
        render_normalization_tab()
    
    with tabs[4]:
        render_transformations_tab()
    
    with tabs[5]:
        preview_cleaned_data()
    
    with tabs[6]:
        render_cleaning_history_tab()


if __name__ == "__main__":
    main()
