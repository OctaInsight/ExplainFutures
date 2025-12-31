"""
Page 3: Data Cleaning & Preprocessing - COMPLETE UPDATE

UPDATES:
1. Metrics use comprehensive_health_report (comparable to Page 2)
2. "Go to Visualization" button added
3. Parameters health table with status colors
4. Replaced "Smoothing" tab with "Summary" tab
5. Enhanced scientific methods for cleaning
6. Two-column layout in tabs (methods + parameter selection)
7. Before/After comparison plots with export (PNG, PDF, HTML)
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
# SECTION 1: SESSION STATE RESET AND HELPER FUNCTIONS
# =============================================================================

def reset_page_session_state():
    """Reset session state, keeping only project/auth/progress keys"""
    keys_to_keep = {
        'current_project_id', 'current_project', 'selected_project',
        'project_name', 'project_description', 'project_created_at',
        'authenticated', 'user_id', 'user_email', 'user_name',
        'user_profile', 'session_id',
        'workflow_state', 'completion_percentage', 'current_page',
        'project_progress', 'step_completion',
        'sidebar_state', 'page_config'
    }
    
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        if key not in keys_to_keep:
            del st.session_state[key]


def calculate_data_hash(df):
    """Calculate hash for change detection"""
    if df is None or len(df) == 0:
        return None
    try:
        return hashlib.md5(df.to_json().encode()).hexdigest()
    except:
        return None


def initialize_cleaning_history():
    """Initialize cleaning operation history"""
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
# SECTION 2: GET COMPREHENSIVE HEALTH REPORT (FROM PAGE 2)
# =============================================================================

def get_comprehensive_health_report():
    """Get comprehensive health report from parameters (SAME as Page 2)"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return None
    
    try:
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if not parameters:
            return None
        
        total_missing = 0
        total_count = 0
        missing_values_detail = {}
        
        for param in parameters:
            param_name = param['parameter_name']
            missing_count = param.get('missing_count', 0)
            param_total = param.get('total_count', 0)
            
            total_missing += missing_count
            total_count += param_total
            
            if param_total > 0:
                missing_pct = missing_count / param_total
                missing_values_detail[param_name] = {
                    'count': missing_count,
                    'percentage': missing_pct,
                    'total': param_total
                }
        
        health_score = 100
        issues = []
        
        for var, info in missing_values_detail.items():
            missing_pct = info['percentage']
            if missing_pct > 0.20:
                health_score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing (critical)")
            elif missing_pct > 0.05:
                health_score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing")
        
        health_score = max(0, min(100, health_score))
        
        if health_score >= 85:
            category = "excellent"
        elif health_score >= 70:
            category = "good"
        elif health_score >= 50:
            category = "fair"
        else:
            category = "poor"
        
        return {
            'health_score': health_score,
            'health_category': category,
            'total_parameters': len(parameters),
            'total_data_points': total_count,
            'total_missing_values': total_missing,
            'missing_percentage': total_missing / total_count if total_count > 0 else 0,
            'critical_issues': len([i for i in issues if 'critical' in i.lower()]),
            'warnings': len([i for i in issues if 'warning' in i.lower()]),
            'missing_values_detail': missing_values_detail,
            'issues_list': issues,
            'parameters_analyzed': [p['parameter_name'] for p in parameters]
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# =============================================================================
# SECTION 3: LOAD DATA FROM DATABASE
# =============================================================================

def load_project_data_from_database():
    """Load ALL project data from database on page load"""
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        st.error("❌ No project selected")
        return False
    
    try:
        # Load RAW data
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
            st.error("❌ No raw data found. Please upload data on Page 2.")
            return False
        
        st.session_state.df_raw = df_raw.copy()
        
        # Load CLEANED data
        df_cleaned_db = None
        try:
            df_cleaned_db = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            if df_cleaned_db is not None and len(df_cleaned_db) > 0:
                st.session_state.df_cleaned_db = df_cleaned_db.copy()
            else:
                st.session_state.df_cleaned_db = pd.DataFrame(columns=df_raw.columns)
        except Exception:
            st.session_state.df_cleaned_db = pd.DataFrame(columns=df_raw.columns)
        
        # Create working dataframe
        if df_cleaned_db is not None and len(df_cleaned_db) > 0:
            st.session_state.df_working = pd.concat([df_raw, df_cleaned_db], ignore_index=True)
        else:
            st.session_state.df_working = df_raw.copy()
        
        # Load parameters
        try:
            parameters = db.get_project_parameters(project_id)
            st.session_state.parameters_df = pd.DataFrame(parameters) if parameters else pd.DataFrame()
        except Exception:
            st.session_state.parameters_df = pd.DataFrame()
        
        # Extract variable lists
        st.session_state.raw_variables = sorted(df_raw['variable'].unique().tolist())
        
        if df_cleaned_db is not None and len(df_cleaned_db) > 0:
            cleaned_vars = df_cleaned_db['variable'].unique().tolist()
            st.session_state.cleaned_variables = sorted([v for v in cleaned_vars if v not in st.session_state.raw_variables])
        else:
            st.session_state.cleaned_variables = []
        
        st.session_state.all_variables = sorted(st.session_state.df_working['variable'].unique().tolist())
        
        # Time column
        time_col = 'timestamp' if 'timestamp' in df_raw.columns else 'time'
        st.session_state.time_column = time_col
        
        # Hash for change tracking
        st.session_state.initial_data_hash = calculate_data_hash(st.session_state.df_working)
        st.session_state.page3_data_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return False


# =============================================================================
# SECTION 4: EXPORT FUNCTION (FROM PAGE 7)
# =============================================================================

def quick_export_buttons(fig, filename_prefix, formats=['png', 'pdf', 'html']):
    """Export figure in multiple formats (from Page 7)"""
    cols = st.columns(len(formats))
    
    for idx, fmt in enumerate(formats):
        with cols[idx]:
            if fmt == 'png':
                png_bytes = pio.to_image(fig, format='png', width=1200, height=600)
                st.download_button(
                    label="📊 PNG",
                    data=png_bytes,
                    file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            elif fmt == 'pdf':
                pdf_bytes = pio.to_image(fig, format='pdf', width=1200, height=600)
                st.download_button(
                    label="📄 PDF",
                    data=pdf_bytes,
                    file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            elif fmt == 'html':
                html_bytes = pio.to_html(fig, include_plotlyjs='cdn').encode()
                st.download_button(
                    label="🌐 HTML",
                    data=html_bytes,
                    file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )


def plot_before_after_comparison(df_all, var_raw, var_cleaned):
    """Create before/after comparison plot"""
    try:
        time_col = st.session_state.get('time_column', 'timestamp')
        
        data_raw = df_all[df_all['variable'] == var_raw].copy().sort_values(time_col)
        data_cleaned = df_all[df_all['variable'] == var_cleaned].copy().sort_values(time_col)
        
        fig = go.Figure()
        
        if len(data_raw) > 0:
            fig.add_trace(go.Scatter(
                x=data_raw[time_col],
                y=data_raw['value'],
                mode='lines+markers',
                name=f'{var_raw} (Raw)',
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
            title=f"Before vs After: {var_raw} → {var_cleaned}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Plot error: {str(e)}")
        return None


# =============================================================================
# SECTION 5: ENHANCED CLEANING FUNCTIONS (MORE SCIENTIFIC METHODS)
# =============================================================================

def handle_missing_values(df, variables, method, new_var_names=None, **kwargs):
    """
    Handle missing values by creating NEW cleaned variables
    Returns: (updated_df, list_of_new_variable_names)
    """
    df_copy = df.copy()
    new_columns = []
    
    for idx, var in enumerate(variables):
        # Create new variable name
        if new_var_names and idx < len(new_var_names):
            new_var_name = new_var_names[idx]
        else:
            method_short = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
            new_var_name = f"{var}_{method_short}_missing"
        
        new_columns.append(new_var_name)
        
        # Get original variable data
        var_data = df[df['variable'] == var].copy()
        values = var_data['value'].copy()
        
        # Create new cleaned variable
        cleaned_data = var_data.copy()
        cleaned_data['variable'] = new_var_name
        
        # Apply cleaning method
        if method == "Drop Rows":
            cleaned_data = cleaned_data[cleaned_data['value'].notna()]
        elif method == "Forward Fill (LOCF)":
            cleaned_data.loc[:, 'value'] = values.fillna(method='ffill')
        elif method == "Backward Fill (NOCB)":
            cleaned_data.loc[:, 'value'] = values.fillna(method='bfill')
        elif method == "Linear Interpolation":
            cleaned_data.loc[:, 'value'] = values.interpolate(method='linear')
        elif method == "Cubic Spline Interpolation":
            if len(values.dropna()) >= 4:
                cleaned_data.loc[:, 'value'] = values.interpolate(method='cubic')
            else:
                cleaned_data.loc[:, 'value'] = values.interpolate(method='linear')
        elif method == "Polynomial Interpolation":
            order = kwargs.get('poly_order', 2)
            if len(values.dropna()) >= order + 1:
                cleaned_data.loc[:, 'value'] = values.interpolate(method='polynomial', order=order)
            else:
                cleaned_data.loc[:, 'value'] = values.interpolate(method='linear')
        elif method == "Mean Imputation":
            cleaned_data.loc[:, 'value'] = values.fillna(values.mean())
        elif method == "Median Imputation":
            cleaned_data.loc[:, 'value'] = values.fillna(values.median())
        elif method == "Mode Imputation":
            mode_val = values.mode()[0] if len(values.mode()) > 0 else values.mean()
            cleaned_data.loc[:, 'value'] = values.fillna(mode_val)
        elif method == "KNN Imputation":
            from sklearn.impute import KNNImputer
            k = kwargs.get('k_neighbors', 5)
            imputer = KNNImputer(n_neighbors=k)
            vals_array = values.values.reshape(-1, 1)
            imputed = imputer.fit_transform(vals_array)
            cleaned_data.loc[:, 'value'] = imputed.flatten()
        elif method == "Constant Value":
            fill_val = kwargs.get('fill_value', 0)
            cleaned_data.loc[:, 'value'] = values.fillna(fill_val)
        
        # Add cleaned data to dataframe
        df_copy = pd.concat([df_copy, cleaned_data], ignore_index=True)
    
    return df_copy, new_columns


def handle_outliers(df, variables, method, new_var_names=None, **kwargs):
    """
    Handle outliers by creating NEW cleaned variables
    Returns: (updated_df, list_of_new_variable_names)
    """
    df_copy = df.copy()
    new_columns = []
    
    for idx, var in enumerate(variables):
        # Create new variable name
        if new_var_names and idx < len(new_var_names):
            new_var_name = new_var_names[idx]
        else:
            method_short = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
            new_var_name = f"{var}_{method_short}_outlier"
        
        new_columns.append(new_var_name)
        
        # Get original variable data
        var_data = df[df['variable'] == var].copy()
        values = var_data['value'].dropna()
        
        if len(values) == 0:
            continue
        
        # Create new cleaned variable
        cleaned_data = var_data.copy()
        cleaned_data['variable'] = new_var_name
        
        # Apply outlier detection and handling
        if method == "IQR Method":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            factor = kwargs.get('iqr_factor', 1.5)
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outlier_mask = (cleaned_data['value'] < lower) | (cleaned_data['value'] > upper)
            
            action = kwargs.get('action', 'remove')
            if action == 'remove':
                cleaned_data = cleaned_data[~outlier_mask]
            elif action == 'cap':
                cleaned_data.loc[outlier_mask & (cleaned_data['value'] < lower), 'value'] = lower
                cleaned_data.loc[outlier_mask & (cleaned_data['value'] > upper), 'value'] = upper
        
        elif method == "Z-Score Method":
            threshold = kwargs.get('z_threshold', 3.0)
            mean = values.mean()
            std = values.std()
            if std > 0:
                z_scores = np.abs((cleaned_data['value'] - mean) / std)
                outlier_mask = z_scores > threshold
                
                action = kwargs.get('action', 'remove')
                if action == 'remove':
                    cleaned_data = cleaned_data[~outlier_mask]
                elif action == 'cap':
                    cap_val_low = mean - threshold * std
                    cap_val_high = mean + threshold * std
                    cleaned_data.loc[outlier_mask & (cleaned_data['value'] < cap_val_low), 'value'] = cap_val_low
                    cleaned_data.loc[outlier_mask & (cleaned_data['value'] > cap_val_high), 'value'] = cap_val_high
        
        elif method == "Modified Z-Score":
            threshold = kwargs.get('modified_z_threshold', 3.5)
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad > 0:
                modified_z = 0.6745 * (cleaned_data['value'] - median) / mad
                outlier_mask = np.abs(modified_z) > threshold
                
                action = kwargs.get('action', 'remove')
                if action == 'remove':
                    cleaned_data = cleaned_data[~outlier_mask]
        
        elif method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            contamination = kwargs.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            vals_clean = cleaned_data['value'].dropna()
            if len(vals_clean) > 0:
                vals_array = vals_clean.values.reshape(-1, 1)
                predictions = clf.fit_predict(vals_array)
                outlier_mask = predictions == -1
                
                if kwargs.get('action', 'remove') == 'remove':
                    valid_indices = cleaned_data[cleaned_data['value'].notna()].index[~outlier_mask]
                    cleaned_data = cleaned_data.loc[valid_indices]
        
        # Add cleaned data to dataframe
        df_copy = pd.concat([df_copy, cleaned_data], ignore_index=True)
    
    return df_copy, new_columns


def apply_transformations(df, variables, transformation, new_var_names=None, **kwargs):
    """
    Apply mathematical transformations by creating NEW variables
    Returns: (updated_df, list_of_new_variable_names)
    """
    df_copy = df.copy()
    new_columns = []
    
    for idx, var in enumerate(variables):
        # Create new variable name
        if new_var_names and idx < len(new_var_names):
            new_var_name = new_var_names[idx]
        else:
            transform_short = transformation.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            new_var_name = f"{var}_{transform_short}_transform"
        
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
        
        elif "Box-Cox" in transformation:
            from scipy import stats
            vals_clean = values.dropna()
            if len(vals_clean) > 0 and vals_clean.min() > 0:
                transformed_vals, _ = stats.boxcox(vals_clean)
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = transformed_vals
        
        elif "Yeo-Johnson" in transformation:
            from scipy import stats
            vals_clean = values.dropna()
            if len(vals_clean) > 0:
                transformed_vals, _ = stats.yeojohnson(vals_clean)
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = transformed_vals
        
        elif "Standardize" in transformation or "Z-Score" in transformation:
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
        
        elif "Robust Scaling" in transformation:
            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                transformed_data.loc[transformed_data['value'].notna(), 'value'] = \
                    (transformed_data.loc[transformed_data['value'].notna(), 'value'] - median) / iqr
        
        df_copy = pd.concat([df_copy, transformed_data], ignore_index=True)
    
    return df_copy, new_columns


# =============================================================================
# SECTION 6: SAVE TO DATABASE
# =============================================================================

def save_to_database():
    """
    Save cleaned data with append-only logic
    FIXED: Properly identifies cleaned variables and updates session state
    """
    project_id = st.session_state.get('current_project_id')
    
    if not project_id or not DB_AVAILABLE:
        st.error("❌ Cannot save: No project or database unavailable")
        return
    
    with st.spinner("💾 Saving to database..."):
        try:
            # Get current working data
            df_working = st.session_state.get('df_working')
            if df_working is None or len(df_working) == 0:
                st.warning("⚠️ No working data available")
                return
            
            # Identify NEW cleaned variables (not in raw data)
            all_vars_in_working = df_working['variable'].unique().tolist()
            raw_vars = st.session_state.get('raw_variables', [])
            
            # New cleaned variables are those in working data but NOT in raw data
            new_cleaned_vars = [v for v in all_vars_in_working if v not in raw_vars]
            
            if not new_cleaned_vars:
                st.warning("⚠️ No new cleaned variables to save")
                st.info("💡 Cleaned variables are created when you apply transformations or derive new columns")
                return
            
            st.info(f"📊 Found {len(new_cleaned_vars)} new cleaned variables: {', '.join(new_cleaned_vars[:3])}{'...' if len(new_cleaned_vars) > 3 else ''}")
            
            # Extract data for new cleaned variables
            time_col = st.session_state.get('time_column', 'timestamp')
            cleaned_df_new = df_working[
                df_working['variable'].isin(new_cleaned_vars)
            ].copy()
            
            # Check for existing cleaned data to avoid duplicates
            existing_cleaned = st.session_state.get('df_cleaned_db')
            if existing_cleaned is not None and len(existing_cleaned) > 0:
                st.info(f"🔍 Checking for duplicates in {len(existing_cleaned)} existing cleaned records...")
                
                for var in new_cleaned_vars:
                    var_existing = existing_cleaned[existing_cleaned['variable'] == var]
                    if len(var_existing) > 0:
                        existing_timestamps = set(var_existing[time_col].values)
                        # Remove rows that already exist
                        cleaned_df_new = cleaned_df_new[
                            ~((cleaned_df_new['variable'] == var) & 
                              (cleaned_df_new[time_col].isin(existing_timestamps)))
                        ]
            
            if len(cleaned_df_new) == 0:
                st.info("ℹ️ All cleaned data already exists in database")
                return
            
            st.info(f"💾 Saving {len(cleaned_df_new):,} new records...")
            
            # Save to database
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
                st.error("❌ Failed to save to timeseries_data table")
                return
            
            # Update parameters table
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
                st.info(f"✅ Updated {len(cleaned_params)} parameters")
            
            # Update progress: step_key='data_cleaned', step_percent=7
            try:
                # Method 1: Use upsert_progress_step if available
                if hasattr(db, "upsert_progress_step"):
                    db.upsert_progress_step(
                        project_id=project_id,
                        step_key="data_cleaned",
                        step_percent=7
                    )
                    st.info("✅ Progress step updated: data_cleaned = 7%")
                else:
                    # Method 2: Direct database upsert
                    db.client.table("project_progress_steps").upsert({
                        "project_id": project_id,
                        "step_key": "data_cleaned",
                        "step_percent": 7,
                        "updated_at": datetime.now().isoformat()
                    }, on_conflict="project_id,step_key").execute()
                    st.info("✅ Progress step updated (direct): data_cleaned = 7%")
                
                # Recompute total progress
                if hasattr(db, "recompute_and_update_project_progress"):
                    db.recompute_and_update_project_progress(
                        project_id=project_id,
                        workflow_state="preprocessing",
                        current_page=3
                    )
                    st.info("✅ Total project progress recomputed")
            except Exception as e:
                st.warning(f"⚠️ Progress update failed: {e}")
            
            # Update session state
            st.session_state.has_unsaved_changes = False
            
            # CRITICAL: Clear the cleaning history after successful save
            st.session_state.cleaning_history = []
            
            # Update cleaned variables in database list
            # After save, the new variables are now in the database
            current_db_cleaned = st.session_state.get('df_cleaned_db')
            if current_db_cleaned is not None and len(current_db_cleaned) > 0:
                # Merge new cleaned data with existing
                st.session_state.df_cleaned_db = pd.concat([current_db_cleaned, cleaned_df_new], ignore_index=True)
            else:
                st.session_state.df_cleaned_db = cleaned_df_new.copy()
            
            # Success message
            st.success(f"""
            🎉 **Successfully saved to database!**
            - **Records:** {len(cleaned_df_new):,}
            - **Variables:** {len(new_cleaned_vars)}
            - **Parameters:** {len(cleaned_params)} updated
            - **Progress:** Step 'data_cleaned' = 7%
            """)
            
            st.balloons()
            
            time.sleep(2)
            
            # Reload data from database to refresh state
            st.session_state.page3_data_loaded = False
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Save error: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())



# =============================================================================
# SECTION 7: MAIN FUNCTION
# =============================================================================

def main():
    """Main function"""
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Reset and load data
    if not st.session_state.get('page3_data_loaded', False):
        reset_page_session_state()
        initialize_cleaning_history()
        
        with st.spinner("📊 Loading data..."):
            success = load_project_data_from_database()
            if not success:
                st.stop()
    
    # Get comprehensive health report (SAME as Page 2)
    comprehensive_report = get_comprehensive_health_report()
    
    if not comprehensive_report:
        st.error("❌ No health report available")
        st.stop()
    
    # UPDATE 1: Metrics from comprehensive health report
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Parameters", comprehensive_report.get('total_parameters', 0))
    
    # Count raw vs cleaned data points
    df_raw = st.session_state.get('df_raw')
    df_cleaned_db = st.session_state.get('df_cleaned_db')
    raw_count = len(df_raw) if df_raw is not None else 0
    cleaned_count = len(df_cleaned_db) if df_cleaned_db is not None else 0
    
    col2.metric("Raw Data", f"{raw_count:,}")
    col3.metric("Cleaned Data", f"{cleaned_count:,}")
    col4.metric("Operations", len(st.session_state.get('cleaning_history', [])))
    
    st.markdown("---")
    
    # UPDATE 2: Save button + Go to Visualization button
    st.markdown("### 💾 Save & Continue")
    
    # Get current cleaned variables in session
    cleaned_vars_in_session = st.session_state.get('cleaned_variables', [])
    
    # Get cleaned variables already saved in database
    df_cleaned_db = st.session_state.get('df_cleaned_db')
    cleaned_vars_in_db = []
    if df_cleaned_db is not None and len(df_cleaned_db) > 0:
        cleaned_vars_in_db = df_cleaned_db['variable'].unique().tolist()
    
    # Calculate NEW cleaned variables (in session but NOT in database)
    new_cleaned_vars_count = len([v for v in cleaned_vars_in_session if v not in cleaned_vars_in_db])
    
    # Show save section only if there are NEW cleaned variables to save
    if new_cleaned_vars_count > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Show accurate count of NEW cleaned variables
            st.info(f"📊 {new_cleaned_vars_count} cleaned variable(s) ready to save")
            
            if st.button("💾 Save to Database", type="primary", use_container_width=True, key="save_btn"):
                save_to_database()
        
        with col2:
            if st.button("📊 Go to Data Visualization", type="secondary", use_container_width=True, key="viz_btn"):
                st.switch_page("pages/04_Exploration_and_Visualization.py")
    else:
        # No new cleaned variables - show only visualization button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if cleaned_vars_in_session and len(cleaned_vars_in_session) > 0:
                st.success(f"✅ All {len(cleaned_vars_in_session)} cleaned variable(s) already saved to database")
            else:
                st.info("ℹ️ Apply cleaning operations to create cleaned variables")
        
        with col2:
            if st.button("📊 Go to Visualization", type="secondary", use_container_width=True, key="viz_btn_alt"):
                st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    st.markdown("---")
    
    # UPDATE 3: Parameters Health Table (from Page 2)
    st.markdown("### 📋 Parameters Health Status")
    
    missing_detail = comprehensive_report.get('missing_values_detail', {})
    
    if missing_detail:
        param_health_data = []
        for var, info in missing_detail.items():
            missing_count = info.get('count', 0)
            total_count = info.get('total', 0)
            missing_pct = info.get('percentage', 0)
            
            # Determine if raw or cleaned
            is_raw = var in st.session_state.get('raw_variables', [])
            is_cleaned = var in st.session_state.get('cleaned_variables', [])
            data_type = "Raw" if is_raw and not is_cleaned else "Cleaned" if is_cleaned else "Both"
            
            # Health status with color
            if missing_pct > 0.20:
                status = "🔴 Critical"
                problem = f"High missing: {missing_pct*100:.1f}%"
            elif missing_pct > 0.05:
                status = "🟡 Warning"
                problem = f"Missing: {missing_pct*100:.1f}%"
            else:
                status = "🟢 Healthy"
                problem = "No issues"
            
            param_health_data.append({
                "Parameter": var,
                "Type": data_type,
                "Missing": f"{missing_count:,}",
                "Total": f"{total_count:,}",
                "Missing %": f"{missing_pct*100:.1f}%",
                "Problem": problem,
                "Status": status
            })
        
        st.dataframe(pd.DataFrame(param_health_data), use_container_width=True, hide_index=True)
    else:
        st.success("✅ All parameters healthy!")
    
    st.markdown("---")
    
    # UPDATE 4: Tabs with enhanced layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Missing Values",
        "📊 Outliers",
        "🔄 Transformations",
        "📋 Summary"
    ])
    
    # TAB 1: Missing Values
    with tab1:
        st.header("🔍 Missing Values Handling")
        
        # Two-column layout
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 📐 Method Selection")
            
            method = st.selectbox(
                "Imputation Method",
                [
                    "Drop Rows",
                    "Forward Fill (LOCF)",
                    "Backward Fill (NOCB)",
                    "Linear Interpolation",
                    "Cubic Spline Interpolation",
                    "Polynomial Interpolation",
                    "Mean Imputation",
                    "Median Imputation",
                    "Mode Imputation",
                    "KNN Imputation",
                    "Constant Value"
                ],
                key="miss_method"
            )
            
            kwargs = {}
            if method == "Polynomial Interpolation":
                kwargs['poly_order'] = st.slider("Polynomial Order", 2, 5, 2)
            elif method == "KNN Imputation":
                kwargs['k_neighbors'] = st.slider("K Neighbors", 3, 10, 5)
            elif method == "Constant Value":
                kwargs['fill_value'] = st.number_input("Fill Value", value=0.0)
            
            st.markdown("---")
            st.markdown("**Method Description:**")
            if "Interpolation" in method:
                st.info("Estimates missing values using surrounding data points")
            elif "Imputation" in method:
                st.info("Fills missing values with statistical measures")
            elif "Fill" in method:
                st.info("Propagates values forward or backward in time")
        
        with col_right:
            st.markdown("#### 📊 Parameters with Missing Values")
            
            # Filter parameters with missing values
            params_with_missing = []
            if missing_detail:
                for var, info in missing_detail.items():
                    if info.get('count', 0) > 0:
                        params_with_missing.append(var)
            
            if params_with_missing:
                selected_vars = st.multiselect(
                    "Select parameters to clean",
                    params_with_missing,
                    key="miss_vars"
                )
                
                if selected_vars:
                    st.markdown("**Selected:**")
                    for var in selected_vars:
                        info = missing_detail[var]
                        st.text(f"• {var}: {info['count']:,} missing ({info['percentage']*100:.1f}%)")
            else:
                st.success("✅ No missing values detected!")
                selected_vars = []
        
        # Naming section
        new_var_names = []
        if selected_vars:
            st.markdown("---")
            st.markdown("#### ✏️ Name Your Cleaned Variables")
            st.caption("Edit the suggested names or keep defaults")
            
            for var in selected_vars:
                method_short = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
                default_name = f"{var}_{method_short}_missing"
                
                new_name = st.text_input(
                    f"Cleaned name for '{var}'",
                    value=default_name,
                    key=f"miss_name_{var}"
                )
                new_var_names.append(new_name)
        
        # Apply button
        if selected_vars:
            st.markdown("---")
            if st.button("✅ Apply Cleaning", type="primary", use_container_width=True, key="apply_miss"):
                df_result, new_cols = handle_missing_values(
                    st.session_state.df_working, selected_vars, method, new_var_names, **kwargs
                )
                st.session_state.df_working = df_result
                
                # Update cleaned variables list immediately
                current_cleaned = st.session_state.get('cleaned_variables', [])
                st.session_state.cleaned_variables = list(set(current_cleaned + new_cols))
                
                # Update all_variables list
                st.session_state.all_variables = sorted(df_result['variable'].unique().tolist())
                
                add_to_cleaning_history("missing_values", method, selected_vars, {'new_columns': new_cols})
                
                st.success(f"✅ Created {len(new_cols)} cleaned variables: {', '.join(new_cols)}")
                st.info("💡 Click 'Save to Database' to persist these cleaned variables")
                st.rerun()
    
    # TAB 2: Outliers
    with tab2:
        st.header("📊 Outlier Detection & Handling")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 📐 Method Selection")
            
            method = st.selectbox(
                "Detection Method",
                [
                    "IQR Method",
                    "Z-Score Method",
                    "Modified Z-Score",
                    "Isolation Forest"
                ],
                key="out_method"
            )
            
            action = st.selectbox("Action", ["remove", "cap"], key="out_action")
            
            kwargs = {'action': action}
            if method == "IQR Method":
                kwargs['iqr_factor'] = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
            elif method == "Z-Score Method":
                kwargs['z_threshold'] = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
            elif method == "Modified Z-Score":
                kwargs['modified_z_threshold'] = st.slider("Modified Z Threshold", 2.5, 5.0, 3.5, 0.1)
            elif method == "Isolation Forest":
                kwargs['contamination'] = st.slider("Contamination", 0.01, 0.2, 0.1, 0.01)
        
        with col_right:
            st.markdown("#### 📊 Select Parameters")
            
            raw_vars = st.session_state.get('raw_variables', [])
            selected_vars = st.multiselect(
                "Parameters to analyze",
                raw_vars,
                key="out_vars"
            )
        
        # Naming section
        new_var_names = []
        if selected_vars:
            st.markdown("---")
            st.markdown("#### ✏️ Name Your Cleaned Variables")
            st.caption("Edit the suggested names or keep defaults")
            
            for var in selected_vars:
                method_short = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
                default_name = f"{var}_{method_short}_outlier"
                
                new_name = st.text_input(
                    f"Cleaned name for '{var}'",
                    value=default_name,
                    key=f"out_name_{var}"
                )
                new_var_names.append(new_name)
        
        if selected_vars:
            st.markdown("---")
            if st.button("✅ Apply Cleaning", type="primary", use_container_width=True, key="apply_out"):
                df_result, new_cols = handle_outliers(
                    st.session_state.df_working, selected_vars, method, new_var_names, **kwargs
                )
                st.session_state.df_working = df_result
                
                # Update cleaned variables list immediately
                current_cleaned = st.session_state.get('cleaned_variables', [])
                st.session_state.cleaned_variables = list(set(current_cleaned + new_cols))
                
                # Update all_variables list
                st.session_state.all_variables = sorted(df_result['variable'].unique().tolist())
                
                add_to_cleaning_history("outliers", method, selected_vars, {'new_columns': new_cols, **kwargs})
                
                st.success(f"✅ Created {len(new_cols)} cleaned variables: {', '.join(new_cols)}")
                st.info("💡 Click 'Save to Database' to persist these cleaned variables")
                st.rerun()
    
    # TAB 3: Transformations
    with tab3:
        st.header("🔄 Mathematical Transformations")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 📐 Transformation Type")
            
            transformation = st.selectbox(
                "Transform",
                [
                    "Log Transform",
                    "Square Root Transform",
                    "Box-Cox Transform",
                    "Yeo-Johnson Transform",
                    "Standardize (Z-Score)",
                    "Min-Max Normalize",
                    "Robust Scaling"
                ],
                key="trans_method"
            )
            
            st.markdown("---")
            st.markdown("**Why Transform?**")
            if "Log" in transformation:
                st.info("Reduces right skewness, stabilizes variance")
            elif "Box-Cox" in transformation:
                st.info("Automatically finds best power transformation")
            elif "Standardize" in transformation:
                st.info("Centers data with mean=0, std=1")
            elif "Min-Max" in transformation:
                st.info("Scales data to [0, 1] range")
        
        with col_right:
            st.markdown("#### 📊 Select Parameters")
            
            raw_vars = st.session_state.get('raw_variables', [])
            selected_vars = st.multiselect(
                "Parameters to transform",
                raw_vars,
                key="trans_vars"
            )
        
        # Naming section
        new_var_names = []
        if selected_vars:
            st.markdown("---")
            st.markdown("#### ✏️ Name Your Transformed Variables")
            st.caption("Edit the suggested names or keep defaults")
            
            for var in selected_vars:
                transform_short = transformation.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                default_name = f"{var}_{transform_short}_transform"
                
                new_name = st.text_input(
                    f"Transformed name for '{var}'",
                    value=default_name,
                    key=f"trans_name_{var}"
                )
                new_var_names.append(new_name)
        
        if selected_vars:
            st.markdown("---")
            if st.button("✅ Apply Transformation", type="primary", use_container_width=True, key="apply_trans"):
                df_result, new_cols = apply_transformations(
                    st.session_state.df_working, selected_vars, transformation, new_var_names
                )
                st.session_state.df_working = df_result
                
                # Update cleaned variables list immediately
                current_cleaned = st.session_state.get('cleaned_variables', [])
                st.session_state.cleaned_variables = list(set(current_cleaned + new_cols))
                
                # Update all_variables list
                st.session_state.all_variables = sorted(df_result['variable'].unique().tolist())
                
                add_to_cleaning_history("transformation", transformation, selected_vars, {'new_columns': new_cols})
                
                st.success(f"✅ Created {len(new_cols)} transformed variables: {', '.join(new_cols)}")
                st.info("💡 Click 'Save to Database' to persist these cleaned variables")
                st.rerun()
    
    # TAB 4: Summary
    with tab4:
        st.header("📋 Cleaning Operations Summary")
        
        if not st.session_state.get('cleaning_history'):
            st.info("ℹ️ No operations performed yet")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Operations", len(st.session_state.cleaning_history))
            
            modified_vars = set()
            for op in st.session_state.cleaning_history:
                modified_vars.update(op['variables'])
            col2.metric("Parameters Modified", len(modified_vars))
            
            total_new = sum(len(op.get('details', {}).get('new_columns', [])) 
                           for op in st.session_state.cleaning_history)
            col3.metric("New Variables Created", total_new)
            
            st.markdown("---")
            st.subheader("Operation Log")
            
            ops_data = []
            for i, op in enumerate(st.session_state.cleaning_history, 1):
                new_cols = op.get('details', {}).get('new_columns', [])
                ops_data.append({
                    "#": i,
                    "Type": op['type'].replace('_', ' ').title(),
                    "Method": op['method'],
                    "Parameters": ", ".join(op['variables'][:3]) + ("..." if len(op['variables']) > 3 else ""),
                    "New Variables": ", ".join(new_cols[:2]) + ("..." if len(new_cols) > 2 else "") if new_cols else "-"
                })
            
            st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)
    
    # Comparison Plot Section (outside tabs)
    st.markdown("---")
    st.markdown("### 📊 Before/After Comparison")
    
    raw_vars = st.session_state.get('raw_variables', [])
    cleaned_vars = st.session_state.get('cleaned_variables', [])
    
    if cleaned_vars and raw_vars:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_raw = st.selectbox("Raw Data", raw_vars, key="cmp_raw")
        
        with col2:
            selected_cleaned = st.selectbox("Cleaned Data", cleaned_vars, key="cmp_clean")
        
        with col3:
            if st.button("🔍 Plot", use_container_width=True, type="primary"):
                st.session_state.show_comparison = True
        
        if st.session_state.get('show_comparison', False):
            fig = plot_before_after_comparison(
                st.session_state.df_working,
                selected_raw,
                selected_cleaned
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("💾 Export Figure"):
                    quick_export_buttons(fig, f"comparison_{selected_raw}_vs_{selected_cleaned}", ['png', 'pdf', 'html'])
            else:
                st.error("❌ Could not create comparison plot")
    elif not cleaned_vars:
        st.info("ℹ️ **No cleaned variables available yet**")
        st.markdown("""
        **To create cleaned variables:**
        1. Go to any cleaning tab (Missing Values, Outliers, or Transformations)
        2. Select parameters and choose a method
        3. Edit variable names if desired
        4. Click **Apply Cleaning/Transformation**
        5. New cleaned variables will appear here for comparison
        """)
    else:
        st.info("ℹ️ No raw variables available")


if __name__ == "__main__":
    main()
