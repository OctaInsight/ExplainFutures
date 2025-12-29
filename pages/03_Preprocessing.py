"""
Page 3: Data Cleaning and Preprocessing (FULLY FIXED)
✅ ALWAYS reads ALL data from database (old + new uploads)
✅ Saves cleaned data with suffix to timeseries_data table
✅ Uses 'raw' and 'cleaned' as data_source
✅ Updates ALL progress indicators (sidebar, home, workflow dot)
✅ Before/After figures read from database
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Data Cleaning and Preprocessing",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

# Import database manager
try:
    from core.database.supabase_manager import get_db_manager
    DB_AVAILABLE = True
    db = get_db_manager()
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database import error: {str(e)}")

# Initialize
initialize_session_state()
config = get_config()

# Render sidebar
render_app_sidebar()

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("*Clean and prepare your data for analysis*")
st.markdown("---")

# Authentication check
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()


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


def load_all_data_from_database():
    """
    Load ALL data from database (raw data from all uploads)
    This ensures we ALWAYS have ALL data, not just session state
    """
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return None
    
    try:
        # Load ALL raw data from database
        df_long = db.load_timeseries_data(
            project_id=st.session_state.current_project_id,
            data_source='raw'  # Load raw data
        )
        
        if df_long is None or len(df_long) == 0:
            return None
        
        return df_long
        
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        return None


def plot_comparison_from_database(project_id, original_variable, cleaned_variable):
    """
    Create comparison plot reading DIRECTLY from database
    Compares raw vs cleaned data
    """
    try:
        # Load raw data for this variable
        raw_data = db.load_timeseries_data(
            project_id=project_id,
            data_source='raw',
            variables=[original_variable]
        )
        
        # Load cleaned data for this variable
        cleaned_data = db.load_timeseries_data(
            project_id=project_id,
            data_source='cleaned',
            variables=[cleaned_variable]
        )
        
        if raw_data is None or len(raw_data) == 0:
            st.error(f"❌ No raw data found for: {original_variable}")
            return None
        
        if cleaned_data is None or len(cleaned_data) == 0:
            st.error(f"❌ No cleaned data found for: {cleaned_variable}")
            return None
        
        time_col = 'timestamp' if 'timestamp' in raw_data.columns else 'time'
        
        raw_data = raw_data.sort_values(time_col)
        cleaned_data = cleaned_data.sort_values(time_col)
        
        # Create figure
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=raw_data[time_col],
            y=raw_data['value'],
            mode='lines+markers',
            name=f'Raw ({original_variable})',
            line=dict(color='#3498db', width=2),
            marker=dict(size=4, color='#3498db'),
            hovertemplate='<b>Raw</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Cleaned data
        fig.add_trace(go.Scatter(
            x=cleaned_data[time_col],
            y=cleaned_data['value'],
            mode='lines+markers',
            name=f'Cleaned ({cleaned_variable})',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4, color='#e74c3c'),
            hovertemplate='<b>Cleaned</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Before vs After: {original_variable} → {cleaned_variable}<br><sub>Data from Database</sub>",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            template='plotly_white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating comparison plot: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def main():
    """Main page function"""
    
    initialize_cleaning_history()
    
    # Check project selected
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ No project selected")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # ============================================================
    # CRITICAL: ALWAYS load ALL data from database
    # This ensures we see ALL uploads (old + new)
    # ============================================================
    
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        st.stop()
    
    with st.spinner("📊 Loading ALL data from database..."):
        df_long = load_all_data_from_database()
        
        if df_long is None or len(df_long) == 0:
            st.warning("⚠️ No data found in database")
            st.info("👈 Please go to **Upload & Data Diagnostics** to upload data first")
            
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            st.stop()
        
        # Store in session state
        st.session_state.df_long = df_long
        st.session_state.data_loaded = True
        
        variables = sorted(df_long['variable'].unique().tolist())
        st.session_state.value_columns = variables
        
        time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
        st.session_state.time_column = time_col
        
        st.success(f"✅ Loaded {len(df_long):,} data points from database (ALL uploads)")
        st.info(f"📊 {len(variables)} variables available")
    
    # Initialize df_clean
    if st.session_state.df_clean is None:
        st.session_state.df_clean = df_long.copy()
    
    variables = sorted(df_long['variable'].unique().tolist())
    
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
            key="missing_suffix",
            help="New columns will be: variable_name + suffix"
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
                st.rerun()
        else:
            st.warning("⚠️ No treatment selected or no variables selected")


def apply_missing_value_treatment(df, variables, method, suffix):
    """Apply missing value treatment and create NEW columns"""
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
        summary_df = pd.DataFrame(outlier_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("⚙️ Select Treatment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outlier_method = st.selectbox(
            "Outlier treatment",
            [
                "None - Keep as is",
                "Remove - Delete outlier rows",
                "Cap - Clip to bounds",
                "Transform - Log transformation",
                "Winsorize - Replace with percentile values"
            ],
            key="outlier_treatment"
        )
    
    with col2:
        apply_to_outliers = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="outlier_apply_to"
        )
    
    if outlier_method != "None - Keep as is":
        outlier_suffix = st.text_input(
            "Suffix for treated columns",
            value="_outlier_treated",
            key="outlier_suffix"
        )
    else:
        outlier_suffix = "_outlier_treated"
    
    if st.button("✨ Apply Outlier Treatment", type="primary", key="apply_outlier"):
        if outlier_method != "None - Keep as is" and apply_to_outliers:
            with st.spinner("Applying treatment..."):
                df_modified, new_columns = apply_outlier_treatment(
                    st.session_state.df_clean.copy(),
                    apply_to_outliers,
                    outlier_method,
                    outlier_suffix
                )
                
                add_to_cleaning_history(
                    "Outliers",
                    outlier_method,
                    apply_to_outliers,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Applied {outlier_method} to {len(apply_to_outliers)} variable(s)")
                st.info(f"📝 Created new columns: {', '.join(new_columns)}")
                
                st.session_state.df_clean = df_modified
                st.session_state.preprocessing_applied = True
                st.rerun()


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
            key="transformation_type"
        )
    
    with col2:
        transform_vars = st.multiselect(
            "Apply to variables",
            variables,
            default=[],
            key="transform_vars"
        )
    
    if transformation != "None":
        suffix = st.text_input(
            "Suffix for new columns",
            value=f"_{transformation.split('-')[0].strip().lower()}",
            key="transform_suffix"
        )
    else:
        suffix = "_transformed"
    
    if st.button("✨ Apply Transformation", type="primary", key="apply_transform"):
        if transformation != "None" and transform_vars:
            with st.spinner("Applying transformation..."):
                df_modified, new_columns = apply_transformation(
                    st.session_state.df_clean.copy(),
                    transform_vars,
                    transformation,
                    suffix
                )
                
                add_to_cleaning_history(
                    "Transformation",
                    transformation,
                    transform_vars,
                    {"new_columns": new_columns}
                )
                
                st.success(f"✅ Created {len(new_columns)} new transformed column(s)")
                st.info(f"📝 New columns: {', '.join(new_columns)}")
                
                st.session_state.df_clean = df_modified
                st.session_state.preprocessing_applied = True
                st.rerun()


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


def save_cleaned_data_to_database():
    """
    Save ONLY cleaned data (with suffixes) to database
    Uses data_source='cleaned'
    Updates ALL progress indicators
    """
    
    with st.spinner("💾 Saving cleaned data to database..."):
        try:
            project_id = st.session_state.current_project_id
            
            # Get ONLY new cleaned variables (with suffixes)
            all_vars = st.session_state.df_clean['variable'].unique()
            original_vars = st.session_state.df_long['variable'].unique()
            new_cleaned_vars = [v for v in all_vars if v not in original_vars]
            
            if not new_cleaned_vars:
                st.warning("⚠️ No cleaned variables to save")
                return
            
            # Extract ONLY cleaned data
            cleaned_df = st.session_state.df_clean[
                st.session_state.df_clean['variable'].isin(new_cleaned_vars)
            ].copy()
            
            # STEP 1: Save cleaned data to database (data_source='cleaned')
            st.info("📊 Step 1/4: Saving cleaned data...")
            
            success = db.save_timeseries_data(
                project_id=project_id,
                df_long=cleaned_df,
                data_source='cleaned',  # Use 'cleaned' not 'original'
                batch_size=1000
            )
            
            if not success:
                st.error("❌ Failed to save")
                return
            
            summary = db.get_timeseries_summary(project_id, data_source='cleaned')
            st.success(f"✅ Saved {summary['total_records']:,} cleaned data points!")
            
            # STEP 2: Update parameters table
            st.info("📊 Step 2/4: Updating parameters...")
            
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
                st.success(f"✅ Updated {len(cleaned_params)} parameters")
            
            # STEP 3: Update database progress
            st.info("📊 Step 3/4: Updating workflow...")
            
            # Mark step complete (turns 2nd dot green!)
            db.update_step_completion(
                project_id=project_id,
                step_key='data_cleaned',
                completed=True
            )
            
            # Update project progress
            db.update_project_progress(
                project_id=project_id,
                workflow_state="preprocessing_complete",
                current_page=3,
                completion_percentage=15
            )
            
            st.success("✅ Database updated")
            
            # STEP 4: Update session state
            st.info("📊 Step 4/4: Updating session...")
            
            st.session_state.preprocessing_applied = True
            st.session_state.data_cleaned = True  # Green dot trigger!
            
            st.success("✅ Session updated")
            
            # SUCCESS
            st.markdown("---")
            st.success("🎉 **All done! Cleaned data saved!**")
            st.balloons()
            
            st.info(f"""
            **✅ Completed:**
            - Saved {summary['total_records']:,} cleaned data points
            - Updated {len(cleaned_params)} parameters
            - Progress: 15%
            - 2nd workflow dot: GREEN ✨
            """)
            
            # Next steps
            st.markdown("---")
            st.markdown("### 🎯 Ready for Next Step!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Visualize Data", use_container_width=True, type="primary"):
                    st.switch_page("pages/04_Exploration_and_Visualization.py")
            
            with col2:
                if st.button("🔬 Analyze", use_container_width=True):
                    st.switch_page("pages/05_Understand_The_System_(Dimensionality_Reduction).py")
            
            with col3:
                if st.button("🏠 Home", use_container_width=True):
                    st.switch_page("pages/01_Home.py")
            
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


def show_cleaning_summary():
    """Show summary and save"""
    
    st.header("📋 Cleaning Summary & Save")
    
    if not st.session_state.cleaning_history:
        st.info("ℹ️ No cleaning operations yet")
        return
    
    # Summary
    st.subheader("📊 Operation Summary")
    
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
    
    # SAVE SECTION
    st.subheader("💾 Save Cleaned Data")
    
    if DB_AVAILABLE:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **What gets saved:**
            - ✅ ONLY cleaned variables (with suffixes)
            - ✅ Saved to timeseries_data (source='cleaned')
            - ✅ Raw data stays unchanged (source='raw')
            - ✅ Progress updated everywhere
            - ✅ 2nd workflow dot turns GREEN ✨
            """)
        
        with col2:
            st.write("")
            st.write("")
            
            try:
                summary = db.get_timeseries_summary(
                    st.session_state.current_project_id,
                    data_source='cleaned'
                )
                
                if summary['total_records'] > 0:
                    st.warning(f"⚠️ Cleaned data exists ({summary['total_records']:,})")
                    st.caption("Will be replaced")
            except:
                pass
            
            if st.button("💾 Save Cleaned Data", 
                        type="primary", 
                        use_container_width=True):
                save_cleaned_data_to_database()
    
    st.markdown("---")
    
    # Before/After Comparison from DATABASE
    st.subheader("📈 Before/After Comparison (from Database)")
    
    all_cleaned = []
    all_original = []
    
    for op in st.session_state.cleaning_history:
        new_cols = op.get('details', {}).get('new_columns', [])
        orig_vars = op.get('variables', [])
        all_cleaned.extend(new_cols)
        all_original.extend(orig_vars)
    
    if all_cleaned and all_original:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            options = [f"{o} → {c}" for o, c in zip(all_original, all_cleaned)]
            
            selected_idx = st.selectbox(
                "Select variable",
                range(len(options)),
                format_func=lambda i: options[i],
                key="compare_var"
            )
            
            orig_var = all_original[selected_idx]
            clean_var = all_cleaned[selected_idx]
            
            st.caption(f"**Raw:** {orig_var}")
            st.caption(f"**Cleaned:** {clean_var}")
        
        with col2:
            # Read from DATABASE
            fig = plot_comparison_from_database(
                st.session_state.current_project_id,
                orig_var,
                clean_var
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
