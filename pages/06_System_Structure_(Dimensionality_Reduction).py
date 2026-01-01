"""
Page 6: Understand The System (Dimensionality Reduction)
Reduce complexity, stabilize models, and improve explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Understand The System",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import dimensionality reduction modules
from core.dim_reduction.pca_module import run_pca_analysis
from core.dim_reduction.factor_analysis_module import run_factor_analysis
from core.dim_reduction.ica_module import run_ica_analysis
from core.dim_reduction.clustering_module import run_hierarchical_clustering

# Initialize
initialize_session_state()
config = get_config()

# Render shared sidebar
render_app_sidebar()

# Authentication check
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

st.title("🔬 Understand The System")
st.markdown("*Dimensionality Reduction for System Understanding & Model Stabilization*")
st.markdown("---")


def initialize_reduction_state():
    """Initialize session state for dimensionality reduction"""
    if "reduction_results" not in st.session_state:
        st.session_state.reduction_results = {}
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = []
    if "pca_model" not in st.session_state:
        st.session_state.pca_model = None
    if "fa_model" not in st.session_state:
        st.session_state.fa_model = None
    if "ica_model" not in st.session_state:
        st.session_state.ica_model = None
    if "pca_accepted" not in st.session_state:
        st.session_state.pca_accepted = False
    if "reduced_data" not in st.session_state:
        st.session_state.reduced_data = None
    if "component_names" not in st.session_state:
        st.session_state.component_names = {}


def get_wide_format_data():
    """Convert long format to wide format for dimensionality reduction"""
    if (st.session_state.get('preprocessing_applied', False) and 
        st.session_state.get('df_clean') is not None):
        df_long = st.session_state.df_clean
        data_source = "cleaned"
    else:
        df_long = st.session_state.df_long
        data_source = "original"
    
    time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
    
    df_wide = df_long.pivot(
        index=time_col,
        columns='variable',
        values='value'
    ).reset_index()
    
    return df_wide, data_source


def reset_page_session_state():
    """Reset all session state except project/auth/progress keys AND reduction results"""
    keys_to_keep = {
        # Project and auth keys
        'current_project_id', 'current_project', 'selected_project',
        'project_name', 'project_description', 'project_created_at',
        'authenticated', 'user_id', 'user_email', 'user_name',
        'user_profile', 'session_id',
        'workflow_state', 'completion_percentage', 'current_page',
        'project_progress', 'step_completion',
        'sidebar_state', 'page_config',
        # CRITICAL: Keep reduction results between reruns
        'reduction_results', 'component_names', 'selected_features',
        'pca_model', 'fa_model', 'ica_model', 'pca_accepted', 'reduced_data',
        # Keep page-specific flags
        'page6_data_loaded', 'data_loaded', 'df_long', 'df_clean',
        'value_columns', 'time_column', 'project_parameters',
        'preprocessing_applied'
    }
    
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        if key not in keys_to_keep:
            del st.session_state[key]


def load_project_data_from_database():
    """Load ALL project data from database"""
    try:
        from core.database.supabase_manager import get_db_manager
        db = get_db_manager()
    except:
        st.error("❌ Database not available")
        return False
    
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        st.error("❌ No project selected")
        return False
    
    try:
        # FIXED: Changed source= to data_source=
        df_raw = db.load_timeseries_data(project_id, data_source='raw')
        if df_raw is None or len(df_raw) == 0:
            df_raw = db.load_timeseries_data(project_id, data_source='original')
        
        df_cleaned = db.load_timeseries_data(project_id, data_source='cleaned')
        
        if df_raw is not None and len(df_raw) > 0:
            df_all = pd.concat([df_raw, df_cleaned]) if df_cleaned is not None and len(df_cleaned) > 0 else df_raw
        elif df_cleaned is not None and len(df_cleaned) > 0:
            df_all = df_cleaned
        else:
            st.error("❌ No data found")
            return False
        
        st.session_state.df_long = df_all
        st.session_state.data_loaded = True
        
        try:
            parameters = db.get_project_parameters(project_id)
            if parameters:
                st.session_state.project_parameters = parameters
                st.session_state.value_columns = [p['parameter_name'] for p in parameters]
            else:
                st.session_state.value_columns = sorted(df_all['variable'].unique().tolist())
        except Exception:
            st.session_state.value_columns = sorted(df_all['variable'].unique().tolist())
        
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'time'
        st.session_state.time_column = time_col
        
        return True
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return False


def generate_transformed_data_if_missing(method_type, method_data, df_wide, feature_cols):
    """
    Generate transformed data from stored model if it's missing from method_data.
    This fixes the issue where modules don't include transformed_data in results.
    """
    if 'transformed_data' in method_data and method_data['transformed_data'] is not None:
        st.info(f"✅ Transformed data already exists for {method_type}")
        return method_data  # Already has transformed data
    
    st.warning(f"⚠️ Regenerating transformed data for {method_type}...")
    
    try:
        # Get the time column
        time_col = st.session_state.get('time_column', 'timestamp')
        
        # Get feature data
        X = df_wide[feature_cols].fillna(0)  # Fill NaN with 0 for transformation
        
        # Check if transformed data exists under different key names
        if method_type == 'factor_analysis' and 'factors' in method_data:
            # FA modules often store transformed data as 'factors'
            factors_data = method_data['factors']
            if isinstance(factors_data, (pd.DataFrame, np.ndarray)):
                st.info("✅ Found transformed data in 'factors' key")
                
                if isinstance(factors_data, np.ndarray):
                    n_factors = method_data.get('n_factors', factors_data.shape[1])
                    output_vars = method_data.get('output_variables', [f'Factor{i+1}' for i in range(n_factors)])
                    df_transformed = pd.DataFrame(
                        factors_data[:, :n_factors],
                        columns=output_vars,
                        index=df_wide.index
                    )
                else:
                    df_transformed = factors_data.copy()
                
                df_transformed[time_col] = df_wide[time_col].values
                method_data['transformed_data'] = df_transformed
                st.success(f"✅ Converted 'factors' to transformed_data")
                return method_data
        
        if method_type == 'ica' and 'components' in method_data:
            # ICA might store as 'components' 
            components_data = method_data['components']
            # Note: 'components' in ICA is usually the unmixing matrix, not transformed data
            # We need the actual transformed signals
            pass
        
        if method_type == 'pca':
            # Get PCA model from session state or method_data
            model = st.session_state.get('pca_model') or method_data.get('model')
            if model is not None:
                # Store model in method_data for persistence
                method_data['model'] = model
                
                # Transform the data
                transformed = model.transform(X)
                n_components = method_data.get('n_components', transformed.shape[1])
                
                # Create dataframe with transformed data
                output_vars = method_data.get('output_variables', [f'PC{i+1}' for i in range(n_components)])
                df_transformed = pd.DataFrame(
                    transformed[:, :n_components],
                    columns=output_vars,
                    index=df_wide.index
                )
                df_transformed[time_col] = df_wide[time_col].values
                
                method_data['transformed_data'] = df_transformed
                st.success(f"✅ Generated {n_components} PCA components")
            else:
                st.error("❌ PCA model not found in session state")
                
        elif method_type == 'factor_analysis':
            model = st.session_state.get('fa_model') or method_data.get('model')
            if model is not None:
                # Store model in method_data for persistence
                method_data['model'] = model
                
                transformed = model.transform(X)
                n_factors = method_data.get('n_factors', transformed.shape[1])
                
                output_vars = method_data.get('output_variables', [f'Factor{i+1}' for i in range(n_factors)])
                df_transformed = pd.DataFrame(
                    transformed[:, :n_factors],
                    columns=output_vars,
                    index=df_wide.index
                )
                df_transformed[time_col] = df_wide[time_col].values
                
                method_data['transformed_data'] = df_transformed
                st.success(f"✅ Generated {n_factors} factors")
            else:
                st.error("❌ Factor Analysis model not found. The module needs to save the fitted model to session state.")
                st.info("💡 Try re-running Factor Analysis to generate the model.")
                
        elif method_type == 'ica':
            model = st.session_state.get('ica_model') or method_data.get('model')
            if model is not None:
                # Store model in method_data for persistence
                method_data['model'] = model
                
                transformed = model.transform(X)
                n_components = method_data.get('n_components', transformed.shape[1])
                
                output_vars = method_data.get('output_variables', [f'IC{i+1}' for i in range(n_components)])
                df_transformed = pd.DataFrame(
                    transformed[:, :n_components],
                    columns=output_vars,
                    index=df_wide.index
                )
                df_transformed[time_col] = df_wide[time_col].values
                
                method_data['transformed_data'] = df_transformed
                st.success(f"✅ Generated {n_components} independent components")
            else:
                st.error("❌ ICA model not found. The module needs to save the fitted model to session state.")
                st.info("💡 Try re-running ICA to generate the model.")
                
        elif method_type == 'clustering':
            # Clustering doesn't produce transformed data, just assignments
            st.info("ℹ️ Clustering doesn't produce timeseries data (only groupings)")
            
    except Exception as e:
        st.error(f"❌ Error generating transformed data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    return method_data


def save_dimensionality_reduction_results(method_type, method_data, renamed_components=None, df_wide=None, feature_cols=None):
    """Save dimensionality reduction results to database"""
    try:
        from core.database.supabase_manager import get_db_manager
        db = get_db_manager()
        
        project_id = st.session_state.get('current_project_id')
        user_id = st.session_state.get('user_id')
        
        if not project_id or not user_id:
            return False, "Missing project or user ID"
        
        # DEBUG: Show what's in method_data
        st.info(f"🔍 Checking {method_type} data...")
        st.write(f"**Keys in method_data:** {list(method_data.keys())}")
        
        # Try to generate transformed data if missing
        if df_wide is not None and feature_cols is not None:
            method_data = generate_transformed_data_if_missing(method_type, method_data, df_wide, feature_cols)
        
        # CRITICAL FIX: Auto-generate output_variables and input_variables if missing
        output_variables = method_data.get('output_variables', [])
        input_variables = method_data.get('input_variables', [])
        
        # If output_variables not defined, create from transformed data or n_components
        if not output_variables:
            st.warning(f"⚠️ No output_variables in {method_type} results. Auto-generating...")
            
            if method_type == 'pca':
                n_comp = method_data.get('n_components', 3)
                output_variables = [f'PC{i+1}' for i in range(n_comp)]
            elif method_type == 'factor_analysis':
                n_factors = method_data.get('n_factors', 3)
                output_variables = [f'Factor{i+1}' for i in range(n_factors)]
            elif method_type == 'ica':
                n_comp = method_data.get('n_components', 3)
                output_variables = [f'IC{i+1}' for i in range(n_comp)]
            elif method_type == 'clustering':
                n_clusters = method_data.get('n_clusters', 3)
                output_variables = [f'Cluster{i+1}' for i in range(n_clusters)]
            
            method_data['output_variables'] = output_variables
            st.info(f"✅ Created output_variables: {output_variables}")
        
        # If input_variables not defined, use 'features' key or feature_cols
        if not input_variables:
            if 'features' in method_data:
                input_variables = method_data['features']
            elif feature_cols is not None:
                input_variables = list(feature_cols)
            else:
                input_variables = []
            
            method_data['input_variables'] = input_variables
            st.info(f"✅ Created input_variables: {len(input_variables)} features")
        
        if not output_variables:
            st.error(f"❌ Could not determine output variables for {method_type}")
            return False, "No output variables to save"
        
        # Update session state with the corrected method_data
        st.session_state.reduction_results[method_type] = method_data
        
        # IMPORTANT: Apply user's custom names FIRST (before checking for duplicates)
        if renamed_components:
            # Apply renamed components
            original_to_renamed = {}
            for orig_name in output_variables:
                new_name = renamed_components.get(orig_name, orig_name)
                original_to_renamed[orig_name] = new_name
            
            # Update output_variables list
            output_variables = [original_to_renamed[v] for v in output_variables]
            
            # Also rename columns in transformed_data if it exists
            if 'transformed_data' in method_data and method_data['transformed_data'] is not None:
                df_transformed = method_data['transformed_data']
                time_col = st.session_state.get('time_column', 'timestamp')
                
                # Rename columns (except time column)
                rename_dict = {old: new for old, new in original_to_renamed.items() 
                             if old in df_transformed.columns and old != new}
                if rename_dict:
                    df_transformed = df_transformed.rename(columns=rename_dict)
                    method_data['transformed_data'] = df_transformed
                    st.info(f"✅ Applied user renames: {list(rename_dict.values())}")
        
        input_variables = method_data.get('input_variables', [])
        
        # Check for existing timeseries data to avoid duplicates
        existing_timeseries = db.client.table('timeseries_data').select('variable').eq(
            'project_id', project_id
        ).eq('data_source', method_type).execute()
        
        existing_ts_vars = {row['variable'] for row in (existing_timeseries.data or [])}
        
        # If data already exists for these variables with this data_source, skip saving
        if any(var in existing_ts_vars for var in output_variables):
            overlap = [var for var in output_variables if var in existing_ts_vars]
            st.warning(f"⚠️ Timeseries data already exists for {overlap}. Skipping duplicate save.")
            st.info("💡 If you want to save new data, rename the components differently or delete old data first.")
            
            # Still save metadata and parameters, but skip timeseries
            skip_timeseries = True
        else:
            skip_timeseries = False
        
        # Check for existing parameters (for metadata only)
        existing_params = db.get_project_parameters(project_id)
        existing_param_names = {p['parameter_name'] for p in existing_params}
        
        # Adjust output variable names if they already exist IN PARAMETERS
        # But only if we're not skipping timeseries (if we skip, keep original names)
        final_output_variables = []
        name_mapping = {}  # Original -> Final name mapping
        
        if not skip_timeseries:
            for var in output_variables:
                original_var = var
                if var in existing_param_names:
                    counter = 1
                    new_var = f"{var}_v{counter}"
                    while new_var in existing_param_names or new_var in existing_ts_vars:
                        counter += 1
                        new_var = f"{var}_v{counter}"
                    st.warning(f"⚠️ Parameter '{var}' already exists. Saving as '{new_var}'")
                    final_output_variables.append(new_var)
                    name_mapping[original_var] = new_var
                else:
                    final_output_variables.append(var)
                    name_mapping[original_var] = var
            
            output_variables = final_output_variables
            
            # CRITICAL FIX: Also rename columns in transformed_data DataFrame
            if name_mapping and 'transformed_data' in method_data and method_data['transformed_data'] is not None:
                df_transformed = method_data['transformed_data']
                time_col = st.session_state.get('time_column', 'timestamp')
                
                # Create rename dict for columns that exist
                rename_dict = {old: new for old, new in name_mapping.items() 
                             if old in df_transformed.columns and old != new}
                
                if rename_dict:
                    df_transformed = df_transformed.rename(columns=rename_dict)
                    method_data['transformed_data'] = df_transformed
                    st.info(f"✅ Renamed columns in transformed_data: {list(rename_dict.values())}")
        else:
            # Keep original names if skipping
            for var in output_variables:
                name_mapping[var] = var
        
        # Convert any numpy arrays to lists before saving
        def safe_convert(value):
            """Convert numpy arrays to lists, handle None"""
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [safe_convert(v) for v in value]
            return value
        
        # Save to dimensionality_reduction_results
        reduction_data = {
            'project_id': project_id,
            'user_id': user_id,
            'method_type': method_type,
            'method_name': method_data.get('method_name', method_type),
            'input_variables': input_variables,
            'output_variables': output_variables,
            'n_components': int(method_data['n_components']) if method_data.get('n_components') is not None else None,
            'explained_variance': safe_convert(method_data.get('explained_variance')),
            'cumulative_variance': safe_convert(method_data.get('cumulative_variance')),
            'total_variance_explained': float(method_data['total_variance_explained']) if method_data.get('total_variance_explained') is not None else None,
            'loadings': json.dumps(safe_convert(method_data.get('loadings'))) if method_data.get('loadings') is not None else None,
            'transformation_matrix': json.dumps(safe_convert(method_data.get('transformation_matrix'))) if method_data.get('transformation_matrix') is not None else None,
            'component_equations': json.dumps(safe_convert(method_data.get('component_equations'))) if method_data.get('component_equations') is not None else None,
            'n_clusters': int(method_data['n_clusters']) if method_data.get('n_clusters') is not None else None,
            'cluster_assignments': json.dumps(safe_convert(method_data.get('cluster_assignments'))) if method_data.get('cluster_assignments') is not None else None,
            'removed_variables': method_data.get('removed_variables'),
            'correlation_threshold': float(method_data['correlation_threshold']) if method_data.get('correlation_threshold') is not None else None,
            'is_accepted': True,
            'accepted_at': datetime.now().isoformat(),
            'config': json.dumps(method_data.get('config', {})),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = db.client.table('dimensionality_reduction_results').insert(reduction_data).execute()
        st.info(f"✅ Saved metadata to dimensionality_reduction_results table")
        
        # IMPROVED: Save timeseries data for new components
        total_inserted = 0
        if skip_timeseries:
            st.info("ℹ️ Skipping timeseries save - data already exists for these variables")
        elif 'transformed_data' in method_data and method_data['transformed_data'] is not None:
            df_transformed = method_data['transformed_data']
            time_col = st.session_state.get('time_column', 'timestamp')
            
            st.write(f"**Transformed data shape:** {df_transformed.shape}")
            st.write(f"**Transformed data columns:** {list(df_transformed.columns)}")
            st.write(f"**Output variables to save:** {output_variables}")
            
            if time_col in df_transformed.columns and len(output_variables) > 0:
                # Prepare data in long format for batch insert
                records = []
                
                # Now columns in df_transformed should match output_variables after renaming
                for var in output_variables:
                    if var in df_transformed.columns:
                        for idx, row in df_transformed.iterrows():
                            timestamp_val = row[time_col]
                            if hasattr(timestamp_val, 'isoformat'):
                                ts_str = timestamp_val.isoformat()
                            elif isinstance(timestamp_val, str):
                                ts_str = timestamp_val
                            else:
                                ts_str = pd.Timestamp(timestamp_val).isoformat()
                            
                            value_val = row[var]
                            if pd.notna(value_val):
                                records.append({
                                    'project_id': project_id,
                                    'timestamp': ts_str,
                                    'variable': var,  # Variable name is already correct
                                    'value': float(value_val),
                                    'data_source': method_type
                                })
                    else:
                        st.warning(f"⚠️ Column '{var}' not found in transformed_data. Available: {list(df_transformed.columns)}")
                
                # Batch insert timeseries data
                if records:
                    batch_size = 1000
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        try:
                            db.client.table('timeseries_data').insert(batch).execute()
                            total_inserted += len(batch)
                        except Exception as e:
                            st.error(f"⚠️ Error inserting batch {i//batch_size + 1}: {str(e)}")
                    
                    st.info(f"✅ Saved {total_inserted} timeseries records to timeseries_data table")
                else:
                    st.warning("⚠️ No timeseries data to save (empty records list)")
            else:
                st.warning(f"⚠️ Cannot save timeseries: time column '{time_col}' not found or no output variables")
        else:
            st.warning("⚠️ No transformed_data found in method results")
        
        # IMPROVED: Save parameters metadata
        params_saved = 0
        params_skipped = 0
        for i, var in enumerate(output_variables):
            try:
                description = f"{method_data.get('method_name', method_type)} component {i+1}"
                
                # Add variance info for PCA
                if method_type == 'pca' and 'explained_variance' in method_data and method_data['explained_variance'] is not None:
                    ev = method_data['explained_variance']
                    if isinstance(ev, (list, np.ndarray)) and len(ev) > i:
                        variance_pct = float(ev[i]) * 100
                        description += f" - Explains {variance_pct:.1f}% variance"
                
                # Check if parameter already exists
                existing_check = db.client.table('parameters').select('parameter_id').eq(
                    'project_id', project_id
                ).eq('parameter_name', var).execute()
                
                if existing_check.data and len(existing_check.data) > 0:
                    # Update existing parameter (only metadata, not create new)
                    db.client.table('parameters').update({
                        'data_type': f"{method_type}_component",
                        'description': description,
                        'updated_at': datetime.now().isoformat()
                    }).eq('parameter_id', existing_check.data[0]['parameter_id']).execute()
                    params_skipped += 1
                else:
                    # Insert new parameter
                    db.client.table('parameters').insert({
                        'project_id': project_id,
                        'parameter_name': var,
                        'data_type': f"{method_type}_component",
                        'description': description,
                        'unit': None,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }).execute()
                    params_saved += 1
            except Exception as e:
                st.error(f"⚠️ Error saving parameter '{var}': {str(e)}")
        
        if params_saved > 0:
            st.info(f"✅ Saved {params_saved} new parameters to parameters table")
        if params_skipped > 0:
            st.info(f"ℹ️ Updated {params_skipped} existing parameters (no duplicates created)")
        
        # Update project progress tracking (only if we actually saved something new)
        if total_inserted > 0 or params_saved > 0:
            db.upsert_progress_step(project_id, "dim_reduction_done", 10)
            db.recompute_and_update_project_progress(project_id)
        
        # Create detailed success message
        msg_parts = []
        if total_inserted > 0:
            msg_parts.append(f"{total_inserted} data points")
        if params_saved > 0:
            msg_parts.append(f"{params_saved} new parameters")
        if params_skipped > 0:
            msg_parts.append(f"{params_skipped} updated parameters")
        if skip_timeseries:
            msg_parts.append("(timeseries skipped - already exists)")
        
        final_msg = f"✅ Saved: {', '.join(msg_parts)}" if msg_parts else "✅ Metadata saved"
        
        return True, final_msg
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        return False, error_msg


def handle_correlation_filtering(df_wide, feature_cols):
    """Handle correlation-based filtering"""
    st.markdown("*Remove highly correlated (redundant) variables*")
    
    st.markdown("""
    ### 📊 How It Works
    
    Correlation filtering identifies and removes redundant variables that are highly correlated with each other.
    This is the simplest dimensionality reduction method.
    
    **When to use:**
    - You have many highly correlated variables
    - You want a simple, interpretable reduction
    - You don't need to create new composite variables
    """)
    
    # Calculate correlation matrix
    df_features = df_wide[feature_cols]
    corr_matrix = df_features.corr().abs()
    
    # Threshold selection
    threshold = st.slider(
        "Correlation threshold (remove variables above this value):",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.05,
        help="Variables with correlation above this threshold will be candidates for removal"
    )
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = set()
    correlated_pairs = []
    
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column] > threshold]
        for index in high_corr.index:
            correlated_pairs.append({
                'Variable 1': column,
                'Variable 2': index,
                'Correlation': high_corr[index]
            })
            # Remove the second variable (you can change this logic)
            to_remove.add(index)
    
    if correlated_pairs:
        st.subheader(f"🔍 Found {len(correlated_pairs)} highly correlated pairs")
        
        df_pairs = pd.DataFrame(correlated_pairs)
        st.dataframe(df_pairs, use_container_width=True)
        
        st.subheader(f"🗑️ Variables to remove: {len(to_remove)}")
        st.write(sorted(list(to_remove)))
        
        remaining = sorted([v for v in feature_cols if v not in to_remove])
        st.subheader(f"✅ Remaining variables: {len(remaining)}")
        st.write(remaining)
        
        # Visualization
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("ℹ️ **Note:** Correlation filtering removes variables but doesn't create new ones, so there's nothing to save to database. You can use the remaining variables list for other analyses.")
        
    else:
        st.success(f"✅ No variables exceed correlation threshold of {threshold}")
        st.info("Try lowering the threshold to find correlated variables.")


def handle_pca_analysis(df_wide, feature_cols):
    """Handle PCA analysis with renaming option"""
    st.markdown("*Find orthogonal components that explain variance*")
    
    # Run PCA analysis
    run_pca_analysis(df_wide, feature_cols)
    
    # Check if PCA results exist
    if 'pca' in st.session_state.reduction_results:
        pca = st.session_state.reduction_results['pca']
        
        st.markdown("---")
        st.subheader("✏️ Rename Components")
        st.markdown("Give meaningful names to your principal components (e.g., 'Economic_Index', 'Social_Factor')")
        
        renamed_components = {}
        cols = st.columns(min(3, pca['n_components']))
        
        for i in range(pca['n_components']):
            col_idx = i % 3
            with cols[col_idx]:
                default_name = f"PC{i+1}"
                new_name = st.text_input(
                    f"Rename {default_name}:",
                    value=default_name,
                    key=f"rename_pca_{i}"
                )
                renamed_components[default_name] = new_name
        
        st.session_state.component_names['pca'] = renamed_components


def handle_factor_analysis(df_wide, feature_cols):
    """Handle Factor Analysis with renaming option"""
    st.markdown("*Discover latent factors underlying your data*")
    
    run_factor_analysis(df_wide, feature_cols)
    
    if 'factor_analysis' in st.session_state.reduction_results:
        fa = st.session_state.reduction_results['factor_analysis']
        
        st.markdown("---")
        st.subheader("✏️ Rename Factors")
        
        renamed_components = {}
        cols = st.columns(min(3, fa['n_factors']))
        
        for i in range(fa['n_factors']):
            col_idx = i % 3
            with cols[col_idx]:
                default_name = f"Factor{i+1}"
                new_name = st.text_input(
                    f"Rename {default_name}:",
                    value=default_name,
                    key=f"rename_fa_{i}"
                )
                renamed_components[default_name] = new_name
        
        st.session_state.component_names['factor_analysis'] = renamed_components


def handle_ica_analysis(df_wide, feature_cols):
    """Handle ICA with renaming option"""
    st.markdown("*Find statistically independent sources*")
    
    run_ica_analysis(df_wide, feature_cols)
    
    if 'ica' in st.session_state.reduction_results:
        ica = st.session_state.reduction_results['ica']
        
        st.markdown("---")
        st.subheader("✏️ Rename Independent Components")
        
        renamed_components = {}
        cols = st.columns(min(3, ica['n_components']))
        
        for i in range(ica['n_components']):
            col_idx = i % 3
            with cols[col_idx]:
                default_name = f"IC{i+1}"
                new_name = st.text_input(
                    f"Rename {default_name}:",
                    value=default_name,
                    key=f"rename_ica_{i}"
                )
                renamed_components[default_name] = new_name
        
        st.session_state.component_names['ica'] = renamed_components


def handle_hierarchical_clustering(df_wide, feature_cols):
    """Handle clustering"""
    st.markdown("*Group similar variables together*")
    
    run_hierarchical_clustering(df_wide, feature_cols)


def handle_summary(df_wide, feature_cols, all_feature_cols):
    """Display summary of results"""
    
    if not st.session_state.reduction_results:
        st.info("ℹ️ No analysis results yet. Please run at least one method from the tabs above.")
        return
    
    st.success(f"✅ {len(st.session_state.reduction_results)} method(s) applied")
    
    # Show summary
    for method, results in st.session_state.reduction_results.items():
        with st.expander(f"📊 {method.upper()} Results", expanded=False):
            st.write(f"**Keys in results:** {list(results.keys())}")
            
            # Show if transformed_data exists
            if 'transformed_data' in results and results['transformed_data'] is not None:
                st.success("✅ Has transformed data")
                st.write(f"Shape: {results['transformed_data'].shape}")
            else:
                st.warning("⚠️ No transformed data - will try to regenerate on save")
            
            st.json({k: v for k, v in results.items() if k != 'transformed_data'})


def main():
    # Check project
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Initialize reduction state FIRST (before any reset)
    initialize_reduction_state()
    
    # Reset and load data on first page load ONLY
    if not st.session_state.get('page6_data_loaded', False):
        # Don't reset! Just load data
        with st.spinner("📊 Loading project data from database..."):
            success = load_project_data_from_database()
            
            if not success:
                if st.button("📁 Go to Upload Page"):
                    st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                st.stop()
            
            st.session_state.page6_data_loaded = True
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        return
    
    # Get data
    df_wide, data_source = get_wide_format_data()
    time_col = st.session_state.get('time_column', 'timestamp')
    
    # Separate original and transformed variables
    original_vars = sorted(st.session_state.df_long['variable'].unique().tolist())
    cleaned_suffixes = ['_missing', '_outlier', '_transform', '_cleaned', '_filled', 
                       '_interpolated', '_normalized', '_scaled', '_imputed']
    
    cleaned_vars = [v for v in original_vars if any(suffix in v.lower() for suffix in cleaned_suffixes)]
    raw_vars = [v for v in original_vars if v not in cleaned_vars]
    
    all_feature_cols = [col for col in df_wide.columns if col != time_col]
    
    st.success(f"✅ Data loaded: {len(original_vars)} variables ({len(raw_vars)} original + {len(cleaned_vars)} cleaned)")
    
    # DEBUG: Show session state status
    with st.expander("🔍 Debug: Session State Status", expanded=False):
        st.write(f"**reduction_results keys:** {list(st.session_state.reduction_results.keys())}")
        st.write(f"**component_names:** {st.session_state.component_names}")
        st.write(f"**pca_model exists:** {st.session_state.pca_model is not None}")
        st.write(f"**fa_model exists:** {st.session_state.fa_model is not None}")
        st.write(f"**ica_model exists:** {st.session_state.ica_model is not None}")
    
    # COLLAPSIBLE SECTION 1: What is Dimensionality Reduction
    with st.expander("ℹ️ What is Dimensionality Reduction?", expanded=False):
        st.markdown("""
        ### 🎯 Purpose
        
        Dimensionality reduction helps you:
        
        1. 🎯 **Reduce complexity** - Work with fewer variables while retaining information
        2. 🎯 **Improve explainability** - Identify dominant patterns and components
        3. 🎯 **Stabilize models** - Remove redundancy and noise
        4. 🎯 **Visualize patterns** - Plot high-dimensional data in 2D/3D
        
        ### 🛠️ Methods Available
        
        **🔗 Correlation Filtering:** Remove highly correlated variables (most simple)
        **🧮 PCA:** Find orthogonal components (most common)
        **🎯 Factor Analysis:** Discover latent factors (most interpretable)
        **🔬 ICA:** Find independent sources (most sophisticated)
        **🌳 Clustering:** Group similar variables (most visual)
        
        ### 📊 When to Use
        
        - Too many variables (>20)
        - Variables are highly correlated
        - Need to understand system structure
        - Want to improve model performance
        - Need to visualize complex relationships
        """)
    
    # COLLAPSIBLE SECTION 2: Intelligent System Component Mapping
    with st.expander("🧠 Intelligent System Component Mapping", expanded=False):
        st.markdown("""
        ### 💡 Map Your System Components
        
        If your system has distinct **conceptual components** (e.g., Economic, Social, Environmental), 
        you can use dimensionality reduction to create **mathematical representations** of these components.
        
        ### 🎯 How It Works
        
        1. **Identify System Components**
           - Example: Economic, Social, Environmental components
           - Or: Production, Quality, Efficiency components
        
        2. **Select Related Variables** (in Step 1 below)
           - Economic: GDP, income, employment, prices
           - Social: education, health, demographics
           - Environmental: emissions, resources, pollution
        
        3. **Run Dimensionality Reduction**
           - **PCA:** Creates linear combinations (PC1, PC2, ...)
           - **Factor Analysis:** Discovers latent factors (Factor1, Factor2, ...)
           - **ICA:** Finds independent signals (IC1, IC2, ...)
        
        4. **Rename Components** (in each analysis tab)
           - PC1 → "Economic_Index"
           - Factor1 → "Social_Factor"
           - IC1 → "Environmental_Signal"
        
        5. **Use in Trajectory Analysis** (Page 8+)
           - These indices can represent system components
           - Track how components evolve over time
           - Compare baseline vs scenarios
        
        ### ✅ Benefits
        
        - ✅ Reduce 20 variables to 3 meaningful components
        - ✅ Each component has clear interpretation
        - ✅ Components can be used in forecasting
        - ✅ Easier to communicate and understand
        - ✅ Use equations to compute components for new data
        
        ### 📝 Example
        
        **Sustainability System:**
        ```
        Economic Variables: GDP, Employment, Income
        → Run PCA → Rename PC1 to "Economic_Health"
        
        Social Variables: Education, Health, Equality
        → Run Factor Analysis → Rename Factor1 to "Social_Wellbeing"
        
        Environmental Variables: CO2, Waste, Resources
        → Run ICA → Rename IC1 to "Environmental_Impact"
        ```
        
        **Result:** 9 variables → 3 interpretable system components
        """)
    
    # Step 1: Select Variables
    st.subheader("📋 Step 1: Select Variables for Analysis")
    
    st.markdown("""
    Select the variables you want to analyze. You can:
    - Select all variables for general dimensionality reduction
    - Select specific groups (e.g., only economic variables) for component mapping
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User must select from dropdown
        feature_cols = st.multiselect(
            "Select variables for analysis:",
            options=all_feature_cols,
            default=all_feature_cols,  # All selected by default
            key="selected_features_manual"
        )
    
    with col2:
        st.metric("Selected Features", len(feature_cols))
        st.metric("Total Available", len(all_feature_cols))
    
    if not feature_cols or len(feature_cols) < 2:
        st.warning("⚠️ Please select at least 2 variables for analysis")
        return
    
    st.success(f"✅ {len(feature_cols)} features selected")
    
    # Analysis Tabs
    st.markdown("---")
    st.subheader("📊 Step 2: Run Dimensionality Reduction Analysis")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔗 Correlation Filtering",
        "🧮 PCA Analysis",
        "🎯 Factor Analysis",
        "🔬 Independent Component Analysis",
        "🌳 Hierarchical Clustering",
        "📋 Summary"
    ])
    
    with tab1:
        handle_correlation_filtering(df_wide, feature_cols)
    
    with tab2:
        handle_pca_analysis(df_wide, feature_cols)
    
    with tab3:
        handle_factor_analysis(df_wide, feature_cols)
    
    with tab4:
        handle_ica_analysis(df_wide, feature_cols)
    
    with tab5:
        handle_hierarchical_clustering(df_wide, feature_cols)
    
    with tab6:
        handle_summary(df_wide, feature_cols, all_feature_cols)
    
    # Save button outside tabs
    st.markdown("---")
    st.subheader("💾 Save Results to Database")
    
    st.markdown("""
    This will save:
    - **Timeseries data** for new components (PC1, Factor1, etc.) to `timeseries_data` table
    - **Parameter metadata** for new components to `parameters` table
    - **Reduction details** (loadings, equations, variance) to `dimensionality_reduction_results` table
    
    **Note:** If a component name already exists, it will be saved with a version suffix (e.g., PC1_v2)
    """)
    
    if st.button("💾 Save All Results to Database", type="primary", use_container_width=True):
        with st.spinner("Saving to database..."):
            results_saved = []
            errors = []
            
            for method, method_data in st.session_state.reduction_results.items():
                renamed_components = st.session_state.component_names.get(method, {})
                success, message = save_dimensionality_reduction_results(
                    method, method_data, renamed_components, df_wide, feature_cols
                )
                
                if success:
                    results_saved.append(f"✅ {method}: {message}")
                else:
                    errors.append(f"❌ {method}: {message}")
            
            if results_saved:
                for msg in results_saved:
                    st.success(msg)
                st.balloons()
            
            if errors:
                for msg in errors:
                    st.error(msg)
    
    # Navigation buttons at the bottom
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Go Back to Variable Relationships", type="secondary", use_container_width=True):
            st.switch_page("pages/05_Variable_Relationships.py")
    
    with col2:
        if st.button("➡️ Go to Time Modeling & Training", type="primary", use_container_width=True):
            st.switch_page("pages/07_Time_Modeling_&_Training.py")


if __name__ == "__main__":
    main()
