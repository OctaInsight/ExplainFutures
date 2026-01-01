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
    """Reset all session state except project/auth/progress keys"""
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


def save_dimensionality_reduction_results(method_type, method_data, renamed_components=None):
    """Save dimensionality reduction results to database"""
    try:
        from core.database.supabase_manager import get_db_manager
        db = get_db_manager()
        
        project_id = st.session_state.get('current_project_id')
        user_id = st.session_state.get('user_id')
        
        if not project_id or not user_id:
            return False, "Missing project or user ID"
        
        # Check if results already exist
        existing = db.client.table('dimensionality_reduction_results').select('*').eq(
            'project_id', project_id
        ).eq('method_type', method_type).order('created_at', desc=True).limit(1).execute()
        
        # Prepare component names
        output_variables = method_data.get('output_variables', [])
        if renamed_components:
            output_variables = [renamed_components.get(v, v) for v in output_variables]
        
        # Check if input variables match (parameter names involved)
        input_variables = method_data.get('input_variables', [])
        
        if existing.data and len(existing.data) > 0:
            old_record = existing.data[0]
            old_inputs = set(old_record.get('input_variables', []))
            new_inputs = set(input_variables)
            
            if old_inputs == new_inputs:
                # Same variables - ask user
                st.warning(f"⚠️ A {method_type} analysis with the same variables already exists (created {old_record['created_at']})")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Overwrite", key=f"overwrite_{method_type}"):
                        # Delete old record
                        db.client.table('dimensionality_reduction_results').delete().eq(
                            'result_id', old_record['result_id']
                        ).execute()
                        
                        # Delete old timeseries
                        for var in old_record.get('output_variables', []):
                            db.client.table('timeseries_data').delete().eq(
                                'project_id', project_id
                            ).eq('variable', var).eq('data_source', method_type).execute()
                            
                            db.client.table('parameters').delete().eq(
                                'project_id', project_id
                            ).eq('parameter_name', var).execute()
                        
                        st.info("✅ Old data deleted. Proceeding with save...")
                    else:
                        return False, "User needs to confirm overwrite"
                
                with col2:
                    new_suffix = st.text_input("Or save with suffix:", value="_v2", key=f"suffix_{method_type}")
                    if st.button("💾 Save as New", key=f"save_new_{method_type}"):
                        # Add suffix to output variables
                        output_variables = [f"{v}{new_suffix}" for v in output_variables]
                        method_data['output_variables'] = output_variables
        
        # Save to dimensionality_reduction_results
        reduction_data = {
            'project_id': project_id,
            'user_id': user_id,
            'method_type': method_type,
            'method_name': method_data.get('method_name', method_type),
            'input_variables': input_variables,
            'output_variables': output_variables,
            'n_components': method_data.get('n_components'),
            'explained_variance': method_data.get('explained_variance'),
            'cumulative_variance': method_data.get('cumulative_variance'),
            'total_variance_explained': method_data.get('total_variance_explained'),
            'loadings': json.dumps(method_data.get('loadings')) if method_data.get('loadings') else None,
            'transformation_matrix': json.dumps(method_data.get('transformation_matrix')) if method_data.get('transformation_matrix') else None,
            'component_equations': json.dumps(method_data.get('component_equations')) if method_data.get('component_equations') else None,
            'n_clusters': method_data.get('n_clusters'),
            'cluster_assignments': json.dumps(method_data.get('cluster_assignments')) if method_data.get('cluster_assignments') else None,
            'removed_variables': method_data.get('removed_variables'),
            'correlation_threshold': method_data.get('correlation_threshold'),
            'is_accepted': True,
            'accepted_at': datetime.now().isoformat(),
            'config': json.dumps(method_data.get('config', {})),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = db.client.table('dimensionality_reduction_results').insert(reduction_data).execute()
        
        # Save timeseries data
        if 'transformed_data' in method_data and method_data['transformed_data'] is not None:
            df_transformed = method_data['transformed_data']
            time_col = st.session_state.get('time_column', 'timestamp')
            
            if time_col in df_transformed.columns:
                for var in output_variables:
                    if var in df_transformed.columns:
                        for idx, row in df_transformed.iterrows():
                            db.client.table('timeseries_data').insert({
                                'project_id': project_id,
                                'timestamp': row[time_col].isoformat() if hasattr(row[time_col], 'isoformat') else str(row[time_col]),
                                'variable': var,
                                'value': float(row[var]),
                                'data_source': method_type,
                                'created_at': datetime.now().isoformat()
                            }).execute()
        
        # Save parameters metadata
        for i, var in enumerate(output_variables):
            description = f"{method_data.get('method_name', method_type)} component {i+1}"
            if method_type == 'pca' and 'explained_variance' in method_data:
                description += f" - Explains {method_data['explained_variance'][i]*100:.1f}% variance"
            
            db.client.table('parameters').insert({
                'project_id': project_id,
                'parameter_name': var,
                'data_type': f"{method_type}_component",
                'description': description,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).execute()
        
        # ADDED: Update project progress tracking
        # Add "dim_reduction_done" step with 10% contribution
        db.upsert_progress_step(project_id, "dim_reduction_done", 10)
        # Recompute total project progress
        db.recompute_and_update_project_progress(project_id)
        
        return True, f"✅ Saved {len(output_variables)} components to database"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    # Check project
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Reset and load data on first page load
    if not st.session_state.get('page6_data_loaded', False):
        reset_page_session_state()
        
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
    
    initialize_reduction_state()
    
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
    
    # COLLAPSIBLE SECTION 1: What is Dimensionality Reduction (MOVED TO TOP)
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
    
    # COLLAPSIBLE SECTION 2: Intelligent System Component Mapping (NEW)
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
        if st.checkbox("📋 Use all variables", value=True, key="use_all_vars"):
            feature_cols = all_feature_cols
        else:
            feature_cols = st.multiselect(
                "Select variables:",
                options=all_feature_cols,
                default=all_feature_cols[:5] if len(all_feature_cols) > 5 else all_feature_cols,
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
        "📋 Summary & Save"
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
        handle_summary_and_save(df_wide, feature_cols, all_feature_cols)


# Analysis handler functions would go here
# (I'll continue with these in the next part)

def handle_correlation_filtering(df_wide, feature_cols):
    """Handle correlation-based filtering"""
    st.header("🔗 Correlation-Based Feature Filtering")
    st.markdown("*Remove highly correlated (redundant) variables*")
    
    st.info("This method removes variables but doesn't create new ones, so there's nothing to save to database.")
    
    # Rest of correlation filtering code...
    st.markdown("Implementation of correlation filtering here...")


def handle_pca_analysis(df_wide, feature_cols):
    """Handle PCA analysis with renaming option"""
    st.header("🧮 Principal Component Analysis (PCA)")
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
    st.header("🎯 Factor Analysis")
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
    st.header("🔬 Independent Component Analysis (ICA)")
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
    st.header("🌳 Hierarchical Clustering")
    st.markdown("*Group similar variables together*")
    
    run_hierarchical_clustering(df_wide, feature_cols)


def handle_summary_and_save(df_wide, feature_cols, all_feature_cols):
    """Summary and save results"""
    st.header("📋 Summary & Save Results")
    
    if not st.session_state.reduction_results:
        st.info("ℹ️ No analysis results yet. Please run at least one method from the tabs above.")
        return
    
    st.success(f"✅ {len(st.session_state.reduction_results)} method(s) applied")
    
    # Show summary
    st.subheader("📊 Analysis Summary")
    
    for method, results in st.session_state.reduction_results.items():
        with st.expander(f"📊 {method.upper()} Results", expanded=True):
            st.json(results)
    
    # Save button
    st.markdown("---")
    st.subheader("💾 Save Results to Database")
    
    st.markdown("""
    This will save:
    - **Timeseries data** for new components (PC1, Factor1, etc.) to `timeseries_data` table
    - **Parameter metadata** for new components to `parameters` table
    - **Reduction details** (loadings, equations, variance) to `dimensionality_reduction_results` table
    """)
    
    if st.button("💾 Save All Results to Database", type="primary", use_container_width=True):
        with st.spinner("Saving to database..."):
            results_saved = []
            errors = []
            
            for method, method_data in st.session_state.reduction_results.items():
                renamed_components = st.session_state.component_names.get(method, {})
                success, message = save_dimensionality_reduction_results(method, method_data, renamed_components)
                
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
