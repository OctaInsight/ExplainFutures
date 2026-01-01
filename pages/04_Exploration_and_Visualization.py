"""
Page 04: Data Exploration & Visualization (FIXED)
- Loads data from session state OR database
- Updates progress when leaving page
- Marks visualization step as complete
"""

import streamlit as st
import pandas as pd
import sys
import time
from pathlib import Path
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Data Exploration & Visualization",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Authentication check FIRST
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_warning, display_info, get_color_for_index
from core.shared_sidebar import render_app_sidebar  
from core.viz.plot_time import plot_single_variable
from core.viz.export import export_figure, quick_export_buttons
from core.database.supabase_manager import get_db_manager

# Initialize
initialize_session_state()
config = get_config()

# Get database
try:
    db = get_db_manager()
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False
    st.error("⚠️ Database not available")

# Render shared sidebar
render_app_sidebar()

st.title("📊 Data Exploration & Visualization")
st.markdown("*Create interactive plots of your time-series data*")
st.markdown("---")


def load_data_from_database():
    """Load data from database if not in session state"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return False
    
    # If data already loaded, skip
    if st.session_state.get('df_long') is not None:
        return True
    
    st.info("📊 Loading data from database...")
    
    try:
        # Get parameters from database
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if not parameters:
            st.warning("⚠️ No parameters found in database")
            return False
        
        # Store parameter names
        st.session_state.value_columns = [p['parameter_name'] for p in parameters]
        st.session_state.project_parameters = parameters
        
        # We don't have the actual time series data from database
        # User needs to upload data first
        st.warning("⚠️ Parameter metadata loaded, but time-series data is not available.")
        st.info("💡 Please go to **Upload & Data Diagnostics** to upload your data file first.")
        return False
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


def get_available_variables():
    """
    Get all available variables (original + cleaned if available)
    Returns variables that have actual data in df_long
    """
    # Check if we have data in session state
    if st.session_state.get('df_long') is None:
        return [], [], [], False
    
    df_long = st.session_state.df_long
    
    # Check if 'variable' column exists
    if 'variable' not in df_long.columns:
        return [], [], [], False
    
    # Get all unique variables from the actual data
    unique_vars = df_long['variable'].unique()
    all_vars_in_data = sorted(unique_vars.tolist() if hasattr(unique_vars, 'tolist') else list(unique_vars))
    
    # Separate raw vs cleaned variables
    # Raw variables are those without cleaning suffixes
    cleaned_suffixes = ['_missing', '_outlier', '_transform', '_cleaned', '_filled', 
                       '_interpolated', '_normalized', '_scaled', '_imputed']
    
    cleaned_vars = [v for v in all_vars_in_data 
                   if any(suffix in v.lower() for suffix in cleaned_suffixes)]
    original_vars = [v for v in all_vars_in_data if v not in cleaned_vars]
    
    has_cleaned_data = len(cleaned_vars) > 0
    
    return all_vars_in_data, original_vars, cleaned_vars, has_cleaned_data


def get_total_parameter_count():
    """
    Get total parameter count from parameters table (like Page 3)
    This includes ALL parameters, even those without data
    """
    try:
        from core.database.supabase_manager import get_db_manager
        db = get_db_manager()
        project_id = st.session_state.get('current_project_id')
        
        if not project_id:
            return 0
        
        parameters = db.get_project_parameters(project_id)
        return len(parameters) if parameters else 0
    except:
        # Fallback to counting unique variables in data
        df_long = st.session_state.get('df_long')
        if df_long is not None:
            return len(df_long['variable'].unique())
        return 0


def get_data_for_variable(variable):
    """Get data for a specific variable from the appropriate dataframe"""
    # Check if variable is in cleaned data
    if (st.session_state.get('df_clean') is not None and 
        variable in st.session_state.df_clean['variable'].values):
        df = st.session_state.df_clean
    else:
        df = st.session_state.df_long
    
    return df[df['variable'] == variable].copy()


def mark_visualization_complete():
    """Mark visualization step as complete in database"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return
    
    try:
        # Update progress: Page 4 = 21% (3 steps complete out of 13, each ~7%)
        db.update_project_progress(
            project_id=st.session_state.current_project_id,
            workflow_state="visualization_complete",
            current_page=4,
            completion_percentage=21
        )
        
        # Mark visualization step as complete
        db.update_step_completion(
            project_id=st.session_state.current_project_id,
            step_key='visualization_done',
            completed=True
        )
        
        # Update session state for immediate UI feedback
        st.session_state.visualization_done = True
        
    except Exception as e:
        st.warning(f"Could not update progress: {str(e)}")




# =============================================================================
# SESSION STATE RESET AND DATABASE LOADING
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


def load_project_data_from_database():
    """
    Load ALL project data from database (timeseries_data + parameters)
    AND update progress step: data_explored = 7%
    """
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
        # Load data from ALL sources (original + cleaned + dimensionality reduction results)
        data_sources = [
            ('raw', 'original'),     # Try both names for raw data
            'cleaned',                # Preprocessed data
            'pca',                    # PCA components
            'factor_analysis',        # Factor analysis components
            'ica',                    # ICA components
            'clustering'              # Clustering results (if any)
        ]
        
        all_dfs = []
        sources_loaded = []
        
        # Try to load raw/original first
        df_raw = None
        for src_label in ('raw', 'original'):
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_temp is not None and len(df_temp) > 0:
                    df_raw = df_temp
                    all_dfs.append(df_temp)
                    sources_loaded.append(src_label)
                    break
            except Exception:
                continue
        
        if df_raw is None or len(df_raw) == 0:
            st.error("❌ No raw data found. Please upload data on Page 2.")
            return False
        
        # Load all other sources
        for source in ['cleaned', 'pca', 'factor_analysis', 'ica', 'clustering']:
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=source)
                if df_temp is not None and len(df_temp) > 0:
                    all_dfs.append(df_temp)
                    sources_loaded.append(source)
            except Exception:
                continue
        
        # Combine all data sources
        if len(all_dfs) > 1:
            df_all = pd.concat(all_dfs, ignore_index=True)
            # Remove duplicates (keep last occurrence)
            df_all = df_all.drop_duplicates(subset=['timestamp', 'variable'], keep='last')
        else:
            df_all = df_raw.copy()
        
        # Info about what was loaded
        if len(sources_loaded) > 1:
            st.success(f"✅ Loaded data from {len(sources_loaded)} sources: {', '.join(sources_loaded)}")
        
        # Store in session state
        st.session_state.df_long = df_all
        st.session_state.data_loaded = True
        
        # Validate data structure
        if 'variable' not in df_all.columns:
            st.error(f"❌ Data format error: 'variable' column not found. Columns: {df_all.columns.tolist()}")
            return False
        
        # Load parameters
        try:
            parameters = db.get_project_parameters(project_id)
            if parameters:
                st.session_state.project_parameters = parameters
                st.session_state.value_columns = [p['parameter_name'] for p in parameters]
            else:
                unique_vars = df_all['variable'].unique()
                st.session_state.value_columns = sorted(unique_vars.tolist() if hasattr(unique_vars, 'tolist') else list(unique_vars))
        except Exception:
            unique_vars = df_all['variable'].unique()
            st.session_state.value_columns = sorted(unique_vars.tolist() if hasattr(unique_vars, 'tolist') else list(unique_vars))
        
        # Time column
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'time'
        st.session_state.time_column = time_col
        
        # Update progress: step_key='data_explored', step_percent=7
        try:
            from datetime import datetime
            
            # Method 1: Use upsert_progress_step if available
            if hasattr(db, "upsert_progress_step"):
                db.upsert_progress_step(
                    project_id=project_id,
                    step_key="data_explored",
                    step_percent=7
                )
            else:
                # Method 2: Direct database upsert
                db.client.table("project_progress_steps").upsert({
                    "project_id": project_id,
                    "step_key": "data_explored",
                    "step_percent": 7,
                    "updated_at": datetime.now().isoformat()
                }, on_conflict="project_id,step_key").execute()
            
            # Recompute total progress
            if hasattr(db, "recompute_and_update_project_progress"):
                db.recompute_and_update_project_progress(
                    project_id=project_id,
                    workflow_state="exploration",
                    current_page=4
                )
        except Exception as e:
            # Don't fail if progress update fails
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return False



def main():
    # Check project
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Reset and load data on first page load
    if not st.session_state.get('page4_data_loaded', False):
        reset_page_session_state()
        
        with st.spinner("📊 Loading project data from database..."):
            success = load_project_data_from_database()
            
            if not success:
                if st.button("📁 Go to Upload Page"):
                    st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
                st.stop()
            
            st.session_state.page4_data_loaded = True
    

    """Main page function"""
    
    # Check if project is selected
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project first")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Try to load data if not in session state
    if not st.session_state.get('data_loaded') or st.session_state.get('df_long') is None:
        success = load_data_from_database()
        
        if not success:
            st.warning("⚠️ No data loaded yet!")
            st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
            
            if st.button("📁 Go to Upload Page"):
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
            st.stop()
    
    # Get all available variables
    all_vars, original_vars, cleaned_vars, has_cleaned_data = get_available_variables()
    
    if not all_vars:
        st.error("❌ No variables available for visualization")
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        st.stop()
    
    # Display data info
    with st.expander("ℹ️ Current Dataset Info"):
        col1, col2, col3, col4 = st.columns(4)
        
        total_params = get_total_parameter_count()
        col1.metric("Total Parameters", total_params)
        col1.caption(f"With data: {len(all_vars)}")
        if cleaned_vars:
            col1.caption(f"Cleaned: {len(cleaned_vars)}")
        
        # Use df_long which contains both raw and cleaned data
        df_to_show = st.session_state.df_long
        col2.metric("Data Points", len(df_to_show))
        
        # Check if timestamp column exists
        time_col = st.session_state.get('time_column', 'timestamp')
        if time_col in df_to_show.columns:
            time_range = f"{df_to_show[time_col].min().date()} to {df_to_show[time_col].max().date()}"
            col3.caption("Time Range")
            col3.write(time_range)
        else:
            col3.caption("Time Range")
            col3.write("N/A")
        
        if st.session_state.get('uploaded_file_name'):
            col4.caption("Source File")
            col4.write(st.session_state.uploaded_file_name)
    
    # Main visualization tabs
    tab1, tab2 = st.tabs(["📈 Single Variable", "📊 Multi-Variable Comparison"])
    
    with tab1:
        render_single_variable_plot(all_vars)
    
    with tab2:
        render_multi_variable_plot(all_vars)
    
    # Navigation footer
    st.markdown("---")
    st.markdown("### 🎯 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Data Cleaning and Preprocess", use_container_width=True):
            st.switch_page("pages/03_Preprocessing.py")
    
    with col2:
        st.markdown("**Continue Workflow →**")
    
    with col3:
        if st.button("Go to Variable Relationships →", type="primary", use_container_width=True):
            # Mark this step as complete before moving on
            mark_visualization_complete()
            st.switch_page("pages/05_Variable_Relationships.py")


def render_single_variable_plot(variables):
    """Render single variable time plot interface"""
    
    st.header("Single Variable Time Plot")
    st.markdown("Plot one variable over time with customization options")
    
    # Configuration columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Variable selection
        selected_var = st.selectbox(
            "Select variable",
            variables,
            help="Choose which variable to plot",
            key="single_var_select"
        )
    
    with col2:
        # Color selection
        color = st.color_picker(
            "Line color",
            value=get_color_for_index(0),
            help="Choose line color",
            key="single_var_color"
        )
    
    # Advanced options
    with st.expander("⚙️ Plot Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_markers = st.checkbox("Show markers", value=False, key="single_markers")
            line_width = st.slider("Line width", 1, 5, 2, key="single_line_width")
        
        with col2:
            y_scale = st.selectbox("Y-axis scale", ["linear", "log"], key="single_y_scale")
            show_grid = st.checkbox("Show grid", value=True, key="single_show_grid")
        
        with col3:
            plot_height = st.slider("Plot height", 300, 800, 500, step=50, key="single_plot_height")
    
    # Create plot button
    if st.button("📈 Generate Plot", type="primary", key="single_plot_button"):
        with st.spinner("Creating plot..."):
            # Get data for selected variable
            var_data = get_data_for_variable(selected_var)
            
            if len(var_data) == 0:
                display_error("No data available for selected variable")
                return
            
            # Create plot
            fig = plot_single_variable(
                var_data,
                variable=selected_var,
                color=color,
                show_markers=show_markers,
                line_width=line_width,
                y_scale=y_scale,
                show_grid=show_grid,
                height=plot_height
            )
            
            if fig is None:
                display_error("Failed to create plot")
                return
            
            # Store figure and metadata in session state
            st.session_state.single_var_fig = fig
            st.session_state.single_var_name = selected_var
            st.session_state.single_var_data = var_data
    
    # Display plot if it exists in session state
    if st.session_state.get('single_var_fig') is not None:
        st.plotly_chart(st.session_state.single_var_fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        with st.expander("💾 Export Figure", expanded=True):
            quick_export_buttons(
                st.session_state.single_var_fig, 
                filename_prefix=f"single_var_{st.session_state.single_var_name}",
                show_formats=['png', 'pdf', 'html']
            )
        
        # Display statistics
        with st.expander("📊 Variable Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            values = st.session_state.single_var_data['value'].dropna()
            col1.metric("Count", len(values))
            col2.metric("Mean", f"{values.mean():.2f}")
            col3.metric("Std Dev", f"{values.std():.2f}")
            col4.metric("Range", f"{values.min():.2f} to {values.max():.2f}")


def render_multi_variable_plot(variables):
    """Render multi-variable comparison plot interface"""
    
    st.header("Multi-Variable Comparison")
    st.markdown("Compare multiple variables with **independent Y-axes**")
    
    # Variable selection
    selected_vars = st.multiselect(
        "Choose 2 or more variables to compare",
        variables,
        default=variables[:2] if len(variables) >= 2 else variables,
        help="Each variable will have its own Y-axis",
        key="multi_var_select"
    )
    
    if len(selected_vars) < 2:
        st.info("ℹ️ Select at least 2 variables to create a comparison plot")
        return
    
    st.markdown("---")
    
    # Configuration for each variable
    st.subheader("Configure Each Variable")
    
    axis_config = {}
    
    for idx, var in enumerate(selected_vars):
        with st.expander(f"⚙️ {var} Settings", expanded=(idx < 2)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                color = st.color_picker(
                    "Line Color",
                    value=get_color_for_index(idx),
                    key=f"multi_color_{var}_{idx}"
                )
            
            with col2:
                scale = st.selectbox(
                    "Y-Axis Scale",
                    ["linear", "log"],
                    key=f"multi_scale_{var}_{idx}"
                )
            
            with col3:
                line_width = st.slider(
                    "Line Width",
                    1, 5, 2,
                    key=f"multi_width_{var}_{idx}"
                )
            
            # Store configuration
            axis_config[var] = {
                "color": color,
                "scale": scale,
                "line_width": line_width,
                "axis_index": idx
            }
    
    st.markdown("---")
    
    # Global plot options
    col1, col2 = st.columns(2)
    
    with col1:
        show_legend = st.checkbox("Show legend", value=True, key="multi_show_legend")
        plot_height = st.slider("Plot height", 400, 900, 600, step=50, key="multi_plot_height")
    
    with col2:
        show_grid = st.checkbox("Show grid", value=True, key="multi_show_grid")
        plot_title = st.text_input("Plot title (optional)", "", key="multi_plot_title")
    
    # Create plot button
    if st.button("📊 Generate Comparison Plot", type="primary", key="multi_plot_button"):
        with st.spinner("Creating multi-variable plot..."):
            try:
                # Create plot with independent axes
                fig = create_multi_axis_plot(
                    selected_vars,
                    axis_config=axis_config,
                    show_legend=show_legend,
                    show_grid=show_grid,
                    height=plot_height,
                    title=plot_title if plot_title else None
                )
                
                if fig:
                    st.session_state.multi_var_fig = fig
                    st.session_state.multi_var_names = selected_vars
                else:
                    display_error("Failed to create plot")
                    
            except Exception as e:
                display_error(f"Error creating plot: {str(e)}")
    
    # Display plot
    if st.session_state.get('multi_var_fig') is not None:
        st.plotly_chart(st.session_state.multi_var_fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        with st.expander("💾 Export Figure", expanded=True):
            quick_export_buttons(
                st.session_state.multi_var_fig,
                filename_prefix=f"multi_var_comparison",
                show_formats=['png', 'pdf', 'html']
            )


def create_multi_axis_plot(variables, axis_config, show_legend=True, 
                           show_grid=True, height=600, title=None):
    """Create a plot with independent Y-axis for each variable"""
    try:
        fig = go.Figure()
        
        # Add traces for each variable
        for idx, var in enumerate(variables):
            var_data = get_data_for_variable(var)
            var_data = var_data.sort_values('timestamp').dropna(subset=['timestamp', 'value'])
            
            if len(var_data) == 0:
                continue
            
            config = axis_config.get(var, {})
            color = config.get('color', get_color_for_index(idx))
            line_width = config.get('line_width', 2)
            
            yaxis = 'y' if idx == 0 else f'y{idx+1}'
            
            trace = go.Scatter(
                x=var_data['timestamp'],
                y=var_data['value'],
                mode='lines',
                name=var,
                line=dict(color=color, width=line_width),
                yaxis=yaxis,
                hovertemplate=f'<b>{var}</b><br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            )
            
            fig.add_trace(trace)
        
        # Configure layout
        layout_config = {
            'xaxis': dict(
                title="Time",
                showgrid=show_grid,
                domain=[0.15, 1]
            ),
            'height': height,
            'hovermode': 'x unified',
            'showlegend': show_legend
        }
        
        if title:
            layout_config['title'] = title
        
        # Configure each y-axis
        for idx, var in enumerate(variables):
            config = axis_config.get(var, {})
            scale = config.get('scale', 'linear')
            color = config.get('color', get_color_for_index(idx))
            
            if idx == 0:
                yaxis_config = {
                    'title': var,
                    'type': scale,
                    'showgrid': show_grid,
                    'tickfont': dict(color=color)
                }
                layout_config['yaxis'] = yaxis_config
            else:
                position = 0.15 - (idx * 0.08)
                yaxis_config = {
                    'title': var,
                    'type': scale,
                    'showgrid': False,
                    'overlaying': 'y',
                    'side': 'left',
                    'position': max(0, position),
                    'tickfont': dict(color=color)
                }
                layout_config[f'yaxis{idx+1}'] = yaxis_config
        
        fig.update_layout(**layout_config)
        
        return fig
        
    except Exception as e:
        st.error(f"Error in create_multi_axis_plot: {str(e)}")
        return None



if __name__ == "__main__":
    main()
