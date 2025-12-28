"""
Page 3: Data Exploration & Visualization
Interactive visualization of time-series data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Data Exploration & Visualization",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"  # CRITICAL: Use full page width
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_warning, display_info, get_color_for_index
from core.shared_sidebar import render_app_sidebar  
from core.viz.plot_time import plot_single_variable
from core.viz.export import export_figure, quick_export_buttons

# Initialize
initialize_session_state()
config = get_config()


# Render shared sidebar
render_app_sidebar()  

st.title("📊 Data Exploration & Visualization")
st.markdown("*Create interactive plots of your time-series data*")
st.markdown("---")

# Copy these 6 lines to the TOP of each page (02-13)
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Then your existing code continues...


def get_available_variables():
    """
    Get all available variables (original + cleaned if available)
    Returns: (all_variables, original_variables, cleaned_variables, has_cleaned_data)
    """
    # Always have original data
    df_long = st.session_state.df_long
    original_vars = sorted(df_long['variable'].unique().tolist())
    
    # Check if cleaned data exists
    has_cleaned_data = (
        st.session_state.get('preprocessing_applied', False) and 
        st.session_state.get('df_clean') is not None
    )
    
    if has_cleaned_data:
        df_clean = st.session_state.df_clean
        all_vars = sorted(df_clean['variable'].unique().tolist())
        cleaned_vars = [v for v in all_vars if v not in original_vars]
    else:
        all_vars = original_vars
        cleaned_vars = []
    
    return all_vars, original_vars, cleaned_vars, has_cleaned_data


def get_data_for_variable(variable):
    """
    Get data for a specific variable from the appropriate dataframe
    """
    # Check if variable is in cleaned data
    if (st.session_state.get('df_clean') is not None and 
        variable in st.session_state.df_clean['variable'].values):
        df = st.session_state.df_clean
    else:
        df = st.session_state.df_long
    
    return df[df['variable'] == variable].copy()


def main():
    """Main page function"""
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Health.py")
        return
    
    # Get all available variables
    all_vars, original_vars, cleaned_vars, has_cleaned_data = get_available_variables()
    
    # Display data info
    with st.expander("ℹ️ Current Dataset Info"):
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Variables", len(all_vars))
        col1.caption(f"Original: {len(original_vars)}")
        if cleaned_vars:
            col1.caption(f"Cleaned/Transformed: {len(cleaned_vars)}")
        
        # Use cleaned data if available, otherwise original
        df_to_show = st.session_state.df_clean if has_cleaned_data else st.session_state.df_long
        col2.metric("Data Points", len(df_to_show))
        
        time_range = f"{df_to_show['timestamp'].min().date()} to {df_to_show['timestamp'].max().date()}"
        col3.caption("Time Range")
        col3.write(time_range)
        
        if st.session_state.uploaded_file_name:
            col4.caption("Source File")
            col4.write(st.session_state.uploaded_file_name)
        
        # Show variable categorization if cleaned data exists
        if has_cleaned_data and cleaned_vars:
            st.markdown("---")
            st.markdown("**Available Variables:**")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("📊 Original Variables:")
                for var in original_vars[:5]:
                    st.text(f"  • {var}")
                if len(original_vars) > 5:
                    st.caption(f"  ... and {len(original_vars) - 5} more")
            
            with col2:
                st.caption("✨ Cleaned/Transformed:")
                for var in cleaned_vars[:5]:
                    st.text(f"  • {var}")
                if len(cleaned_vars) > 5:
                    st.caption(f"  ... and {len(cleaned_vars) - 5} more")
    
    # Main visualization tabs
    tab1, tab2 = st.tabs(["📈 Single Variable", "📊 Multi-Variable Comparison"])
    
    with tab1:
        render_single_variable_plot(all_vars)
    
    with tab2:
        render_multi_variable_plot(all_vars)


def render_single_variable_plot(variables):
    """Render single variable time plot interface"""
    
    st.header("Single Variable Time Plot")
    st.markdown("Plot one variable over time with customization options")
    
    # Configuration columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Variable selection with grouping if cleaned data exists
        all_vars, original_vars, cleaned_vars, has_cleaned_data = get_available_variables()
        
        if has_cleaned_data and cleaned_vars:
            # Create grouped selection
            st.markdown("**Select variable:**")
            
            var_type = st.radio(
                "Variable type",
                ["Original", "Cleaned/Transformed", "All"],
                horizontal=True,
                key="single_var_type"
            )
            
            if var_type == "Original":
                available_vars = original_vars
                help_text = "Original variables from uploaded data"
            elif var_type == "Cleaned/Transformed":
                available_vars = cleaned_vars
                help_text = "Variables after cleaning or transformation"
            else:
                available_vars = all_vars
                help_text = "All available variables"
            
            selected_var = st.selectbox(
                "Choose variable",
                available_vars,
                help=help_text,
                key="single_var_select"
            )
        else:
            # Simple selection if no cleaned data
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
    st.markdown("Compare multiple variables with **independent Y-axes** for each variable")
    
    # Get variable info
    all_vars, original_vars, cleaned_vars, has_cleaned_data = get_available_variables()
    
    # Variable selection
    st.subheader("Step 1: Select Variables")
    
    # Add filter if cleaned data exists
    if has_cleaned_data and cleaned_vars:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            var_filter = st.multiselect(
                "Filter by type",
                ["Original", "Cleaned/Transformed"],
                default=["Original", "Cleaned/Transformed"],
                key="multi_var_filter"
            )
        
        # Filter variables based on selection
        filtered_vars = []
        if "Original" in var_filter:
            filtered_vars.extend(original_vars)
        if "Cleaned/Transformed" in var_filter:
            filtered_vars.extend(cleaned_vars)
        
        available_vars = sorted(filtered_vars)
        
        with col2:
            st.info(f"📊 Showing {len(available_vars)} variables ({len([v for v in available_vars if v in original_vars])} original, {len([v for v in available_vars if v in cleaned_vars])} cleaned)")
    else:
        available_vars = all_vars
    
    selected_vars = st.multiselect(
        "Choose 2 or more variables to compare",
        available_vars,
        default=available_vars[:2] if len(available_vars) >= 2 else available_vars,
        help="Each variable will have its own Y-axis for independent control",
        key="multi_var_select"
    )
    
    if len(selected_vars) < 2:
        st.info("ℹ️ Select at least 2 variables to create a comparison plot")
        return
    
    st.markdown("---")
    
    # Configuration for each variable
    st.subheader("Step 2: Configure Each Variable (Independent Y-Axes)")
    
    st.info(f"📊 Creating {len(selected_vars)} independent Y-axes - one for each variable")
    
    axis_config = {}
    
    for idx, var in enumerate(selected_vars):
        with st.expander(f"⚙️ {var} Settings (Y-axis {idx+1})", expanded=(idx < 2)):
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
                    key=f"multi_scale_{var}_{idx}",
                    help="Independent scale for this variable"
                )
            
            with col3:
                line_width = st.slider(
                    "Line Width",
                    1, 5, 2,
                    key=f"multi_width_{var}_{idx}"
                )
            
            # Axis range (optional)
            st.markdown(f"**Y-Axis Range for {var}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_custom_range = st.checkbox(
                    "Custom range",
                    key=f"multi_custom_range_{var}_{idx}",
                    help="Set specific min/max values"
                )
            
            if use_custom_range:
                # Get data for this variable
                var_data = get_data_for_variable(var)
                data_values = var_data['value'].dropna()
                
                if len(data_values) > 0:
                    data_min, data_max = data_values.min(), data_values.max()
                    
                    with col2:
                        y_min = st.number_input(
                            "Min",
                            value=float(data_min),
                            key=f"multi_ymin_{var}_{idx}",
                            format="%.2f"
                        )
                    
                    with col3:
                        y_max = st.number_input(
                            "Max",
                            value=float(data_max),
                            key=f"multi_ymax_{var}_{idx}",
                            format="%.2f"
                        )
                else:
                    y_min, y_max = None, None
            else:
                y_min, y_max = None, None
            
            # Store configuration
            axis_config[var] = {
                "color": color,
                "scale": scale,
                "line_width": line_width,
                "y_min": y_min,
                "y_max": y_max,
                "axis_index": idx
            }
    
    st.markdown("---")
    
    # Global plot options
    st.subheader("Step 3: Plot Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_legend = st.checkbox("Show legend", value=True, key="multi_show_legend")
        plot_height = st.slider("Plot height", 400, 900, 600, step=50, key="multi_plot_height")
    
    with col2:
        show_grid = st.checkbox("Show grid", value=True, key="multi_show_grid")
    
    with col3:
        plot_title = st.text_input("Plot title (optional)", "", key="multi_plot_title")
    
    # Create plot button
    if st.button("📊 Generate Comparison Plot", type="primary", key="multi_plot_button"):
        with st.spinner("Creating multi-variable plot with independent axes..."):
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
                
                if fig is None:
                    display_error("Failed to create plot - figure is None")
                    return
                
                # Store figure and metadata in session state
                st.session_state.multi_var_fig = fig
                st.session_state.multi_var_names = selected_vars
                
            except Exception as e:
                display_error(f"Error creating plot: {str(e)}")
                st.exception(e)
    
    # Display plot if it exists in session state
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
        
        # Display summary statistics
        with st.expander("📊 Summary Statistics"):
            stats_data = []
            
            for var in st.session_state.multi_var_names:
                var_data = get_data_for_variable(var)
                values = var_data['value'].dropna()
                
                stats_data.append({
                    "Variable": var,
                    "Count": len(values),
                    "Mean": f"{values.mean():.2f}" if len(values) > 0 else "N/A",
                    "Std Dev": f"{values.std():.2f}" if len(values) > 0 else "N/A",
                    "Min": f"{values.min():.2f}" if len(values) > 0 else "N/A",
                    "Max": f"{values.max():.2f}" if len(values) > 0 else "N/A"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


def create_multi_axis_plot(variables, axis_config, show_legend=True, 
                           show_grid=True, height=600, title=None):
    """
    Create a plot with independent Y-axis for each variable
    """
    try:
        n_vars = len(variables)
        
        if n_vars == 0:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each variable
        for idx, var in enumerate(variables):
            # Get variable data from appropriate dataframe
            var_data = get_data_for_variable(var)
            var_data = var_data.sort_values('timestamp')
            var_data = var_data.dropna(subset=['timestamp', 'value'])
            
            if len(var_data) == 0:
                continue
            
            # Get configuration
            config = axis_config.get(var, {})
            color = config.get('color', get_color_for_index(idx))
            line_width = config.get('line_width', 2)
            
            # Determine which y-axis to use
            if idx == 0:
                yaxis = 'y'
            else:
                yaxis = f'y{idx+1}'
            
            # Create trace
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
                gridcolor='lightgray',
                zeroline=False,
                domain=[0.15, 1]  # Leave space for y-axes on left
            ),
            'height': height,
            'hovermode': 'x unified',
            'template': 'plotly_white',
            'showlegend': show_legend,
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        }
        
        if title:
            layout_config['title'] = dict(
                text=title,
                x=0.5,
                xanchor='center'
            )
        
        # Configure each y-axis
        for idx, var in enumerate(variables):
            config = axis_config.get(var, {})
            scale = config.get('scale', 'linear')
            y_min = config.get('y_min')
            y_max = config.get('y_max')
            color = config.get('color', get_color_for_index(idx))
            
            if idx == 0:
                # Primary y-axis (left side)
                yaxis_config = {
                    'title': dict(
                        text=var,
                        font=dict(color=color)
                    ),
                    'tickfont': dict(color=color),
                    'type': scale,
                    'showgrid': show_grid,
                    'gridcolor': 'lightgray',
                    'zeroline': False,
                    'anchor': 'x'
                }
                
                if y_min is not None and y_max is not None:
                    yaxis_config['range'] = [y_min, y_max]
                
                layout_config['yaxis'] = yaxis_config
            else:
                # Additional y-axes (overlaid on left side)
                position = 0.15 - (idx * 0.08)  # Stack on left side
                
                yaxis_config = {
                    'title': dict(
                        text=var,
                        font=dict(color=color)
                    ),
                    'tickfont': dict(color=color),
                    'type': scale,
                    'showgrid': False,  # Only primary axis shows grid
                    'zeroline': False,
                    'anchor': 'free',
                    'overlaying': 'y',
                    'side': 'left',
                    'position': max(0, position)  # Ensure position is not negative
                }
                
                if y_min is not None and y_max is not None:
                    yaxis_config['range'] = [y_min, y_max]
                
                layout_config[f'yaxis{idx+1}'] = yaxis_config
        
        fig.update_layout(**layout_config)
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error in create_multi_axis_plot: {str(e)}")
        import traceback
        traceback.print_exc()
        st.code(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()
