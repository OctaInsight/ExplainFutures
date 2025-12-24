"""
Page 3: Data Exploration & Visualization
Interactive visualization of time-series data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_warning, display_info, get_color_for_index
from core.shared_sidebar import render_app_sidebar  
from core.viz.plot_time import plot_single_variable

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Data Exploration & Visualization", page_icon="📊", layout="wide")

st.title("📊 Data Exploration & Visualization")
st.markdown("*Create interactive plots of your time-series data*")
st.markdown("---")


def main():
    """Main page function"""
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Health** to load your data first")
        
        st.page_link("pages/1_Upload_and_Data_Health.py", label="Go to Upload Page", icon="📁")
        return
    
    # Get data
    df_long = st.session_state.df_long
    variables = sorted(df_long['variable'].unique().tolist())
    
    # Display data info
    with st.expander("ℹ️ Current Dataset Info"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Variables", len(variables))
        col2.metric("Data Points", len(df_long))
        
        time_range = f"{df_long['timestamp'].min().date()} to {df_long['timestamp'].max().date()}"
        col3.caption("Time Range")
        col3.write(time_range)
        
        if st.session_state.uploaded_file_name:
            col4.caption("Source File")
            col4.write(st.session_state.uploaded_file_name)
    
    # Main visualization tabs
    tab1, tab2 = st.tabs(["📈 Single Variable", "📊 Multi-Variable Comparison"])
    
    with tab1:
        render_single_variable_plot(df_long, variables)
    
    with tab2:
        render_multi_variable_plot(df_long, variables)


def render_single_variable_plot(df_long, variables):
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
            # Filter data for selected variable
            var_data = df_long[df_long['variable'] == selected_var].copy()
            
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
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            with st.expander("📊 Variable Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                
                values = var_data['value'].dropna()
                col1.metric("Count", len(values))
                col2.metric("Mean", f"{values.mean():.2f}")
                col3.metric("Std Dev", f"{values.std():.2f}")
                col4.metric("Range", f"{values.min():.2f} to {values.max():.2f}")


def render_multi_variable_plot(df_long, variables):
    """Render multi-variable comparison plot interface"""
    
    st.header("Multi-Variable Comparison")
    st.markdown("Compare multiple variables with **independent Y-axes** for each variable")
    
    # Variable selection
    st.subheader("Step 1: Select Variables")
    
    selected_vars = st.multiselect(
        "Choose 2 or more variables to compare",
        variables,
        default=variables[:2] if len(variables) >= 2 else variables,
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
                    key=f"multi_color_{var}"
                )
            
            with col2:
                scale = st.selectbox(
                    "Y-Axis Scale",
                    ["linear", "log"],
                    key=f"multi_scale_{var}",
                    help="Independent scale for this variable"
                )
            
            with col3:
                line_width = st.slider(
                    "Line Width",
                    1, 5, 2,
                    key=f"multi_width_{var}"
                )
            
            # Axis range (optional)
            st.markdown(f"**Y-Axis Range for {var}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_custom_range = st.checkbox(
                    "Custom range",
                    key=f"multi_custom_range_{var}",
                    help="Set specific min/max values"
                )
            
            if use_custom_range:
                with col2:
                    # Get data range for defaults
                    var_data = df_long[df_long['variable'] == var]['value'].dropna()
                    data_min, data_max = var_data.min(), var_data.max()
                    
                    y_min = st.number_input(
                        "Min",
                        value=float(data_min),
                        key=f"multi_ymin_{var}",
                        format="%.2f"
                    )
                
                with col3:
                    y_max = st.number_input(
                        "Max",
                        value=float(data_max),
                        key=f"multi_ymax_{var}",
                        format="%.2f"
                    )
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
                    df_long,
                    variables=selected_vars,
                    axis_config=axis_config,
                    show_legend=show_legend,
                    show_grid=show_grid,
                    height=plot_height,
                    title=plot_title if plot_title else None
                )
                
                if fig is None:
                    display_error("Failed to create plot - figure is None")
                    return
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                with st.expander("📊 Summary Statistics"):
                    stats_data = []
                    
                    for var in selected_vars:
                        var_data = df_long[df_long['variable'] == var]['value'].dropna()
                        stats_data.append({
                            "Variable": var,
                            "Count": len(var_data),
                            "Mean": f"{var_data.mean():.2f}",
                            "Std Dev": f"{var_data.std():.2f}",
                            "Min": f"{var_data.min():.2f}",
                            "Max": f"{var_data.max():.2f}"
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                display_error(f"Error creating plot: {str(e)}")
                st.exception(e)


def create_multi_axis_plot(df_long, variables, axis_config, show_legend=True, 
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
            # Get variable data
            var_data = df_long[df_long['variable'] == var].copy()
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
