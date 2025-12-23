"""
Page 2: Explore and Visualize
Interactive visualization of time-series data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_warning, display_info, get_color_for_index
from core.viz.plot_time import plot_single_variable
from core.viz.plot_multi_axis import plot_multi_variable

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Explore & Visualize", page_icon="📊", layout="wide")

st.title("📊 Explore & Visualize")
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
            help="Choose which variable to plot"
        )
    
    with col2:
        # Color selection
        color = st.color_picker(
            "Line color",
            value=get_color_for_index(0),
            help="Choose line color"
        )
    
    # Advanced options
    with st.expander("⚙️ Plot Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_markers = st.checkbox("Show markers", value=False)
            line_width = st.slider("Line width", 1, 5, 2)
        
        with col2:
            y_scale = st.selectbox("Y-axis scale", ["Linear", "Log"])
            show_grid = st.checkbox("Show grid", value=True)
        
        with col3:
            plot_height = st.slider("Plot height", 300, 800, 500, step=50)
    
    # Create plot button
    if st.button("📈 Generate Plot", type="primary"):
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
                y_scale=y_scale.lower(),
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
    st.markdown("Compare multiple variables with independent Y-axes")
    
    # Variable selection
    st.subheader("Step 1: Select Variables")
    
    selected_vars = st.multiselect(
        "Choose 2 or more variables to compare",
        variables,
        default=variables[:2] if len(variables) >= 2 else variables,
        help="Select variables to plot together"
    )
    
    if len(selected_vars) < 2:
        st.info("ℹ️ Select at least 2 variables to create a comparison plot")
        return
    
    st.markdown("---")
    
    # Configuration for each variable
    st.subheader("Step 2: Configure Each Variable")
    
    axis_config = {}
    
    for idx, var in enumerate(selected_vars):
        with st.expander(f"⚙️ {var} Settings", expanded=(idx < 2)):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                color = st.color_picker(
                    "Color",
                    value=get_color_for_index(idx),
                    key=f"color_{var}"
                )
            
            with col2:
                axis_side = st.selectbox(
                    "Y-axis",
                    ["Left", "Right"],
                    key=f"axis_{var}",
                    help="Which side to place the Y-axis"
                )
            
            with col3:
                scale = st.selectbox(
                    "Scale",
                    ["Linear", "Log"],
                    key=f"scale_{var}"
                )
            
            with col4:
                line_width = st.slider(
                    "Line width",
                    1, 5, 2,
                    key=f"width_{var}"
                )
            
            # Axis range (optional)
            col1, col2 = st.columns(2)
            
            with col1:
                use_custom_range = st.checkbox(
                    "Custom Y-axis range",
                    key=f"custom_range_{var}"
                )
            
            if use_custom_range:
                with col2:
                    # Get data range for defaults
                    var_data = df_long[df_long['variable'] == var]['value'].dropna()
                    data_min, data_max = var_data.min(), var_data.max()
                    
                    y_min = st.number_input(
                        "Min",
                        value=float(data_min),
                        key=f"ymin_{var}"
                    )
                    y_max = st.number_input(
                        "Max",
                        value=float(data_max),
                        key=f"ymax_{var}"
                    )
            else:
                y_min, y_max = None, None
            
            # Store configuration
            axis_config[var] = {
                "color": color,
                "axis_side": axis_side.lower(),
                "scale": scale.lower(),
                "line_width": line_width,
                "y_min": y_min,
                "y_max": y_max
            }
    
    st.markdown("---")
    
    # Global plot options
    st.subheader("Step 3: Plot Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_legend = st.checkbox("Show legend", value=True)
        plot_height = st.slider("Plot height", 400, 900, 600, step=50)
    
    with col2:
        show_grid = st.checkbox("Show grid", value=True)
        sync_zoom = st.checkbox("Synchronized zoom", value=True)
    
    with col3:
        plot_title = st.text_input("Plot title (optional)", "")
    
    # Create plot button
    if st.button("📊 Generate Comparison Plot", type="primary"):
        with st.spinner("Creating multi-variable plot..."):
            # Create plot
            fig = plot_multi_variable(
                df_long,
                variables=selected_vars,
                axis_config=axis_config,
                show_legend=show_legend,
                show_grid=show_grid,
                height=plot_height,
                title=plot_title if plot_title else None
            )
            
            if fig is None:
                display_error("Failed to create plot")
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


if __name__ == "__main__":
    main()
