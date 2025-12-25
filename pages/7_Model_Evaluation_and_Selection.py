"""
Page 7: Model Evaluation & Selection
Interactive matrix for evaluating and selecting best models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import visualization
from core.models.model_visualization import create_model_comparison_plot
from core.models.data_preparation import get_series_data

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Model Evaluation & Selection", 
    page_icon="📊", 
    layout="wide"
)

# Render shared sidebar
render_app_sidebar()

st.title("📊 Model Evaluation & Selection")
st.markdown("*Evaluate trained models and select the best for each variable*")
st.markdown("---")


def initialize_selection_state():
    """Initialize session state for model selection"""
    if "selected_models_for_forecast" not in st.session_state:
        st.session_state.selected_models_for_forecast = {}
    if "model_selection_complete" not in st.session_state:
        st.session_state.model_selection_complete = False


def calculate_health_index(metrics: dict) -> float:
    """
    Calculate health index (0-100) from R², MAE, RMSE, MAPE
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with 'r2', 'mae', 'rmse', 'mape'
        
    Returns:
    --------
    health_index : float
        Score from 0-100 (100 = best)
    """
    # Extract metrics
    r2 = metrics.get('r2', 0)
    mae = metrics.get('mae', np.inf)
    rmse = metrics.get('rmse', np.inf)
    mape = metrics.get('mape', np.inf)
    
    # Handle NaN/inf
    if np.isnan(r2) or np.isinf(r2):
        r2 = 0
    if np.isnan(mae) or np.isinf(mae):
        mae = 100
    if np.isnan(rmse) or np.isinf(rmse):
        rmse = 100
    if np.isnan(mape) or np.isinf(mape):
        mape = 100
    
    # R² contribution (0-1 → 0-40 points)
    # Higher R² is better
    r2_score = max(0, min(r2, 1)) * 40
    
    # MAE contribution (0-20 points)
    # Lower MAE is better - use inverse scaling
    # Assume MAE < 10 is excellent, MAE > 50 is poor
    mae_score = max(0, 20 * (1 - min(mae / 50, 1)))
    
    # RMSE contribution (0-20 points)
    # Lower RMSE is better
    # Assume RMSE < 10 is excellent, RMSE > 50 is poor
    rmse_score = max(0, 20 * (1 - min(rmse / 50, 1)))
    
    # MAPE contribution (0-20 points)
    # Lower MAPE is better
    # Assume MAPE < 10% is excellent, MAPE > 50% is poor
    mape_score = max(0, 20 * (1 - min(mape / 50, 1)))
    
    # Total health index
    health_index = r2_score + mae_score + rmse_score + mape_score
    
    return round(health_index, 2)


def get_metric_color(value: float, metric: str, all_values: list) -> str:
    """
    Get color for metric value (gradient from red to green)
    
    Parameters:
    -----------
    value : float
        Metric value
    metric : str
        Metric name ('r2', 'mae', 'rmse', 'mape')
    all_values : list
        All values for this metric (for scaling)
        
    Returns:
    --------
    color : str
        RGB color string
    """
    if len(all_values) == 0 or np.isnan(value):
        return 'rgb(200, 200, 200)'  # Gray for missing
    
    # Filter out NaN
    clean_values = [v for v in all_values if not np.isnan(v)]
    if len(clean_values) == 0:
        return 'rgb(200, 200, 200)'
    
    min_val = min(clean_values)
    max_val = max(clean_values)
    
    # Normalize value to 0-1
    if max_val == min_val:
        normalized = 0.5
    else:
        if metric in ['r2', 'health_index']:
            # Higher is better
            normalized = (value - min_val) / (max_val - min_val)
        else:
            # Lower is better (mae, rmse, mape)
            normalized = 1 - (value - min_val) / (max_val - min_val)
    
    # Create gradient from red (0) to yellow (0.5) to green (1)
    if normalized < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0
    
    return f'rgb({r}, {g}, {b})'


def create_evaluation_matrix():
    """
    Create interactive evaluation matrix DataFrame
    
    Returns:
    --------
    matrix_data : dict
        Dictionary with matrix info for rendering
    """
    if not st.session_state.get('trained_models'):
        return None
    
    # Get all variables and models
    variables = list(st.session_state.trained_models.keys())
    
    # Get all unique model names across all variables
    all_model_names = set()
    for var_results in st.session_state.trained_models.values():
        all_model_names.update(var_results.get('tier1_math', {}).keys())
        all_model_names.update(var_results.get('tier2_timeseries', {}).keys())
        all_model_names.update(var_results.get('tier3_ml', {}).keys())
    
    all_model_names = sorted(list(all_model_names))
    
    # Create matrix data
    matrix_data = {
        'variables': variables,
        'models': all_model_names,
        'metrics': {}  # Store metrics for each variable-model pair
    }
    
    # Populate metrics
    for var_name in variables:
        var_results = st.session_state.trained_models[var_name]
        matrix_data['metrics'][var_name] = {}
        
        for model_name in all_model_names:
            # Find model in tiers
            model_data = None
            for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                if model_name in var_results.get(tier_name, {}):
                    model_data = var_results[tier_name][model_name]
                    break
            
            if model_data:
                test_metrics = model_data.get('test_metrics', {})
                
                # Calculate health index
                health_index = calculate_health_index(test_metrics)
                
                matrix_data['metrics'][var_name][model_name] = {
                    'health_index': health_index,
                    'r2': test_metrics.get('r2', np.nan),
                    'mae': test_metrics.get('mae', np.nan),
                    'rmse': test_metrics.get('rmse', np.nan),
                    'mape': test_metrics.get('mape', np.nan),
                    'equation': model_data.get('equation', 'N/A'),
                    'model_data': model_data
                }
            else:
                matrix_data['metrics'][var_name][model_name] = None
    
    return matrix_data


def display_cell_details(variable: str, model: str, model_data: dict):
    """
    Display detailed information for a selected cell
    
    Parameters:
    -----------
    variable : str
        Variable name
    model : str
        Model name
    model_data : dict
        Model data with metrics and predictions
    """
    st.markdown(f"### 📊 {variable} - {model}")
    
    # Health Index (big display)
    metrics_info = model_data.get('test_metrics', {})
    health_index = calculate_health_index(metrics_info)
    
    col_health = st.columns(1)[0]
    
    # Color code health index
    if health_index >= 80:
        health_color = "🟢"
        health_label = "Excellent"
    elif health_index >= 60:
        health_color = "🟡"
        health_label = "Good"
    elif health_index >= 40:
        health_color = "🟠"
        health_label = "Fair"
    else:
        health_color = "🔴"
        health_label = "Poor"
    
    st.markdown(f"## {health_color} Health Index: {health_index}/100 ({health_label})")
    st.progress(health_index / 100)
    
    st.markdown("---")
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    test_metrics = model_data.get('test_metrics', {})
    
    with col1:
        r2 = test_metrics.get('r2', 0)
        st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        mae = test_metrics.get('mae', 0)
        st.metric("MAE", f"{mae:.4f}")
    
    with col3:
        rmse = test_metrics.get('rmse', 0)
        st.metric("RMSE", f"{rmse:.4f}")
    
    with col4:
        mape = test_metrics.get('mape', np.nan)
        if not np.isnan(mape):
            st.metric("MAPE", f"{mape:.2f}%")
        else:
            st.metric("MAPE", "N/A")
    
    # Model equation
    equation = model_data.get('equation', 'N/A')
    if equation != 'N/A':
        st.markdown("**Model Equation:**")
        st.code(equation, language="text")
    
    st.markdown("---")
    
    # Train vs Test comparison
    train_metrics = model_data.get('train_metrics', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set:**")
        st.text(f"R²:   {train_metrics.get('r2', 0):.4f}")
        st.text(f"MAE:  {train_metrics.get('mae', 0):.4f}")
        st.text(f"RMSE: {train_metrics.get('rmse', 0):.4f}")
    
    with col2:
        st.markdown("**Test Set:**")
        st.text(f"R²:   {test_metrics.get('r2', 0):.4f}")
        st.text(f"MAE:  {test_metrics.get('mae', 0):.4f}")
        st.text(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
    
    # Overfitting check
    train_r2 = train_metrics.get('r2', 0)
    test_r2 = test_metrics.get('r2', 0)
    gap = train_r2 - test_r2
    
    if gap > 0.2:
        st.warning(f"⚠️ Possible overfitting (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    elif gap > 0.1:
        st.info(f"💡 Moderate train-test gap (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    else:
        st.success(f"✅ Good generalization (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    
    st.markdown("---")
    
    # Visualization
    st.markdown("**Model Performance Visualization**")
    
    # Customization options
    col_a, col_b = st.columns(2)
    with col_a:
        scatter_size = st.slider(
            "Point Size",
            min_value=1,
            max_value=15,
            value=6,
            key=f"scatter_size_{variable}_{model}"
        )
    with col_b:
        scatter_color = st.color_picker(
            "Point Color",
            value="#000000",
            key=f"scatter_color_{variable}_{model}"
        )
    
    try:
        # Get variable results
        var_results = st.session_state.trained_models[variable]
        split_data = var_results.get('train_test_split', {})
        
        train_values = split_data.get('train_values', np.array([]))
        test_values = split_data.get('test_values', np.array([]))
        train_timestamps = split_data.get('train_timestamps', pd.DatetimeIndex([]))
        test_timestamps = split_data.get('test_timestamps', pd.DatetimeIndex([]))
        split_index = split_data.get('split_index', 0)
        
        # Create plot
        fig = create_model_comparison_plot(
            variable_name=variable,
            train_values=train_values,
            test_values=test_values,
            train_timestamps=train_timestamps,
            test_timestamps=test_timestamps,
            model_results=var_results,
            selected_models=[model],
            split_index=split_index
        )
        
        # Apply customizations
        fig.data[0].marker.size = scatter_size
        fig.data[0].marker.color = scatter_color
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        with st.expander("💾 Export Figure"):
            quick_export_buttons(fig, f"{variable}_{model}_evaluation", ['png', 'pdf', 'html'])
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")


def main():
    """Main page function"""
    
    # Initialize state
    initialize_selection_state()
    
    # Check if models are trained
    if not st.session_state.get('training_complete', False):
        st.warning("⚠️ No models trained yet!")
        st.info("👈 Please go to **Time-Based Models & ML Training** to train models first")
        
        if st.button("🤖 Go to Training Page"):
            st.switch_page("pages/6_Time-Based_Models_&_ML_Training.py")
        return
    
    if not st.session_state.trained_models:
        st.error("❌ No model results found!")
        return
    
    # Get matrix data
    matrix_data = create_evaluation_matrix()
    
    if matrix_data is None:
        st.error("❌ Could not create evaluation matrix")
        return
    
    variables = matrix_data['variables']
    models = matrix_data['models']
    
    st.success(f"✅ Ready to evaluate {len(variables)} variables with {len(models)} model types")
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ How to Use This Page", expanded=False):
        st.markdown("""
        ### Model Evaluation Matrix
        
        **Color Coding:**
        - 🟢 **Green:** Best performing models
        - 🟡 **Yellow:** Average performance
        - 🔴 **Red:** Poor performance
        - 🔵 **Blue:** Selected for forecasting
        - ⚪ **Gray:** Model not available
        
        **How to Select Models:**
        1. Review the color-coded matrix below
        2. Click on a cell to see detailed metrics
        3. Click "Select for Forecasting" to choose this model
        4. Selected cells turn blue
        5. Review your selections in the summary
        6. Proceed to forecasting when ready
        
        **Metrics Used:**
        - **R²:** Variance explained (higher is better)
        - Green cells = high R² relative to other models
        """)
    
    st.markdown("---")
    
    # === METRIC SELECTOR ===
    st.subheader("📊 Evaluation Matrix")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        metric_display = st.selectbox(
            "Select metric to display in cells:",
            ['health_index', 'r2', 'mae', 'rmse', 'mape'],
            format_func=lambda x: {
                'health_index': '🏥 Health Index (0-100)',
                'r2': 'R² (Variance Explained)',
                'mae': 'MAE (Mean Absolute Error)',
                'rmse': 'RMSE (Root Mean Squared Error)',
                'mape': 'MAPE (Mean Absolute % Error)'
            }[x],
            index=0,  # Default to health_index
            key='metric_selector'
        )
    
    with col2:
        if st.button("🔄 Apply & Update Colors", type="primary", use_container_width=True):
            st.session_state.current_metric = metric_display
            st.rerun()
    
    with col3:
        st.metric("Variables", len(variables))
        st.metric("Model Types", len(models))
    
    # Get current metric (use session state to persist after Apply button)
    if 'current_metric' not in st.session_state:
        st.session_state.current_metric = 'health_index'
    
    current_metric = st.session_state.current_metric
    
    # === INTERACTIVE MATRIX ===
    st.markdown("#### Interactive Evaluation Matrix")
    st.caption("Click on any cell to see details and select model")
    
    # Create grid using columns
    # Header row
    header_cols = st.columns([2] + [1] * len(models))
    header_cols[0].markdown("**Variable**")
    for idx, model in enumerate(models):
        header_cols[idx + 1].markdown(f"**{model}**")
    
    st.markdown("---")
    
    # Data rows
    for var_idx, variable in enumerate(variables):
        cols = st.columns([2] + [1] * len(models))
        
        # Variable name
        cols[0].markdown(f"**{variable}**")
        
        # Model cells
        for model_idx, model in enumerate(models):
            with cols[model_idx + 1]:
                metrics = matrix_data['metrics'][variable].get(model)
                
                if metrics is None:
                    # Model not available
                    st.markdown("⚪")
                else:
                    # Get metric value for coloring
                    metric_value = metrics.get(current_metric, np.nan)
                    
                    # Get all values for this metric to scale colors
                    all_metric_values = []
                    for v in variables:
                        for m in models:
                            m_data = matrix_data['metrics'][v].get(m)
                            if m_data:
                                val = m_data.get(current_metric, np.nan)
                                if not np.isnan(val):
                                    all_metric_values.append(val)
                    
                    # Check if selected
                    is_selected = (variable in st.session_state.selected_models_for_forecast and 
                                 st.session_state.selected_models_for_forecast[variable] == model)
                    
                    # Get color
                    if is_selected:
                        # Blue for selected - use markdown with colored background
                        bg_color = '#4682B4'  # Steel blue
                        text_color = 'white'
                    elif np.isnan(metric_value):
                        bg_color = '#C8C8C8'  # Gray
                        text_color = 'black'
                    else:
                        color_rgb = get_metric_color(metric_value, current_metric, all_metric_values)
                        bg_color = color_rgb.replace('rgb', '').replace('(', '').replace(')', '')
                        rgb_parts = [int(x.strip()) for x in bg_color.split(',')]
                        bg_color = f'#{rgb_parts[0]:02x}{rgb_parts[1]:02x}{rgb_parts[2]:02x}'
                        # Use white text on dark backgrounds
                        brightness = (rgb_parts[0] * 299 + rgb_parts[1] * 587 + rgb_parts[2] * 114) / 1000
                        text_color = 'white' if brightness < 128 else 'black'
                    
                    # Format button label based on metric
                    if current_metric == 'health_index':
                        button_label = f"{int(metric_value)}" if not np.isnan(metric_value) else "N/A"
                    else:
                        button_label = f"{metric_value:.3f}" if not np.isnan(metric_value) else "N/A"
                    
                    # Create colored button using HTML/CSS
                    button_html = f"""
                    <div style="
                        background-color: {bg_color};
                        color: {text_color};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        font-weight: bold;
                        cursor: pointer;
                        border: 2px solid {'#000' if is_selected else '#ccc'};
                        margin: 2px;
                    ">
                        {button_label}
                    </div>
                    """
                    
                    st.markdown(button_html, unsafe_allow_html=True)
                    
                    # Add invisible button for click detection
                    if st.button(
                        "Select",
                        key=f"cell_{variable}_{model}",
                        help=f"Click to view details for {variable} - {model}",
                        use_container_width=True
                    ):
                        # Store selection for modal
                        st.session_state.selected_cell = {
                            'variable': variable,
                            'model': model,
                            'metrics': metrics
                        }
                        st.rerun()
    
    st.markdown("---")
    
    # === CELL DETAILS MODAL ===
    if 'selected_cell' in st.session_state and st.session_state.selected_cell:
        cell_info = st.session_state.selected_cell
        
        with st.container():
            st.markdown("---")
            
            # Close button
            col1, col2 = st.columns([5, 1])
            with col2:
                if st.button("✖ Close", key="close_modal"):
                    st.session_state.selected_cell = None
                    st.rerun()
            
            # Display details
            display_cell_details(
                cell_info['variable'],
                cell_info['model'],
                cell_info['metrics']['model_data']
            )
            
            # Selection button
            st.markdown("---")
            
            variable = cell_info['variable']
            model = cell_info['model']
            
            is_already_selected = (variable in st.session_state.selected_models_for_forecast and 
                                  st.session_state.selected_models_for_forecast[variable] == model)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if is_already_selected:
                    st.success(f"✅ Currently selected for {variable}")
                    if st.button("🔄 Unselect This Model", type="secondary", use_container_width=True):
                        del st.session_state.selected_models_for_forecast[variable]
                        st.session_state.selected_cell = None
                        st.rerun()
                else:
                    if st.button("✅ Select for Forecasting", type="primary", use_container_width=True):
                        st.session_state.selected_models_for_forecast[variable] = model
                        st.success(f"Selected {model} for {variable}!")
                        st.session_state.selected_cell = None
                        st.rerun()
    
    # === SELECTION SUMMARY ===
    if st.session_state.selected_models_for_forecast:
        st.markdown("---")
        st.subheader("✅ Selected Models Summary")
        
        summary_data = []
        for variable, model in st.session_state.selected_models_for_forecast.items():
            # Get metrics
            metrics = matrix_data['metrics'][variable].get(model)
            if metrics:
                test_metrics = metrics['model_data'].get('test_metrics', {})
                health_index = metrics.get('health_index', 0)
                summary_data.append({
                    'Variable': variable,
                    'Selected Model': model,
                    'Health Index': f"{int(health_index)}/100",
                    'R²': f"{test_metrics.get('r2', 0):.4f}",
                    'MAE': f"{test_metrics.get('mae', 0):.4f}",
                    'RMSE': f"{test_metrics.get('rmse', 0):.4f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Export summary
        with st.expander("💾 Export Selection Summary"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Selection Summary (CSV)",
                data=csv,
                file_name=f"model_selections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Check if all variables have selections
        if len(st.session_state.selected_models_for_forecast) >= len(variables):
            st.session_state.model_selection_complete = True
            
            st.success(f"✅ All {len(variables)} variables have selected models!")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔮 Proceed to Future Projections →", type="primary", use_container_width=True):
                    st.switch_page("pages/8_Future_Projections.py")
        else:
            missing = len(variables) - len(st.session_state.selected_models_for_forecast)
            st.warning(f"⚠️ {missing} variable(s) still need model selection")


if __name__ == "__main__":
    main()
