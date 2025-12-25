"""
Page 7: Model Evaluation & Selection
Compare models with interactive evaluation matrix and select best models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import evaluation modules
from core.models.evaluation import (
    get_metric_color,
    create_actual_vs_predicted_plot,
    create_residual_plot,
    create_time_series_plot,
    create_metrics_comparison_plot
)

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Model Evaluation", 
    page_icon="📊", 
    layout="wide"
)

# Render shared sidebar
render_app_sidebar()

st.title("📊 Model Evaluation & Selection")
st.markdown("*Compare models and select the best for forecasting*")
st.markdown("---")


def initialize_selection_state():
    """Initialize session state for model selection"""
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = {}
    if "auto_selected" not in st.session_state:
        st.session_state.auto_selected = False


def create_evaluation_matrix(series_list: list, metric: str = 'reliability'):
    """
    Create evaluation matrix DataFrame
    
    Parameters:
    -----------
    series_list : list
        List of series names
    metric : str
        Metric to display ('reliability', 'mae', 'rmse', 'r2', 'mape')
        
    Returns:
    --------
    matrix_df : pd.DataFrame
        Matrix with series as rows, models as columns
    """
    # Get all unique model names
    all_model_names = set()
    for series_name in series_list:
        if series_name in st.session_state.model_results:
            all_model_names.update(st.session_state.model_results[series_name].keys())
    
    all_model_names = sorted(list(all_model_names))
    
    # Create matrix
    matrix_data = []
    
    for series_name in series_list:
        row = {'Series': series_name}
        
        if series_name in st.session_state.model_results:
            models = st.session_state.model_results[series_name]
            
            for model_name in all_model_names:
                if model_name in models:
                    artifact = models[model_name]
                    value = artifact.test_data['metrics'].get(metric, np.nan)
                    row[model_name] = value
                else:
                    row[model_name] = np.nan
        else:
            for model_name in all_model_names:
                row[model_name] = np.nan
        
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df = matrix_df.set_index('Series')
    
    return matrix_df


def create_colored_matrix_html(matrix_df: pd.DataFrame, metric: str):
    """
    Create HTML table with color coding
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        Matrix dataframe
    metric : str
        Metric name
        
    Returns:
    --------
    html : str
        HTML string
    """
    html = ['<table style="width:100%; border-collapse: collapse; font-size:14px;">']
    
    # Header row
    html.append('<tr style="background-color:#f0f0f0;">')
    html.append('<th style="border:1px solid #ddd; padding:8px; text-align:left;">Series</th>')
    for col in matrix_df.columns:
        html.append(f'<th style="border:1px solid #ddd; padding:8px; text-align:center;">{col}</th>')
    html.append('</tr>')
    
    # Data rows
    for series_name, row in matrix_df.iterrows():
        html.append('<tr>')
        html.append(f'<td style="border:1px solid #ddd; padding:8px; font-weight:bold;">{series_name}</td>')
        
        # Get all values for color scaling
        all_values = [v for v in row.values if not np.isnan(v)]
        
        for col in matrix_df.columns:
            value = row[col]
            
            if np.isnan(value):
                cell_html = '<td style="border:1px solid #ddd; padding:8px; text-align:center; background-color:#f9f9f9;">-</td>'
            else:
                # Get color
                color = get_metric_color(value, metric, all_values)
                
                # Format value
                if metric == 'reliability':
                    formatted = f"{value:.1f}"
                elif metric in ['r2']:
                    formatted = f"{value:.3f}"
                elif metric in ['mae', 'rmse']:
                    formatted = f"{value:.3f}"
                elif metric == 'mape':
                    formatted = f"{value:.1f}%"
                else:
                    formatted = f"{value:.3f}"
                
                cell_html = f'<td style="border:1px solid #ddd; padding:8px; text-align:center; background-color:{color}; color:white; font-weight:bold; cursor:pointer;" data-series="{series_name}" data-model="{col}">{formatted}</td>'
            
            html.append(cell_html)
        
        html.append('</tr>')
    
    html.append('</table>')
    
    return ''.join(html)


def display_model_details(series_name: str, model_name: str):
    """
    Display detailed metrics and plots for a specific model
    
    Parameters:
    -----------
    series_name : str
        Series name
    model_name : str
        Model name
    """
    if series_name not in st.session_state.model_results:
        st.error(f"No results for series: {series_name}")
        return
    
    if model_name not in st.session_state.model_results[series_name]:
        st.error(f"No results for model: {model_name}")
        return
    
    artifact = st.session_state.model_results[series_name][model_name]
    
    st.markdown(f"### {series_name} - {model_name}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    test_metrics = artifact.test_data['metrics']
    
    with col1:
        st.metric("MAE", f"{test_metrics.get('mae', 0):.3f}")
    with col2:
        st.metric("RMSE", f"{test_metrics.get('rmse', 0):.3f}")
    with col3:
        r2 = test_metrics.get('r2', 0)
        st.metric("R²", f"{r2:.3f}")
    with col4:
        mape = test_metrics.get('mape', np.nan)
        if not np.isnan(mape):
            st.metric("MAPE", f"{mape:.1f}%")
        else:
            st.metric("MAPE", "N/A")
    with col5:
        reliability = test_metrics.get('reliability', 0)
        st.metric("Reliability", f"{reliability:.1f}/100")
    
    # Train vs Test comparison
    train_metrics = artifact.train_data['metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set:**")
        st.text(f"  R²:   {train_metrics.get('r2', 0):.3f}")
        st.text(f"  MAE:  {train_metrics.get('mae', 0):.3f}")
        st.text(f"  RMSE: {train_metrics.get('rmse', 0):.3f}")
    
    with col2:
        st.markdown("**Test Set:**")
        st.text(f"  R²:   {test_metrics.get('r2', 0):.3f}")
        st.text(f"  MAE:  {test_metrics.get('mae', 0):.3f}")
        st.text(f"  RMSE: {test_metrics.get('rmse', 0):.3f}")
    
    # Overfitting check
    train_r2 = train_metrics.get('r2', 0)
    test_r2 = test_metrics.get('r2', 0)
    gap = train_r2 - test_r2
    
    if gap > 0.2:
        st.warning(f"⚠️ Possible overfitting detected (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    elif gap > 0.1:
        st.info(f"💡 Moderate train-test gap (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    else:
        st.success(f"✅ Good generalization (Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
    
    # Model equation (if available)
    equation = artifact.model.get_equation()
    if equation and equation != "Model not fitted":
        st.markdown("**Model Equation:**")
        st.code(equation, language="text")
    
    # Plots
    st.markdown("---")
    st.markdown("#### Diagnostic Plots")
    
    # Actual vs Predicted
    fig_scatter = create_actual_vs_predicted_plot(
        artifact.test_data['y_true'],
        artifact.test_data['y_pred'],
        title=f"{series_name} - {model_name}: Actual vs Predicted"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    with st.expander("💾 Export Actual vs Predicted"):
        quick_export_buttons(fig_scatter, f"actual_vs_predicted_{series_name}_{model_name}", ['png', 'pdf', 'html'])
    
    # Residual plot
    fig_residual = create_residual_plot(
        artifact.test_data['y_true'],
        artifact.test_data['y_pred'],
        title=f"{series_name} - {model_name}: Residuals"
    )
    st.plotly_chart(fig_residual, use_container_width=True)
    
    with st.expander("💾 Export Residual Plot"):
        quick_export_buttons(fig_residual, f"residuals_{series_name}_{model_name}", ['png', 'pdf', 'html'])
    
    # Time series plot (if timestamps available)
    if artifact.config.get('split_info') and 'train_end' in artifact.config['split_info']:
        # Reconstruct full series with timestamps
        # This is a simplified version - you may need to adapt based on your data structure
        pass  # TODO: Add time series plot if needed


def auto_select_best_models():
    """Automatically select best model for each series based on reliability"""
    for series_name, models in st.session_state.model_results.items():
        best_model = None
        best_reliability = 0
        
        for model_name, artifact in models.items():
            reliability = artifact.test_data['metrics'].get('reliability', 0)
            
            if reliability > best_reliability:
                best_reliability = reliability
                best_model = model_name
        
        if best_model:
            st.session_state.selected_models[series_name] = best_model
    
    st.session_state.auto_selected = True


def main():
    """Main page function"""
    
    # Initialize state
    initialize_selection_state()
    
    # Check if models are trained
    if not st.session_state.get('model_training_complete', False):
        st.warning("⚠️ No models trained yet!")
        st.info("👈 Please go to **Time-Based Models & Training** to train models first")
        
        if st.button("🤖 Go to Training Page"):
            st.switch_page("pages/6_Time_Based_Models_and_Training.py")
        return
    
    if not st.session_state.model_results:
        st.error("❌ No model results found!")
        return
    
    # Get series list
    series_list = list(st.session_state.model_results.keys())
    
    st.success(f"✅ Models trained for {len(series_list)} time series")
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown("""
        ### Model Evaluation Matrix
        
        **Color Coding:**
        - 🟢 **Green:** Best performing models (top 25%)
        - 🟡 **Yellow:** Average performance (middle 50%)
        - 🔴 **Red:** Poor performance (bottom 25%)
        
        **How to evaluate:**
        1. View the matrix below (switch between metrics)
        2. Click on any cell to see detailed analysis
        3. Use auto-select or manually choose best models
        4. Selected models will be used for forecasting
        
        **Metrics explained:**
        - **Reliability (0-100):** Overall model quality score
        - **R²:** Variance explained (higher is better, 0-1)
        - **MAE:** Mean Absolute Error (lower is better)
        - **RMSE:** Root Mean Squared Error (lower is better)
        - **MAPE:** Mean Absolute Percentage Error (lower is better)
        """)
    
    st.markdown("---")
    
    # === METRIC SELECTOR ===
    st.subheader("📊 Evaluation Matrix")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        metric_display = st.selectbox(
            "Select metric to display",
            ['reliability', 'r2', 'mae', 'rmse', 'mape'],
            format_func=lambda x: {
                'reliability': 'Reliability Score (0-100)',
                'r2': 'R² (Variance Explained)',
                'mae': 'MAE (Mean Absolute Error)',
                'rmse': 'RMSE (Root Mean Squared Error)',
                'mape': 'MAPE (Mean Absolute % Error)'
            }[x]
        )
    
    with col2:
        st.metric("Total Series", len(series_list))
    
    with col3:
        # Count total models
        total_models = sum(len(models) for models in st.session_state.model_results.values())
        st.metric("Total Models", total_models)
    
    # Create and display matrix
    matrix_df = create_evaluation_matrix(series_list, metric=metric_display)
    
    # Display as styled dataframe
    st.markdown("#### Interactive Evaluation Matrix")
    st.markdown("*Click cells below to view detailed analysis*")
    
    # Create color-coded display
    def color_cells(val, metric=metric_display):
        """Color code cells based on metric value"""
        if pd.isna(val):
            return 'background-color: #f9f9f9'
        
        # Get all values for context
        all_vals = matrix_df.values.flatten()
        all_vals = [v for v in all_vals if not np.isnan(v)]
        
        color = get_metric_color(val, metric, all_vals)
        return f'background-color: {color}; color: white; font-weight: bold'
    
    # Format and style
    styled_df = matrix_df.style.applymap(color_cells).format(precision=3)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # === INTERACTIVE CELL SELECTION ===
    st.subheader("🔍 Detailed Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_series = st.selectbox(
            "Select series",
            series_list,
            key="detail_series"
        )
    
    with col2:
        if selected_series and selected_series in st.session_state.model_results:
            available_models = list(st.session_state.model_results[selected_series].keys())
            selected_model = st.selectbox(
                "Select model",
                available_models,
                key="detail_model"
            )
        else:
            selected_model = None
            st.info("Select a series first")
    
    if selected_series and selected_model:
        st.markdown("---")
        display_model_details(selected_series, selected_model)
    
    st.markdown("---")
    
    # === MODEL SELECTION ===
    st.subheader("✅ Model Selection for Forecasting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Select the best model for each time series. These models will be used for:
        - Future forecasting
        - Scenario analysis
        - What-if simulations
        """)
    
    with col2:
        if st.button("🎯 Auto-Select Best Models", type="primary", use_container_width=True):
            auto_select_best_models()
            st.success(f"✅ Auto-selected best models for {len(series_list)} series!")
            st.rerun()
    
    # Manual selection interface
    st.markdown("#### Current Selections")
    
    selection_data = []
    
    for series_name in series_list:
        if series_name in st.session_state.model_results:
            available_models = list(st.session_state.model_results[series_name].keys())
            
            # Get current selection
            current_selection = st.session_state.selected_models.get(series_name, None)
            
            # Model selector
            selected = st.selectbox(
                f"Best model for **{series_name}**",
                available_models,
                index=available_models.index(current_selection) if current_selection in available_models else 0,
                key=f"select_{series_name}"
            )
            
            # Update selection
            st.session_state.selected_models[series_name] = selected
            
            # Get metrics
            artifact = st.session_state.model_results[series_name][selected]
            reliability = artifact.test_data['metrics'].get('reliability', 0)
            r2 = artifact.test_data['metrics'].get('r2', 0)
            mae = artifact.test_data['metrics'].get('mae', 0)
            
            selection_data.append({
                'Series': series_name,
                'Selected Model': selected,
                'Reliability': f"{reliability:.1f}",
                'R²': f"{r2:.3f}",
                'MAE': f"{mae:.3f}"
            })
    
    # Display selections summary
    if selection_data:
        st.markdown("---")
        st.markdown("#### Selection Summary")
        selection_df = pd.DataFrame(selection_data)
        st.dataframe(selection_df, use_container_width=True, hide_index=True)
        
        # Export selection summary
        with st.expander("💾 Export Selection Summary"):
            csv = selection_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Selection Summary (CSV)",
                data=csv,
                file_name=f"model_selections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # === COMPARISON VISUALIZATIONS ===
    st.markdown("---")
    st.subheader("📈 Model Comparison Visualizations")
    
    # Select series for comparison
    comparison_series = st.selectbox(
        "Select series to compare models",
        series_list,
        key="comparison_series"
    )
    
    if comparison_series and comparison_series in st.session_state.model_results:
        # Create metrics dataframe
        models_data = []
        
        for model_name, artifact in st.session_state.model_results[comparison_series].items():
            metrics = artifact.test_data['metrics']
            models_data.append({
                'Model': model_name,
                'reliability': metrics.get('reliability', 0),
                'r2': metrics.get('r2', 0),
                'mae': metrics.get('mae', 0),
                'rmse': metrics.get('rmse', 0),
                'mape': metrics.get('mape', np.nan)
            })
        
        models_df = pd.DataFrame(models_data).set_index('Model')
        
        # Comparison plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_reliability = create_metrics_comparison_plot(
                models_df, 
                'reliability',
                title=f"{comparison_series}: Reliability Comparison"
            )
            st.plotly_chart(fig_reliability, use_container_width=True)
            
            with st.expander("💾 Export Reliability Comparison"):
                quick_export_buttons(fig_reliability, f"reliability_comparison_{comparison_series}", ['png', 'pdf', 'html'])
        
        with col2:
            fig_mae = create_metrics_comparison_plot(
                models_df,
                'mae',
                title=f"{comparison_series}: MAE Comparison"
            )
            st.plotly_chart(fig_mae, use_container_width=True)
            
            with st.expander("💾 Export MAE Comparison"):
                quick_export_buttons(fig_mae, f"mae_comparison_{comparison_series}", ['png', 'pdf', 'html'])
    
    # === NEXT STEPS ===
    st.markdown("---")
    
    if len(st.session_state.selected_models) >= len(series_list):
        st.success(f"✅ All {len(series_list)} series have selected models!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Proceed to Future Projections →", type="primary", use_container_width=True):
                st.switch_page("pages/8_Future_Projections.py")
    else:
        missing = len(series_list) - len(st.session_state.selected_models)
        st.warning(f"⚠️ {missing} series still need model selection")


if __name__ == "__main__":
    main()
