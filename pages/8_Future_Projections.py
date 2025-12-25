"""
Page 8: Future Projections
Generate future forecasts using selected models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Future Projections", 
    page_icon="🔮", 
    layout="wide"
)

# Render shared sidebar
render_app_sidebar()

st.title("🔮 Future Projections")
st.markdown("*Generate forecasts using your selected models*")
st.markdown("---")


def initialize_forecast_state():
    """Initialize session state for forecasting"""
    if "forecasts_generated" not in st.session_state:
        st.session_state.forecasts_generated = False
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = {}


def validate_forecast_horizon(last_date: pd.Timestamp, forecast_date: pd.Timestamp, data_length: int) -> dict:
    """
    Validate if forecast horizon is reasonable
    
    Parameters:
    -----------
    last_date : pd.Timestamp
        Last date in training data
    forecast_date : pd.Timestamp
        Target forecast date
    data_length : int
        Number of data points in training
        
    Returns:
    --------
    validation : dict
        {
            'valid': bool,
            'warning_level': str ('safe', 'caution', 'risky'),
            'message': str,
            'days_ahead': int,
            'ratio': float (forecast_period / training_period)
        }
    """
    days_ahead = (forecast_date - last_date).days
    
    # Calculate data span
    data_span_days = data_length  # Approximate (depends on frequency)
    ratio = days_ahead / data_span_days if data_span_days > 0 else 0
    
    # Validation rules
    if days_ahead <= 0:
        return {
            'valid': False,
            'warning_level': 'error',
            'message': '❌ Forecast date must be in the future!',
            'days_ahead': days_ahead,
            'ratio': ratio
        }
    
    elif ratio <= 0.5:
        # Forecasting less than 50% of training period
        return {
            'valid': True,
            'warning_level': 'safe',
            'message': f'✅ Safe forecast horizon ({days_ahead} days ahead)',
            'days_ahead': days_ahead,
            'ratio': ratio
        }
    
    elif ratio <= 1.0:
        # Forecasting 50-100% of training period
        return {
            'valid': True,
            'warning_level': 'caution',
            'message': f'⚠️ Moderate extrapolation ({days_ahead} days ahead - be cautious)',
            'days_ahead': days_ahead,
            'ratio': ratio
        }
    
    else:
        # Forecasting more than training period
        return {
            'valid': True,
            'warning_level': 'risky',
            'message': f'🔴 High uncertainty ({days_ahead} days ahead - {ratio:.1f}x training period)',
            'days_ahead': days_ahead,
            'ratio': ratio
        }


def generate_forecast(variable: str, model_name: str, target_date: pd.Timestamp, last_date: pd.Timestamp) -> dict:
    """
    Generate forecast for a variable using selected model
    
    Parameters:
    -----------
    variable : str
        Variable name
    model_name : str
        Selected model name
    target_date : pd.Timestamp
        Target forecast date (user-specified)
    last_date : pd.Timestamp
        Last date in historical data
        
    Returns:
    --------
    forecast : dict
        {
            'forecast_values': np.ndarray,
            'forecast_timestamps': pd.DatetimeIndex,
            'model_type': str,
            'confidence_interval': dict (optional)
        }
    """
    # Get model results
    var_results = st.session_state.trained_models[variable]
    
    # Find model in tiers
    model_data = None
    model_type = None
    
    for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
        if model_name in var_results.get(tier_name, {}):
            model_data = var_results[tier_name][model_name]
            model_type = tier_name
            break
    
    if model_data is None:
        raise ValueError(f"Model {model_name} not found for {variable}")
    
    # Get split data
    split_data = var_results.get('train_test_split', {})
    
    # Calculate exact number of steps from last_date to target_date
    forecast_steps = (target_date - last_date).days
    
    # Generate future timestamps - EXACT range from last_date to target_date
    train_timestamps = split_data['train_timestamps']
    if len(train_timestamps) > 1:
        freq = pd.infer_freq(train_timestamps)
        if freq is None:
            # Calculate median difference
            median_diff = pd.Series(train_timestamps[1:] - train_timestamps[:-1]).median()
            future_timestamps = pd.date_range(
                start=last_date + median_diff,
                end=target_date,
                freq=median_diff
            )
        else:
            future_timestamps = pd.date_range(
                start=last_date,
                end=target_date,
                freq=freq
            )[1:]  # Exclude last_date (already in historical)
    else:
        # Default to daily
        future_timestamps = pd.date_range(
            start=last_date + timedelta(days=1),
            end=target_date,
            freq='D'
        )
    
    # Recalculate forecast_steps based on actual timestamps generated
    forecast_steps = len(future_timestamps)
    
    # Generate forecast based on model type
    if model_type == 'tier2_timeseries':
        # Time series models have forecast method
        model = model_data['model']
        forecast_values = model.forecast(steps=forecast_steps)
        
    elif model_type == 'tier1_math':
        # Mathematical models - extrapolate
        start_time = split_data['train_timestamps'][0]
        time_since_start = (future_timestamps - start_time).total_seconds().values / 86400
        
        # Use model to predict
        if 'poly_transformer' in model_data:
            # Polynomial model
            X_future = model_data['poly_transformer'].transform(time_since_start.reshape(-1, 1))
            forecast_values = model_data['model'].predict(X_future)
        else:
            # Linear or other
            X_future = time_since_start.reshape(-1, 1)
            forecast_values = model_data['model'].predict(X_future)
    
    elif model_type == 'tier3_ml':
        # ML models need features - use last known values for lag features
        last_value = split_data['test_values'][-1] if len(split_data.get('test_values', [])) > 0 else split_data['train_values'][-1]
        forecast_values = np.full(forecast_steps, last_value)
        st.warning(f"ML model forecasting simplified for {variable} - using last known value")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'forecast_values': np.array(forecast_values),
        'forecast_timestamps': future_timestamps,
        'model_type': model_type,
        'model_name': model_name,
        'target_date': target_date
    }


def create_forecast_plot(variable: str, 
                        historical_values: np.ndarray,
                        historical_timestamps: pd.DatetimeIndex,
                        forecast_values: np.ndarray,
                        forecast_timestamps: pd.DatetimeIndex,
                        split_index: int,
                        customize: dict,
                        target_date: pd.Timestamp) -> go.Figure:
    """
    Create forecast visualization showing ONLY the forecast timeframe
    
    Parameters:
    -----------
    variable : str
        Variable name
    historical_values : np.ndarray
        Historical data
    historical_timestamps : pd.DatetimeIndex
        Historical timestamps
    forecast_values : np.ndarray
        Forecast values
    forecast_timestamps : pd.DatetimeIndex
        Forecast timestamps
    split_index : int
        Where train/test split occurred
    customize : dict
        Customization options (colors, sizes, visibility)
    target_date : pd.Timestamp
        User-specified target forecast date
        
    Returns:
    --------
    fig : go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Determine timeframe to show
    # Show last portion of historical + all forecast up to target_date
    last_historical_date = historical_timestamps[-1] if len(historical_timestamps) > 0 else forecast_timestamps[0]
    
    # Calculate how much historical to show (20% or last 30 days, whichever is less)
    days_to_show_hist = min(30, int(len(historical_values) * 0.2))
    hist_start_idx = max(0, len(historical_values) - days_to_show_hist)
    
    # Filter historical data to show
    hist_values_to_show = historical_values[hist_start_idx:]
    hist_timestamps_to_show = historical_timestamps[hist_start_idx:]
    
    # Filter forecast to show only up to target_date
    forecast_mask = forecast_timestamps <= target_date
    forecast_values_to_show = forecast_values[forecast_mask]
    forecast_timestamps_to_show = forecast_timestamps[forecast_mask]
    
    # Historical data
    if customize.get('show_historical', True):
        fig.add_trace(go.Scatter(
            x=hist_timestamps_to_show,
            y=hist_values_to_show,
            mode='markers',
            name='Historical Data',
            marker=dict(
                size=customize.get('hist_size', 6),
                color=customize.get('hist_color', '#000000')
            ),
            showlegend=True
        ))
    
    # Forecast data
    if customize.get('show_forecast', True):
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_to_show,
            y=forecast_values_to_show,
            mode='markers',
            name='Forecast',
            marker=dict(
                size=customize.get('forecast_size', 8),
                color=customize.get('forecast_color', '#FF6B6B'),
                symbol='diamond'
            ),
            showlegend=True
        ))
    
    # Model line (connect historical and forecast)
    if customize.get('show_model_line', True):
        all_times = list(hist_timestamps_to_show) + list(forecast_timestamps_to_show)
        all_values = list(hist_values_to_show) + list(forecast_values_to_show)
        
        fig.add_trace(go.Scatter(
            x=all_times,
            y=all_values,
            mode='lines',
            name='Model Trend',
            line=dict(
                color=customize.get('line_color', '#4A90E2'),
                width=2,
                dash='dash'
            ),
            showlegend=True
        ))
    
    # Vertical line at forecast start
    if len(historical_timestamps) > 0:
        split_time = historical_timestamps[-1]
        
        fig.add_shape(
            type="line",
            x0=split_time,
            x1=split_time,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=split_time,
            y=1.0,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )
    
    # Layout with explicit X-axis range
    fig.update_layout(
        title=f"{variable} - Historical Data & Forecast (to {target_date.strftime('%Y-%m-%d')})",
        xaxis_title="Time",
        yaxis_title=f"{variable}",
        height=600,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(color='black', size=11)
        ),
        xaxis=dict(
            range=[hist_timestamps_to_show[0] if len(hist_timestamps_to_show) > 0 else forecast_timestamps_to_show[0],
                   target_date]
        )
    )
    
    return fig


def main():
    """Main page function"""
    
    # Initialize state
    initialize_forecast_state()
    
    # Check if model selection is complete
    if not st.session_state.get('model_selection_complete', False):
        st.warning("⚠️ No models selected yet!")
        st.info("👈 Please go to **Model Evaluation & Selection** to select models first")
        
        if st.button("📊 Go to Model Selection"):
            st.switch_page("pages/7_Model_Evaluation_and_Selection.py")
        return
    
    # === SELECTED MODELS SUMMARY (Collapsible) ===
    with st.expander("📋 Selected Models Summary", expanded=False):
        st.markdown("### Models Selected for Forecasting")
        
        summary_data = []
        for variable, model in st.session_state.selected_models_for_forecast.items():
            # Get metrics from trained models
            var_results = st.session_state.trained_models.get(variable, {})
            
            # Find model metrics
            model_data = None
            for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                if model in var_results.get(tier_name, {}):
                    model_data = var_results[tier_name][model]
                    break
            
            if model_data:
                test_metrics = model_data.get('test_metrics', {})
                summary_data.append({
                    'Variable': variable,
                    'Selected Model': model,
                    'R²': f"{test_metrics.get('r2', 0):.4f}",
                    'MAE': f"{test_metrics.get('mae', 0):.4f}",
                    'RMSE': f"{test_metrics.get('rmse', 0):.4f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No model data available")
    
    # === HOW TO USE THIS PAGE (Collapsible) ===
    with st.expander("ℹ️ How to Use This Page", expanded=False):
        st.markdown("""
        ### Forecasting Workflow
        
        **Step 1: Set Forecast Horizon**
        - Choose a future date for forecasting
        - The system will validate if the forecast is reasonable
        - ✅ Green: Safe (< 50% of training period)
        - ⚠️ Yellow: Caution (50-100% of training period)  
        - 🔴 Red: Risky (> training period)
        
        **Step 2: Generate Forecasts**
        - Click "Generate Forecasts" button
        - System uses selected models to predict future values
        - Forecasts appear in separate tabs (one per variable)
        
        **Step 3: Review & Customize**
        - Each tab shows a forecast visualization
        - Customize colors and sizes of data points
        - Toggle visibility of historical data, forecasts, and trend lines
        - Export figures as PNG, PDF, or HTML
        
        **Step 4: Export Results**
        - Download forecast data as CSV
        - Export all visualizations
        - Use forecasts for decision-making
        
        **Important Notes:**
        - Forecasts are extrapolations based on historical patterns
        - Longer forecasts = higher uncertainty
        - Always consider domain knowledge when interpreting results
        """)
    
    st.markdown("---")
    
    # === STEP 1: FORECAST CONFIGURATION ===
    st.subheader("🎯 Step 1: Configure Forecast")
    
    # Get date range from data
    if st.session_state.trained_models:
        first_var = list(st.session_state.trained_models.keys())[0]
        split_data = st.session_state.trained_models[first_var]['train_test_split']
        
        first_date = split_data['train_timestamps'][0]
        last_date = split_data['test_timestamps'][-1] if len(split_data.get('test_timestamps', [])) > 0 else split_data['train_timestamps'][-1]
        data_length = len(split_data['train_values']) + len(split_data['test_values'])
    else:
        first_date = pd.Timestamp('2020-01-01')
        last_date = pd.Timestamp('2024-12-31')
        data_length = 100
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Data Range:**")
        st.text(f"First date: {first_date.strftime('%Y-%m-%d')}")
        st.text(f"Last date:  {last_date.strftime('%Y-%m-%d')}")
        st.text(f"Data points: {data_length}")
    
    with col2:
        forecast_date = st.date_input(
            "Forecast until:",
            value=last_date + timedelta(days=30),
            min_value=last_date + timedelta(days=1),
            help="Select the future date to forecast to"
        )
        
        forecast_date = pd.Timestamp(forecast_date)
    
    with col3:
        st.markdown("**Forecast Horizon:**")
        days_ahead = (forecast_date - last_date).days
        st.metric("Days Ahead", days_ahead)
    
    # Validate forecast horizon
    validation = validate_forecast_horizon(last_date, forecast_date, data_length)
    
    if validation['warning_level'] == 'error':
        st.error(validation['message'])
    elif validation['warning_level'] == 'safe':
        st.success(validation['message'])
    elif validation['warning_level'] == 'caution':
        st.warning(validation['message'])
    else:  # risky
        st.error(validation['message'])
        st.warning("⚠️ Proceeding with this forecast may produce unreliable results!")
    
    # Show forecast/training ratio
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Forecast Period", f"{validation['days_ahead']} days")
    with col2:
        st.metric("Forecast/Training Ratio", f"{validation['ratio']:.2f}x")
    
    st.markdown("---")
    
    # === STEP 2: GENERATE FORECASTS ===
    st.subheader("🚀 Step 2: Generate Forecasts")
    
    if st.button("🔮 Generate Forecasts for All Variables", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate forecasts
        forecast_results = {}
        
        for idx, (variable, model_name) in enumerate(st.session_state.selected_models_for_forecast.items()):
            status_text.text(f"Generating forecast for: {variable}...")
            
            try:
                forecast = generate_forecast(variable, model_name, forecast_date, last_date)
                forecast_results[variable] = forecast
                
            except Exception as e:
                st.error(f"Error forecasting {variable}: {str(e)}")
                continue
            
            # Update progress
            progress = (idx + 1) / len(st.session_state.selected_models_for_forecast)
            progress_bar.progress(progress)
        
        # Store results AND target date
        st.session_state.forecast_results = forecast_results
        st.session_state.forecast_target_date = forecast_date
        st.session_state.forecasts_generated = True
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Generated forecasts for {len(forecast_results)} variables!")
        st.rerun()
    
    # === STEP 3: DISPLAY FORECASTS ===
    if st.session_state.forecasts_generated and st.session_state.forecast_results:
        
        st.markdown("---")
        st.subheader("📊 Step 3: Review Forecasts")
        
        # Create tabs for each variable
        tabs = st.tabs(list(st.session_state.forecast_results.keys()))
        
        for tab, variable in zip(tabs, st.session_state.forecast_results.keys()):
            with tab:
                display_forecast_tab(variable)


def display_forecast_tab(variable: str):
    """Display forecast results for a single variable"""
    
    st.markdown(f"### {variable}")
    
    forecast = st.session_state.forecast_results[variable]
    var_results = st.session_state.trained_models[variable]
    split_data = var_results['train_test_split']
    
    # Get target date
    target_date = st.session_state.get('forecast_target_date', forecast['forecast_timestamps'][-1])
    
    # Get historical data
    train_values = split_data['train_values']
    test_values = split_data['test_values']
    train_timestamps = split_data['train_timestamps']
    test_timestamps = split_data['test_timestamps']
    
    historical_values = np.concatenate([train_values, test_values])
    historical_timestamps = pd.DatetimeIndex(list(train_timestamps) + list(test_timestamps))
    
    # Filter forecast to target_date
    forecast_mask = forecast['forecast_timestamps'] <= target_date
    forecast_values_filtered = forecast['forecast_values'][forecast_mask]
    forecast_timestamps_filtered = forecast['forecast_timestamps'][forecast_mask]
    
    # === CUSTOMIZATION OPTIONS ===
    st.markdown("#### 🎨 Customize Visualization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Historical Data:**")
        show_hist = st.checkbox("Show", value=True, key=f"show_hist_{variable}")
        hist_size = st.slider("Size", 1, 15, 6, key=f"hist_size_{variable}")
        hist_color = st.color_picker("Color", "#000000", key=f"hist_color_{variable}")
    
    with col2:
        st.markdown("**Forecast Data:**")
        show_forecast = st.checkbox("Show", value=True, key=f"show_forecast_{variable}")
        forecast_size = st.slider("Size", 1, 15, 8, key=f"forecast_size_{variable}")
        forecast_color = st.color_picker("Color", "#FF6B6B", key=f"forecast_color_{variable}")
    
    with col3:
        st.markdown("**Model Trend Line:**")
        show_line = st.checkbox("Show", value=True, key=f"show_line_{variable}")
        line_color = st.color_picker("Color", "#4A90E2", key=f"line_color_{variable}")
    
    with col4:
        st.markdown("**Export Options:**")
        st.markdown("")  # Spacing
        st.markdown("")  # Spacing
    
    st.markdown("---")
    
    # === VISUALIZATION ===
    st.markdown("#### 📈 Forecast Visualization")
    
    customize = {
        'show_historical': show_hist,
        'hist_size': hist_size,
        'hist_color': hist_color,
        'show_forecast': show_forecast,
        'forecast_size': forecast_size,
        'forecast_color': forecast_color,
        'show_model_line': show_line,
        'line_color': line_color
    }
    
    fig = create_forecast_plot(
        variable=variable,
        historical_values=historical_values,
        historical_timestamps=historical_timestamps,
        forecast_values=forecast_values_filtered,
        forecast_timestamps=forecast_timestamps_filtered,
        split_index=len(train_values),
        customize=customize,
        target_date=target_date
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === EXPORT ===
    with st.expander("💾 Export Forecast"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Figure:**")
            quick_export_buttons(fig, f"forecast_{variable}", ['png', 'pdf', 'html'])
        
        with col2:
            st.markdown("**Export Data:**")
            
            # Create forecast DataFrame - ONLY up to target_date
            forecast_df = pd.DataFrame({
                'Date': forecast_timestamps_filtered,
                'Forecast': forecast_values_filtered,
                'Variable': variable,
                'Model': forecast['model_name']
            })
            
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"forecast_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # === FORECAST STATISTICS ===
    with st.expander("📊 Forecast Statistics"):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Forecast Steps", len(forecast_values_filtered))
        
        with col2:
            st.metric("Mean Forecast", f"{np.mean(forecast_values_filtered):.3f}")
        
        with col3:
            st.metric("Forecast Range", f"{np.ptp(forecast_values_filtered):.3f}")
        
        # Forecast table
        st.markdown("**Forecast Values:**")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # ADD EXPORT BUTTON FOR TABLE
        st.markdown("---")
        st.markdown("**Export Table Data:**")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            csv_table = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_table,
                file_name=f"forecast_table_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"table_csv_{variable}"
            )
        
        with col_b:
            excel_buffer = io.BytesIO()
            forecast_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"forecast_table_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"table_excel_{variable}"
            )
        
        with col_c:
            json_data = forecast_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"forecast_table_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key=f"table_json_{variable}"
            )


if __name__ == "__main__":
    main()
