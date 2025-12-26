"""
Page 8: Future Projections (with Iterative Retraining)
Generate future forecasts using selected models with smart iteration
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from copy import deepcopy

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

# Render shared sidebar
render_app_sidebar()

st.title("🔮 Future Projections")
st.markdown("*Generate forecasts using your selected models with intelligent iteration*")
st.markdown("---")


# ============================================================================
# MODEL SAFETY CLASSIFICATION
# ============================================================================

# Model types that are SAFE for iterative retraining
SAFE_FOR_ITERATION = [
    'ARIMA', 'SARIMAX', 'AutoARIMA', 'ExponentialSmoothing', 
    'Holt-Winters', 'Linear', 'Ridge', 'Lasso', 'ElasticNet'
]

# Model types that are RISKY for iterative retraining
RISKY_FOR_ITERATION = [
    'Polynomial', 'RandomForest', 'GradientBoosting', 
    'XGBoost', 'NeuralNetwork', 'SVR', 'KNN'
]


def initialize_forecast_state():
    """Initialize session state for forecasting"""
    if "forecasts_generated" not in st.session_state:
        st.session_state.forecasts_generated = False
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = {}
    if "forecast_iterations" not in st.session_state:
        st.session_state.forecast_iterations = {}


def is_model_safe_for_iteration(model_name: str) -> bool:
    """
    Check if model type is safe for iterative retraining
    
    Parameters:
    -----------
    model_name : str
        Model name
        
    Returns:
    --------
    bool
        True if safe, False if risky
    """
    # Check if any safe keyword in model name
    model_upper = model_name.upper()
    
    for safe_type in SAFE_FOR_ITERATION:
        if safe_type.upper() in model_upper:
            return True
    
    # If not in safe list, assume risky
    return False


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
            'ratio': float (forecast_period / training_period),
            'use_iterative': bool (whether to use iterative approach)
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
            'ratio': ratio,
            'use_iterative': False
        }
    
    elif ratio <= 0.5:
        # Forecasting less than 50% of training period - SAFE
        return {
            'valid': True,
            'warning_level': 'safe',
            'message': f'✅ Safe forecast horizon ({days_ahead} days ahead) - Direct forecasting',
            'days_ahead': days_ahead,
            'ratio': ratio,
            'use_iterative': False  # No need for iteration
        }
    
    elif ratio <= 1.5:
        # Forecasting 50-150% of training period - USE ITERATION
        return {
            'valid': True,
            'warning_level': 'caution',
            'message': f'⚠️ Moderate extrapolation ({days_ahead} days ahead) - Using iterative retraining for safety',
            'days_ahead': days_ahead,
            'ratio': ratio,
            'use_iterative': True  # Use iterative approach
        }
    
    else:
        # Forecasting more than 150% training period - RISKY
        return {
            'valid': True,
            'warning_level': 'risky',
            'message': f'🔴 High uncertainty ({days_ahead} days ahead - {ratio:.1f}x training period) - Iterative forecast with high uncertainty',
            'days_ahead': days_ahead,
            'ratio': ratio,
            'use_iterative': True  # Use iteration but with warnings
        }


def clone_model_for_retraining(model, model_type: str):
    """
    Clone a model for retraining
    
    Parameters:
    -----------
    model : object
        Original trained model
    model_type : str
        Type of model (tier1_math, tier2_timeseries, tier3_ml)
        
    Returns:
    --------
    cloned_model : object
        Cloned model ready for retraining
    """
    try:
        # Try to use copy/clone methods
        if hasattr(model, 'clone'):
            return model.clone()
        elif hasattr(model, 'copy'):
            return model.copy()
        else:
            # Deep copy as fallback
            return deepcopy(model)
    except:
        # If cloning fails, return original (will train from scratch)
        return model


def iterative_forecast_with_retraining(
    variable: str,
    model_name: str,
    model_data: dict,
    model_type: str,
    split_data: dict,
    target_date: pd.Timestamp,
    last_date: pd.Timestamp,
    safe_ratio: float = 0.5,
    confidence_decay: float = 0.05,
    max_iterations: int = 10
) -> dict:
    """
    Generate forecast using iterative retraining approach
    
    This method:
    1. Forecasts in chunks (each chunk = safe_ratio * current_data_length)
    2. Retrains model on original + forecasted data after each chunk
    3. Repeats until target_date is reached
    
    Parameters:
    -----------
    variable : str
        Variable name
    model_name : str
        Model name
    model_data : dict
        Model data from training
    model_type : str
        Model tier (tier1_math, tier2_timeseries, tier3_ml)
    split_data : dict
        Train/test split data
    target_date : pd.Timestamp
        Target forecast date
    last_date : pd.Timestamp
        Last date in historical data
    safe_ratio : float
        Ratio for safe forecast horizon (default 0.5)
    confidence_decay : float
        Confidence reduction per iteration (default 0.05 = 5%)
    max_iterations : int
        Maximum number of iterations allowed
        
    Returns:
    --------
    result : dict
        {
            'forecast_values': np.ndarray,
            'forecast_timestamps': pd.DatetimeIndex,
            'iterations': int,
            'confidence_scores': np.ndarray,
            'iteration_points': list of int (where iterations occurred)
        }
    """
    st.info(f"🔄 Using iterative retraining for {variable} (safer for long forecasts)")
    
    # Initialize
    all_forecast_values = []
    all_forecast_timestamps = []
    confidence_scores = []
    iteration_points = []
    
    # Get current data
    train_values = split_data['train_values']
    test_values = split_data['test_values']
    current_data = np.concatenate([train_values, test_values])
    
    train_timestamps = split_data['train_timestamps']
    test_timestamps = split_data['test_timestamps']
    current_timestamps = pd.DatetimeIndex(list(train_timestamps) + list(test_timestamps))
    
    # Current model
    current_model = model_data['model']
    
    # Starting point
    current_date = last_date
    iteration = 0
    
    # Progress tracking
    progress_placeholder = st.empty()
    iteration_info = st.empty()
    
    while current_date < target_date and iteration < max_iterations:
        iteration += 1
        
        # Calculate safe forecast steps for this iteration
        current_data_length = len(current_data)
        safe_steps = max(1, int(current_data_length * safe_ratio))
        
        # Don't overshoot target
        days_remaining = (target_date - current_date).days
        steps_this_iteration = min(safe_steps, days_remaining)
        
        # Update progress
        progress = 1 - (days_remaining / (target_date - last_date).days)
        progress_placeholder.progress(min(progress, 1.0))
        iteration_info.text(f"Iteration {iteration}: Forecasting {steps_this_iteration} steps...")
        
        # === FORECAST THIS CHUNK ===
        try:
            if model_type == 'tier2_timeseries':
                # Time series models
                chunk_forecast = current_model.forecast(steps=steps_this_iteration)
                
            elif model_type == 'tier1_math':
                # Mathematical models
                # Generate timestamps for this chunk
                freq = pd.infer_freq(current_timestamps)
                if freq is None:
                    median_diff = pd.Series(current_timestamps[1:] - current_timestamps[:-1]).median()
                    chunk_timestamps = pd.date_range(
                        start=current_date + median_diff,
                        periods=steps_this_iteration,
                        freq=median_diff
                    )
                else:
                    chunk_timestamps = pd.date_range(
                        start=current_date,
                        periods=steps_this_iteration + 1,
                        freq=freq
                    )[1:]
                
                # Calculate time features
                start_time = train_timestamps[0]
                time_since_start = (chunk_timestamps - start_time).total_seconds().values / 86400
                
                # Predict
                if 'poly_transformer' in model_data:
                    X_future = model_data['poly_transformer'].transform(time_since_start.reshape(-1, 1))
                    chunk_forecast = current_model.predict(X_future)
                else:
                    X_future = time_since_start.reshape(-1, 1)
                    chunk_forecast = current_model.predict(X_future)
                    
            elif model_type == 'tier3_ml':
                # ML models - simplified
                last_value = current_data[-1]
                chunk_forecast = np.full(steps_this_iteration, last_value)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
        except Exception as e:
            st.error(f"Error in iteration {iteration}: {str(e)}")
            break
        
        # Generate timestamps for this chunk
        freq = pd.infer_freq(current_timestamps)
        if freq is None:
            median_diff = pd.Series(current_timestamps[1:] - current_timestamps[:-1]).median()
            chunk_timestamps = pd.date_range(
                start=current_date + median_diff,
                periods=steps_this_iteration,
                freq=median_diff
            )
        else:
            chunk_timestamps = pd.date_range(
                start=current_date,
                periods=steps_this_iteration + 1,
                freq=freq
            )[1:]
        
        # === APPLY CONFIDENCE DECAY ===
        # Reduce forecast values slightly to account for increasing uncertainty
        confidence_factor = 1.0 - (iteration - 1) * confidence_decay
        confidence_factor = max(confidence_factor, 0.5)  # Don't go below 50%
        
        chunk_forecast_adjusted = chunk_forecast * confidence_factor
        chunk_confidences = np.full(len(chunk_forecast), confidence_factor)
        
        # Store results
        all_forecast_values.extend(chunk_forecast_adjusted)
        all_forecast_timestamps.extend(chunk_timestamps)
        confidence_scores.extend(chunk_confidences)
        iteration_points.append(len(all_forecast_values))
        
        # === RETRAIN MODEL (if not last iteration) ===
        current_date = chunk_timestamps[-1]
        
        if current_date < target_date and iteration < max_iterations:
            # Append forecasted values to current data
            current_data = np.append(current_data, chunk_forecast_adjusted)
            current_timestamps = pd.DatetimeIndex(list(current_timestamps) + list(chunk_timestamps))
            
            # Retrain the model
            try:
                if model_type == 'tier2_timeseries':
                    # Clone and refit time series model
                    current_model = clone_model_for_retraining(current_model, model_type)
                    current_model.fit(current_data)
                    
                elif model_type == 'tier1_math':
                    # Retrain mathematical model
                    start_time = train_timestamps[0]
                    time_since_start = (current_timestamps - start_time).total_seconds().values / 86400
                    
                    if 'poly_transformer' in model_data:
                        X_train = model_data['poly_transformer'].transform(time_since_start.reshape(-1, 1))
                    else:
                        X_train = time_since_start.reshape(-1, 1)
                    
                    current_model = clone_model_for_retraining(current_model, model_type)
                    current_model.fit(X_train, current_data)
                    
            except Exception as e:
                st.warning(f"Could not retrain model in iteration {iteration}: {str(e)}")
                # Continue with current model
    
    # Clear progress indicators
    progress_placeholder.empty()
    iteration_info.empty()
    
    # Success message
    if iteration > 1:
        avg_confidence = np.mean(confidence_scores)
        st.success(f"✅ Completed iterative forecast in {iteration} iterations (avg confidence: {avg_confidence:.1%})")
    
    return {
        'forecast_values': np.array(all_forecast_values),
        'forecast_timestamps': pd.DatetimeIndex(all_forecast_timestamps),
        'iterations': iteration,
        'confidence_scores': np.array(confidence_scores),
        'iteration_points': iteration_points
    }


def generate_forecast(variable: str, model_name: str, target_date: pd.Timestamp, last_date: pd.Timestamp, use_iterative: bool = False) -> dict:
    """
    Generate forecast for a variable using selected model
    
    Chooses between direct forecasting or iterative retraining based on forecast ratio
    
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
    use_iterative : bool
        Whether to use iterative retraining approach
        
    Returns:
    --------
    forecast : dict
        {
            'forecast_values': np.ndarray,
            'forecast_timestamps': pd.DatetimeIndex,
            'model_type': str,
            'model_name': str,
            'target_date': pd.Timestamp,
            'method': str ('direct' or 'iterative'),
            'iterations': int (if iterative),
            'confidence_scores': np.ndarray (if iterative)
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
    
    # Calculate forecast ratio
    train_values = split_data['train_values']
    test_values = split_data['test_values']
    total_data_length = len(train_values) + len(test_values)
    forecast_days = (target_date - last_date).days
    forecast_ratio = forecast_days / total_data_length
    
    # === DECISION: Direct vs Iterative ===
    
    if use_iterative and forecast_ratio > 0.5:
        # Check if model is safe for iteration
        is_safe = is_model_safe_for_iteration(model_name)
        
        if not is_safe and forecast_ratio > 1.0:
            st.warning(f"⚠️ Model type '{model_name}' is not ideal for iterative retraining. Proceeding with caution.")
        
        # === ITERATIVE FORECAST ===
        result = iterative_forecast_with_retraining(
            variable=variable,
            model_name=model_name,
            model_data=model_data,
            model_type=model_type,
            split_data=split_data,
            target_date=target_date,
            last_date=last_date,
            safe_ratio=0.5,
            confidence_decay=0.05,
            max_iterations=10
        )
        
        result.update({
            'model_type': model_type,
            'model_name': model_name,
            'target_date': target_date,
            'method': 'iterative'
        })
        
        return result
        
    else:
        # === DIRECT FORECAST ===
        
        # Calculate exact number of steps
        forecast_steps = (target_date - last_date).days
        
        # Generate future timestamps
        train_timestamps = split_data['train_timestamps']
        if len(train_timestamps) > 1:
            freq = pd.infer_freq(train_timestamps)
            if freq is None:
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
                )[1:]
        else:
            future_timestamps = pd.date_range(
                start=last_date + timedelta(days=1),
                end=target_date,
                freq='D'
            )
        
        forecast_steps = len(future_timestamps)
        
        # Generate forecast based on model type
        if model_type == 'tier2_timeseries':
            model = model_data['model']
            forecast_values = model.forecast(steps=forecast_steps)
            
        elif model_type == 'tier1_math':
            start_time = split_data['train_timestamps'][0]
            time_since_start = (future_timestamps - start_time).total_seconds().values / 86400
            
            if 'poly_transformer' in model_data:
                X_future = model_data['poly_transformer'].transform(time_since_start.reshape(-1, 1))
                forecast_values = model_data['model'].predict(X_future)
            else:
                X_future = time_since_start.reshape(-1, 1)
                forecast_values = model_data['model'].predict(X_future)
        
        elif model_type == 'tier3_ml':
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
            'target_date': target_date,
            'method': 'direct',
            'iterations': 1,
            'confidence_scores': np.ones(len(forecast_values))
        }




def create_forecast_plot(variable: str, 
                        historical_values: np.ndarray,
                        historical_timestamps: pd.DatetimeIndex,
                        forecast_values: np.ndarray,
                        forecast_timestamps: pd.DatetimeIndex,
                        split_index: int,
                        customize: dict,
                        target_date: pd.Timestamp,
                        iteration_points: list = None,
                        confidence_scores: np.ndarray = None) -> go.Figure:
    """
    Create forecast visualization with iteration markers and confidence bands
    
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
        Customization options
    target_date : pd.Timestamp
        Target forecast date
    iteration_points : list
        Indices where iterations occurred (for iterative forecast)
    confidence_scores : np.ndarray
        Confidence scores for each forecast point
        
    Returns:
    --------
    fig : go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Filter forecast to target_date
    forecast_mask = forecast_timestamps <= target_date
    forecast_values_to_show = forecast_values[forecast_mask]
    forecast_timestamps_to_show = forecast_timestamps[forecast_mask]
    
    if confidence_scores is not None:
        confidence_to_show = confidence_scores[forecast_mask]
    else:
        confidence_to_show = np.ones(len(forecast_values_to_show))
    
    # Historical data
    if customize.get('show_historical', True):
        fig.add_trace(go.Scatter(
            x=historical_timestamps,
            y=historical_values,
            mode='markers',
            name='Historical Data',
            marker=dict(
                size=customize.get('hist_size', 6),
                color=customize.get('hist_color', '#000000')
            ),
            showlegend=True
        ))
    
    # Forecast data - SOLID COLOR (no colorbar)
    if customize.get('show_forecast', True):
        # Always use solid color - no confusing colorbar
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_to_show,
            y=forecast_values_to_show,
            mode='markers',
            name='Forecast',
            marker=dict(
                size=customize.get('forecast_size', 8),
                color=customize.get('forecast_color', '#FF6B6B'),
                symbol='diamond',
                line=dict(width=1, color='white')  # White border for visibility
            ),
            showlegend=True,
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    # Model line
    if customize.get('show_model_line', True):
        all_times = list(historical_timestamps) + list(forecast_timestamps_to_show)
        all_values = list(historical_values) + list(forecast_values_to_show)
        
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
            font=dict(color="red", size=10)
        )
    
    # Iteration markers (for iterative forecasts)
    if iteration_points is not None and len(iteration_points) > 1:
        for i, point_idx in enumerate(iteration_points[:-1], 1):  # Skip last point
            if point_idx < len(forecast_timestamps_to_show):
                iter_time = forecast_timestamps_to_show[point_idx]
                
                fig.add_shape(
                    type="line",
                    x0=iter_time,
                    x1=iter_time,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="orange", width=1, dash="dot")
                )
                
                fig.add_annotation(
                    x=iter_time,
                    y=0.95 - (i % 3) * 0.05,  # Stagger annotations
                    yref="paper",
                    text=f"Iter {i}",
                    showarrow=False,
                    font=dict(color="orange", size=8)
                )
    
    # Calculate X-axis range with margin
    x_start = historical_timestamps[0] if len(historical_timestamps) > 0 else forecast_timestamps_to_show[0]
    total_span = (target_date - x_start).days
    margin_days = int(total_span * 0.1)
    x_end = target_date + timedelta(days=margin_days)
    
    # Layout
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
            range=[x_start, x_end]
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
            st.switch_page("pages/07_Model_Evaluation_and_Selection.py")
        return
    
    # === SELECTED MODELS SUMMARY ===
    with st.expander("📋 Selected Models Summary", expanded=False):
        st.markdown("### Models Selected for Forecasting")
        
        summary_data = []
        for variable, model in st.session_state.selected_models_for_forecast.items():
            var_results = st.session_state.trained_models.get(variable, {})
            
            model_data = None
            for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                if model in var_results.get(tier_name, {}):
                    model_data = var_results[tier_name][model]
                    break
            
            if model_data:
                test_metrics = model_data.get('test_metrics', {})
                is_safe = is_model_safe_for_iteration(model)
                summary_data.append({
                    'Variable': variable,
                    'Selected Model': model,
                    'R²': f"{test_metrics.get('r2', 0):.4f}",
                    'Safe for Iteration': '✅' if is_safe else '⚠️'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Add re-select button
            st.markdown("---")
            if st.button("🔄 Change Model Selection", type="secondary", use_container_width=True, key="reselect_models_top"):
                st.info("Navigating to Model Evaluation & Selection page...")
                st.switch_page("pages/07_Model_Evaluation_and_Selection.py")
    
    # === HOW TO USE ===
    with st.expander("ℹ️ How to Use This Page (NEW: Iterative Forecasting)", expanded=False):
        st.markdown("""
        ### Smart Forecasting with Iterative Retraining
        
        **🆕 What's New:**
        - System automatically detects forecast horizon safety
        - For long forecasts (> 50% training period), uses **iterative retraining**
        - Model retrains on its own forecasts in safe chunks
        - Much more reliable for long-term projections!
        
        **How It Works:**
        
        **1. Safe Forecasts (ratio ≤ 0.5):**
        - ✅ Uses direct forecasting
        - Single prediction step
        - Fast and reliable
        
        **2. Moderate Forecasts (0.5 < ratio ≤ 1.5):**
        - 🔄 Uses iterative retraining
        - Forecasts in 0.5 ratio chunks
        - Retrains after each chunk
        - 2-3 iterations typically
        
        **3. Risky Forecasts (ratio > 1.5):**
        - ⚠️ Uses iteration with strong warnings
        - High uncertainty
        - Results should be interpreted cautiously
        
        **Confidence Scores:**
        - Each forecast point has a confidence score
        - Color-coded: Green (high) → Yellow (medium) → Red (low)
        - Confidence decreases with each iteration
        
        **Iteration Markers:**
        - Orange dashed lines show where model retrained
        - Helps understand forecast quality
        """)
    
    st.markdown("---")
    
    # === STEP 1: FORECAST CONFIGURATION ===
    st.subheader("🎯 Step 1: Configure Forecast")
    
    # Get date range
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
    
    # Display validation with method info
    if validation['warning_level'] == 'error':
        st.error(validation['message'])
    elif validation['warning_level'] == 'safe':
        st.success(validation['message'])
        if not validation['use_iterative']:
            st.info("📊 Method: **Direct Forecasting** (single step)")
    elif validation['warning_level'] == 'caution':
        st.warning(validation['message'])
        if validation['use_iterative']:
            st.info("🔄 Method: **Iterative Retraining** (forecast in safe chunks)")
    else:  # risky
        st.error(validation['message'])
        if validation['use_iterative']:
            st.warning("🔄 Using iterative retraining, but uncertainty is HIGH!")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast Period", f"{validation['days_ahead']} days")
    with col2:
        st.metric("Forecast/Training Ratio", f"{validation['ratio']:.2f}x")
    with col3:
        if validation['use_iterative']:
            est_iterations = max(1, int(np.ceil(validation['ratio'] / 0.5)))
            st.metric("Estimated Iterations", est_iterations)
        else:
            st.metric("Method", "Direct")
    
    st.markdown("---")
    
    # === STEP 2: GENERATE FORECASTS ===
    st.subheader("🚀 Step 2: Generate Forecasts")
    
    if st.button("🔮 Generate Forecasts for All Variables", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        forecast_results = {}
        iteration_info = {}
        
        for idx, (variable, model_name) in enumerate(st.session_state.selected_models_for_forecast.items()):
            status_text.text(f"Generating forecast for: {variable}...")
            
            try:
                forecast = generate_forecast(
                    variable, 
                    model_name, 
                    forecast_date, 
                    last_date,
                    use_iterative=validation['use_iterative']
                )
                forecast_results[variable] = forecast
                iteration_info[variable] = {
                    'iterations': forecast.get('iterations', 1),
                    'method': forecast.get('method', 'direct')
                }
                
            except Exception as e:
                st.error(f"Error forecasting {variable}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                continue
            
            progress = (idx + 1) / len(st.session_state.selected_models_for_forecast)
            progress_bar.progress(progress)
        
        # Store results
        st.session_state.forecast_results = forecast_results
        st.session_state.forecast_target_date = forecast_date
        st.session_state.forecast_iterations = iteration_info
        st.session_state.forecasts_generated = True
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Generated forecasts for {len(forecast_results)} variables!")
        
        # Show iteration summary
        total_iterations = sum([info['iterations'] for info in iteration_info.values()])
        iterative_count = sum([1 for info in iteration_info.values() if info['method'] == 'iterative'])
        
        if iterative_count > 0:
            st.info(f"🔄 Used iterative retraining for {iterative_count} variables (total {total_iterations} iterations)")
        
        st.rerun()
    
    # === STEP 3: DISPLAY FORECASTS ===
    if st.session_state.forecasts_generated and st.session_state.forecast_results:
        
        st.markdown("---")
        st.subheader("📊 Step 3: Review Forecasts")
        
        # Add option to re-do forecast with different models
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Not Satisfied? Change Models & Re-forecast", type="secondary", use_container_width=True, key="reforecast_button"):
                # Clear forecast results
                st.session_state.forecasts_generated = False
                st.session_state.forecast_results = {}
                st.session_state.forecast_iterations = {}
                st.info("Navigating to Model Evaluation & Selection page...")
                st.switch_page("pages/07_Model_Evaluation_and_Selection.py")
        
        st.markdown("---")
        
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
    
    # Show forecast method info
    method = forecast.get('method', 'direct')
    iterations = forecast.get('iterations', 1)
    
    if method == 'iterative':
        st.info(f"🔄 **Iterative Forecast:** Generated in {iterations} iterations with retraining")
    else:
        st.info(f"📊 **Direct Forecast:** Single-step prediction (safe ratio)")
    
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
    
    # Get confidence scores
    confidence_scores = forecast.get('confidence_scores')
    if confidence_scores is not None:
        confidence_filtered = confidence_scores[forecast_mask]
        avg_confidence = np.mean(confidence_filtered)
        min_confidence = np.min(confidence_filtered)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col2:
            st.metric("Min Confidence", f"{min_confidence:.1%}")
        with col3:
            st.metric("Iterations", iterations)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", "~100%")
        with col2:
            st.metric("Method", "Direct Forecast")
    
    st.markdown("---")
    
    # === CUSTOMIZATION ===
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
        st.markdown("**Display:**")
        show_iterations = st.checkbox("Show Iteration Markers", value=True, key=f"show_iter_{variable}")
    
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
    
    # Get iteration points
    iteration_points = forecast.get('iteration_points') if show_iterations else None
    
    fig = create_forecast_plot(
        variable=variable,
        historical_values=historical_values,
        historical_timestamps=historical_timestamps,
        forecast_values=forecast_values_filtered,
        forecast_timestamps=forecast_timestamps_filtered,
        split_index=len(train_values),
        customize=customize,
        target_date=target_date,
        iteration_points=iteration_points,
        confidence_scores=confidence_filtered if confidence_scores is not None else None
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
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast_timestamps_filtered,
                'Forecast': forecast_values_filtered,
                'Variable': variable,
                'Model': forecast['model_name'],
                'Confidence': confidence_filtered if confidence_scores is not None else 1.0
            })
            
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"forecast_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # === STATISTICS ===
    with st.expander("📊 Forecast Statistics"):
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Forecast Steps", len(forecast_values_filtered))
        
        with col2:
            st.metric("Mean Forecast", f"{np.mean(forecast_values_filtered):.3f}")
        
        with col3:
            st.metric("Forecast Range", f"{np.ptp(forecast_values_filtered):.3f}")
        
        with col4:
            if method == 'iterative':
                st.metric("Method", f"Iterative ({iterations}x)")
            else:
                st.metric("Method", "Direct")
        
        # Forecast table
        st.markdown("**Forecast Values:**")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Export table
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
