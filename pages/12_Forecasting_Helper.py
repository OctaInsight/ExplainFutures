"""
Page 12: Forecasting Helper (HIDDEN - Not in sidebar)
Train, evaluate, and forecast parameters needed for trajectory analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Forecast Missing Parameters",
    page_icon="🔮",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import initialize_session_state
from core.shared_sidebar import render_app_sidebar
from core.models.data_preparation import get_series_data
from core.models.model_trainer import train_all_models_for_variable

# Import visualization functions
try:
    from core.models.model_visualization import create_model_comparison_plot
    from core.viz.export import quick_export_buttons
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Initialize
initialize_session_state()

# Render shared sidebar
render_app_sidebar()

# === PAGE TITLE ===
st.title("🔮 Train & Forecast Missing Parameters")
st.markdown("*Complete workflow: Training → Evaluation → Forecasting*")
st.markdown("---")


def calculate_health_index(metrics: dict) -> float:
    """Calculate health index (0-100) from metrics"""
    r2 = metrics.get('r2', 0)
    mae = metrics.get('mae', np.inf)
    rmse = metrics.get('rmse', np.inf)
    mape = metrics.get('mape', np.inf)
    
    if np.isnan(r2) or np.isinf(r2):
        r2 = 0
    if np.isnan(mae) or np.isinf(mae):
        mae = 100
    if np.isnan(rmse) or np.isinf(rmse):
        rmse = 100
    if np.isnan(mape) or np.isinf(mape):
        mape = 100
    
    r2_score = max(0, min(r2, 1)) * 40
    mae_score = max(0, 20 * (1 - min(mae / 50, 1)))
    rmse_score = max(0, 20 * (1 - min(rmse / 50, 1)))
    mape_score = max(0, 20 * (1 - min(mape / 50, 1)))
    
    health_index = r2_score + mae_score + rmse_score + mape_score
    
    return round(health_index, 2)


def get_metric_color(value: float, metric: str, all_values: list) -> str:
    """Get color for metric value (gradient from red to green)"""
    if len(all_values) == 0 or np.isnan(value):
        return 'rgb(200, 200, 200)'
    
    clean_values = [v for v in all_values if not np.isnan(v)]
    if len(clean_values) == 0:
        return 'rgb(200, 200, 200)'
    
    min_val = min(clean_values)
    max_val = max(clean_values)
    
    if max_val == min_val:
        normalized = 0.5
    else:
        if metric in ['r2', 'health_index']:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 1 - (value - min_val) / (max_val - min_val)
    
    if normalized < 0.5:
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0
    
    return f'rgb({r}, {g}, {b})'


def main():
    """Main function"""
    
    # Check if we were sent here from Page 11
    if 'trajectory_forecast_params' not in st.session_state:
        st.error("❌ This page should be accessed from Trajectory-Scenario Space")
        
        if st.button("🎯 Go to Trajectory Analysis"):
            st.switch_page("pages/11_Trajectory-Scenario_Space.py")
        return
    
    params_to_forecast = st.session_state.trajectory_forecast_params
    target_date = st.session_state.trajectory_forecast_target
    
    st.info(f"""
    🎯 **Goal:** Forecast {len(params_to_forecast)} parameter(s) to **{target_date.strftime('%Y-%m-%d')}**
    
    **Workflow:**
    1. Train models (if needed)
    2. Evaluate with quality matrix
    3. Select best models
    4. Generate forecasts
    """)
    
    # Show parameters
    with st.expander("📋 Parameters to Forecast", expanded=True):
        for param in params_to_forecast:
            st.caption(f"• {param}")
    
    st.markdown("---")
    
    # === STEP 1: TRAIN MODELS ===
    st.subheader("🤖 Step 1: Train Models")
    
    # Check which parameters need training
    params_need_training = []
    params_already_trained = []
    
    for param in params_to_forecast:
        if 'trained_models' in st.session_state and param in st.session_state.trained_models:
            params_already_trained.append(param)
        else:
            params_need_training.append(param)
    
    if params_already_trained:
        st.success(f"✅ {len(params_already_trained)} parameter(s) already have trained models")
        with st.expander("Show trained parameters"):
            for param in params_already_trained:
                st.caption(f"• {param}")
    
    if params_need_training:
        st.warning(f"⚠️ {len(params_need_training)} parameter(s) need model training")
        
        with st.expander("Show parameters needing training", expanded=True):
            for param in params_need_training:
                st.caption(f"• {param}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            train_split = st.slider(
                "Training set size",
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Fraction of data to use for training"
            )
        
        with col2:
            st.metric("Train %", f"{train_split*100:.0f}%")
            st.metric("Test %", f"{(1-train_split)*100:.0f}%")
        
        if st.button("🤖 Train Models for All Parameters", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, param in enumerate(params_need_training):
                status_text.text(f"Training models for: {param}...")
                
                try:
                    series_data = get_series_data(param)
                    
                    results = train_all_models_for_variable(
                        variable_name=param,
                        series_data=series_data,
                        train_split=train_split,
                        detect_seasonality=True
                    )
                    
                    if results.get('success', False):
                        st.session_state.trained_models[param] = results
                        st.success(f"✓ Trained {results.get('total_models', 0)} models for {param}")
                    else:
                        st.error(f"✗ Training failed for {param}")
                
                except Exception as e:
                    st.error(f"✗ Error training {param}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(params_need_training))
            
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ Training complete!")
            
            # Mark that models are trained
            st.session_state.models_evaluated = True
            
            st.rerun()
        
        # Don't continue until training is done
        return
    
    # === STEP 2: MODEL EVALUATION MATRIX ===
    st.markdown("---")
    st.subheader("📊 Step 2: Evaluate Models & Select Best")
    st.caption("Review model quality and select the best model for each parameter")
    
    # Initialize selection state
    if 'trajectory_selected_models' not in st.session_state:
        st.session_state.trajectory_selected_models = {}
    
    # Get all models for all parameters
    all_model_names = set()
    
    for param in params_to_forecast:
        if param not in st.session_state.trained_models:
            continue
        
        var_results = st.session_state.trained_models[param]
        all_model_names.update(var_results.get('tier1_math', {}).keys())
        all_model_names.update(var_results.get('tier2_timeseries', {}).keys())
        all_model_names.update(var_results.get('tier3_ml', {}).keys())
    
    all_model_names = sorted(list(all_model_names))
    
    if not all_model_names:
        st.error("❌ No models found. Please train models first.")
        return
    
    # Metric selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metric_display = st.selectbox(
            "Display metric in cells:",
            ['health_index', 'r2', 'mae', 'rmse', 'mape'],
            format_func=lambda x: {
                'health_index': '🏥 Health Index (0-100)',
                'r2': 'R² (Variance Explained)',
                'mae': 'MAE (Mean Absolute Error)',
                'rmse': 'RMSE (Root Mean Squared Error)',
                'mape': 'MAPE (Mean Absolute % Error)'
            }[x],
            index=0
        )
    
    with col2:
        st.metric("Parameters", len(params_to_forecast))
        st.metric("Model Types", len(all_model_names))
    
    st.markdown("---")
    
    # === INTERACTIVE MATRIX ===
    st.markdown("#### 🎯 Model Evaluation Matrix")
    st.caption("Click 'Select' button to choose model and view details. Selected models highlighted in blue.")
    
    # Build matrix data
    matrix_data = {}
    for param in params_to_forecast:
        if param not in st.session_state.trained_models:
            continue
        
        var_results = st.session_state.trained_models[param]
        matrix_data[param] = {}
        
        for model_name in all_model_names:
            model_data = None
            for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                if model_name in var_results.get(tier_name, {}):
                    model_data = var_results[tier_name][model_name]
                    break
            
            if model_data:
                test_metrics = model_data.get('test_metrics', {})
                health_index = calculate_health_index(test_metrics)
                
                matrix_data[param][model_name] = {
                    'health_index': health_index,
                    'r2': test_metrics.get('r2', np.nan),
                    'mae': test_metrics.get('mae', np.nan),
                    'rmse': test_metrics.get('rmse', np.nan),
                    'mape': test_metrics.get('mape', np.nan),
                    'model_data': model_data
                }
            else:
                matrix_data[param][model_name] = None
    
    # === AUTO-SELECT BEST MODELS (before displaying matrix) ===
    for param in params_to_forecast:
        if param not in st.session_state.trajectory_selected_models:
            if param in matrix_data:
                # Find best model by health_index
                best_model = None
                best_health = -np.inf
                
                for model, metrics in matrix_data[param].items():
                    if metrics:
                        health = metrics.get('health_index', -np.inf)
                        if health > best_health:
                            best_health = health
                            best_model = model
                
                if best_model:
                    st.session_state.trajectory_selected_models[param] = best_model
    
    # Collect all metric values for color scaling
    all_metric_values = []
    for param_metrics in matrix_data.values():
        for model_metrics in param_metrics.values():
            if model_metrics:
                val = model_metrics.get(metric_display, np.nan)
                if not np.isnan(val):
                    all_metric_values.append(val)
    
    # Header row
    header_cols = st.columns([2] + [1] * len(all_model_names))
    header_cols[0].markdown("**Parameter**")
    for idx, model in enumerate(all_model_names):
        header_cols[idx + 1].markdown(f"**{model}**")
    
    st.markdown("---")
    
    # Data rows
    for param in params_to_forecast:
        if param not in matrix_data:
            continue
        
        cols = st.columns([2] + [1] * len(all_model_names))
        
        # Parameter name
        cols[0].markdown(f"**{param}**")
        
        # Model cells
        for model_idx, model in enumerate(all_model_names):
            with cols[model_idx + 1]:
                metrics = matrix_data[param].get(model)
                
                if metrics is None:
                    st.markdown("⚪")
                else:
                    metric_value = metrics.get(metric_display, np.nan)
                    
                    # Check if selected (NOW THIS WORKS IMMEDIATELY)
                    is_selected = (param in st.session_state.trajectory_selected_models and 
                                 st.session_state.trajectory_selected_models[param] == model)
                    
                    # Get color
                    if is_selected:
                        bg_color = '#4682B4'  # Blue for selected
                        text_color = 'white'
                    elif np.isnan(metric_value):
                        bg_color = '#C8C8C8'
                        text_color = 'black'
                    else:
                        color_rgb = get_metric_color(metric_value, metric_display, all_metric_values)
                        bg_color = color_rgb.replace('rgb', '').replace('(', '').replace(')', '')
                        rgb_parts = [int(x.strip()) for x in bg_color.split(',')]
                        bg_color = f'#{rgb_parts[0]:02x}{rgb_parts[1]:02x}{rgb_parts[2]:02x}'
                        brightness = (rgb_parts[0] * 299 + rgb_parts[1] * 587 + rgb_parts[2] * 114) / 1000
                        text_color = 'white' if brightness < 128 else 'black'
                    
                    # Format label
                    if metric_display == 'health_index':
                        button_label = f"{int(metric_value)}" if not np.isnan(metric_value) else "N/A"
                    else:
                        button_label = f"{metric_value:.3f}" if not np.isnan(metric_value) else "N/A"
                    
                    # Colored display
                    button_html = f"""
                    <div style="
                        background-color: {bg_color};
                        color: {text_color};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        font-weight: bold;
                        border: 2px solid {'#000' if is_selected else '#ccc'};
                        margin: 2px;
                    ">
                        {button_label}
                    </div>
                    """
                    
                    st.markdown(button_html, unsafe_allow_html=True)
                    
                    # Selection button - NOW SHOWS VISUALIZATION
                    if st.button(
                        "Select" if not is_selected else "✓ Selected",
                        key=f"select_{param}_{model}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True
                    ):
                        # SELECT THIS MODEL
                        st.session_state.trajectory_selected_models[param] = model
                        
                        # SHOW DETAILS BELOW
                        st.session_state.show_model_details = {
                            'variable': param,
                            'model': model,
                            'metrics': metrics
                        }
                        st.rerun()
    
    st.markdown("---")
    
    # === SHOW MODEL VISUALIZATION (if clicked) ===
    if 'show_model_details' in st.session_state and st.session_state.show_model_details:
        details = st.session_state.show_model_details
        variable = details['variable']
        model = details['model']
        metrics = details['metrics']
        
        st.markdown("---")
        st.markdown(f"### 📊 {variable} - {model} Performance")
        
        col_close1, col_close2 = st.columns([5, 1])
        with col_close2:
            if st.button("✖ Close", key="close_viz"):
                st.session_state.show_model_details = None
                st.rerun()
        
        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Health Index", f"{int(metrics['health_index'])}/100")
        with col2:
            st.metric("R²", f"{metrics['r2']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col4:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        
        # Create visualization (same as Page 7)
        try:
            var_results = st.session_state.trained_models[variable]
            split_data = var_results.get('train_test_split', {})
            
            train_values = split_data.get('train_values', np.array([]))
            test_values = split_data.get('test_values', np.array([]))
            train_timestamps = split_data.get('train_timestamps', pd.DatetimeIndex([]))
            test_timestamps = split_data.get('test_timestamps', pd.DatetimeIndex([]))
            split_index = split_data.get('split_index', 0)
            
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export buttons
            if VISUALIZATION_AVAILABLE:
                with st.expander("💾 Export Figure"):
                    quick_export_buttons(fig, f"{variable}_{model}_trajectory_eval", ['png', 'pdf', 'html'])
        
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    # === SELECTION SUMMARY ===
    if st.session_state.trajectory_selected_models:
        st.markdown("---")
        st.markdown("### ✅ Selected Models Summary")
        
        summary_data = []
        for param, model in st.session_state.trajectory_selected_models.items():
            if param in matrix_data and model in matrix_data[param]:
                metrics = matrix_data[param][model]
                if metrics:
                    summary_data.append({
                        'Parameter': param,
                        'Selected Model': model,
                        'Health Index': f"{int(metrics['health_index'])}/100",
                        'R²': f"{metrics['r2']:.4f}",
                        'MAE': f"{metrics['mae']:.4f}"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # === STEP 3: GENERATE FORECASTS ===
    st.markdown("---")
    st.subheader("🔮 Step 3: Generate Forecasts")
    
    # Check if all parameters have selected models
    missing_selections = [p for p in params_to_forecast if p not in st.session_state.trajectory_selected_models]
    
    # Check if forecasts already completed
    forecasts_completed = st.session_state.get('page12_forecasts_completed', False)
    
    if missing_selections:
        st.warning(f"⚠️ {len(missing_selections)} parameter(s) need model selection")
        with st.expander("Show missing"):
            for param in missing_selections:
                st.caption(f"• {param}")
    elif forecasts_completed:
        # FORECASTS ALREADY DONE
        st.success("✅ **All forecasts have been generated successfully!**")
        st.info("All forecast data has been saved and is available in Page 11")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Return to Trajectory Analysis", type="primary", use_container_width=True):
                # Clear the helper state
                st.session_state.pop('page12_forecasts_completed', None)
                st.session_state.pop('show_model_details', None)
                
                # Navigate to Page 11
                st.switch_page("pages/11_Trajectory-Scenario_Space.py")
    else:
        # NOT YET FORECASTED - SHOW BUTTON
        st.info(f"Forecasting to: **{target_date.strftime('%Y-%m-%d')}**")
        
        if st.button("🔮 Generate Forecasts for All Parameters", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, param in enumerate(params_to_forecast):
                status_text.text(f"Forecasting: {param}...")
                
                try:
                    model_name = st.session_state.trajectory_selected_models.get(param)
                    
                    if not model_name:
                        st.error(f"✗ No model selected for {param}")
                        continue
                    
                    # Get model data
                    var_results = st.session_state.trained_models[param]
                    
                    model_data = None
                    model_type = None
                    
                    for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                        if model_name in var_results.get(tier_name, {}):
                            model_data = var_results[tier_name][model_name]
                            model_type = tier_name
                            break
                    
                    if not model_data:
                        st.error(f"✗ Model not found for {param}")
                        continue
                    
                    # Get split data
                    split_data = var_results.get('train_test_split', {})
                    
                    train_timestamps = split_data.get('train_timestamps', pd.DatetimeIndex([]))
                    test_timestamps = split_data.get('test_timestamps', pd.DatetimeIndex([]))
                    
                    if len(train_timestamps) == 0:
                        st.error(f"✗ No timestamp data for {param}")
                        continue
                    
                    last_date = test_timestamps[-1] if len(test_timestamps) > 0 else train_timestamps[-1]
                    
                    # Generate timestamps
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
                    
                    forecast_steps = len(future_timestamps)
                    
                    # Generate forecast
                    if model_type == 'tier2_timeseries':
                        model = model_data['model']
                        forecast_values = model.forecast(steps=forecast_steps)
                        
                    elif model_type == 'tier1_math':
                        start_time = train_timestamps[0]
                        time_since_start = (future_timestamps - start_time).total_seconds().values / 86400
                        
                        if 'poly_transformer' in model_data:
                            X_future = model_data['poly_transformer'].transform(time_since_start.reshape(-1, 1))
                            forecast_values = model_data['model'].predict(X_future)
                        else:
                            X_future = time_since_start.reshape(-1, 1)
                            forecast_values = model_data['model'].predict(X_future)
                    
                    elif model_type == 'tier3_ml':
                        last_value = split_data.get('test_values', [split_data.get('train_values', [0])[-1]])[-1]
                        forecast_values = np.full(forecast_steps, last_value)
                    
                    else:
                        st.error(f"✗ Unknown model type for {param}")
                        continue
                    
                    # Store forecast
                    if 'forecast_results' not in st.session_state:
                        st.session_state.forecast_results = {}
                    
                    st.session_state.forecast_results[param] = {
                        'forecast_values': np.array(forecast_values),
                        'forecast_timestamps': future_timestamps,
                        'model_type': model_type,
                        'model_name': model_name,
                        'target_date': target_date,
                        'method': 'direct',
                        'iterations': 1,
                        'confidence_scores': np.ones(len(forecast_values))
                    }
                    
                    st.success(f"✓ Forecasted {param} ({len(forecast_values)} steps)")
                
                except Exception as e:
                    st.error(f"✗ Error forecasting {param}: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
                
                progress_bar.progress((idx + 1) / len(params_to_forecast))
            
            status_text.empty()
            progress_bar.empty()
            
            st.success("✅ All forecasts generated!")
            
            # CRITICAL: Mark forecasts as completed
            st.session_state.page12_forecasts_completed = True
            
            # CRITICAL: Update all necessary session states
            st.session_state.trajectory_forecasts_updated = True
            st.session_state.master_parameter_df = None  # Force rebuild
            st.session_state.projections_done = True  # Update workflow progress
            
            # Rerun to show return button
            st.rerun()


if __name__ == "__main__":
    main()
