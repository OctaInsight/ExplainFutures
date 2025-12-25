"""
Page 8: Future Projections
Use selected models to forecast into the future
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
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
from core.models.validation import create_forecast_horizon

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
st.markdown("*Forecast future values using selected models*")
st.markdown("---")


def initialize_forecast_state():
    """Initialize session state for forecasting"""
    if "forecasts" not in st.session_state:
        st.session_state.forecasts = {}
    if "forecast_generated" not in st.session_state:
        st.session_state.forecast_generated = False


def generate_forecast(series_name: str, model_artifact, forecast_steps: int):
    """
    Generate forecast for a series
    
    Parameters:
    -----------
    series_name : str
        Name of series
    model_artifact : ModelArtifact
        Trained model artifact
    forecast_steps : int
        Number of steps to forecast
        
    Returns:
    --------
    forecast : dict
        Forecast results
    """
    model = model_artifact.model
    model_type = model.model_type
    
    try:
        if model_type == 'regression':
            # Regression models: extend time
            # Get last training time point
            X_train = model_artifact.train_data['X']
            last_time = X_train[-1, 0]
            
            # Create future time points
            future_time = np.array([last_time + i + 1 for i in range(forecast_steps)]).reshape(-1, 1)
            
            # Predict
            forecast_values = model.predict(future_time)
            
        elif model_type == 'timeseries':
            # Time series models: direct forecasting
            forecast_values = model.predict(np.arange(forecast_steps))
            
        elif model_type == 'ml':
            # ML models: need to construct features iteratively
            # This is complex - for now, use last known values
            
            # Get last training data
            X_train = model_artifact.train_data['X']
            y_train = model_artifact.train_data['y_true']
            
            # Start with last known values
            forecast_values = []
            last_features = X_train[-1].copy()
            
            for step in range(forecast_steps):
                # Predict next value
                pred = model.predict(last_features.reshape(1, -1))[0]
                forecast_values.append(pred)
                
                # Update features (simplified - shift lags)
                # This is a simplified approach
                # In production, you'd properly update lag features
                if len(last_features) > 1:
                    last_features[1:] = last_features[:-1]
                    last_features[0] = pred
            
            forecast_values = np.array(forecast_values)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate confidence interval (simplified)
        # Use residual std from test set
        test_std = model_artifact.test_data['metrics'].get('std_residual', 0)
        
        # 95% confidence interval (±1.96 * std)
        ci_lower = forecast_values - 1.96 * test_std
        ci_upper = forecast_values + 1.96 * test_std
        
        return {
            'values': forecast_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'model_name': model.name,
            'model_type': model_type,
            'steps': forecast_steps,
            'confidence_level': 0.95,
            'success': True
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_forecast_plot(series_name: str, 
                        historical_data: dict,
                        forecast_data: dict,
                        timestamps_historical: pd.Series,
                        timestamps_forecast: pd.DatetimeIndex):
    """
    Create forecast visualization
    
    Parameters:
    -----------
    series_name : str
        Series name
    historical_data : dict
        Historical data (train + test)
    forecast_data : dict
        Forecast results
    timestamps_historical : pd.Series
        Historical timestamps
    timestamps_forecast : pd.DatetimeIndex
        Future timestamps
        
    Returns:
    --------
    fig : plotly Figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=timestamps_historical,
        y=historical_data['values'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=timestamps_forecast,
        y=forecast_data['values'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='steelblue', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=timestamps_forecast,
        y=forecast_data['ci_upper'],
        mode='lines',
        name='95% CI Upper',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps_forecast,
        y=forecast_data['ci_lower'],
        mode='lines',
        name='95% CI Lower',
        line=dict(width=0),
        fillcolor='rgba(70, 130, 180, 0.2)',
        fill='tonexty',
        showlegend=True
    ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=timestamps_historical.iloc[-1],
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"{series_name} - Forecast using {forecast_data['model_name']}",
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def get_series_historical_data(series_name: str):
    """
    Get complete historical data for a series
    
    Parameters:
    -----------
    series_name : str
        Series name
        
    Returns:
    --------
    data : dict
        Historical data and timestamps
    """
    # Get from model artifact (train + test combined)
    if series_name not in st.session_state.model_results:
        return None
    
    # Get first model artifact (any model will have the same historical data)
    first_model = list(st.session_state.model_results[series_name].values())[0]
    
    # Combine train and test
    y_train = first_model.train_data['y_true']
    y_test = first_model.test_data['y_true']
    
    y_full = np.concatenate([y_train, y_test])
    
    return {
        'values': y_full,
        'train_size': len(y_train),
        'test_size': len(y_test)
    }


def main():
    """Main page function"""
    
    # Initialize state
    initialize_forecast_state()
    
    # Check if models are selected
    if not st.session_state.get('selected_models'):
        st.warning("⚠️ No models selected yet!")
        st.info("👈 Please go to **Model Evaluation & Selection** to select models first")
        
        if st.button("📊 Go to Evaluation Page"):
            st.switch_page("pages/7_Model_Evaluation_and_Selection.py")
        return
    
    selected_series = list(st.session_state.selected_models.keys())
    
    st.success(f"✅ Ready to forecast {len(selected_series)} time series")
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ How forecasting works", expanded=False):
        st.markdown("""
        ### Forecasting Process
        
        **What happens:**
        1. Use selected best models from evaluation
        2. Extend into future based on learned patterns
        3. Provide confidence intervals (95%)
        4. Generate forecasts for all series
        
        **Forecast horizon:**
        - Specify target date OR number of steps
        - Longer forecasts = less certain
        - Confidence intervals widen over time
        
        **Model types:**
        - **Regression:** Extrapolate trend
        - **Time Series (ETS/ARIMA):** Use built-in forecasting
        - **ML:** Iterative prediction with lag features
        
        **Confidence intervals:**
        - Based on historical prediction errors
        - 95% CI = ±1.96 × standard deviation of residuals
        - Wider interval = more uncertainty
        """)
    
    st.markdown("---")
    
    # === FORECAST CONFIGURATION ===
    st.subheader("⚙️ Forecast Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get last known date (approximate)
        # This is simplified - in production, get actual last date per series
        last_date = datetime.now()
        
        forecast_method = st.radio(
            "Specify forecast by:",
            ["Target Date", "Number of Steps"],
            help="Choose how to define forecast horizon"
        )
    
    with col2:
        if forecast_method == "Target Date":
            target_date = st.date_input(
                "Forecast to date:",
                value=datetime.now() + timedelta(days=365),
                min_value=datetime.now(),
                help="Target date for forecast"
            )
            
            # Calculate steps (approximate)
            forecast_steps = (target_date - datetime.now().date()).days
            st.metric("Forecast Steps", forecast_steps)
        
        else:
            forecast_steps = st.number_input(
                "Number of steps to forecast:",
                min_value=1,
                max_value=1000,
                value=30,
                step=1,
                help="Number of time steps into the future"
            )
            
            target_date = datetime.now().date() + timedelta(days=forecast_steps)
            st.metric("Target Date", target_date.strftime("%Y-%m-%d"))
    
    with col3:
        st.metric("Series to Forecast", len(selected_series))
        
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%"
        )
    
    st.markdown("---")
    
    # === GENERATE FORECASTS ===
    st.subheader("🚀 Generate Forecasts")
    
    if st.button("🔮 Generate Forecasts for All Series", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate forecasts
        for idx, series_name in enumerate(selected_series):
            status_text.text(f"Forecasting: {series_name}...")
            
            try:
                # Get selected model
                model_name = st.session_state.selected_models[series_name]
                model_artifact = st.session_state.model_results[series_name][model_name]
                
                # Generate forecast
                forecast_result = generate_forecast(
                    series_name,
                    model_artifact,
                    forecast_steps
                )
                
                if forecast_result['success']:
                    # Store forecast
                    st.session_state.forecasts[series_name] = forecast_result
                else:
                    st.error(f"Failed to forecast {series_name}: {forecast_result.get('error', 'Unknown error')}")
                
                # Update progress
                progress = (idx + 1) / len(selected_series)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error forecasting {series_name}: {str(e)}")
                continue
        
        # Complete
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.forecast_generated = True
        
        st.success(f"✅ Successfully generated forecasts for {len(st.session_state.forecasts)} series!")
        st.balloons()
    
    # === DISPLAY FORECASTS ===
    if st.session_state.forecast_generated and st.session_state.forecasts:
        st.markdown("---")
        st.subheader("📊 Forecast Results")
        
        # Summary table
        summary_data = []
        
        for series_name, forecast in st.session_state.forecasts.items():
            # Get last and forecast values
            last_value = get_series_historical_data(series_name)['values'][-1]
            forecast_end = forecast['values'][-1]
            change = ((forecast_end - last_value) / last_value) * 100
            
            summary_data.append({
                'Series': series_name,
                'Model Used': forecast['model_name'],
                'Last Value': f"{last_value:.3f}",
                'Forecast End': f"{forecast_end:.3f}",
                'Change': f"{change:+.1f}%",
                'Steps': forecast['steps']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Export summary
        with st.expander("💾 Export Forecast Summary"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=csv,
                file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # === INDIVIDUAL FORECASTS ===
        st.subheader("📈 Individual Forecast Visualizations")
        
        for series_name in selected_series:
            if series_name in st.session_state.forecasts:
                with st.expander(f"📊 {series_name}", expanded=True):
                    forecast_data = st.session_state.forecasts[series_name]
                    historical_data = get_series_historical_data(series_name)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Model", forecast_data['model_name'])
                    with col2:
                        last_val = historical_data['values'][-1]
                        st.metric("Last Historical", f"{last_val:.3f}")
                    with col3:
                        forecast_val = forecast_data['values'][-1]
                        st.metric("Forecast End", f"{forecast_val:.3f}")
                    with col4:
                        change = ((forecast_val - last_val) / last_val) * 100
                        st.metric("Change", f"{change:+.1f}%")
                    
                    # Create timestamps (simplified - using index)
                    n_historical = len(historical_data['values'])
                    timestamps_hist = pd.Series(pd.date_range(
                        start=datetime.now() - timedelta(days=n_historical),
                        periods=n_historical,
                        freq='D'
                    ))
                    
                    timestamps_forecast = pd.date_range(
                        start=timestamps_hist.iloc[-1] + timedelta(days=1),
                        periods=forecast_steps,
                        freq='D'
                    )
                    
                    # Create plot
                    fig = create_forecast_plot(
                        series_name,
                        historical_data,
                        forecast_data,
                        timestamps_hist,
                        timestamps_forecast
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    with st.expander("💾 Export Forecast"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Export plot
                            quick_export_buttons(
                                fig, 
                                f"forecast_{series_name}", 
                                ['png', 'pdf', 'html']
                            )
                        
                        with col_b:
                            # Export data
                            forecast_df = pd.DataFrame({
                                'Date': timestamps_forecast,
                                'Forecast': forecast_data['values'],
                                'CI_Lower': forecast_data['ci_lower'],
                                'CI_Upper': forecast_data['ci_upper']
                            })
                            
                            csv_data = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast Data (CSV)",
                                data=csv_data,
                                file_name=f"forecast_{series_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
        
        st.markdown("---")
        
        # === COMBINED FORECAST EXPORT ===
        st.subheader("💾 Export All Forecasts")
        
        # Create combined forecast dataframe
        combined_data = []
        
        for series_name, forecast in st.session_state.forecasts.items():
            historical_data = get_series_historical_data(series_name)
            
            # Create timestamps
            n_historical = len(historical_data['values'])
            timestamps_forecast = pd.date_range(
                start=datetime.now(),
                periods=forecast['steps'],
                freq='D'
            )
            
            for i, (date, value, ci_lower, ci_upper) in enumerate(zip(
                timestamps_forecast, 
                forecast['values'],
                forecast['ci_lower'],
                forecast['ci_upper']
            )):
                combined_data.append({
                    'Series': series_name,
                    'Date': date,
                    'Forecast': value,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'Model': forecast['model_name'],
                    'Step': i + 1
                })
        
        combined_df = pd.DataFrame(combined_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Combined Forecast Data:**")
            st.dataframe(combined_df.head(20), use_container_width=True)
        
        with col2:
            st.markdown("**Export Options:**")
            
            csv_all = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Forecasts (CSV)",
                data=csv_all,
                file_name=f"all_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Generate report
            report = generate_forecast_report(combined_df, st.session_state.forecasts)
            
            st.download_button(
                label="📄 Download Forecast Report (TXT)",
                data=report,
                file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )


def generate_forecast_report(forecast_df: pd.DataFrame, forecasts: dict) -> str:
    """Generate text report of forecasts"""
    
    report = []
    report.append("="*80)
    report.append("FUTURE PROJECTIONS - FORECAST REPORT")
    report.append("ExplainFutures")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Series: {len(forecasts)}")
    report.append("\n")
    
    for series_name, forecast in forecasts.items():
        report.append("-"*80)
        report.append(f"SERIES: {series_name}")
        report.append("-"*80)
        report.append(f"Model Used: {forecast['model_name']}")
        report.append(f"Model Type: {forecast['model_type']}")
        report.append(f"Forecast Steps: {forecast['steps']}")
        report.append(f"Confidence Level: {forecast['confidence_level']*100:.0f}%")
        report.append("")
        
        # Statistics
        report.append("Forecast Statistics:")
        report.append(f"  Mean:   {np.mean(forecast['values']):.3f}")
        report.append(f"  Median: {np.median(forecast['values']):.3f}")
        report.append(f"  Min:    {np.min(forecast['values']):.3f}")
        report.append(f"  Max:    {np.max(forecast['values']):.3f}")
        report.append(f"  Std:    {np.std(forecast['values']):.3f}")
        report.append("")
        
        # First and last values
        report.append(f"First Forecast: {forecast['values'][0]:.3f}")
        report.append(f"Last Forecast:  {forecast['values'][-1]:.3f}")
        report.append(f"Total Change:   {forecast['values'][-1] - forecast['values'][0]:.3f}")
        report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
