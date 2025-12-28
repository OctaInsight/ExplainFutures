"""
Page 13: Trajectory-Scenario Space
Visualize trajectory-scenario analysis with baseline values and interactive plots
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Trajectory-Scenario Space",
    page_icon="🎯",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import initialize_session_state
from core.shared_sidebar import render_app_sidebar

# Try to import export functions
try:
    from core.viz.export import quick_export_buttons
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Initialize
initialize_session_state()
render_app_sidebar()

# === PAGE TITLE ===
st.title("🎯 Trajectory-Scenario Space Visualization")
st.markdown("*Analyze parameter trajectories from baseline to scenario targets*")
st.markdown("---")

# Copy these 6 lines to the TOP of each page (02-13)
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Copy these 6 lines to the TOP of each page (02-13)
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Then your existing code continues...


def get_data_time_range(master_df: pd.DataFrame) -> dict:
    """Get time range from data"""
    
    min_date = None
    max_date = None
    forecast_start = None
    
    # Get from raw data only
    if master_df is not None and len(master_df) > 0:
        raw_entries = master_df[master_df['source'] == 'raw']
        
        if len(raw_entries) > 0:
            all_timestamps = []
            for _, entry in raw_entries.iterrows():
                timestamps_list = entry['timestamps']
                if timestamps_list is not None and len(timestamps_list) > 0:
                    all_timestamps.extend([pd.to_datetime(t) for t in timestamps_list])
            
            if all_timestamps:
                min_date = min(all_timestamps)
                max_date = max(all_timestamps)
    
    # Get forecast start
    if master_df is not None and len(master_df) > 0:
        forecast_entries = master_df[master_df['source'] == 'forecasted']
        
        if len(forecast_entries) > 0:
            all_forecast_starts = []
            for _, entry in forecast_entries.iterrows():
                timestamps_list = entry['timestamps']
                if timestamps_list is not None and len(timestamps_list) > 0:
                    all_forecast_starts.append(pd.to_datetime(timestamps_list[0]))
            
            if all_forecast_starts:
                forecast_start = min(all_forecast_starts)
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'forecast_start': forecast_start
    }


def extract_baseline_value(param_name: str, baseline_date: pd.Timestamp, master_df: pd.DataFrame) -> dict:
    """Extract baseline value at specific date"""
    
    param_entry = master_df[master_df['name'] == param_name]
    
    if len(param_entry) == 0:
        return {'value': None, 'year': None, 'source': 'not_found'}
    
    param_entry = param_entry.iloc[0]
    
    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
    values = param_entry['values']
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    df = df.sort_values('timestamp')
    df['year'] = df['timestamp'].dt.year
    
    baseline_year = baseline_date.year
    
    year_data = df[df['year'] == baseline_year]
    
    if len(year_data) > 0:
        baseline_value = year_data['value'].mean()
        return {
            'value': float(baseline_value),
            'year': baseline_year,
            'source': param_entry['source_full']
        }
    
    closest_year = df['year'].iloc[(df['year'] - baseline_year).abs().argsort()[0]]
    closest_data = df[df['year'] == closest_year]
    baseline_value = closest_data['value'].mean()
    
    return {
        'value': float(baseline_value),
        'year': closest_year,
        'source': f"{param_entry['source_full']} (closest: {closest_year})"
    }


def extract_target_value(param_name: str, target_date: pd.Timestamp, master_df: pd.DataFrame) -> dict:
    """Extract forecasted value at target date"""
    
    param_entry = master_df[master_df['name'] == param_name]
    
    if len(param_entry) == 0:
        return {'value': None, 'year': None, 'source': 'not_found'}
    
    param_entry = param_entry.iloc[0]
    
    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
    values = param_entry['values']
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    df = df.sort_values('timestamp')
    df['year'] = df['timestamp'].dt.year
    
    target_year = target_date.year
    
    year_data = df[df['year'] == target_year]
    
    if len(year_data) > 0:
        target_value = year_data['value'].mean()
        return {
            'value': float(target_value),
            'year': target_year,
            'source': param_entry['source_full']
        }
    
    # If exact year not found, try closest
    closest_year = df['year'].iloc[(df['year'] - target_year).abs().argsort()[0]]
    closest_data = df[df['year'] == closest_year]
    target_value = closest_data['value'].mean()
    
    return {
        'value': float(target_value),
        'year': closest_year,
        'source': f"{param_entry['source_full']} (closest: {closest_year})"
    }


def get_forecast_error(param_name: str, target_date: pd.Timestamp, master_df: pd.DataFrame) -> dict:
    """
    Get forecast error/uncertainty for parameter at target date
    
    Returns error as percentage of value (for plotting ellipses)
    """
    
    # Check if parameter has forecast data
    if 'forecast_results' not in st.session_state or param_name not in st.session_state.forecast_results:
        # No forecast - estimate from model metrics
        if 'trained_models' in st.session_state and param_name in st.session_state.trained_models:
            model_results = st.session_state.trained_models[param_name]
            
            # Get test metrics from best model (if available)
            best_rmse = None
            best_mae = None
            
            for tier in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
                if tier in model_results:
                    for model_name, model_data in model_results[tier].items():
                        test_metrics = model_data.get('test_metrics', {})
                        rmse = test_metrics.get('rmse')
                        mae = test_metrics.get('mae')
                        
                        if rmse is not None and (best_rmse is None or rmse < best_rmse):
                            best_rmse = rmse
                        if mae is not None and (best_mae is None or mae < best_mae):
                            best_mae = mae
            
            if best_rmse is not None:
                return {'error': best_rmse, 'type': 'RMSE', 'source': 'model_test'}
            elif best_mae is not None:
                return {'error': best_mae, 'type': 'MAE', 'source': 'model_test'}
        
        # Default: 5% error if no info available
        param_entry = master_df[master_df['name'] == param_name]
        if len(param_entry) > 0:
            timestamps = [pd.to_datetime(t) for t in param_entry.iloc[0]['timestamps']]
            values = param_entry.iloc[0]['values']
            
            if len(values) > 0:
                mean_val = np.mean([v for v in values if not np.isnan(v)])
                default_error = abs(mean_val * 0.05)  # 5% default
                return {'error': default_error, 'type': 'Estimated (5%)', 'source': 'default'}
        
        return {'error': 0, 'type': 'Unknown', 'source': 'none'}
    
    # Has forecast - get confidence/error from forecast
    forecast_data = st.session_state.forecast_results[param_name]
    
    # Check if confidence scores available
    if 'confidence_scores' in forecast_data:
        confidence_scores = forecast_data['confidence_scores']
        
        # Find confidence at target date
        forecast_timestamps = forecast_data['forecast_timestamps']
        target_year = target_date.year
        
        # Get confidence for target year
        target_confidences = []
        for i, ts in enumerate(forecast_timestamps):
            if pd.to_datetime(ts).year == target_year and i < len(confidence_scores):
                target_confidences.append(confidence_scores[i])
        
        if target_confidences:
            avg_confidence = np.mean(target_confidences)
            # Convert confidence to error (lower confidence = higher error)
            # Assume: confidence = 1 is perfect, 0 is worst
            # Error = (1 - confidence) * mean_value * scale_factor
            
            forecast_values = forecast_data['forecast_values']
            mean_forecast = np.mean([v for v in forecast_values if not np.isnan(v)])
            
            # Scale error: 0% at confidence=1, 20% at confidence=0
            error_pct = (1 - avg_confidence) * 0.20
            error_abs = abs(mean_forecast * error_pct)
            
            return {'error': error_abs, 'type': f'Confidence ({avg_confidence:.2f})', 'source': 'forecast'}
    
    # Check if model data available
    if 'trained_models' in st.session_state and param_name in st.session_state.trained_models:
        model_results = st.session_state.trained_models[param_name]
        
        # Get RMSE from test set
        best_rmse = None
        for tier in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
            if tier in model_results:
                for model_name, model_data in model_results[tier].items():
                    test_metrics = model_data.get('test_metrics', {})
                    rmse = test_metrics.get('rmse')
                    
                    if rmse is not None and (best_rmse is None or rmse < best_rmse):
                        best_rmse = rmse
        
        if best_rmse is not None:
            # Increase error for longer forecast horizon
            forecast_timestamps = forecast_data.get('forecast_timestamps', [])
            if forecast_timestamps:
                last_timestamp = pd.to_datetime(forecast_timestamps[-1])
                first_timestamp = pd.to_datetime(forecast_timestamps[0])
                horizon_years = (last_timestamp - first_timestamp).days / 365.25
                
                # Scale error with horizon (longer = more uncertain)
                scaled_rmse = best_rmse * (1 + horizon_years * 0.1)
                
                return {'error': scaled_rmse, 'type': 'RMSE (scaled)', 'source': 'model_test'}
            
            return {'error': best_rmse, 'type': 'RMSE', 'source': 'model_test'}
    
    # Default 5% error
    forecast_values = forecast_data.get('forecast_values', [])
    if forecast_values:
        mean_val = np.mean([v for v in forecast_values if not np.isnan(v)])
        default_error = abs(mean_val * 0.05)
        return {'error': default_error, 'type': 'Estimated (5%)', 'source': 'default'}
    
    return {'error': 0, 'type': 'Unknown', 'source': 'none'}


def convert_scenario_to_absolute(baseline_value: float, scenario_value: float, direction: str, unit: str) -> float:
    """Convert scenario percentage/direction to absolute value"""
    
    if unit == '%' or unit == 'percent':
        if direction == 'increase':
            return baseline_value * (1 + scenario_value / 100)
        elif direction == 'decrease':
            return baseline_value * (1 - scenario_value / 100)
        elif direction == 'target':
            return scenario_value
        else:
            return baseline_value
    elif direction == 'double':
        return baseline_value * 2
    elif direction == 'halve':
        return baseline_value * 0.5
    elif direction == 'stable':
        return baseline_value
    else:
        # If absolute value given
        return scenario_value


def create_error_ellipse(x_center: float, y_center: float, x_error: float, y_error: float, 
                         color: str, name: str, n_points: int = 50) -> dict:
    """
    Create an error ellipse for uncertainty visualization
    
    Parameters:
    -----------
    x_center : float
        Center X coordinate
    y_center : float
        Center Y coordinate
    x_error : float
        Error/uncertainty in X direction (±)
    y_error : float
        Error/uncertainty in Y direction (±)
    color : str
        Color for the ellipse
    name : str
        Name for legend
    n_points : int
        Number of points to draw ellipse
    
    Returns:
    --------
    dict : Plotly trace dict for the ellipse
    """
    
    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, n_points)
    
    # Ellipse equation: x = x_center + a*cos(θ), y = y_center + b*sin(θ)
    x_ellipse = x_center + x_error * np.cos(theta)
    y_ellipse = y_center + y_error * np.sin(theta)
    
    # Create trace
    trace = go.Scatter(
        x=x_ellipse,
        y=y_ellipse,
        mode='lines',
        name=name,
        line=dict(color=color, width=1, dash='dot'),
        fill='toself',
        fillcolor=color,
        opacity=0.15,
        hoverinfo='skip',
        showlegend=False
    )
    
    return trace


def main():
    """Main function"""
    
    # Check prerequisites
    if 'parameter_mappings' not in st.session_state or not st.session_state.parameter_mappings:
        st.error("❌ No parameter mappings found!")
        st.info("Please go to **Parameter Mapping & Validation** first")
        
        if st.button("🔗 Go to Parameter Mapping"):
            st.switch_page("pages/11_Parameter_Mapping_&_Validation.py")
        return
    
    if 'master_parameter_df' not in st.session_state or st.session_state.master_parameter_df is None:
        st.error("❌ No parameter database found!")
        st.info("Please go to **Parameter Mapping & Validation** first")
        
        if st.button("🔗 Go to Parameter Mapping"):
            st.switch_page("pages/11_Parameter_Mapping_&_Validation.py")
        return
    
    master_df = st.session_state.master_parameter_df
    scenario_date = st.session_state.get('scenario_target_date')
    
    if not scenario_date:
        st.error("❌ No scenario target date found!")
        return
    
    # Get scenario parameters
    scenario_params = []
    if 'detected_scenarios' in st.session_state and st.session_state.detected_scenarios:
        scenarios = st.session_state.detected_scenarios
        params = {}
        
        for scenario in scenarios:
            for item in scenario['items']:
                param_name = item.get('parameter_canonical', item.get('parameter', ''))
                
                if param_name not in params:
                    params[param_name] = {
                        'name': param_name,
                        'category': item.get('category', 'Other'),
                        'unit': item.get('unit', ''),
                        'scenarios': []
                    }
                
                params[param_name]['scenarios'].append({
                    'scenario': scenario['title'],
                    'value': item.get('value'),
                    'direction': item.get('direction', ''),
                    'unit': item.get('unit', '')
                })
        
        scenario_params = list(params.values())
    
    scenarios_list = st.session_state.detected_scenarios
    
    st.success(f"✅ Ready to visualize {len(scenario_params)} parameters across {len(scenarios_list)} scenarios")
    
    # === STEP 1: SELECT BASELINE DATE ===
    st.subheader("📍 Step 1: Select Baseline Date")
    st.caption("Choose the reference date for calculating changes")
    
    time_range = get_data_time_range(master_df)
    
    if time_range['min_date'] is None:
        st.error("❌ No time range data available")
        return
    
    st.info(f"""
    📅 **Data Time Range:**
    - Original data: {time_range['min_date'].strftime('%Y-%m-%d')} to {time_range['max_date'].strftime('%Y-%m-%d')}
    {'- Forecast starts: ' + time_range['forecast_start'].strftime('%Y-%m-%d') if time_range['forecast_start'] else ''}
    - Scenario target: {scenario_date.strftime('%Y-%m-%d')}
    """)
    
    baseline_min = time_range['min_date']
    baseline_max = time_range['forecast_start'] if time_range['forecast_start'] else time_range['max_date']
    
    baseline_date = st.date_input(
        "Enter Baseline Date:",
        value=baseline_max.date(),
        min_value=baseline_min.date(),
        max_value=baseline_max.date(),
        help=f"Select reference date between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}"
    )
    
    baseline_date = pd.Timestamp(baseline_date)
    
    st.caption(f"✅ Selected baseline: **{baseline_date.strftime('%Y-%m-%d')}**")
    
    st.markdown("---")
    
    # === TABLE 1: PARAMETER MAPPING & FORECAST STATUS ===
    st.subheader("📋 Table 1: Parameter Mapping & Forecast Status")
    st.caption("Verify that all scenario parameters are mapped and have forecast data")
    
    mapping_table_data = []
    
    for param_info in scenario_params:
        param_name = param_info['name']
        historical_param = st.session_state.parameter_mappings.get(param_name, 'None')
        
        if historical_param == 'None':
            status = '❌ Not Mapped'
            forecast_status = 'N/A'
        else:
            # Check if parameter exists in master_df
            param_entry = master_df[master_df['name'] == historical_param]
            
            if len(param_entry) == 0:
                status = '❌ Not Found'
                forecast_status = 'N/A'
            else:
                param_entry = param_entry.iloc[0]
                source = param_entry['source']
                
                if source == 'forecasted':
                    # Check if forecast extends to scenario_date
                    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
                    max_forecast_date = max(timestamps)
                    
                    if max_forecast_date >= scenario_date:
                        status = '✅ Mapped'
                        forecast_status = f'✅ Forecast to {max_forecast_date.strftime("%Y")}'
                    else:
                        status = '✅ Mapped'
                        forecast_status = f'⚠️ Only to {max_forecast_date.strftime("%Y")}'
                else:
                    status = '✅ Mapped'
                    forecast_status = '❌ No Forecast'
        
        mapping_table_data.append({
            'Scenario Parameter': param_name,
            'Historical Parameter': historical_param,
            'Category': param_info['category'],
            'Mapping Status': status,
            'Forecast Status': forecast_status
        })
    
    df_mapping = pd.DataFrame(mapping_table_data)
    
    st.dataframe(df_mapping, use_container_width=True, hide_index=True)
    
    # Check if any problems
    problems = df_mapping[
        (df_mapping['Mapping Status'].str.contains('❌')) | 
        (df_mapping['Forecast Status'].str.contains('❌'))
    ]
    
    if len(problems) > 0:
        st.error(f"⚠️ **{len(problems)} parameter(s) have issues!**")
        st.warning("Please fix mapping or forecasting issues before proceeding.")
        
        if st.button("🔗 Go Back to Parameter Mapping"):
            st.switch_page("pages/11_Parameter_Mapping_&_Validation.py")
        return
    else:
        st.success("✅ All parameters mapped and forecasted!")
    
    st.markdown("---")
    
    # === TABLE 2: SCENARIO VALUES TABLE ===
    st.subheader("📊 Table 2: Scenario Parameter Values")
    st.caption("Baseline values, scenario changes, and target values for each parameter")
    
    # Extract all data
    all_data = {}
    
    # First, get baseline values for all mapped parameters
    baseline_row = {'Scenario': 'BASELINE', 'Year': baseline_date.year}
    
    for param_info in scenario_params:
        param_name = param_info['name']
        historical_param = st.session_state.parameter_mappings.get(param_name, 'None')
        
        if historical_param != 'None':
            baseline_info = extract_baseline_value(historical_param, baseline_date, master_df)
            baseline_row[param_name] = baseline_info.get('value')
    
    # Then, get scenario values
    scenario_rows = []
    
    for scenario in scenarios_list:
        scenario_name = scenario['title']
        scenario_year = scenario.get('horizon', scenario_date.year)
        
        scenario_row = {
            'Scenario': scenario_name,
            'Year': scenario_year
        }
        
        # For each parameter in this scenario
        for param_info in scenario_params:
            param_name = param_info['name']
            historical_param = st.session_state.parameter_mappings.get(param_name, 'None')
            
            if historical_param == 'None':
                scenario_row[param_name] = None
                continue
            
            baseline_value = baseline_row.get(param_name)
            
            if baseline_value is None:
                scenario_row[param_name] = None
                continue
            
            # Find scenario data for this parameter
            scenario_data = None
            for sc_data in param_info['scenarios']:
                if sc_data['scenario'] == scenario_name:
                    scenario_data = sc_data
                    break
            
            if scenario_data:
                scenario_value = scenario_data.get('value')
                direction = scenario_data.get('direction', '')
                unit = scenario_data.get('unit', '')
                
                if scenario_value is not None:
                    # Convert to absolute value
                    absolute_value = convert_scenario_to_absolute(
                        baseline_value,
                        scenario_value,
                        direction,
                        unit
                    )
                    scenario_row[param_name] = absolute_value
                else:
                    scenario_row[param_name] = baseline_value
            else:
                # Parameter not in this scenario - use baseline
                scenario_row[param_name] = baseline_value
        
        scenario_rows.append(scenario_row)
    
    # Combine into single table
    all_rows = [baseline_row] + scenario_rows
    df_values = pd.DataFrame(all_rows)
    
    # Reorder columns: Scenario, Year, then all parameters
    param_columns = [p['name'] for p in scenario_params if p['name'] in df_values.columns]
    df_values = df_values[['Scenario', 'Year'] + param_columns]
    
    st.dataframe(
        df_values.style.format({col: '{:.2f}' for col in param_columns}, na_rep='-'),
        use_container_width=True,
        hide_index=True
    )
    
    # Export table
    with st.expander("💾 Export Scenario Values Table"):
        csv = df_values.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"scenario_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # === TABLE 3: DETAILED CHANGE TABLE ===
    st.subheader("📈 Table 3: Detailed Change Analysis")
    st.caption("Shows baseline, % change, direction, and absolute target value for each scenario-parameter combination")
    
    detailed_data = []
    
    for scenario in scenarios_list:
        scenario_name = scenario['title']
        
        for param_info in scenario_params:
            param_name = param_info['name']
            historical_param = st.session_state.parameter_mappings.get(param_name, 'None')
            
            if historical_param == 'None':
                continue
            
            baseline_value = baseline_row.get(param_name)
            
            if baseline_value is None:
                continue
            
            # Find scenario data
            scenario_data = None
            for sc_data in param_info['scenarios']:
                if sc_data['scenario'] == scenario_name:
                    scenario_data = sc_data
                    break
            
            if scenario_data:
                scenario_value = scenario_data.get('value')
                direction = scenario_data.get('direction', '')
                unit = scenario_data.get('unit', '')
                
                if scenario_value is not None:
                    # Convert to absolute
                    absolute_value = convert_scenario_to_absolute(
                        baseline_value,
                        scenario_value,
                        direction,
                        unit
                    )
                    
                    # Calculate actual change %
                    if unit in ['%', 'percent'] and direction in ['increase', 'decrease']:
                        change_pct = scenario_value
                    else:
                        change_pct = ((absolute_value - baseline_value) / baseline_value) * 100
                    
                    detailed_data.append({
                        'Scenario': scenario_name,
                        'Parameter': param_name,
                        'Category': param_info['category'],
                        'Baseline Value': baseline_value,
                        'Change (%)': change_pct,
                        'Direction': direction,
                        'Target Value': absolute_value,
                        'Absolute Change': absolute_value - baseline_value
                    })
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        
        st.dataframe(
            df_detailed.style.format({
                'Baseline Value': '{:.2f}',
                'Change (%)': '{:.1f}',
                'Target Value': '{:.2f}',
                'Absolute Change': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Export
        with st.expander("💾 Export Detailed Change Table"):
            csv = df_detailed.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"detailed_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # === INTERACTIVE PLOTTING ===
    st.subheader("📈 Step 2: Interactive Trajectory Visualization")
    
    tab1, tab2 = st.tabs(["📊 Parameter vs Parameter", "🗂️ Category Analysis"])
    
    with tab1:
        st.markdown("### Parameter vs Parameter Plot")
        st.caption("Visualize relationships between parameters across scenarios with baseline reference")
        
        # Get parameter list
        param_columns = [p['name'] for p in scenario_params]
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "X-Axis Parameter:",
                options=['None'] + param_columns,
                index=0,
                key='x_param_plot'
            )
        
        with col2:
            y_param = st.selectbox(
                "Y-Axis Parameter:",
                options=param_columns,
                index=0 if param_columns else None,
                key='y_param_plot'
            )
        
        st.markdown("---")
        st.markdown("**Customize Visualization:**")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            baseline_marker_color = st.color_picker(
                "Baseline Color",
                value="#000000",
                help="Color for baseline reference point"
            )
        
        with col_b:
            scenario_color_scheme = st.selectbox(
                "Scenario Colors",
                options=['Plotly', 'Pastel', 'Bold', 'Vivid', 'D3', 'Set2'],
                index=0
            )
        
        with col_c:
            x_axis_scale = st.selectbox(
                "X-Axis Scale",
                options=['Linear', 'Log'],
                index=0
            )
        
        with col_d:
            y_axis_scale = st.selectbox(
                "Y-Axis Scale",
                options=['Linear', 'Log'],
                index=0
            )
        
        col_e, col_f = st.columns(2)
        
        with col_e:
            marker_size = st.slider(
                "Marker Size",
                min_value=5,
                max_value=20,
                value=10,
                key='marker_size_plot'
            )
        
        with col_f:
            line_width = st.slider(
                "Line Width",
                min_value=1,
                max_value=5,
                value=2,
                key='line_width_plot'
            )
        
        st.markdown("---")
        
        # Create plot
        if y_param and y_param in param_columns:
            fig = go.Figure()
            
            # Color schemes mapping
            color_schemes = {
                'Plotly': px.colors.qualitative.Plotly,
                'Pastel': px.colors.qualitative.Pastel,
                'Bold': px.colors.qualitative.Bold,
                'Vivid': px.colors.qualitative.Vivid,
                'D3': px.colors.qualitative.D3,
                'Set2': px.colors.qualitative.Set2
            }
            
            colors = color_schemes.get(scenario_color_scheme, px.colors.qualitative.Plotly)
            
            if x_param == 'None':
                # === SINGLE PARAMETER VARIATION ===
                
                # Get historical param name for error extraction
                historical_y_param = None
                for param_info in scenario_params:
                    if param_info['name'] == y_param:
                        historical_y_param = st.session_state.parameter_mappings.get(y_param, 'None')
                        break
                
                # Get Y error
                y_error_info = get_forecast_error(historical_y_param, scenario_date, master_df) if historical_y_param and historical_y_param != 'None' else {'error': 0}
                y_error = y_error_info.get('error', 0)
                
                # Plot baseline
                baseline_value_y = baseline_row.get(y_param)
                
                if baseline_value_y is not None:
                    # Baseline has minimal error (it's measured data)
                    baseline_error_y = abs(baseline_value_y * 0.02)  # 2% error for baseline
                    
                    # Add baseline error ellipse
                    if baseline_error_y > 0:
                        ellipse = create_error_ellipse(
                            x_center=0,
                            y_center=baseline_value_y,
                            x_error=0.15,  # Small X error for visual clarity
                            y_error=baseline_error_y,
                            color=baseline_marker_color,
                            name='Baseline Error'
                        )
                        fig.add_trace(ellipse)
                    
                    # Add baseline marker
                    fig.add_trace(go.Scatter(
                        x=[0],
                        y=[baseline_value_y],
                        mode='markers',
                        name='Baseline',
                        marker=dict(
                            size=marker_size + 5,
                            color=baseline_marker_color,
                            symbol='star',
                            line=dict(color='white', width=2)
                        ),
                        showlegend=True,
                        hovertemplate=f'<b>Baseline</b><br>{y_param}: %{{y:.2f}}<br>Error: ±{baseline_error_y:.2f}<extra></extra>'
                    ))
                
                # Plot scenarios with error ellipses
                for idx, row in enumerate(scenario_rows):
                    scenario_name = row['Scenario']
                    y_value = row.get(y_param)
                    
                    if y_value is not None:
                        scenario_color = colors[idx % len(colors)]
                        
                        # Add error ellipse
                        if y_error > 0:
                            ellipse = create_error_ellipse(
                                x_center=idx + 1,
                                y_center=y_value,
                                x_error=0.2,  # X spread for visual clarity
                                y_error=y_error,
                                color=scenario_color,
                                name=f'{scenario_name} Error'
                            )
                            fig.add_trace(ellipse)
                        
                        # Add scenario marker
                        fig.add_trace(go.Scatter(
                            x=[idx + 1],
                            y=[y_value],
                            mode='markers+text',
                            name=scenario_name,
                            marker=dict(
                                size=marker_size,
                                color=scenario_color
                            ),
                            text=[scenario_name],
                            textposition='top center',
                            textfont=dict(size=9),
                            showlegend=True,
                            hovertemplate=f'<b>{scenario_name}</b><br>{y_param}: %{{y:.2f}}<br>Error: ±{y_error:.2f}<br>({y_error_info.get("type", "Unknown")})<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f"{y_param} Across Scenarios (with Forecast Uncertainty)",
                    xaxis_title="Scenario Index",
                    yaxis_title=y_param,
                    yaxis_type='log' if y_axis_scale == 'Log' else 'linear',
                    hovermode='closest',
                    showlegend=True,
                    height=600
                )
            
            else:
                # === TWO PARAMETER COMPARISON ===
                
                # Get historical param names for error extraction
                historical_x_param = None
                historical_y_param = None
                
                for param_info in scenario_params:
                    if param_info['name'] == x_param:
                        historical_x_param = st.session_state.parameter_mappings.get(x_param, 'None')
                    if param_info['name'] == y_param:
                        historical_y_param = st.session_state.parameter_mappings.get(y_param, 'None')
                
                # Get X and Y errors
                x_error_info = get_forecast_error(historical_x_param, scenario_date, master_df) if historical_x_param and historical_x_param != 'None' else {'error': 0}
                y_error_info = get_forecast_error(historical_y_param, scenario_date, master_df) if historical_y_param and historical_y_param != 'None' else {'error': 0}
                
                x_error = x_error_info.get('error', 0)
                y_error = y_error_info.get('error', 0)
                
                # Plot baseline
                baseline_value_x = baseline_row.get(x_param)
                baseline_value_y = baseline_row.get(y_param)
                
                if baseline_value_x is not None and baseline_value_y is not None:
                    # Baseline has minimal error (measured data)
                    baseline_error_x = abs(baseline_value_x * 0.02)  # 2% error
                    baseline_error_y = abs(baseline_value_y * 0.02)
                    
                    # Add baseline error ellipse
                    if baseline_error_x > 0 and baseline_error_y > 0:
                        ellipse = create_error_ellipse(
                            x_center=baseline_value_x,
                            y_center=baseline_value_y,
                            x_error=baseline_error_x,
                            y_error=baseline_error_y,
                            color=baseline_marker_color,
                            name='Baseline Error'
                        )
                        fig.add_trace(ellipse)
                    
                    # Add baseline marker
                    fig.add_trace(go.Scatter(
                        x=[baseline_value_x],
                        y=[baseline_value_y],
                        mode='markers',
                        name='Baseline',
                        marker=dict(
                            size=marker_size + 8,
                            color=baseline_marker_color,
                            symbol='star',
                            line=dict(color='white', width=2)
                        ),
                        showlegend=True,
                        hovertemplate=f'<b>Baseline</b><br>{x_param}: %{{x:.2f}} (±{baseline_error_x:.2f})<br>{y_param}: %{{y:.2f}} (±{baseline_error_y:.2f})<extra></extra>'
                    ))
                
                # Plot trajectories for each scenario with error ellipses
                for idx, row in enumerate(scenario_rows):
                    scenario_name = row['Scenario']
                    x_value = row.get(x_param)
                    y_value = row.get(y_param)
                    
                    if x_value is not None and y_value is not None and baseline_value_x is not None and baseline_value_y is not None:
                        scenario_color = colors[idx % len(colors)]
                        
                        # Add error ellipse at scenario target
                        if x_error > 0 and y_error > 0:
                            ellipse = create_error_ellipse(
                                x_center=x_value,
                                y_center=y_value,
                                x_error=x_error,
                                y_error=y_error,
                                color=scenario_color,
                                name=f'{scenario_name} Error'
                            )
                            fig.add_trace(ellipse)
                        
                        # Draw arrow from baseline to scenario
                        fig.add_trace(go.Scatter(
                            x=[baseline_value_x, x_value],
                            y=[baseline_value_y, y_value],
                            mode='lines+markers',
                            name=scenario_name,
                            line=dict(
                                color=scenario_color,
                                width=line_width
                            ),
                            marker=dict(size=marker_size),
                            showlegend=True,
                            hovertemplate=f'<b>{scenario_name}</b><br>{x_param}: %{{x:.2f}} (±{x_error:.2f})<br>{y_param}: %{{y:.2f}} (±{y_error:.2f})<br>X Error: {x_error_info.get("type", "Unknown")}<br>Y Error: {y_error_info.get("type", "Unknown")}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f"Trajectory: {y_param} vs {x_param} (with Forecast Uncertainty)",
                    xaxis_title=x_param,
                    yaxis_title=y_param,
                    xaxis_type='log' if x_axis_scale == 'Log' else 'linear',
                    yaxis_type='log' if y_axis_scale == 'Log' else 'linear',
                    hovermode='closest',
                    showlegend=True,
                    height=600
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export plot
            if EXPORT_AVAILABLE:
                with st.expander("💾 Export Figure"):
                    filename = f"trajectory_{y_param}_vs_{x_param if x_param != 'None' else 'scenarios'}"
                    quick_export_buttons(fig, filename, ['png', 'pdf', 'html'])
        else:
            st.info("Please select a Y-axis parameter to create the plot")
    
    with tab2:
        st.markdown("### Category-Level Analysis")
        st.info("🚧 **Category aggregation is under development.**")
        
        st.markdown("""
        **Possible approaches for category visualization:**
        
        1. **Average Values per Category:**
           - Calculate mean of all parameters in each category
           - Plot category averages across scenarios
        
        2. **Category Radar Charts:**
           - Show all categories for each scenario
           - Compare scenario profiles
        
        3. **Category Heatmap:**
           - Rows = Scenarios
           - Columns = Categories
           - Values = Average change %
        
        4. **Category Box Plots:**
           - Distribution of parameter values within each category
           - Compare across scenarios
        
        **Which would you prefer? Or suggest another approach!**
        """)
        
        # Show available categories
        categories = sorted(set(p['category'] for p in scenario_params))
        
        st.markdown("**Available Categories:**")
        for cat in categories:
            params_in_cat = [p['name'] for p in scenario_params if p['category'] == cat]
            st.caption(f"• **{cat}**: {len(params_in_cat)} parameters ({', '.join(params_in_cat[:3])}{'...' if len(params_in_cat) > 3 else ''})")


if __name__ == "__main__":
    main()
