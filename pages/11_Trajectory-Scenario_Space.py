"""
Page 11: Trajectory-Scenario Space
Map historical parameters to scenario parameters with forecast validation
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
st.title("🎯 Trajectory-Scenario Space Analysis")
st.markdown("*Bridge historical data with scenario planning through parameter mapping and compatibility analysis*")
st.markdown("---")


def initialize_trajectory_state():
    """Initialize session state"""
    if "parameter_mappings" not in st.session_state:
        st.session_state.parameter_mappings = {}
    if "compatibility_score" not in st.session_state:
        st.session_state.compatibility_score = None
    if "baseline_references" not in st.session_state:
        st.session_state.baseline_references = {}
    if "absolute_values" not in st.session_state:
        st.session_state.absolute_values = {}
    if "master_parameter_df" not in st.session_state:
        st.session_state.master_parameter_df = None
    if "scenario_target_date" not in st.session_state:
        st.session_state.scenario_target_date = None


def build_master_parameter_dataframe():
    """
    Build master dataframe following EXACT priority:
    1. Forecasted data
    2. Cleaned data (skip if in forecasted)
    3. Raw data (skip if in forecasted/cleaned)
    4. Calculated components (skip if in forecasted/cleaned/raw)
    5. Drop short-name duplicates
    6. Organize: Parameters first, then Components
    """
    
    master_data = []
    existing_names = set()
    
    # === STEP 1: FORECASTED DATA ===
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        for param_name, forecast in st.session_state.forecast_results.items():
            is_component = any(comp in param_name.upper() for comp in ['PC', 'IC', 'FACTOR', 'COMPONENT'])
            
            master_data.append({
                'name': param_name,
                'type': 'component' if is_component else 'parameter',
                'source': 'forecasted',
                'symbol': '🔮',
                'source_full': 'Forecasted',
                'timestamps': forecast['forecast_timestamps'],
                'values': forecast['forecast_values']
            })
            
            existing_names.add(param_name)
    
    # === STEP 2: CLEANED DATA ===
    if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
        df_clean = st.session_state.df_clean
        
        for param_name in df_clean['variable'].unique():
            if param_name not in existing_names:
                param_data = df_clean[df_clean['variable'] == param_name]
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                is_component = any(comp in param_name.upper() for comp in ['PC', 'IC', 'FACTOR', 'COMPONENT'])
                
                master_data.append({
                    'name': param_name,
                    'type': 'component' if is_component else 'parameter',
                    'source': 'cleaned',
                    'symbol': '🧹',
                    'source_full': 'Cleaned',
                    'timestamps': param_data[time_col].tolist(),
                    'values': param_data['value'].tolist()
                })
                
                existing_names.add(param_name)
    
    # === STEP 3: RAW DATA ===
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        df_long = st.session_state.df_long
        
        for param_name in df_long['variable'].unique():
            if param_name not in existing_names:
                param_data = df_long[df_long['variable'] == param_name]
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                is_component = any(comp in param_name.upper() for comp in ['PC', 'IC', 'FACTOR', 'COMPONENT'])
                
                master_data.append({
                    'name': param_name,
                    'type': 'component' if is_component else 'parameter',
                    'source': 'raw',
                    'symbol': '📊',
                    'source_full': 'Raw',
                    'timestamps': param_data[time_col].tolist(),
                    'values': param_data['value'].tolist()
                })
                
                existing_names.add(param_name)
    
    # === STEP 4: CALCULATED COMPONENTS ===
    if 'reduction_results' in st.session_state and st.session_state.reduction_results:
        df_long = st.session_state.get('df_long')
        if df_long is not None:
            time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
            unique_times = df_long[time_col].unique().tolist()
            
            # PCA
            if 'pca' in st.session_state.reduction_results and 'pca_components' in st.session_state:
                pca_results = st.session_state.reduction_results['pca']
                n_components = pca_results.get('n_components', 0)
                
                for i in range(n_components):
                    comp_name = f"PC{i+1}"
                    if comp_name not in existing_names:
                        component_values = st.session_state.pca_components[:, i].tolist()
                        
                        master_data.append({
                            'name': comp_name,
                            'type': 'component',
                            'source': 'calculated',
                            'symbol': '🔬',
                            'source_full': 'Calculated PCA',
                            'timestamps': unique_times[:len(component_values)],
                            'values': component_values
                        })
                        existing_names.add(comp_name)
            
            # ICA
            if 'ica' in st.session_state.reduction_results and 'ica_components' in st.session_state:
                ica_results = st.session_state.reduction_results['ica']
                n_components = ica_results.get('n_components', 0)
                
                for i in range(n_components):
                    comp_name = f"IC{i+1}"
                    if comp_name not in existing_names:
                        component_values = st.session_state.ica_components[:, i].tolist()
                        
                        master_data.append({
                            'name': comp_name,
                            'type': 'component',
                            'source': 'calculated',
                            'symbol': '🔬',
                            'source_full': 'Calculated ICA',
                            'timestamps': unique_times[:len(component_values)],
                            'values': component_values
                        })
                        existing_names.add(comp_name)
            
            # Factor Analysis
            if 'factor_analysis' in st.session_state.reduction_results and 'factor_scores' in st.session_state:
                fa_results = st.session_state.reduction_results['factor_analysis']
                n_factors = fa_results.get('n_factors', 0)
                
                for i in range(n_factors):
                    comp_name = f"Factor{i+1}"
                    if comp_name not in existing_names:
                        factor_values = st.session_state.factor_scores[:, i].tolist()
                        
                        master_data.append({
                            'name': comp_name,
                            'type': 'component',
                            'source': 'calculated',
                            'symbol': '🔬',
                            'source_full': 'Calculated Factor',
                            'timestamps': unique_times[:len(factor_values)],
                            'values': factor_values
                        })
                        existing_names.add(comp_name)
    
    # === STEP 5: DROP SHORT-NAME DUPLICATES ===
    names_to_drop = set()
    
    for entry1 in master_data:
        name1 = entry1['name']
        prefix1 = name1[:3].upper() if len(name1) >= 3 else name1.upper()
        
        for entry2 in master_data:
            name2 = entry2['name']
            prefix2 = name2[:3].upper() if len(name2) >= 3 else name2.upper()
            
            if prefix1 == prefix2 and name1 != name2:
                if len(name1) < len(name2):
                    names_to_drop.add(name1)
                elif len(name2) < len(name1):
                    names_to_drop.add(name2)
    
    master_data = [entry for entry in master_data if entry['name'] not in names_to_drop]
    
    # === STEP 6: ORGANIZE ===
    parameters = [entry for entry in master_data if entry['type'] == 'parameter']
    components = [entry for entry in master_data if entry['type'] == 'component']
    
    parameters = sorted(parameters, key=lambda x: x['name'])
    components = sorted(components, key=lambda x: x['name'])
    
    final_data = parameters + components
    
    df = pd.DataFrame(final_data) if final_data else pd.DataFrame()
    
    return df


def get_scenario_target_date():
    """
    Extract scenario target date from Page 9 detected scenarios
    
    Returns:
    --------
    pd.Timestamp or None
        Scenario target date (horizon year)
    """
    
    if 'detected_scenarios' not in st.session_state or not st.session_state.detected_scenarios:
        return None
    
    scenarios = st.session_state.detected_scenarios
    
    # Get all horizon years
    horizon_years = []
    for scenario in scenarios:
        horizon = scenario.get('horizon')
        if horizon is not None:
            try:
                year = int(horizon)
                horizon_years.append(year)
            except:
                pass
    
    if not horizon_years:
        return None
    
    # Use the latest scenario year
    max_year = max(horizon_years)
    
    return pd.Timestamp(year=max_year, month=12, day=31)


def get_scenario_parameters():
    """Get parameters from detected scenarios"""
    
    if 'detected_scenarios' not in st.session_state or not st.session_state.detected_scenarios:
        return []
    
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
    
    return list(params.values())


def calculate_compatibility(scenario_params: list, master_df: pd.DataFrame, mappings: dict) -> dict:
    """Calculate compatibility score"""
    
    if not scenario_params:
        return {'score': 0, 'mapped': 0, 'total': 0, 'unmapped': []}
    
    mapped_count = sum(1 for p in scenario_params if mappings.get(p['name'], 'None') != 'None')
    total_params = len(scenario_params)
    
    score = mapped_count / total_params if total_params > 0 else 0
    unmapped = [p['name'] for p in scenario_params if mappings.get(p['name'], 'None') == 'None']
    
    return {
        'score': score,
        'mapped': mapped_count,
        'total': total_params,
        'unmapped': unmapped
    }


def get_data_time_range(master_df: pd.DataFrame) -> dict:
    """
    Get time range from data - FIXED to avoid pandas Index truth value error
    """
    
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
                # FIXED: Check list length instead of pandas Index truth value
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
                # FIXED: Check list length
                if timestamps_list is not None and len(timestamps_list) > 0:
                    all_forecast_starts.append(pd.to_datetime(timestamps_list[0]))
            
            if all_forecast_starts:
                forecast_start = min(all_forecast_starts)
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'forecast_start': forecast_start
    }


def check_forecast_coverage(mapped_params: dict, master_df: pd.DataFrame, scenario_date: pd.Timestamp) -> dict:
    """
    Check if all mapped parameters have forecast data up to scenario date
    
    Returns:
    --------
    dict with:
        - all_covered: bool
        - missing_params: list of parameter names needing forecast/reforecast
    """
    
    missing_params = []
    
    for scenario_param, historical_param in mapped_params.items():
        if historical_param == 'None':
            continue
        
        # Find parameter in master_df
        param_entry = master_df[master_df['name'] == historical_param]
        
        if len(param_entry) == 0:
            missing_params.append(historical_param)
            continue
        
        param_entry = param_entry.iloc[0]
        source = param_entry['source']
        
        if source == 'forecasted':
            # Check if forecast extends to scenario_date
            timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
            max_forecast_date = max(timestamps)
            
            if max_forecast_date < scenario_date:
                # Forecast too short - needs reforecasting
                missing_params.append(historical_param)
        else:
            # Not forecasted - needs forecasting
            missing_params.append(historical_param)
    
    all_covered = (len(missing_params) == 0)
    
    return {
        'all_covered': all_covered,
        'missing_params': missing_params
    }


def check_data_at_baseline(param_name: str, baseline_date: pd.Timestamp, master_df: pd.DataFrame) -> dict:
    """Check if data exists for parameter at baseline date"""
    
    param_entry = master_df[master_df['name'] == param_name]
    
    if len(param_entry) == 0:
        return {
            'exists': False,
            'source': 'none',
            'message': '❌ Parameter not found'
        }
    
    param_entry = param_entry.iloc[0]
    
    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
    
    if not timestamps:
        return {
            'exists': False,
            'source': 'none',
            'message': '❌ No data available'
        }
    
    min_time = min(timestamps)
    max_time = max(timestamps)
    
    if baseline_date < min_time or baseline_date > max_time:
        return {
            'exists': False,
            'source': param_entry['source'],
            'message': f"❌ No data at {baseline_date.year} (available: {min_time.year}-{max_time.year})"
        }
    
    source = param_entry['source']
    
    if source == 'forecasted':
        message = '✅ Data available from forecasted'
    elif source == 'cleaned':
        message = '✅ Data available from cleaned'
    elif source == 'calculated':
        message = '✅ Data available from calculated'
    elif source == 'raw':
        message = '✅ Data available from raw'
    else:
        message = '✅ Data available'
    
    return {
        'exists': True,
        'source': source,
        'message': message
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


def convert_scenario_to_absolute(param_name: str, scenario_value: float, direction: str, 
                                 unit: str, baseline_value: float) -> float:
    """Convert scenario to absolute value"""
    
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
        return scenario_value


def main():
    """Main function"""
    
    initialize_trajectory_state()
    
    # Check prerequisites
    if 'detected_scenarios' not in st.session_state or not st.session_state.detected_scenarios:
        st.warning("⚠️ No scenarios detected!")
        st.info("👈 Please go to **Scenario Analysis (NLP)** to create scenarios first")
        
        if st.button("📝 Go to Scenario Analysis"):
            st.switch_page("pages/9_Scenario_Analysis_(NLP).py")
        return
    
    # Get scenario parameters
    scenario_params = get_scenario_parameters()
    
    if not scenario_params:
        st.error("❌ No parameters found in scenarios")
        st.info("Please ensure your scenarios have parameters defined in Page 9, Step 3")
        return
    
    # Get scenario target date
    scenario_date = get_scenario_target_date()
    st.session_state.scenario_target_date = scenario_date
    
    # === BUILD MASTER PARAMETER DATAFRAME ===
    with st.spinner("Building master parameter database..."):
        master_df = build_master_parameter_dataframe()
        st.session_state.master_parameter_df = master_df
    
    if master_df is None or len(master_df) == 0:
        st.warning("⚠️ No historical data available!")
        st.info("Please upload data first in **Upload & Data Diagnostics**")
        return
    
    st.success(f"✅ Found {len(scenario_params)} scenario parameters and {len(master_df)} historical parameters/components")
    
    # Show scenario date
    if scenario_date:
        st.info(f"🎯 **Scenario Target Date:** {scenario_date.strftime('%Y-%m-%d')} (extracted from scenario horizons)")
    else:
        st.warning("⚠️ No scenario target date found. Please ensure scenarios have horizon years defined.")
    
    # Show master df summary
    with st.expander("📊 Available Parameters Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        n_forecasted = len(master_df[master_df['source'] == 'forecasted'])
        n_cleaned = len(master_df[master_df['source'] == 'cleaned'])
        n_raw = len(master_df[master_df['source'] == 'raw'])
        n_calculated = len(master_df[master_df['source'] == 'calculated'])
        
        col1.metric("🔮 Forecasted", n_forecasted)
        col2.metric("🧹 Cleaned", n_cleaned)
        col3.metric("📊 Raw", n_raw)
        col4.metric("🔬 Calculated", n_calculated)
        
        st.dataframe(
            master_df[['name', 'type', 'source', 'symbol', 'source_full']],
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # === STEP 1: PARAMETER MAPPING ===
    st.subheader("🔗 Step 1: Map Scenario Parameters to Historical Data")
    st.caption("Match each scenario parameter to its equivalent in historical data")
    
    st.markdown("---")
    
    # Display mapping table FIRST (so selectboxes update session_state)
    st.markdown("### 📋 Parameter Mapping Table")
    
    params_by_category = {}
    for param in scenario_params:
        cat = param['category']
        if cat not in params_by_category:
            params_by_category[cat] = []
        params_by_category[cat].append(param)
    
    for category in sorted(params_by_category.keys()):
        params_in_category = params_by_category[category]
        
        with st.expander(f"**{category}** ({len(params_in_category)} parameters)", expanded=True):
            
            col_h1, col_h2, col_h3 = st.columns([3, 4, 1])
            with col_h1:
                st.markdown("**Scenario Parameter**")
            with col_h2:
                st.markdown("**Historical Parameter / Component**")
            with col_h3:
                st.markdown("**Status**")
            
            st.markdown("---")
            
            for param in params_in_category:
                param_name = param['name']
                
                col1, col2, col3 = st.columns([3, 4, 1])
                
                with col1:
                    st.markdown(f"**{param_name}**")
                    if param.get('unit'):
                        st.caption(f"Unit: {param['unit']}")
                
                with col2:
                    current_mapping = st.session_state.parameter_mappings.get(param_name, 'None')
                    
                    options = ['None']
                    
                    parameters = master_df[master_df['type'] == 'parameter']
                    
                    if len(parameters) > 0:
                        options.append('─── PARAMETERS ───')
                        
                        for _, row in parameters.iterrows():
                            options.append(f"{row['symbol']} {row['name']}")
                    
                    components = master_df[master_df['type'] == 'component']
                    
                    if len(components) > 0:
                        options.append('─── COMPONENTS ───')
                        
                        for _, row in components.iterrows():
                            options.append(f"{row['symbol']} {row['name']}")
                    
                    default_idx = 0
                    
                    for i, opt in enumerate(options):
                        if opt.startswith(('🔮', '🧹', '📊', '🔬')):
                            clean_opt = opt.split(' ', 1)[1] if ' ' in opt else opt
                            if clean_opt == current_mapping:
                                default_idx = i
                                break
                    
                    if default_idx == 0 and current_mapping in options:
                        default_idx = options.index(current_mapping)
                    
                    selected = st.selectbox(
                        "Select:",
                        options=options,
                        index=default_idx,
                        key=f"mapping_{param_name}_{category}",
                        label_visibility="collapsed"
                    )
                    
                    if not selected.startswith('───'):
                        if selected.startswith(('🔮', '🧹', '📊', '🔬')):
                            clean_selected = selected.split(' ', 1)[1] if ' ' in selected else selected
                            st.session_state.parameter_mappings[param_name] = clean_selected
                        else:
                            st.session_state.parameter_mappings[param_name] = selected
                
                with col3:
                    if selected != 'None' and not selected.startswith('───'):
                        st.success("✓")
                    else:
                        st.warning("⚠️")
                
                st.markdown("")
    
    st.markdown("---")
    
    # NOW calculate compatibility AFTER all selectboxes have been displayed and updated
    compatibility = calculate_compatibility(scenario_params, master_df, 
                                           st.session_state.parameter_mappings)
    
    # Display compatibility metrics
    col1, col2, col3, col4 = st.columns(4)
    
    score_pct = compatibility['score'] * 100
    
    if score_pct >= 80:
        score_color = "🟢"
        score_label = "Excellent"
    elif score_pct >= 60:
        score_color = "🟡"
        score_label = "Good"
    elif score_pct >= 40:
        score_color = "🟠"
        score_label = "Fair"
    else:
        score_color = "🔴"
        score_label = "Poor"
    
    col1.metric("Compatibility", f"{score_pct:.0f}%", score_label)
    col2.metric("Mapped Parameters", f"{compatibility['mapped']}/{compatibility['total']}")
    col3.metric("Coverage", f"{compatibility['mapped']}/{compatibility['total']}")
    col4.metric("Unmapped", len(compatibility['unmapped']))
    
    st.markdown(f"### {score_color} Compatibility: {score_label}")
    
    if compatibility['unmapped']:
        st.markdown("---")
        st.warning(f"⚠️ **{len(compatibility['unmapped'])} Unmapped Parameters**")
        st.caption(f"Unmapped: {', '.join(compatibility['unmapped'])}")
    else:
        st.success("🎉 **Perfect! All parameters mapped!**")
    
    # === STEP 1b: CHECK FORECAST COVERAGE ===
    if compatibility['score'] > 0 and scenario_date:
        st.markdown("---")
        st.subheader("✅ Step 1b: Validate Forecast Coverage")
        st.caption("Checking if all parameters have forecasts extending to scenario date...")
        
        coverage = check_forecast_coverage(
            st.session_state.parameter_mappings,
            master_df,
            scenario_date
        )
        
        if coverage['all_covered']:
            st.success(f"✅ All parameters have forecast data extending to {scenario_date.strftime('%Y-%m-%d')}!")
        else:
            st.error(f"❌ {len(coverage['missing_params'])} parameter(s) need forecasting to reach scenario date")
            
            st.warning(f"**Parameters needing forecast:**")
            for param in coverage['missing_params']:
                st.caption(f"• {param}")
            
            st.markdown("---")
            
            # Store parameters for Page 11b
            st.session_state.trajectory_forecast_params = coverage['missing_params']
            st.session_state.trajectory_forecast_target = scenario_date
            
            st.info("**Next Step:** Train models (if needed), evaluate, and forecast these parameters")
            
            if st.button("🔮 Train & Forecast Missing Parameters", type="primary", use_container_width=True):
                # Navigate to helper page in main directory (not in sidebar)
                st.switch_page("pages/12_Forecasting_Helper.py")
            
            # Don't continue to baseline selection
            return
    
    # === STEP 2: BASELINE SELECTION ===
    if compatibility['score'] > 0:
        st.markdown("---")
        st.subheader("📍 Step 2: Select Baseline Date")
        st.caption("Choose the reference date for calculating percentage changes")
        
        time_range = get_data_time_range(master_df)
        
        if time_range['min_date'] is None:
            st.error("❌ No time range data available")
            return
        
        st.info(f"""
        📅 **Data Time Range:**
        - Original data: {time_range['min_date'].strftime('%Y-%m-%d')} to {time_range['max_date'].strftime('%Y-%m-%d')}
        {'- Forecast starts: ' + time_range['forecast_start'].strftime('%Y-%m-%d') if time_range['forecast_start'] else '- No forecasts available'}
        {f"- Scenario target: {scenario_date.strftime('%Y-%m-%d')}" if scenario_date else ''}
        """)
        
        baseline_min = time_range['min_date']
        
        if time_range['forecast_start']:
            baseline_max = time_range['forecast_start']
        else:
            baseline_max = time_range['max_date']
        
        st.success(f"✅ Baseline must be between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}")
        
        baseline_date = st.date_input(
            "Enter Baseline Date:",
            value=baseline_max.date(),
            min_value=baseline_min.date(),
            max_value=baseline_max.date(),
            help=f"Enter date between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}"
        )
        
        baseline_date = pd.Timestamp(baseline_date)
        
        st.caption(f"Selected baseline: **{baseline_date.strftime('%Y-%m-%d')}**")
        
        st.markdown("---")
        
        if st.button("✅ Confirm Baseline Date", type="primary", use_container_width=True):
            st.session_state.confirmed_baseline_date = baseline_date
            st.success(f"✅ Baseline date confirmed: {baseline_date.strftime('%Y-%m-%d')}")
            st.rerun()
        
        # [TRUNCATED - Rest continues with baseline extraction, absolute conversion, visualization]


if __name__ == "__main__":
    main()
