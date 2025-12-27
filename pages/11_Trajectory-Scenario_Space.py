"""
Page 11: Trajectory-Scenario Space
Map historical parameters to scenario parameters following strict data collection priority
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


def build_master_parameter_dataframe():
    """
    Build master dataframe of ALL available parameters/components following EXACT priority:
    
    Step 1: Get forecasted data (parameters + components)
    Step 2: Get cleaned data (skip if already in df from step 1)
    Step 3: Get raw data (skip if already in df from steps 1-2)
    Step 4: Get calculated components (skip if already in df from steps 1-3)
    Step 5: Drop short-name duplicates (if param starts with same 3 letters)
    Step 6: Organize: Parameters first, then Components
    
    Returns:
    --------
    pd.DataFrame with columns:
        - name: parameter/component name
        - type: 'parameter' or 'component'
        - source: 'forecasted', 'cleaned', 'raw', 'calculated'
        - symbol: emoji for UI
        - data: actual time series data (as list of dicts)
    """
    
    master_data = []
    existing_names = set()
    
    # === STEP 1: FORECASTED DATA (Parameters + Components) ===
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        for param_name, forecast in st.session_state.forecast_results.items():
            # Determine if parameter or component
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
    
    # === STEP 2: CLEANED DATA (only if NOT in existing_names) ===
    if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
        df_clean = st.session_state.df_clean
        
        for param_name in df_clean['variable'].unique():
            if param_name not in existing_names:  # Skip if already exists
                # Get data
                param_data = df_clean[df_clean['variable'] == param_name]
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                # Determine type
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
    
    # === STEP 3: RAW DATA (only if NOT in existing_names) ===
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        df_long = st.session_state.df_long
        
        for param_name in df_long['variable'].unique():
            if param_name not in existing_names:  # Skip if already exists
                # Get data
                param_data = df_long[df_long['variable'] == param_name]
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                # Determine type
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
    
    # === STEP 4: CALCULATED COMPONENTS (only if NOT in existing_names) ===
    if 'reduction_results' in st.session_state and st.session_state.reduction_results:
        
        # Get timestamps from original data
        df_long = st.session_state.get('df_long')
        if df_long is not None:
            time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
            unique_times = df_long[time_col].unique().tolist()
            
            # PCA components
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
            
            # ICA components
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
    # If two params start with same 3 letters, drop the shorter one
    names_to_drop = set()
    
    for entry1 in master_data:
        name1 = entry1['name']
        prefix1 = name1[:3].upper()
        
        for entry2 in master_data:
            name2 = entry2['name']
            prefix2 = name2[:3].upper()
            
            # Same 3-letter prefix
            if prefix1 == prefix2 and name1 != name2:
                # Drop the shorter name
                if len(name1) < len(name2):
                    names_to_drop.add(name1)
                elif len(name2) < len(name1):
                    names_to_drop.add(name2)
    
    master_data = [entry for entry in master_data if entry['name'] not in names_to_drop]
    
    # === STEP 6: ORGANIZE - Parameters first, then Components ===
    parameters = [entry for entry in master_data if entry['type'] == 'parameter']
    components = [entry for entry in master_data if entry['type'] == 'component']
    
    # Sort each group alphabetically
    parameters = sorted(parameters, key=lambda x: x['name'])
    components = sorted(components, key=lambda x: x['name'])
    
    # Combine: parameters first, then components
    final_data = parameters + components
    
    # Convert to DataFrame
    df = pd.DataFrame(final_data)
    
    return df


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
    Get time range from data
    
    Returns:
    --------
    dict with:
        - min_date: earliest date in raw data
        - max_date: latest date in raw data
        - forecast_start: date where forecasting started
    """
    
    min_date = None
    max_date = None
    forecast_start = None
    
    # Get from raw data only
    if master_df is not None:
        raw_entries = master_df[master_df['source'] == 'raw']
        
        if len(raw_entries) > 0:
            all_timestamps = []
            for _, entry in raw_entries.iterrows():
                timestamps = entry['timestamps']
                all_timestamps.extend([pd.to_datetime(t) for t in timestamps])
            
            if all_timestamps:
                min_date = min(all_timestamps)
                max_date = max(all_timestamps)
    
    # Get forecast start
    if master_df is not None:
        forecast_entries = master_df[master_df['source'] == 'forecasted']
        
        if len(forecast_entries) > 0:
            all_forecast_starts = []
            for _, entry in forecast_entries.iterrows():
                timestamps = entry['timestamps']
                if timestamps:
                    all_forecast_starts.append(pd.to_datetime(timestamps[0]))
            
            if all_forecast_starts:
                forecast_start = min(all_forecast_starts)
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'forecast_start': forecast_start
    }


def check_data_at_baseline(param_name: str, baseline_date: pd.Timestamp, master_df: pd.DataFrame) -> dict:
    """
    Check if data exists for parameter at baseline date
    
    Returns:
    --------
    dict with:
        - exists: bool
        - source: str ('forecasted', 'cleaned', 'calculated', 'raw', 'none')
        - message: str
    """
    
    # Find parameter in master df
    param_entry = master_df[master_df['name'] == param_name]
    
    if len(param_entry) == 0:
        return {
            'exists': False,
            'source': 'none',
            'message': '❌ Parameter not found'
        }
    
    param_entry = param_entry.iloc[0]
    
    # Convert timestamps
    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
    
    if not timestamps:
        return {
            'exists': False,
            'source': 'none',
            'message': '❌ No data available'
        }
    
    min_time = min(timestamps)
    max_time = max(timestamps)
    
    # Check if baseline is within range
    if baseline_date < min_time or baseline_date > max_time:
        return {
            'exists': False,
            'source': param_entry['source'],
            'message': f"❌ No data at {baseline_date.year} (available: {min_time.year}-{max_time.year})"
        }
    
    # Data exists
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
    
    # Find parameter
    param_entry = master_df[master_df['name'] == param_name]
    
    if len(param_entry) == 0:
        return {'value': None, 'year': None, 'source': 'not_found'}
    
    param_entry = param_entry.iloc[0]
    
    # Get data
    timestamps = [pd.to_datetime(t) for t in param_entry['timestamps']]
    values = param_entry['values']
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    df = df.sort_values('timestamp')
    df['year'] = df['timestamp'].dt.year
    
    baseline_year = baseline_date.year
    
    # Find data for that year
    year_data = df[df['year'] == baseline_year]
    
    if len(year_data) > 0:
        baseline_value = year_data['value'].mean()
        return {
            'value': float(baseline_value),
            'year': baseline_year,
            'source': param_entry['source_full']
        }
    
    # Use closest year
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
    
    # === BUILD MASTER PARAMETER DATAFRAME ===
    with st.spinner("Building master parameter database..."):
        master_df = build_master_parameter_dataframe()
        st.session_state.master_parameter_df = master_df
    
    if master_df is None or len(master_df) == 0:
        st.warning("⚠️ No historical data available!")
        st.info("Please upload data first in **Upload & Data Diagnostics**")
        return
    
    st.success(f"✅ Found {len(scenario_params)} scenario parameters and {len(master_df)} historical parameters/components")
    
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
    
    # Calculate compatibility
    compatibility = calculate_compatibility(scenario_params, master_df, 
                                           st.session_state.parameter_mappings)
    
    # Show compatibility score
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
    
    st.markdown("---")
    
    # Display mapping table
    st.markdown("### 📋 Parameter Mapping Table")
    
    # Group by category
    params_by_category = {}
    for param in scenario_params:
        cat = param['category']
        if cat not in params_by_category:
            params_by_category[cat] = []
        params_by_category[cat].append(param)
    
    # Create mapping interface
    for category in sorted(params_by_category.keys()):
        params_in_category = params_by_category[category]
        
        with st.expander(f"**{category}** ({len(params_in_category)} parameters)", expanded=True):
            
            # Header
            col_h1, col_h2, col_h3 = st.columns([3, 4, 1])
            with col_h1:
                st.markdown("**Scenario Parameter**")
            with col_h2:
                st.markdown("**Historical Parameter / Component**")
            with col_h3:
                st.markdown("**Status**")
            
            st.markdown("---")
            
            # Each parameter
            for param in params_in_category:
                param_name = param['name']
                
                col1, col2, col3 = st.columns([3, 4, 1])
                
                with col1:
                    st.markdown(f"**{param_name}**")
                    if param.get('unit'):
                        st.caption(f"Unit: {param['unit']}")
                
                with col2:
                    current_mapping = st.session_state.parameter_mappings.get(param_name, 'None')
                    
                    # Build dropdown - Parameters first, then Components
                    options = ['None']
                    
                    # === PARAMETERS ===
                    parameters = master_df[master_df['type'] == 'parameter']
                    
                    if len(parameters) > 0:
                        options.append('─── PARAMETERS ───')
                        
                        for _, row in parameters.iterrows():
                            options.append(f"{row['symbol']} {row['name']}")
                    
                    # === COMPONENTS ===
                    components = master_df[master_df['type'] == 'component']
                    
                    if len(components) > 0:
                        options.append('─── COMPONENTS ───')
                        
                        for _, row in components.iterrows():
                            options.append(f"{row['symbol']} {row['name']}")
                    
                    # Find current selection
                    default_idx = 0
                    
                    # Try exact match with symbol
                    for i, opt in enumerate(options):
                        if opt.startswith(('🔮', '🧹', '📊', '🔬')):
                            clean_opt = opt.split(' ', 1)[1] if ' ' in opt else opt
                            if clean_opt == current_mapping:
                                default_idx = i
                                break
                    
                    # If not found, try without symbol
                    if default_idx == 0 and current_mapping in options:
                        default_idx = options.index(current_mapping)
                    
                    selected = st.selectbox(
                        "Select:",
                        options=options,
                        index=default_idx,
                        key=f"mapping_{param_name}_{category}",
                        label_visibility="collapsed"
                    )
                    
                    # Store mapping (clean the selection)
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
    
    # Show unmapped (only if actually unmapped)
    if compatibility['unmapped']:
        st.markdown("---")
        st.warning(f"⚠️ **{len(compatibility['unmapped'])} Unmapped Parameters**")
        st.caption(f"Unmapped: {', '.join(compatibility['unmapped'])}")
    else:
        st.success("🎉 **Perfect! All parameters mapped!**")
    
    # === STEP 2: BASELINE SELECTION ===
    if compatibility['score'] > 0:
        st.markdown("---")
        st.subheader("📍 Step 2: Select Baseline Date")
        st.caption("Choose the reference date for calculating percentage changes")
        
        # Get time range
        time_range = get_data_time_range(master_df)
        
        if time_range['min_date'] is None:
            st.error("❌ No time range data available")
            return
        
        st.info(f"""
        📅 **Data Time Range:**
        - Original data: {time_range['min_date'].strftime('%Y-%m-%d')} to {time_range['max_date'].strftime('%Y-%m-%d')}
        {'- Forecast starts: ' + time_range['forecast_start'].strftime('%Y-%m-%d') if time_range['forecast_start'] else '- No forecasts available'}
        """)
        
        # Determine valid range
        baseline_min = time_range['min_date']
        
        if time_range['forecast_start']:
            baseline_max = time_range['forecast_start']
        else:
            baseline_max = time_range['max_date']
        
        st.success(f"✅ Baseline must be between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}")
        
        # User enters date
        baseline_date = st.date_input(
            "Enter Baseline Date:",
            value=baseline_max.date(),
            min_value=baseline_min.date(),
            max_value=baseline_max.date(),
            help=f"Enter date between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}"
        )
        
        baseline_date = pd.Timestamp(baseline_date)
        
        st.caption(f"Selected baseline: **{baseline_date.strftime('%Y-%m-%d')}**")
        
        # === CONFIRMATION BUTTON ===
        st.markdown("---")
        
        if st.button("✅ Confirm Baseline Date", type="primary", use_container_width=True):
            st.session_state.confirmed_baseline_date = baseline_date
            st.success(f"✅ Baseline date confirmed: {baseline_date.strftime('%Y-%m-%d')}")
            st.rerun()
        
        # === DATA AVAILABILITY CHECK (only after confirmation) ===
        if 'confirmed_baseline_date' in st.session_state:
            confirmed_date = st.session_state.confirmed_baseline_date
            
            st.markdown("---")
            st.markdown("### 📊 Data Availability Check")
            st.caption(f"Checking data availability at baseline: {confirmed_date.strftime('%Y-%m-%d')}")
            
            # Check each mapped parameter
            availability_results = []
            missing_params = []
            
            for scenario_param, historical_param in st.session_state.parameter_mappings.items():
                if historical_param != 'None':
                    avail = check_data_at_baseline(historical_param, confirmed_date, master_df)
                    
                    availability_results.append({
                        'Scenario Parameter': scenario_param,
                        'Historical Parameter': historical_param,
                        'Status': avail['message'],
                        'Source': avail['source']
                    })
                    
                    if not avail['exists']:
                        missing_params.append((scenario_param, historical_param, avail))
            
            # Display table
            df_avail = pd.DataFrame(availability_results)
            st.dataframe(df_avail, use_container_width=True, hide_index=True)
            
            # Show warnings for missing data
            if missing_params:
                st.markdown("---")
                st.warning(f"⚠️ **{len(missing_params)} parameter(s) have no data at baseline date**")
                
                st.markdown("**Parameters with missing baseline data:**")
                for scenario_param, historical_param, avail in missing_params:
                    st.error(f"❌ **{scenario_param}** → {historical_param}: {avail['message']}")
                
                st.info("""
                💡 **To fix this:**
                
                Use the **Forecasting and Future Projections** functionality to model these parameters at the baseline time.
                
                **Steps:**
                1. Go to Time-Based Models & ML Training
                2. Train models for missing parameters
                3. Go to Future Projections to generate forecasts
                4. Return here - forecasted data will be available at baseline
                """)
            else:
                st.success("✅ All parameters have data at baseline!")
            
            # === EXTRACT BASELINES ===
            st.markdown("---")
            
            if st.button("🔍 Extract Baselines", type="primary"):
                with st.spinner("Extracting baseline values..."):
                    baseline_results = {}
                    
                    for scenario_param, historical_param in st.session_state.parameter_mappings.items():
                        if historical_param != 'None':
                            baseline = extract_baseline_value(historical_param, confirmed_date, master_df)
                            
                            baseline_results[scenario_param] = {
                                'historical_param': historical_param,
                                'value': baseline['value'],
                                'year': baseline['year'],
                                'source': baseline['source']
                            }
                    
                    st.session_state.baseline_references = baseline_results
                
                st.success(f"✅ Extracted {len(baseline_results)} baseline values!")
                st.rerun()
            
            # Display baseline results
            if st.session_state.baseline_references:
                st.markdown("### 📊 Baseline Values")
                
                baseline_data = []
                for scenario_param, baseline_info in st.session_state.baseline_references.items():
                    value_str = f"{baseline_info['value']:.2f}" if baseline_info['value'] is not None else "N/A"
                    year_str = str(baseline_info['year']) if baseline_info['year'] is not None else "N/A"
                    
                    baseline_data.append({
                        'Scenario Parameter': scenario_param,
                        'Historical Parameter': baseline_info['historical_param'],
                        'Baseline Value': value_str,
                        'Year': year_str,
                        'Source': baseline_info['source']
                    })
                
                df_baselines = pd.DataFrame(baseline_data)
                st.dataframe(df_baselines, use_container_width=True, hide_index=True)
                
                # === STEP 3: CONVERT TO ABSOLUTE ===
                st.markdown("---")
                st.subheader("🔢 Step 3: Convert Scenarios to Absolute Values")
                st.caption("Transform percentage-based changes to absolute values")
                
                if st.button("⚙️ Convert to Absolute Values", type="primary"):
                    with st.spinner("Converting..."):
                        absolute_results = {}
                        
                        for scenario in st.session_state.detected_scenarios:
                            scenario_name = scenario['title']
                            absolute_results[scenario_name] = {}
                            
                            for item in scenario['items']:
                                param_name = item.get('parameter_canonical', item.get('parameter', ''))
                                
                                if param_name in st.session_state.baseline_references:
                                    baseline_info = st.session_state.baseline_references[param_name]
                                    baseline_value = baseline_info['value']
                                    
                                    if baseline_value is None:
                                        continue
                                    
                                    scenario_value = item.get('value', 0)
                                    direction = item.get('direction', 'target')
                                    unit = item.get('unit', '')
                                    
                                    absolute_value = convert_scenario_to_absolute(
                                        param_name, scenario_value, direction, unit, baseline_value
                                    )
                                    
                                    absolute_results[scenario_name][param_name] = {
                                        'absolute_value': absolute_value,
                                        'original_value': scenario_value,
                                        'direction': direction,
                                        'unit': unit,
                                        'baseline': baseline_value,
                                        'category': item.get('category', 'Other')
                                    }
                        
                        st.session_state.absolute_values = absolute_results
                    
                    st.success(f"✅ Converted {len(absolute_results)} scenarios!")
                    st.rerun()
                
                # Display absolute values
                if st.session_state.absolute_values:
                    st.markdown("### 📊 Absolute Values by Scenario")
                    
                    for scenario_name, params in st.session_state.absolute_values.items():
                        with st.expander(f"**{scenario_name}** ({len(params)} parameters)", expanded=False):
                            abs_data = []
                            for param, info in params.items():
                                abs_data.append({
                                    'Parameter': param,
                                    'Category': info['category'],
                                    'Original': f"{info['direction']} {info['original_value']:.1f} {info['unit']}",
                                    'Baseline': f"{info['baseline']:.2f}",
                                    'Absolute Value': f"{info['absolute_value']:.2f}"
                                })
                            
                            df_abs = pd.DataFrame(abs_data)
                            st.dataframe(df_abs, use_container_width=True, hide_index=True)
                    
                    # === STEP 4: VISUALIZATION ===
                    st.markdown("---")
                    st.subheader("📊 Step 4: Trajectory Visualizations")
                    st.caption("Visualize scenario trajectories in parameter space")
                    
                    display_trajectory_visualizations()


def display_trajectory_visualizations():
    """Display trajectory visualizations"""
    
    if not st.session_state.absolute_values:
        st.info("Convert scenarios to absolute values first")
        return
    
    # Customization
    st.markdown("### 🎨 Visualization Customization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis_scale = st.selectbox("X-axis scale:", ["linear", "log"], key="x_scale")
        y_axis_scale = st.selectbox("Y-axis scale:", ["linear", "log"], key="y_scale")
    
    with col2:
        st.markdown("**Scenario Colors:**")
        scenario_colors = {}
        for scenario_name in st.session_state.absolute_values.keys():
            color = st.color_picker(
                f"{scenario_name[:15]}...",
                value="#1f77b4",
                key=f"color_{scenario_name}"
            )
            scenario_colors[scenario_name] = color
    
    with col3:
        box_opacity = st.slider("Box opacity:", 0.1, 1.0, 0.3, 0.1, key="box_opacity")
        show_forecast_points = st.checkbox("Show forecast points", value=True, key="show_forecast")
    
    st.markdown("---")
    
    # Select parameters
    all_params = set()
    for scenario_params in st.session_state.absolute_values.values():
        all_params.update(scenario_params.keys())
    
    all_params = sorted(list(all_params))
    
    if len(all_params) < 2:
        st.warning("Need at least 2 parameters for visualization")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox("X-axis parameter:", all_params, key="x_param")
    
    with col2:
        y_params_options = [p for p in all_params if p != x_param]
        y_param = st.selectbox("Y-axis parameter:", y_params_options, key="y_param")
    
    # Create plot
    fig = create_trajectory_plot(
        x_param, y_param, scenario_colors, 
        x_axis_scale, y_axis_scale,
        box_opacity, show_forecast_points
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        st.markdown("---")
        with st.expander("💾 Export Figure", expanded=False):
            if EXPORT_AVAILABLE:
                quick_export_buttons(
                    fig,
                    filename_prefix=f"trajectory_{x_param}_vs_{y_param}",
                    show_formats=['png', 'pdf', 'html']
                )
            else:
                import io
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    try:
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        st.download_button(
                            label="📥 PNG",
                            data=img_bytes,
                            file_name=f"trajectory_{x_param}_vs_{y_param}.png",
                            mime="image/png"
                        )
                    except:
                        st.caption("⚠️ Install kaleido")
                
                with col2:
                    try:
                        pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
                        st.download_button(
                            label="📥 PDF",
                            data=pdf_bytes,
                            file_name=f"trajectory_{x_param}_vs_{y_param}.pdf",
                            mime="application/pdf"
                        )
                    except:
                        st.caption("⚠️ Install kaleido")
                
                with col3:
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer)
                    st.download_button(
                        label="📥 HTML",
                        data=html_buffer.getvalue(),
                        file_name=f"trajectory_{x_param}_vs_{y_param}.html",
                        mime="text/html"
                    )


def create_trajectory_plot(x_param: str, y_param: str, scenario_colors: dict,
                          x_scale: str, y_scale: str, box_opacity: float,
                          show_forecast: bool) -> go.Figure:
    """Create trajectory plot"""
    
    fig = go.Figure()
    
    absolute_vals = st.session_state.absolute_values
    
    # Plot scenarios as boxes
    for scenario_name, params in absolute_vals.items():
        if x_param in params and y_param in params:
            x_val = params[x_param]['absolute_value']
            y_val = params[y_param]['absolute_value']
            
            # Box size (±10%)
            x_size = abs(x_val * 0.1)
            y_size = abs(y_val * 0.1)
            
            color = scenario_colors.get(scenario_name, '#1f77b4')
            
            # Add box
            fig.add_shape(
                type="rect",
                x0=x_val - x_size,
                x1=x_val + x_size,
                y0=y_val - y_size,
                y1=y_val + y_size,
                fillcolor=color,
                opacity=box_opacity,
                line=dict(color=color, width=2),
                name=scenario_name
            )
            
            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_val],
                y=[y_val],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='square'),
                text=[scenario_name],
                textposition="top center",
                name=scenario_name,
                showlegend=True
            ))
    
    # Plot forecast points if available
    if show_forecast:
        master_df = st.session_state.master_parameter_df
        
        # Find mapped historical params
        x_hist = st.session_state.parameter_mappings.get(x_param)
        y_hist = st.session_state.parameter_mappings.get(y_param)
        
        if x_hist and y_hist:
            x_entry = master_df[master_df['name'] == x_hist]
            y_entry = master_df[master_df['name'] == y_hist]
            
            if len(x_entry) > 0 and len(y_entry) > 0:
                x_entry = x_entry.iloc[0]
                y_entry = y_entry.iloc[0]
                
                if x_entry['source'] == 'forecasted' and y_entry['source'] == 'forecasted':
                    # Merge data
                    x_df = pd.DataFrame({
                        'timestamp': x_entry['timestamps'],
                        'x_value': x_entry['values']
                    })
                    
                    y_df = pd.DataFrame({
                        'timestamp': y_entry['timestamps'],
                        'y_value': y_entry['values']
                    })
                    
                    merged = pd.merge(x_df, y_df, on='timestamp')
                    
                    fig.add_trace(go.Scatter(
                        x=merged['x_value'],
                        y=merged['y_value'],
                        mode='markers',
                        marker=dict(size=6, color='black', symbol='circle'),
                        name='Forecast Points',
                        showlegend=True,
                        opacity=0.6
                    ))
    
    # Layout
    fig.update_layout(
        title=f"Trajectory Space: {x_param} vs {y_param}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        xaxis_type=x_scale,
        yaxis_type=y_scale,
        height=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


if __name__ == "__main__":
    main()
