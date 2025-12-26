"""
Page 11: Trajectory-Scenario Space
Map historical parameters to scenario parameters with intelligent data source prioritization,
assess compatibility, extract baselines, and visualize the trajectory space
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cosine

# Page configuration - MUST be first Streamlit command
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
    """Initialize session state for trajectory analysis"""
    if "parameter_mappings" not in st.session_state:
        st.session_state.parameter_mappings = {}
    if "compatibility_score" not in st.session_state:
        st.session_state.compatibility_score = None
    if "baseline_references" not in st.session_state:
        st.session_state.baseline_references = {}
    if "absolute_values" not in st.session_state:
        st.session_state.absolute_values = {}


def fuzzy_match_parameter(target: str, candidates: list) -> str:
    """
    Fuzzy match parameter name to find similar names
    E.g., "CO2" matches "CO2_cleaned", "CO2_outlier_treated"
    
    Returns the best match or None
    """
    target_lower = target.lower()
    
    # Exact match first
    for candidate in candidates:
        if candidate.lower() == target_lower:
            return candidate
    
    # Check if target is a prefix of candidate
    for candidate in candidates:
        if candidate.lower().startswith(target_lower):
            return candidate
    
    # Check if candidate starts with target (without underscores)
    target_base = target_lower.replace('_', '').replace('-', '').replace(' ', '')
    for candidate in candidates:
        candidate_base = candidate.lower().replace('_', '').replace('-', '').replace(' ', '')
        if candidate_base.startswith(target_base):
            return candidate
    
    return None


def get_data_source_for_parameter(param_name: str, for_baseline: bool = False) -> dict:
    """
    Get data source for a parameter with STRICT priority:
    
    FOR REGULAR PARAMETERS:
    1. Forecasted data (if exists, STOP - don't check others)
    2. Cleaned data (with fuzzy matching, if exists, STOP)
    3. Raw data (if exists, STOP)
    
    FOR COMPONENTS (PC1, IC1, Factor1, etc):
    1. Forecasted components (if exists, STOP)
    2. Calculated components from Page 5 (if exists, STOP)
    
    Parameters:
    -----------
    param_name : str
        Parameter name
    for_baseline : bool
        If True, returns additional metadata about data availability
    
    Returns:
    --------
    dict with:
        - data: pd.DataFrame with columns [timestamp, value]
        - source: str ('forecasted', 'component', 'cleaned', 'raw')
        - symbol: str emoji
        - full_name: str
        - available_at_baseline: bool (only if for_baseline=True)
    """
    
    # Check if this is a component
    is_component = any(comp in param_name.upper() for comp in ['PC', 'IC', 'FACTOR'])
    
    if is_component:
        # === COMPONENT PRIORITY ===
        
        # Priority 1: Forecasted component
        if 'forecast_results' in st.session_state and param_name in st.session_state.forecast_results:
            forecast = st.session_state.forecast_results[param_name]
            df = pd.DataFrame({
                'timestamp': forecast['forecast_timestamps'],
                'value': forecast['forecast_values']
            })
            return {
                'data': df,
                'source': 'forecasted',
                'symbol': '🔮',
                'full_name': 'Forecasted Component',
                'available_at_baseline': True  # Forecast always has baseline
            }
        
        # Priority 2: Calculated component from Page 5
        if 'reduction_results' in st.session_state and st.session_state.reduction_results:
            # PCA components
            if 'PC' in param_name.upper() and 'pca' in st.session_state.reduction_results:
                pca_results = st.session_state.reduction_results['pca']
                n_components = pca_results.get('n_components', 0)
                
                for i in range(n_components):
                    if param_name in [f"PC{i+1}", f"PC{i+1} [PCA]"]:
                        if 'pca_components' in st.session_state:
                            component_values = st.session_state.pca_components[:, i]
                            
                            df_long = st.session_state.get('df_long')
                            if df_long is not None:
                                time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
                                unique_times = df_long[time_col].unique()
                                
                                df = pd.DataFrame({
                                    'timestamp': unique_times[:len(component_values)],
                                    'value': component_values
                                })
                                
                                return {
                                    'data': df,
                                    'source': 'component',
                                    'symbol': '🔬',
                                    'full_name': 'Calculated PCA Component',
                                    'available_at_baseline': False  # Only historical
                                }
            
            # ICA components
            if 'IC' in param_name.upper() and 'ica' in st.session_state.reduction_results:
                # Similar logic for ICA
                pass
            
            # Factor Analysis
            if 'FACTOR' in param_name.upper() and 'factor_analysis' in st.session_state.reduction_results:
                # Similar logic for Factor
                pass
    
    else:
        # === REGULAR PARAMETER PRIORITY ===
        
        # Priority 1: Forecasted data (STOP if found)
        if 'forecast_results' in st.session_state and param_name in st.session_state.forecast_results:
            forecast = st.session_state.forecast_results[param_name]
            df = pd.DataFrame({
                'timestamp': forecast['forecast_timestamps'],
                'value': forecast['forecast_values']
            })
            return {
                'data': df,
                'source': 'forecasted',
                'symbol': '🔮',
                'full_name': 'Forecasted',
                'available_at_baseline': True
            }
        
        # Priority 2: Cleaned data with fuzzy matching (STOP if found)
        if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
            df_clean = st.session_state.df_clean
            all_cleaned_vars = df_clean['variable'].unique().tolist()
            
            # Try fuzzy match
            matched_var = fuzzy_match_parameter(param_name, all_cleaned_vars)
            
            if matched_var:
                param_data = df_clean[df_clean['variable'] == matched_var].copy()
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                df = pd.DataFrame({
                    'timestamp': param_data[time_col],
                    'value': param_data['value']
                })
                
                return {
                    'data': df,
                    'source': 'cleaned',
                    'symbol': '🧹',
                    'full_name': f'Cleaned ({matched_var})',
                    'available_at_baseline': False
                }
        
        # Priority 3: Raw data (STOP if found)
        if 'df_long' in st.session_state and st.session_state.df_long is not None:
            df_long = st.session_state.df_long
            
            if param_name in df_long['variable'].values:
                param_data = df_long[df_long['variable'] == param_name].copy()
                time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
                
                df = pd.DataFrame({
                    'timestamp': param_data[time_col],
                    'value': param_data['value']
                })
                
                return {
                    'data': df,
                    'source': 'raw',
                    'symbol': '📊',
                    'full_name': 'Raw',
                    'available_at_baseline': False
                }
    
    # Not found
    return {
        'data': None,
        'source': 'not_found',
        'symbol': '❌',
        'full_name': 'Not Found',
        'available_at_baseline': False
    }


def get_all_available_parameters() -> dict:
    """
    Get all available parameters organized by type and source
    
    Returns:
    --------
    dict with:
        - regular_params: list of regular parameter names
        - components: list of component names
        - source_info: dict mapping parameter -> source info
    """
    
    regular_params = set()
    components = set()
    source_info = {}
    
    # Get all from forecast
    if 'forecast_results' in st.session_state:
        for param in st.session_state.forecast_results.keys():
            if any(comp in param.upper() for comp in ['PC', 'IC', 'FACTOR']):
                components.add(param)
            else:
                regular_params.add(param)
    
    # Get all from cleaned
    if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
        for param in st.session_state.df_clean['variable'].unique():
            if param not in regular_params and param not in components:
                if any(comp in param.upper() for comp in ['PC', 'IC', 'FACTOR']):
                    components.add(param)
                else:
                    regular_params.add(param)
    
    # Get all from raw
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        for param in st.session_state.df_long['variable'].unique():
            if param not in regular_params and param not in components:
                if any(comp in param.upper() for comp in ['PC', 'IC', 'FACTOR']):
                    components.add(param)
                else:
                    regular_params.add(param)
    
    # Get all from reduction results
    if 'reduction_results' in st.session_state:
        if 'pca' in st.session_state.reduction_results:
            n = st.session_state.reduction_results['pca'].get('n_components', 0)
            for i in range(n):
                components.add(f"PC{i+1}")
        
        if 'ica' in st.session_state.reduction_results:
            n = st.session_state.reduction_results['ica'].get('n_components', 0)
            for i in range(n):
                components.add(f"IC{i+1}")
        
        if 'factor_analysis' in st.session_state.reduction_results:
            n = st.session_state.reduction_results['factor_analysis'].get('n_factors', 0)
            for i in range(n):
                components.add(f"Factor{i+1}")
    
    # Get source info for each
    for param in list(regular_params) + list(components):
        source_info[param] = get_data_source_for_parameter(param)
    
    return {
        'regular_params': sorted(list(regular_params)),
        'components': sorted(list(components)),
        'source_info': source_info
    }


def get_data_time_range() -> dict:
    """
    Get the time range of original data and forecast point
    
    Returns:
    --------
    dict with:
        - min_date: earliest date in original data
        - max_date: latest date in original data
        - forecast_start: date where forecasting started
        - has_forecast: bool
    """
    
    min_date = None
    max_date = None
    forecast_start = None
    
    # Get from original data
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        df = st.session_state.df_long
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        
        df[time_col] = pd.to_datetime(df[time_col])
        min_date = df[time_col].min()
        max_date = df[time_col].max()
    
    # Get forecast start point
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        # Get from any forecast
        first_forecast = list(st.session_state.forecast_results.values())[0]
        forecast_timestamps = first_forecast['forecast_timestamps']
        forecast_start = pd.to_datetime(forecast_timestamps[0])
    
    return {
        'min_date': min_date,
        'max_date': max_date,
        'forecast_start': forecast_start,
        'has_forecast': forecast_start is not None
    }


def check_baseline_data_availability(param_name: str, baseline_date: pd.Timestamp) -> dict:
    """
    Check if data is available for parameter at baseline date
    
    Returns:
    --------
    dict with:
        - available: bool
        - source: str
        - message: str
        - suggestion: str
    """
    
    source_info = get_data_source_for_parameter(param_name, for_baseline=True)
    
    if source_info['source'] == 'forecasted':
        # Forecasted data - always available
        return {
            'available': True,
            'source': 'forecasted',
            'message': f"✅ Data available from forecast",
            'suggestion': None
        }
    
    elif source_info['source'] in ['component', 'cleaned', 'raw']:
        # Historical data only - check if baseline is within range
        df = source_info['data']
        
        if df is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            
            if baseline_date < min_date or baseline_date > max_date:
                return {
                    'available': False,
                    'source': source_info['source'],
                    'message': f"⚠️ No data at {baseline_date.year} (available: {min_date.year}-{max_date.year})",
                    'suggestion': "Consider forecasting this parameter in 'Future Projections' page"
                }
            else:
                return {
                    'available': True,
                    'source': source_info['source'],
                    'message': f"✅ Data available from {source_info['source']} data",
                    'suggestion': None
                }
    
    return {
        'available': False,
        'source': 'not_found',
        'message': "❌ Parameter not found",
        'suggestion': "Check parameter mapping"
    }


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


def calculate_compatibility(scenario_params: list, available_params: dict, mappings: dict) -> dict:
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


def extract_baseline_value(historical_param: str, baseline_date: pd.Timestamp) -> dict:
    """Extract baseline value at specific date"""
    
    source_info = get_data_source_for_parameter(historical_param)
    
    if source_info['data'] is None:
        return {'value': None, 'year': None, 'source': 'not_found', 'available': False}
    
    df = source_info['data']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Extract year from baseline_date
    baseline_year = baseline_date.year
    df['year'] = df['timestamp'].dt.year
    
    # Find data for that year
    year_data = df[df['year'] == baseline_year]
    
    if len(year_data) > 0:
        baseline_value = year_data['value'].mean()
        return {
            'value': float(baseline_value),
            'year': baseline_year,
            'source': source_info['full_name'],
            'available': True
        }
    
    # Year not found - use closest
    closest_year = df['year'].iloc[(df['year'] - baseline_year).abs().argsort()[0]]
    closest_data = df[df['year'] == closest_year]
    baseline_value = closest_data['value'].mean()
    
    return {
        'value': float(baseline_value),
        'year': closest_year,
        'source': f"{source_info['full_name']} (closest: {closest_year})",
        'available': True
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
    
    # Get all available parameters
    available_params = get_all_available_parameters()
    
    if not available_params['regular_params'] and not available_params['components']:
        st.warning("⚠️ No historical data available!")
        st.info("Please upload data first in **Upload & Data Diagnostics**")
        return
    
    st.success(f"✅ Found {len(scenario_params)} scenario parameters and {len(available_params['regular_params']) + len(available_params['components'])} historical parameters")
    
    st.markdown("---")
    
    # === STEP 1: PARAMETER MAPPING ===
    st.subheader("🔗 Step 1: Map Scenario Parameters to Historical Data")
    st.caption("Match each scenario parameter to its equivalent in historical data")
    
    # Calculate compatibility
    all_available = available_params['regular_params'] + available_params['components']
    source_info_dict = available_params['source_info']
    
    compatibility = calculate_compatibility(scenario_params, available_params, 
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
                    
                    # Build dropdown with STRICT ordering:
                    # 1. Regular parameters (forecasted > cleaned > raw)
                    # 2. Components (forecasted > calculated)
                    
                    options = ['None']
                    
                    # === REGULAR PARAMETERS ===
                    forecasted_params = [p for p in available_params['regular_params'] 
                                        if source_info_dict[p]['source'] == 'forecasted']
                    cleaned_params = [p for p in available_params['regular_params']
                                     if source_info_dict[p]['source'] == 'cleaned']
                    raw_params = [p for p in available_params['regular_params']
                                 if source_info_dict[p]['source'] == 'raw']
                    
                    if forecasted_params:
                        options.append('─── 🔮 Forecasted Parameters ───')
                        for p in sorted(forecasted_params):
                            options.append(f"🔮 {p}")
                    
                    if cleaned_params:
                        options.append('─── 🧹 Cleaned Parameters ───')
                        for p in sorted(cleaned_params):
                            options.append(f"🧹 {p}")
                    
                    if raw_params:
                        options.append('─── 📊 Raw Parameters ───')
                        for p in sorted(raw_params):
                            options.append(f"📊 {p}")
                    
                    # === COMPONENTS ===
                    forecasted_comps = [c for c in available_params['components']
                                       if source_info_dict[c]['source'] == 'forecasted']
                    calculated_comps = [c for c in available_params['components']
                                       if source_info_dict[c]['source'] == 'component']
                    
                    if forecasted_comps:
                        options.append('─── 🔮 Forecasted Components ───')
                        for c in sorted(forecasted_comps):
                            options.append(f"🔮 {c}")
                    
                    if calculated_comps:
                        options.append('─── 🔬 Calculated Components ───')
                        for c in sorted(calculated_comps):
                            if 'PC' in c:
                                options.append(f"🔬 {c} [PCA]")
                            elif 'IC' in c:
                                options.append(f"🔬 {c} [ICA]")
                            elif 'Factor' in c:
                                options.append(f"🔬 {c} [Factor]")
                            else:
                                options.append(f"🔬 {c}")
                    
                    # Find current selection
                    default_idx = 0
                    if current_mapping in options:
                        default_idx = options.index(current_mapping)
                    else:
                        for i, opt in enumerate(options):
                            if opt.startswith(('🔮', '🧹', '📊', '🔬')):
                                clean_opt = opt.split(' ', 1)[1] if ' ' in opt else opt
                                clean_opt = clean_opt.split(' [')[0] if ' [' in clean_opt else clean_opt
                                if clean_opt == current_mapping:
                                    default_idx = i
                                    break
                    
                    selected = st.selectbox(
                        "Select:",
                        options=options,
                        index=default_idx,
                        key=f"mapping_{param_name}_{category}",
                        label_visibility="collapsed"
                    )
                    
                    # Store mapping
                    if not selected.startswith('───'):
                        if selected.startswith(('🔮', '🧹', '📊', '🔬')):
                            clean_selected = selected.split(' ', 1)[1]
                            clean_selected = clean_selected.split(' [')[0] if ' [' in clean_selected else clean_selected
                            st.session_state.parameter_mappings[param_name] = clean_selected
                        else:
                            st.session_state.parameter_mappings[param_name] = selected
                
                with col3:
                    if selected != 'None' and not selected.startswith('───'):
                        st.success("✓")
                    else:
                        st.warning("⚠️")
                
                st.markdown("")
    
    # Show unmapped
    if compatibility['unmapped']:
        st.markdown("---")
        st.warning(f"⚠️ **{len(compatibility['unmapped'])} Unmapped Parameters**")
        
        st.markdown("**Suggested Actions:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Option 1: Add Historical Parameters**
            - Upload additional data
            - Include equivalent parameters
            - Re-run mapping
            """)
        
        with col2:
            st.info("""
            **Option 2: Modify Scenarios**
            - Go to Page 9
            - Remove/merge unmapped parameters
            - Focus on mapped parameters
            """)
    else:
        st.success("🎉 **Perfect! All parameters mapped!**")
    
    # === STEP 2: BASELINE SELECTION ===
    if compatibility['score'] > 0:
        st.markdown("---")
        st.subheader("📍 Step 2: Select Baseline Date")
        st.caption("Choose the reference date for calculating percentage changes")
        
        # Get time range
        time_range = get_data_time_range()
        
        if time_range['min_date'] is None:
            st.error("❌ No time range data available")
            return
        
        st.info(f"""
        📅 **Data Time Range:**
        - Original data: {time_range['min_date'].strftime('%Y-%m-%d')} to {time_range['max_date'].strftime('%Y-%m-%d')}
        {'- Forecast starts: ' + time_range['forecast_start'].strftime('%Y-%m-%d') if time_range['has_forecast'] else '- No forecasts available'}
        """)
        
        # Determine baseline range
        baseline_min = time_range['min_date']
        
        if time_range['has_forecast']:
            baseline_max = time_range['forecast_start']
            st.success(f"✅ Baseline must be between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}")
        else:
            baseline_max = time_range['max_date']
            st.info(f"ℹ️ Baseline can be between {baseline_min.strftime('%Y-%m-%d')} and {baseline_max.strftime('%Y-%m-%d')}")
        
        # Calculate date range in years
        year_min = baseline_min.year
        year_max = baseline_max.year
        
        # Slider for baseline year
        baseline_year = st.slider(
            "Baseline Year:",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            step=1,
            help=f"Select baseline year between {year_min} and {year_max}"
        )
        
        # Convert to timestamp
        baseline_date = pd.Timestamp(year=baseline_year, month=1, day=1)
        
        st.caption(f"Selected baseline: **{baseline_date.strftime('%Y-%m-%d')}**")
        
        # Check data availability for each mapped parameter
        st.markdown("---")
        st.markdown("### 📊 Data Availability Check")
        
        availability_results = {}
        missing_params = []
        
        for scenario_param, historical_param in st.session_state.parameter_mappings.items():
            if historical_param != 'None':
                avail = check_baseline_data_availability(historical_param, baseline_date)
                availability_results[scenario_param] = avail
                
                if not avail['available']:
                    missing_params.append((scenario_param, historical_param, avail))
        
        # Display availability table
        avail_data = []
        for scenario_param, avail in availability_results.items():
            historical_param = st.session_state.parameter_mappings[scenario_param]
            avail_data.append({
                'Scenario Parameter': scenario_param,
                'Historical Parameter': historical_param,
                'Status': avail['message'],
                'Source': avail['source']
            })
        
        df_avail = pd.DataFrame(avail_data)
        st.dataframe(df_avail, use_container_width=True, hide_index=True)
        
        # Show warnings for missing data
        if missing_params:
            st.markdown("---")
            st.warning(f"⚠️ **{len(missing_params)} parameter(s) have no data at baseline year {baseline_year}**")
            
            st.markdown("**Parameters with missing baseline data:**")
            for scenario_param, historical_param, avail in missing_params:
                with st.expander(f"❌ {scenario_param} → {historical_param}"):
                    st.error(avail['message'])
                    if avail['suggestion']:
                        st.info(f"💡 **Suggestion:** {avail['suggestion']}")
                        
                        # Link to relevant pages
                        st.markdown("**Steps to fix:**")
                        st.markdown("1. Go to **Time-Based Models & ML Training** (Page 6)")
                        st.markdown("2. Train a model for this parameter")
                        st.markdown("3. Go to **Future Projections** (Page 8) to generate forecast")
                        st.markdown("4. Return here to extract baseline from forecast")
        
        # Extract baselines button
        st.markdown("---")
        
        if st.button("🔍 Extract Baselines", type="primary"):
            with st.spinner("Extracting baseline values..."):
                baseline_results = {}
                
                for scenario_param, historical_param in st.session_state.parameter_mappings.items():
                    if historical_param != 'None':
                        baseline = extract_baseline_value(historical_param, baseline_date)
                        
                        baseline_results[scenario_param] = {
                            'historical_param': historical_param,
                            'value': baseline['value'],
                            'year': baseline['year'],
                            'source': baseline['source'],
                            'available': baseline['available']
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
                
                # === STEP 4: TRAJECTORY VISUALIZATION ===
                st.markdown("---")
                st.subheader("📊 Step 4: Trajectory Visualizations")
                st.caption("Visualize scenario trajectories in parameter space")
                
                display_trajectory_visualizations()


def display_trajectory_visualizations():
    """Display trajectory visualizations with customization"""
    
    if not st.session_state.absolute_values:
        st.info("Convert scenarios to absolute values first")
        return
    
    # Customization options
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
    
    # Select parameters to visualize
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
        
        # Export options
        st.markdown("---")
        with st.expander("💾 Export Figure", expanded=False):
            if EXPORT_AVAILABLE:
                quick_export_buttons(
                    fig,
                    filename_prefix=f"trajectory_{x_param}_vs_{y_param}",
                    show_formats=['png', 'pdf', 'html']
                )
            else:
                # Fallback export
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
    """Create trajectory plot with boxes for scenarios and points for forecasts"""
    
    fig = go.Figure()
    
    absolute_vals = st.session_state.absolute_values
    
    # Plot each scenario as a box
    for scenario_name, params in absolute_vals.items():
        if x_param in params and y_param in params:
            x_val = params[x_param]['absolute_value']
            y_val = params[y_param]['absolute_value']
            
            # Calculate box size (±10% uncertainty)
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
        x_source = get_data_source_for_parameter(x_param)
        y_source = get_data_source_for_parameter(y_param)
        
        if (x_source['source'] == 'forecasted' and 
            y_source['source'] == 'forecasted'):
            
            x_data = x_source['data']
            y_data = y_source['data']
            
            # Merge on timestamp
            merged = pd.merge(x_data, y_data, on='timestamp', suffixes=('_x', '_y'))
            
            fig.add_trace(go.Scatter(
                x=merged['value_x'],
                y=merged['value_y'],
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
