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

# Try to import export functions (same as page 4)
try:
    from core.viz.export import quick_export_buttons
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Initialize
initialize_session_state()

# Render shared sidebar
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


def get_data_source_for_parameter(param_name: str) -> dict:
    """
    Get data source for a parameter following priority:
    1. Forecasted data (Page 8)
    2. Component data (Page 5) for PCA/ICA/Factor
    3. Cleaned data (Page 2)
    4. Raw data (Page 1)
    
    Returns:
    --------
    dict with:
        - data: pd.DataFrame with columns [timestamp, value]
        - source: str ('forecasted', 'component', 'cleaned', 'raw')
        - symbol: str emoji for UI
        - full_name: str description
    """
    
    # Priority 1: Forecasted data
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
            'full_name': 'Forecasted (Future Projections)'
        }
    
    # Priority 2: Component data (PCA, ICA, Factor)
    if 'reduction_results' in st.session_state and st.session_state.reduction_results:
        # Check if this is a PCA component
        if 'pca' in st.session_state.reduction_results and 'pca_components' in st.session_state:
            pca_results = st.session_state.reduction_results['pca']
            n_components = pca_results.get('n_components', 0)
            
            # Match component names
            for i in range(n_components):
                if param_name in [f"PC{i+1}", f"PC{i+1} [PCA]"]:
                    # Get component data
                    if hasattr(st.session_state, 'pca_components'):
                        component_values = st.session_state.pca_components[:, i]
                        
                        # Get timestamps from original data
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
                                'full_name': 'Component (Dimensionality Reduction)'
                            }
    
    # Priority 3: Cleaned data
    if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
        df_clean = st.session_state.df_clean
        
        # Check if parameter exists in cleaned data
        if param_name in df_clean['variable'].values:
            param_data = df_clean[df_clean['variable'] == param_name].copy()
            time_col = 'timestamp' if 'timestamp' in param_data.columns else 'time'
            
            df = pd.DataFrame({
                'timestamp': param_data[time_col],
                'value': param_data['value']
            })
            
            return {
                'data': df,
                'source': 'cleaned',
                'symbol': '🧹',
                'full_name': 'Cleaned (Data Preprocessing)'
            }
    
    # Priority 4: Raw data
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
                'full_name': 'Raw (Upload & Diagnostics)'
            }
    
    # Not found
    return {
        'data': None,
        'source': 'not_found',
        'symbol': '❌',
        'full_name': 'Not Found'
    }


def get_historical_parameters_with_source():
    """
    Get all parameters from historical data with source information
    
    Returns:
    --------
    dict with parameter names as keys and source info as values
    """
    
    params_with_source = {}
    
    # Get all unique parameter names from all sources
    all_param_names = set()
    
    # From forecasts
    if 'forecast_results' in st.session_state:
        all_param_names.update(st.session_state.forecast_results.keys())
    
    # From components
    if 'reduction_results' in st.session_state and st.session_state.reduction_results:
        if 'pca' in st.session_state.reduction_results:
            n_components = st.session_state.reduction_results['pca'].get('n_components', 0)
            for i in range(n_components):
                all_param_names.add(f"PC{i+1}")
        
        if 'factor_analysis' in st.session_state.reduction_results:
            n_factors = st.session_state.reduction_results['factor_analysis'].get('n_factors', 0)
            for i in range(n_factors):
                all_param_names.add(f"Factor{i+1}")
        
        if 'ica' in st.session_state.reduction_results:
            n_components = st.session_state.reduction_results['ica'].get('n_components', 0)
            for i in range(n_components):
                all_param_names.add(f"IC{i+1}")
    
    # From cleaned data
    if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
        all_param_names.update(st.session_state.df_clean['variable'].unique())
    
    # From raw data
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        all_param_names.update(st.session_state.df_long['variable'].unique())
    
    # Get source for each parameter
    for param in all_param_names:
        source_info = get_data_source_for_parameter(param)
        if source_info['source'] != 'not_found':
            params_with_source[param] = source_info
    
    return params_with_source


def get_scenario_parameters():
    """
    Get parameters from detected scenarios (Page 9)
    Priority: Saved table > Current table
    
    Returns:
    --------
    list of dicts with parameter info
    """
    
    if 'detected_scenarios' not in st.session_state or not st.session_state.detected_scenarios:
        return []
    
    scenarios = st.session_state.detected_scenarios
    
    # Collect unique parameters across all scenarios
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
            
            # Add scenario info
            params[param_name]['scenarios'].append({
                'scenario': scenario['title'],
                'value': item.get('value'),
                'direction': item.get('direction', ''),
                'unit': item.get('unit', '')
            })
    
    return list(params.values())


def calculate_compatibility(scenario_params: list, historical_params_with_source: dict, mappings: dict) -> dict:
    """
    Calculate compatibility between scenario and historical parameters
    
    Returns:
    --------
    dict with score and details
    """
    
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


def extract_baseline_value(historical_param: str, reference_year: int = None, use_forecast: bool = True) -> dict:
    """
    Extract baseline value for a parameter at a specific year
    
    Parameters:
    -----------
    historical_param : str
        Historical parameter name
    reference_year : int
        Year to extract baseline from
    use_forecast : bool
        Whether to use forecast if year is beyond historical data
        
    Returns:
    --------
    dict with:
        - value: float
        - year: int
        - source: str
    """
    
    # Get data source
    source_info = get_data_source_for_parameter(historical_param)
    
    if source_info['data'] is None:
        return {'value': None, 'year': reference_year, 'source': 'not_found'}
    
    df = source_info['data']
    
    # Convert to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Extract year
    df['year'] = df['timestamp'].dt.year
    
    # If reference year not specified, use last year
    if reference_year is None:
        reference_year = df['year'].max()
    
    # Try to find data for that year
    year_data = df[df['year'] == reference_year]
    
    if len(year_data) > 0:
        # Take mean of values in that year
        baseline_value = year_data['value'].mean()
        return {
            'value': float(baseline_value),
            'year': reference_year,
            'source': source_info['full_name']
        }
    
    # Year not found - use closest available
    closest_year = df['year'].iloc[(df['year'] - reference_year).abs().argsort()[0]]
    closest_data = df[df['year'] == closest_year]
    
    baseline_value = closest_data['value'].mean()
    
    return {
        'value': float(baseline_value),
        'year': closest_year,
        'source': f"{source_info['full_name']} (closest: {closest_year})"
    }


def convert_scenario_to_absolute(param_name: str, scenario_value: float, direction: str, 
                                 unit: str, baseline_value: float) -> float:
    """
    Convert scenario description to absolute value
    
    Parameters:
    -----------
    param_name : str
        Parameter name
    scenario_value : float
        Scenario value (e.g., 20 for "20% increase")
    direction : str
        Direction (increase, decrease, target, stable, etc.)
    unit : str
        Unit (%, absolute, etc.)
    baseline_value : float
        Baseline reference value
        
    Returns:
    --------
    absolute_value : float
    """
    
    if unit == '%' or unit == 'percent':
        # Percentage change
        if direction == 'increase':
            return baseline_value * (1 + scenario_value / 100)
        elif direction == 'decrease':
            return baseline_value * (1 - scenario_value / 100)
        elif direction == 'target':
            return scenario_value  # Already absolute
        else:
            return baseline_value
    
    elif direction == 'double':
        return baseline_value * 2
    
    elif direction == 'halve':
        return baseline_value * 0.5
    
    elif direction == 'stable':
        return baseline_value
    
    else:
        # Absolute value
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
    
    # Get historical parameters with source
    historical_params_with_source = get_historical_parameters_with_source()
    
    if not historical_params_with_source:
        st.warning("⚠️ No historical data available!")
        st.info("Please upload data first in **Upload & Data Diagnostics**")
        return
    
    st.success(f"✅ Found {len(scenario_params)} scenario parameters and {len(historical_params_with_source)} historical parameters")
    
    st.markdown("---")
    
    # === STEP 1: PARAMETER MAPPING ===
    st.subheader("🔗 Step 1: Map Scenario Parameters to Historical Data")
    st.caption("Match each scenario parameter to its equivalent in historical data")
    
    # Calculate compatibility
    compatibility = calculate_compatibility(scenario_params, historical_params_with_source, 
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
    
    # Create mapping interface for each category
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
                    
                    # Build options with symbols
                    options = ['None']
                    
                    # Priority order in dropdown
                    forecasted = [(name, info) for name, info in historical_params_with_source.items() 
                                 if info['source'] == 'forecasted']
                    components = [(name, info) for name, info in historical_params_with_source.items() 
                                 if info['source'] == 'component']
                    cleaned = [(name, info) for name, info in historical_params_with_source.items() 
                               if info['source'] == 'cleaned']
                    raw = [(name, info) for name, info in historical_params_with_source.items() 
                          if info['source'] == 'raw']
                    
                    if forecasted:
                        options.append('─── 🔮 Forecasted (Future Projections) ───')
                        for name, info in sorted(forecasted):
                            options.append(f"🔮 {name}")
                    
                    if components:
                        options.append('─── 🔬 Components (PCA/ICA/Factor) ───')
                        for name, info in sorted(components):
                            # Clean up component name
                            if 'Principal Component' in name:
                                clean_name = name.replace(' (Principal Component)', '')
                                options.append(f"🔬 {clean_name} [PCA]")
                            elif 'Latent Factor' in name:
                                clean_name = name.replace(' (Latent Factor)', '')
                                options.append(f"🔬 {clean_name} [Factor]")
                            elif 'Independent Component' in name:
                                clean_name = name.replace(' (Independent Component)', '')
                                options.append(f"🔬 {clean_name} [ICA]")
                            else:
                                options.append(f"🔬 {name}")
                    
                    if cleaned:
                        options.append('─── 🧹 Cleaned (Data Preprocessing) ───')
                        for name, info in sorted(cleaned):
                            options.append(f"🧹 {name}")
                    
                    if raw:
                        options.append('─── 📊 Raw (Upload & Diagnostics) ───')
                        for name, info in sorted(raw):
                            options.append(f"📊 {name}")
                    
                    # Find current selection
                    default_idx = 0
                    if current_mapping in options:
                        default_idx = options.index(current_mapping)
                    else:
                        # Try to find without symbol
                        for i, opt in enumerate(options):
                            if opt.startswith(('🔮', '🔬', '🧹', '📊')):
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
                    
                    # Store mapping (clean the selection)
                    if not selected.startswith('───'):
                        if selected.startswith(('🔮', '🔬', '🧹', '📊')):
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
    
    # Show unmapped parameters
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
    
    # === STEP 2: BASELINE EXTRACTION ===
    if compatibility['score'] > 0:
        st.markdown("---")
        st.subheader("📍 Step 2: Extract Baseline Values")
        st.caption("Define baseline reference year for percentage-based calculations")
        
        st.info("""
        💡 **Baseline Year Selection:**
        - Enter the reference year for calculating percentage changes
        - Typically: last historical year OR a specific policy year
        - Should be BEFORE scenario horizon years
        - Example: If scenarios are for 2040, baseline might be 2020 or 2025
        """)
        
        # Get scenario years for reference
        scenario_years = []
        if 'detected_scenarios' in st.session_state:
            for scenario in st.session_state.detected_scenarios:
                horizon = scenario.get('horizon')
                if horizon and isinstance(horizon, (int, float)):
                    scenario_years.append(int(horizon))
        
        if scenario_years:
            min_year = min(scenario_years)
            max_year = max(scenario_years)
            st.caption(f"📅 Your scenario years range: {min_year} - {max_year}")
            st.caption(f"💡 Recommended baseline: {min_year - 10} or earlier")
        
        # User enters baseline year (NO DEFAULT)
        baseline_year = st.number_input(
            "Enter Baseline Reference Year:",
            min_value=2000,
            max_value=2100,
            value=2020,  # Neutral default
            step=1,
            help="Enter the year to use as baseline for percentage calculations"
        )
        
        # Warning if baseline >= scenario
        if scenario_years and baseline_year >= min(scenario_years):
            st.warning(f"""
            ⚠️ **Warning:** Baseline year ({baseline_year}) is at or after scenario year ({min(scenario_years)}).
            
            **Scientific Issue:** Baseline should be BEFORE the scenario period.
            
            **Recommendation:** Use {min(scenario_years) - 10} or earlier.
            """)
        
        use_forecast = st.checkbox(
            "Use forecast values if year beyond historical data",
            value=True,
            help="If baseline year is in future, use forecasted values"
        )
        
        if st.button("🔍 Extract Baselines", type="primary"):
            with st.spinner("Extracting baseline values..."):
                baseline_results = {}
                
                for scenario_param, historical_param in st.session_state.parameter_mappings.items():
                    if historical_param != 'None':
                        baseline = extract_baseline_value(
                            historical_param,
                            reference_year=baseline_year,
                            use_forecast=use_forecast
                        )
                        
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
                        st.caption("⚠️ Install kaleido for PNG export")
                
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
                        st.caption("⚠️ Install kaleido for PDF export")
                
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
    
    # Get absolute values
    absolute_vals = st.session_state.absolute_values
    
    # Get baseline references for forecast points
    baseline_refs = st.session_state.baseline_references
    
    # Plot each scenario as a box
    for scenario_name, params in absolute_vals.items():
        if x_param in params and y_param in params:
            x_val = params[x_param]['absolute_value']
            y_val = params[y_param]['absolute_value']
            
            # Calculate box size (±10% uncertainty)
            x_size = abs(x_val * 0.1)
            y_size = abs(y_val * 0.1)
            
            color = scenario_colors.get(scenario_name, '#1f77b4')
            
            # Add box (rectangle)
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
        # Get forecast data for x_param and y_param
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
