"""
Page 11: Trajectory-Scenario Space
Map historical parameters to scenario parameters, assess compatibility, 
extract baselines, and visualize the trajectory space
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


def get_historical_parameters():
    """
    Get all parameters from historical data
    
    Returns:
    --------
    dict with two lists:
        - forecast_params: Parameters used in forecasting
        - other_params: Parameters not used in forecasting
    """
    
    forecast_params = []
    other_params = []
    
    # Check if data is loaded
    if 'df_long' not in st.session_state or st.session_state.df_long is None:
        return {'forecast_params': [], 'other_params': []}
    
    df = st.session_state.df_long
    all_variables = df['variable'].unique().tolist()
    
    # Check which parameters were used in forecasting
    if 'forecast_results' in st.session_state and st.session_state.forecast_results:
        forecasted_vars = list(st.session_state.forecast_results.keys())
        
        for var in all_variables:
            if var in forecasted_vars:
                forecast_params.append(var)
            else:
                other_params.append(var)
    else:
        # No forecasts - all are "other"
        other_params = all_variables
    
    return {
        'forecast_params': sorted(forecast_params),
        'other_params': sorted(other_params)
    }


def get_scenario_parameters():
    """
    Get all parameters from scenario analysis (Page 9)
    
    Returns:
    --------
    list: List of scenario parameters with metadata
    """
    
    if 'detected_scenarios' not in st.session_state or not st.session_state.detected_scenarios:
        return []
    
    # Collect all unique parameters across scenarios
    scenario_params = {}
    
    for scenario in st.session_state.detected_scenarios:
        for item in scenario['items']:
            param_name = item.get('parameter_canonical', item.get('parameter', ''))
            
            if param_name not in scenario_params:
                scenario_params[param_name] = {
                    'name': param_name,
                    'category': item.get('category', 'Other'),
                    'unit': item.get('unit', ''),
                    'direction': item.get('direction', ''),
                    'scenarios': []
                }
            
            scenario_params[param_name]['scenarios'].append({
                'scenario': scenario['title'],
                'value': item.get('value'),
                'direction': item.get('direction')
            })
    
    return list(scenario_params.values())


def calculate_compatibility_score(mappings, scenario_params):
    """
    Calculate compatibility score between historical and scenario parameters
    
    Parameters:
    -----------
    mappings : dict
        Mapping of scenario params to historical params
    scenario_params : list
        List of scenario parameters
    
    Returns:
    --------
    dict with:
        - score: float (0-100)
        - covered: int (number of mapped params)
        - total: int (total scenario params)
        - unmapped: list (unmapped scenario params)
    """
    
    total_params = len(scenario_params)
    
    if total_params == 0:
        return {
            'score': 0,
            'covered': 0,
            'total': 0,
            'unmapped': []
        }
    
    # Count mapped parameters
    mapped_params = [p['name'] for p in scenario_params if p['name'] in mappings and mappings[p['name']] != 'None']
    covered = len(mapped_params)
    
    # Calculate score
    score = (covered / total_params) * 100 if total_params > 0 else 0
    
    # Find unmapped
    unmapped = [p['name'] for p in scenario_params if p['name'] not in mappings or mappings[p['name']] == 'None']
    
    return {
        'score': score,
        'covered': covered,
        'total': total_params,
        'unmapped': unmapped
    }


def extract_baseline_value(historical_param, reference_year=None, use_forecast=True):
    """
    Extract baseline value from historical data or forecast
    
    Parameters:
    -----------
    historical_param : str
        Name of historical parameter
    reference_year : int
        Year to use as baseline (None = last historical)
    use_forecast : bool
        If True and forecast available, use forecast value
    
    Returns:
    --------
    dict with:
        - value: float
        - year: int
        - source: str ('historical' or 'forecast')
    """
    
    # Get historical data
    df = st.session_state.df_long
    var_data = df[df['variable'] == historical_param].copy()
    
    if len(var_data) == 0:
        return {'value': None, 'year': None, 'source': None}
    
    # Sort by date
    var_data = var_data.sort_values('date')
    
    if reference_year is None:
        # Use last historical year
        last_date = var_data['date'].max()
        baseline_value = var_data[var_data['date'] == last_date]['value'].iloc[0]
        baseline_year = last_date.year
        source = 'historical'
    else:
        # Try to find exact year
        var_data['year'] = pd.to_datetime(var_data['date']).dt.year
        year_data = var_data[var_data['year'] == reference_year]
        
        if len(year_data) > 0:
            baseline_value = year_data['value'].iloc[0]
            baseline_year = reference_year
            source = 'historical'
        else:
            # Year not in historical - try forecast
            if use_forecast and 'forecast_results' in st.session_state:
                forecast = st.session_state.forecast_results.get(historical_param)
                
                if forecast:
                    # Find closest forecast year
                    forecast_dates = pd.to_datetime(forecast['forecast_timestamps'])
                    forecast_years = forecast_dates.year
                    
                    closest_idx = np.argmin(np.abs(forecast_years - reference_year))
                    baseline_value = forecast['forecast_values'][closest_idx]
                    baseline_year = forecast_years[closest_idx]
                    source = 'forecast'
                else:
                    # No forecast - use last historical
                    baseline_value = var_data['value'].iloc[-1]
                    baseline_year = var_data['year'].iloc[-1]
                    source = 'historical (fallback)'
            else:
                # Use last historical
                baseline_value = var_data['value'].iloc[-1]
                baseline_year = var_data['year'].iloc[-1]
                source = 'historical (fallback)'
    
    return {
        'value': float(baseline_value),
        'year': int(baseline_year),
        'source': source
    }


def convert_scenario_to_absolute(scenario_param, scenario_value, direction, unit, baseline_value):
    """
    Convert scenario parameter from percentage/direction to absolute value
    
    Parameters:
    -----------
    scenario_param : str
        Parameter name
    scenario_value : float
        Value from scenario (could be percentage, absolute, etc.)
    direction : str
        Direction (increase, decrease, target, etc.)
    unit : str
        Unit (%, absolute, etc.)
    baseline_value : float
        Baseline value from historical data
    
    Returns:
    --------
    float: Absolute value
    """
    
    if unit == '%':
        # Percentage change
        if direction == 'increase':
            return baseline_value * (1 + scenario_value / 100)
        elif direction == 'decrease':
            return baseline_value * (1 - scenario_value / 100)
        elif direction == 'target':
            # Target is absolute percentage (e.g., "reach 70%")
            # Interpret as fraction
            return scenario_value
        elif direction == 'double':
            return baseline_value * 2
        elif direction == 'halve':
            return baseline_value * 0.5
        elif direction == 'stable':
            return baseline_value
        else:
            # Default: treat as percentage of baseline
            return baseline_value * (scenario_value / 100)
    
    elif unit == 'absolute' or unit == '':
        # Already absolute
        if direction == 'target':
            return scenario_value
        elif direction == 'increase':
            return baseline_value + scenario_value
        elif direction == 'decrease':
            return baseline_value - scenario_value
        elif direction == 'double':
            return baseline_value * 2
        elif direction == 'halve':
            return baseline_value * 0.5
        elif direction == 'stable':
            return baseline_value
        else:
            return scenario_value
    
    else:
        # Other units (MtCO2, GW, etc.) - assume absolute
        return scenario_value


def aggregate_category_value(parameters_dict, category):
    """
    Aggregate multiple parameters into a single category value
    
    Scientific approach: Weighted normalization + aggregation
    
    Parameters:
    -----------
    parameters_dict : dict
        {param_name: absolute_value}
    category : str
        Category name (e.g., '1. Economic')
    
    Returns:
    --------
    float: Aggregated category value (normalized 0-100 scale)
    """
    
    # Filter parameters for this category
    category_params = {}
    
    if 'detected_scenarios' in st.session_state:
        for scenario in st.session_state.detected_scenarios:
            for item in scenario['items']:
                param_name = item.get('parameter_canonical', item.get('parameter', ''))
                param_category = item.get('category', '')
                
                if param_category == category and param_name in parameters_dict:
                    category_params[param_name] = parameters_dict[param_name]
    
    if not category_params:
        return 0.0
    
    # Normalize each parameter to 0-100 scale
    # Use min-max normalization based on historical ranges
    normalized_values = []
    
    for param_name, value in category_params.items():
        # Get historical range
        if 'df_long' in st.session_state:
            df = st.session_state.df_long
            
            # Try to find corresponding historical parameter
            # (might need mapping)
            if param_name in df['variable'].values:
                hist_data = df[df['variable'] == param_name]['value']
                min_val = hist_data.min()
                max_val = hist_data.max()
                
                # Normalize
                if max_val > min_val:
                    normalized = ((value - min_val) / (max_val - min_val)) * 100
                else:
                    normalized = 50  # Default if no range
            else:
                # No historical data - use value directly
                normalized = value
        else:
            normalized = value
        
        # Clip to 0-100
        normalized = np.clip(normalized, 0, 100)
        normalized_values.append(normalized)
    
    # Aggregate: simple mean (could use weighted mean if weights defined)
    aggregated = np.mean(normalized_values)
    
    return float(aggregated)


def main():
    """Main page function"""
    
    # Initialize state
    initialize_trajectory_state()
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ About Trajectory-Scenario Space Analysis", expanded=False):
        st.markdown("""
        ### Scientific Methodology
        
        **Purpose:**
        Bridge the gap between historical data analysis and scenario planning by:
        1. Mapping scenario parameters to historical parameters
        2. Assessing compatibility between the two systems
        3. Extracting baseline values for percentage-based scenarios
        4. Converting scenario parameters to absolute values
        5. Visualizing trajectories in multi-dimensional parameter space
        
        **Scientific Approach:**
        
        **1. Parameter Mapping**
        - Manual mapping by domain expert (you!)
        - Ensures semantic equivalence between historical and scenario parameters
        - Allows for one-to-one or many-to-one relationships
        
        **2. Compatibility Analysis**
        - Percentage of scenario parameters covered by historical data
        - Identifies gaps and suggests remediation strategies
        
        **3. Baseline Extraction**
        - Automatically extracts baseline values from historical data
        - Uses forecasted values if reference year beyond historical range
        - Provides transparency on baseline source
        
        **4. Absolute Value Conversion**
        - Converts percentage changes to absolute values
        - Handles different directions: increase, decrease, target, stable
        - Ensures consistency with historical units
        
        **5. Category Aggregation**
        - Normalizes parameters to 0-100 scale using min-max normalization
        - Aggregates multiple parameters into category indicators
        - Enables comparison across different measurement units
        
        **6. Trajectory Visualization**
        - Parameter vs Parameter plots show scenario position vs forecast
        - Category vs Category plots enable systems-level comparison
        - Distance metrics quantify scenario divergence from forecast
        
        **Applications:**
        - Policy scenario analysis
        - Climate change trajectory assessment
        - Economic development pathway evaluation
        - Multi-criteria decision analysis
        """)
    
    # Check prerequisites
    has_historical = 'df_long' in st.session_state and st.session_state.df_long is not None
    has_scenarios = 'detected_scenarios' in st.session_state and st.session_state.detected_scenarios
    has_forecasts = 'forecast_results' in st.session_state and st.session_state.forecast_results
    
    if not has_historical:
        st.error("❌ **No historical data loaded.** Please load data in Page 1 first.")
        return
    
    if not has_scenarios:
        st.warning("⚠️ **No scenarios defined.** Please create scenarios in Page 9 (NLP Analysis) first.")
        st.info("💡 You can still explore this page with historical data only.")
    
    st.markdown("---")
    
    # === STEP 1: PARAMETER MAPPING ===
    st.subheader("🔗 Step 1: Map Scenario Parameters to Historical Parameters")
    st.caption("Create semantic equivalence between your scenario definitions and historical data")
    
    # Get parameters
    historical_params = get_historical_parameters()
    scenario_params = get_scenario_parameters()
    
    # Display in two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Historical Parameters")
        
        if historical_params['forecast_params']:
            with st.expander(f"✅ **Forecasted Parameters** ({len(historical_params['forecast_params'])})", expanded=True):
                for param in historical_params['forecast_params']:
                    st.markdown(f"- `{param}`")
        
        if historical_params['other_params']:
            with st.expander(f"📈 **Other Historical Parameters** ({len(historical_params['other_params'])})", expanded=False):
                for param in historical_params['other_params']:
                    st.markdown(f"- `{param}`")
        
        total_historical = len(historical_params['forecast_params']) + len(historical_params['other_params'])
        st.metric("Total Historical Parameters", total_historical)
    
    with col_right:
        st.markdown("### 🎯 Scenario Parameters")
        
        if scenario_params:
            # Group by category
            categories = {}
            for param in scenario_params:
                cat = param['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(param)
            
            for category, params in sorted(categories.items()):
                with st.expander(f"**{category}** ({len(params)} parameters)", expanded=True):
                    for param in params:
                        # Show parameter with metadata
                        scenarios_str = ", ".join([f"{s['scenario']} ({s['direction']})" for s in param['scenarios']])
                        st.markdown(f"- `{param['name']}` - *{scenarios_str}*")
            
            st.metric("Total Scenario Parameters", len(scenario_params))
        else:
            st.info("No scenario parameters available. Create scenarios in Page 9.")
    
    if scenario_params and has_historical:
        st.markdown("---")
        st.markdown("### 🔄 Create Parameter Mappings")
        st.caption("Map each scenario parameter to an equivalent historical parameter")
        
        # Create mapping interface
        all_historical = historical_params['forecast_params'] + historical_params['other_params']
        
        for scenario_param in scenario_params:
            param_name = scenario_param['name']
            
            col_map1, col_map2, col_map3 = st.columns([2, 3, 1])
            
            with col_map1:
                st.markdown(f"**{param_name}**")
                st.caption(f"Category: {scenario_param['category']}")
            
            with col_map2:
                # Dropdown for mapping
                current_mapping = st.session_state.parameter_mappings.get(param_name, 'None')
                
                # Create options: None + all historical parameters
                options = ['None'] + all_historical
                default_idx = options.index(current_mapping) if current_mapping in options else 0
                
                selected = st.selectbox(
                    "Maps to historical parameter:",
                    options=options,
                    index=default_idx,
                    key=f"mapping_{param_name}",
                    label_visibility="collapsed"
                )
                
                # Store mapping
                st.session_state.parameter_mappings[param_name] = selected
            
            with col_map3:
                if selected != 'None':
                    st.success("✓ Mapped")
                else:
                    st.warning("⚠️ Unmapped")
        
        # Calculate compatibility
        st.markdown("---")
        st.markdown("### 📊 Compatibility Analysis")
        
        compatibility = calculate_compatibility_score(
            st.session_state.parameter_mappings,
            scenario_params
        )
        
        st.session_state.compatibility_score = compatibility
        
        # Display compatibility score
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        
        with col_comp1:
            # Color-code the score
            if compatibility['score'] == 100:
                st.metric("Compatibility Score", f"{compatibility['score']:.0f}%", delta="Perfect!")
            elif compatibility['score'] >= 80:
                st.metric("Compatibility Score", f"{compatibility['score']:.0f}%", delta="Good")
            elif compatibility['score'] >= 50:
                st.metric("Compatibility Score", f"{compatibility['score']:.0f}%", delta="Fair")
            else:
                st.metric("Compatibility Score", f"{compatibility['score']:.0f}%", delta="Low")
        
        with col_comp2:
            st.metric("Mapped Parameters", f"{compatibility['covered']}/{compatibility['total']}")
        
        with col_comp3:
            st.metric("Coverage", f"{compatibility['covered']}/{len(scenario_params)}")
        
        with col_comp4:
            unmapped_count = len(compatibility['unmapped'])
            if unmapped_count == 0:
                st.metric("Unmapped", "0", delta="Complete!")
            else:
                st.metric("Unmapped", unmapped_count, delta="Action needed")
        
        # Show recommendations if not 100%
        if compatibility['score'] < 100:
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            st.warning(f"**{len(compatibility['unmapped'])} scenario parameter(s) are not mapped to historical data.**")
            
            with st.expander("📋 View Unmapped Parameters", expanded=True):
                for param in compatibility['unmapped']:
                    st.markdown(f"- `{param}`")
            
            st.markdown("**Suggested Actions:**")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.info("""
                **Option 1: Add Historical Parameters**
                - Upload additional historical data
                - Include parameters equivalent to unmapped scenarios
                - Re-run the mapping process
                """)
            
            with col_rec2:
                st.info("""
                **Option 2: Modify Scenarios**
                - Return to Page 9 (Scenario Analysis)
                - Remove or merge unmapped parameters
                - Focus on parameters with historical equivalents
                """)
            
            st.info("""
            **Option 3: Proceed with Partial Coverage**
            - Continue analysis with mapped parameters only
            - Note: Compatibility score will reflect partial coverage
            - Unmapped parameters will not appear in trajectory visualizations
            """)
        else:
            st.success("🎉 **Perfect Compatibility!** All scenario parameters are mapped to historical data.")
        
        # === STEP 2: BASELINE EXTRACTION ===
        if compatibility['score'] > 0:
            st.markdown("---")
            st.subheader("📍 Step 2: Extract Baseline Values")
            st.caption("Define baseline reference for percentage-based scenario parameters")
            
            # Get scenario horizon years
            scenario_years = []
            if 'detected_scenarios' in st.session_state:
                for scenario in st.session_state.detected_scenarios:
                    horizon = scenario.get('horizon')
                    if horizon and isinstance(horizon, (int, float)):
                        scenario_years.append(int(horizon))
            
            # Determine default reference year
            if scenario_years:
                common_year = max(set(scenario_years), key=scenario_years.count)
                st.info(f"💡 Most common scenario horizon year: **{common_year}**")
            else:
                common_year = datetime.now().year + 10
            
            # Let user choose baseline year
            col_base1, col_base2 = st.columns([1, 3])
            
            with col_base1:
                baseline_year = st.number_input(
                    "Baseline Reference Year:",
                    min_value=2000,
                    max_value=2100,
                    value=common_year,
                    step=1,
                    help="Year to use as baseline for percentage calculations"
                )
            
            with col_base2:
                use_forecast_baseline = st.checkbox(
                    "Use forecast values if year beyond historical data",
                    value=True,
                    help="If baseline year is in the future, use forecasted values instead of last historical"
                )
            
            if st.button("🔍 Extract Baselines", type="primary"):
                with st.spinner("Extracting baseline values..."):
                    # Extract baseline for each mapped parameter
                    baseline_results = {}
                    
                    for scenario_param, historical_param in st.session_state.parameter_mappings.items():
                        if historical_param != 'None':
                            baseline = extract_baseline_value(
                                historical_param,
                                reference_year=baseline_year,
                                use_forecast=use_forecast_baseline
                            )
                            
                            baseline_results[scenario_param] = {
                                'historical_param': historical_param,
                                'value': baseline['value'],
                                'year': baseline['year'],
                                'source': baseline['source']
                            }
                    
                    st.session_state.baseline_references = baseline_results
                
                st.success(f"✅ Extracted {len(baseline_results)} baseline values!")
            
            # Display baseline results if available
            if st.session_state.baseline_references:
                st.markdown("### 📊 Baseline Values")
                
                # Create DataFrame for display
                baseline_data = []
                for scenario_param, baseline_info in st.session_state.baseline_references.items():
                    baseline_data.append({
                        'Scenario Parameter': scenario_param,
                        'Historical Parameter': baseline_info['historical_param'],
                        'Baseline Value': f"{baseline_info['value']:.2f}",
                        'Year': baseline_info['year'],
                        'Source': baseline_info['source']
                    })
                
                df_baselines = pd.DataFrame(baseline_data)
                
                st.dataframe(
                    df_baselines,
                    use_container_width=True,
                    hide_index=True
                )
                
                # === STEP 3: CONVERT TO ABSOLUTE VALUES ===
                st.markdown("---")
                st.subheader("🔢 Step 3: Convert Scenarios to Absolute Values")
                st.caption("Transform percentage-based changes to absolute values using extracted baselines")
                
                if st.button("⚙️ Convert to Absolute Values", type="primary"):
                    with st.spinner("Converting scenario values..."):
                        absolute_results = {}
                        
                        # Process each scenario
                        for scenario in st.session_state.detected_scenarios:
                            scenario_name = scenario['title']
                            absolute_results[scenario_name] = {}
                            
                            for item in scenario['items']:
                                param_name = item.get('parameter_canonical', item.get('parameter', ''))
                                
                                # Check if we have baseline
                                if param_name in st.session_state.baseline_references:
                                    baseline_info = st.session_state.baseline_references[param_name]
                                    baseline_value = baseline_info['value']
                                    
                                    # Convert to absolute
                                    scenario_value = item.get('value', 0)
                                    direction = item.get('direction', 'target')
                                    unit = item.get('unit', '')
                                    
                                    absolute_value = convert_scenario_to_absolute(
                                        param_name,
                                        scenario_value,
                                        direction,
                                        unit,
                                        baseline_value
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
                    
                    st.success(f"✅ Converted {len(absolute_results)} scenarios to absolute values!")
                
                # Display absolute values if available
                if st.session_state.absolute_values:
                    st.markdown("### 📊 Absolute Values by Scenario")
                    
                    for scenario_name, params in st.session_state.absolute_values.items():
                        with st.expander(f"**{scenario_name}** ({len(params)} parameters)", expanded=False):
                            # Create DataFrame
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
                    
                    # === STEP 4: VISUALIZATIONS ===
                    st.markdown("---")
                    st.subheader("📊 Step 4: Trajectory Visualizations")
                    st.caption("Compare scenario trajectories with historical forecasts")
                    
                    # Create tabs for different visualization types
                    viz_tab1, viz_tab2 = st.tabs([
                        "📈 Parameter vs Parameter",
                        "🎯 Category vs Category"
                    ])
                    
                    # === TAB 1: Parameter vs Parameter ===
                    with viz_tab1:
                        st.markdown("### Parameter vs Parameter Comparison")
                        st.caption("Plot scenario positions against forecasted values")
                        
                        # Get list of mapped parameters
                        mapped_params = [p for p, h in st.session_state.parameter_mappings.items() if h != 'None']
                        
                        if len(mapped_params) >= 2:
                            col_viz1, col_viz2 = st.columns(2)
                            
                            with col_viz1:
                                x_param = st.selectbox(
                                    "X-axis parameter:",
                                    options=mapped_params,
                                    key="x_param_select"
                                )
                            
                            with col_viz2:
                                y_param = st.selectbox(
                                    "Y-axis parameter:",
                                    options=[p for p in mapped_params if p != x_param],
                                    key="y_param_select"
                                )
                            
                            if st.button("📊 Generate Parameter Plot"):
                                generate_parameter_plot(x_param, y_param)
                        else:
                            st.warning("Need at least 2 mapped parameters for visualization.")
                    
                    # === TAB 2: Category vs Category ===
                    with viz_tab2:
                        st.markdown("### Category vs Category Comparison")
                        st.caption("Aggregate parameters into categories and visualize system-level trajectories")
                        
                        # Get unique categories
                        categories = set()
                        for params in st.session_state.absolute_values.values():
                            for info in params.values():
                                categories.add(info['category'])
                        
                        categories = sorted(list(categories))
                        
                        if len(categories) >= 2:
                            col_cat1, col_cat2 = st.columns(2)
                            
                            with col_cat1:
                                x_category = st.selectbox(
                                    "X-axis category:",
                                    options=categories,
                                    key="x_cat_select"
                                )
                            
                            with col_cat2:
                                y_category = st.selectbox(
                                    "Y-axis category:",
                                    options=[c for c in categories if c != x_category],
                                    key="y_cat_select"
                                )
                            
                            if st.button("🎯 Generate Category Plot"):
                                generate_category_plot(x_category, y_category)
                        else:
                            st.warning("Need at least 2 categories for visualization.")


def generate_parameter_plot(x_param, y_param):
    """Generate parameter vs parameter scatter plot"""
    
    st.markdown(f"### {x_param} vs {y_param}")
    
    # Collect data points
    plot_data = []
    
    # Add forecast point (if available)
    if 'forecast_results' in st.session_state:
        x_hist_param = st.session_state.parameter_mappings.get(x_param)
        y_hist_param = st.session_state.parameter_mappings.get(y_param)
        
        if x_hist_param != 'None' and y_hist_param != 'None':
            x_forecast = st.session_state.forecast_results.get(x_hist_param)
            y_forecast = st.session_state.forecast_results.get(y_hist_param)
            
            if x_forecast and y_forecast:
                # Use last forecast value
                x_val = x_forecast['forecast_values'][-1]
                y_val = y_forecast['forecast_values'][-1]
                
                plot_data.append({
                    'Scenario': 'Forecast (Historical)',
                    'x': x_val,
                    'y': y_val,
                    'type': 'forecast'
                })
    
    # Add scenario points
    for scenario_name, params in st.session_state.absolute_values.items():
        if x_param in params and y_param in params:
            plot_data.append({
                'Scenario': scenario_name,
                'x': params[x_param]['absolute_value'],
                'y': params[y_param]['absolute_value'],
                'type': 'scenario'
            })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # Create plot
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='Scenario',
            symbol='type',
            title=f"Trajectory Space: {x_param} vs {y_param}",
            labels={'x': x_param, 'y': y_param},
            height=600
        )
        
        fig.update_traces(marker=dict(size=15, line=dict(width=2, color='white')))
        fig.update_layout(
            hovermode='closest',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate distances
        st.markdown("### 📏 Distance from Forecast")
        
        if any(d['type'] == 'forecast' for d in plot_data):
            forecast_point = next(d for d in plot_data if d['type'] == 'forecast')
            
            distances = []
            for d in plot_data:
                if d['type'] == 'scenario':
                    dist = np.sqrt((d['x'] - forecast_point['x'])**2 + (d['y'] - forecast_point['y'])**2)
                    distances.append({
                        'Scenario': d['Scenario'],
                        'Distance': dist
                    })
            
            df_dist = pd.DataFrame(distances).sort_values('Distance')
            st.dataframe(df_dist, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for selected parameters.")


def generate_category_plot(x_category, y_category):
    """Generate category vs category scatter plot"""
    
    st.markdown(f"### {x_category} vs {y_category}")
    
    # Aggregate parameters by category for each scenario
    plot_data = []
    
    for scenario_name, params in st.session_state.absolute_values.items():
        # Aggregate for x_category
        x_val = aggregate_category_value(
            {p: info['absolute_value'] for p, info in params.items()},
            x_category
        )
        
        # Aggregate for y_category
        y_val = aggregate_category_value(
            {p: info['absolute_value'] for p, info in params.items()},
            y_category
        )
        
        plot_data.append({
            'Scenario': scenario_name,
            'x': x_val,
            'y': y_val
        })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # Create plot
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            text='Scenario',
            title=f"Category Space: {x_category} vs {y_category}",
            labels={'x': f'{x_category} (normalized)', 'y': f'{y_category} (normalized)'},
            height=600
        )
        
        fig.update_traces(
            marker=dict(size=20, line=dict(width=2, color='white')),
            textposition='top center'
        )
        
        fig.update_layout(
            hovermode='closest',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Note:** Category values are normalized to 0-100 scale using min-max normalization 
        of constituent parameters based on historical ranges.
        """)
    else:
        st.warning("No data available for selected categories.")


if __name__ == "__main__":
    main()
