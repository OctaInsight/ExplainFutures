"""
Page 9: Scenario Analysis (NLP)
Analyze scenario text, extract parameters, map to variables, and compare scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

# Import NLP modules
from core.nlp.lang_detect import detect_language
from core.nlp.scenario_segment import segment_scenarios
from core.nlp.parameter_extract import extract_parameters_from_scenarios
from core.nlp.value_parse import parse_value_string, normalize_unit, format_value
from core.nlp.mapping import suggest_variable_mapping, create_mapping
from core.nlp.clean_text import generate_cleaned_scenario_text
from core.nlp.schema import Scenario, ScenarioItem

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Scenario Analysis", 
    page_icon="📝", 
    layout="wide"
)

# Render shared sidebar
render_app_sidebar()

st.title("📝 Scenario Analysis (NLP)")
st.markdown("*Analyze scenario text and extract structured parameters*")
st.markdown("---")


def initialize_scenario_state():
    """Initialize session state for scenario analysis"""
    if "scenario_text_input" not in st.session_state:
        st.session_state.scenario_text_input = ""
    if "detected_scenarios" not in st.session_state:
        st.session_state.detected_scenarios = []
    if "scenarios_processed" not in st.session_state:
        st.session_state.scenarios_processed = False
    if "scenario_mappings" not in st.session_state:
        st.session_state.scenario_mappings = {}
    if "cleaned_scenarios" not in st.session_state:
        st.session_state.cleaned_scenarios = {}


def main():
    """Main page function"""
    
    # Initialize state
    initialize_scenario_state()
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ How to Use Scenario Analysis", expanded=False):
        st.markdown("""
        ### NLP-Powered Scenario Analysis
        
        **What This Page Does:**
        1. Analyzes scenario text written in English
        2. Detects and segments multiple scenarios
        3. Extracts parameters, values, and directions
        4. Maps parameters to your dataset variables (if available)
        5. Generates cleaned, structured scenario descriptions
        6. Creates comparison plots between scenarios
        
        **Step-by-Step Guide:**
        
        **Step 1: Input Scenario Text**
        - Write or paste scenario descriptions in English
        - Can contain multiple scenarios
        - Include specific numbers, percentages, or direction (increase/decrease)
        
        **Step 2: Review Extracted Parameters**
        - System shows detected scenarios and parameters
        - Edit parameter names, values, units
        - Add missing parameters
        - Set percentages for "increase/decrease" statements
        
        **Step 3: Map to Dataset Variables** (optional)
        - If you have loaded data, map scenario parameters to variables
        - System suggests matches based on similarity
        - Choose direct mapping or note relationships
        
        **Step 4: Review Cleaned Scenarios**
        - See structured version of your scenarios
        - Export as text or JSON
        
        **Step 5: Compare Scenarios**
        - View X-Y plots comparing parameter values across scenarios
        - Customize which parameters to plot
        
        **Example Scenario Text:**
        ```
        Scenario 1: Optimistic Growth
        GDP increases by 20% by 2040.
        CO2 emissions decrease to 5 MtCO2/year.
        Renewable energy reaches 70% by 2040.
        
        Scenario 2: Business as Usual
        GDP increases by 10% by 2040.
        CO2 emissions remain at current levels.
        Renewable energy reaches 40%.
        ```
        
        **Note:** This module works independently - you don't need loaded data to use it!
        """)
    
    st.markdown("---")
    
    # === STEP 1: INPUT SCENARIO TEXT ===
    st.subheader("📝 Step 1: Input Scenario Text")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scenario_text = st.text_area(
            "Enter or paste scenario text (English only):",
            value=st.session_state.scenario_text_input,
            height=300,
            placeholder="""Example:
Scenario 1: Optimistic Growth
GDP increases by 20% by 2040.
CO2 emissions decrease to 5 MtCO2/year by 2040.
Renewable energy share reaches 70% by 2040.

Scenario 2: Business as Usual
GDP increases by 10% by 2040.
CO2 emissions remain stable.
Renewable energy reaches 40% by 2040.""",
            help="Describe one or more scenarios with specific parameters and values"
        )
    
    with col2:
        st.markdown("**Quick Stats:**")
        word_count = len(scenario_text.split())
        line_count = len(scenario_text.split('\n'))
        
        st.metric("Words", word_count)
        st.metric("Lines", line_count)
        
        # Clear button
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.scenario_text_input = ""
            st.session_state.detected_scenarios = []
            st.session_state.scenarios_processed = False
            st.rerun()
    
    if not scenario_text.strip():
        st.info("👆 Please enter scenario text to begin analysis")
        return
    
    # Store in session state
    st.session_state.scenario_text_input = scenario_text
    
    st.markdown("---")
    
    # === PROCESS BUTTON ===
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Analyze Scenarios", type="primary", use_container_width=True):
            
            with st.spinner("Analyzing scenario text..."):
                
                # Step 1: Language detection
                st.info("🔍 Detecting language...")
                lang_result = detect_language(scenario_text)
                
                if not lang_result['is_english']:
                    st.error(f"❌ Language Detection: {lang_result['detected_language']}")
                    st.warning("⚠️ **This version supports English only.** Please translate your text to English.")
                    return
                
                st.success(f"✅ Language: {lang_result['detected_language']} (confidence: {lang_result['confidence']:.0%})")
                
                # Step 2: Segment scenarios
                st.info("📑 Segmenting scenarios...")
                scenarios = segment_scenarios(scenario_text)
                
                st.success(f"✅ Detected {len(scenarios)} scenario(s)")
                
                # Step 3: Extract parameters from each scenario
                st.info("🔎 Extracting parameters...")
                scenarios_with_params = extract_parameters_from_scenarios(scenarios)
                
                total_params = sum(len(s['items']) for s in scenarios_with_params)
                st.success(f"✅ Extracted {total_params} parameter(s) across all scenarios")
                
                # Store results
                st.session_state.detected_scenarios = scenarios_with_params
                st.session_state.scenarios_processed = True
                
                st.success("🎉 Analysis complete! Review results below.")
                st.rerun()
    
    # === STEP 2: REVIEW EXTRACTED PARAMETERS ===
    if st.session_state.scenarios_processed and st.session_state.detected_scenarios:
        
        st.markdown("---")
        st.subheader("📊 Step 2: Review & Edit Extracted Parameters")
        
        # Create tabs for each scenario
        scenario_tabs = st.tabs([s['title'] for s in st.session_state.detected_scenarios])
        
        for tab_idx, (tab, scenario) in enumerate(zip(scenario_tabs, st.session_state.detected_scenarios)):
            with tab:
                display_scenario_editor(scenario, tab_idx)
        
        # === STEP 3: MAPPING TO DATASET VARIABLES ===
        st.markdown("---")
        st.subheader("🔗 Step 3: Map to Dataset Variables")
        
        # Check if data is available
        if st.session_state.get('df_long') is not None:
            display_mapping_interface()
        else:
            st.info("ℹ️ No dataset loaded. Scenario analysis will work in **standalone mode**.")
            st.markdown("*To enable mapping, load data in the Data Upload page.*")
        
        # === STEP 4: CLEANED SCENARIOS ===
        st.markdown("---")
        st.subheader("📄 Step 4: Review Cleaned Scenarios")
        
        if st.button("✨ Generate Cleaned Scenario Text", type="primary"):
            generate_cleaned_scenarios()
        
        if st.session_state.cleaned_scenarios:
            display_cleaned_scenarios()
        
        # === STEP 5: COMPARISON PLOTS ===
        st.markdown("---")
        st.subheader("📈 Step 5: Compare Scenarios")
        
        display_scenario_comparison()


def display_scenario_editor(scenario: dict, scenario_idx: int):
    """
    Display interactive editor for a single scenario
    
    Parameters:
    -----------
    scenario : dict
        Scenario data with items
    scenario_idx : int
        Index of scenario
    """
    st.markdown(f"### {scenario['title']}")
    
    # Show scenario metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Edit title
        new_title = st.text_input(
            "Scenario Title:",
            value=scenario['title'],
            key=f"title_{scenario_idx}"
        )
        scenario['title'] = new_title
    
    with col2:
        # Edit horizon
        horizon = st.number_input(
            "Horizon (Year):",
            min_value=2020,
            max_value=2100,
            value=scenario.get('horizon', 2050),
            key=f"horizon_{scenario_idx}"
        )
        scenario['horizon'] = horizon
    
    with col3:
        st.metric("Parameters", len(scenario['items']))
    
    st.markdown("---")
    
    # Display items as individual rows for easier editing
    if scenario['items']:
        
        st.markdown(f"#### 📋 Parameters ({len(scenario['items'])} total)")
        
        # Add scrollable container if many parameters
        if len(scenario['items']) > 10:
            st.info(f"💡 This scenario has {len(scenario['items'])} parameters. Scroll down to see all.")
        
        # Show unification info if available
        unified_params = [item.get('parameter_canonical') for item in scenario['items'] if item.get('parameter_canonical')]
        if unified_params:
            with st.expander("🔗 Parameter Unification Info", expanded=False):
                st.markdown("**Unified parameters** (variations merged):")
                unique_canonical = list(set(unified_params))
                for canonical in unique_canonical:
                    # Find all original variations
                    variations = [item.get('parameter_original', item.get('parameter')) 
                                  for item in scenario['items'] 
                                  if item.get('parameter_canonical') == canonical]
                    variations = list(set(variations))
                    
                    if len(variations) > 1:
                        st.markdown(f"- **{canonical}** ← `{', '.join(variations)}`")
                    else:
                        st.markdown(f"- **{canonical}**")
        
        # Edit each item individually
        for item_idx, item in enumerate(scenario['items']):
            with st.container():
                # Show both original and canonical if different
                param_display = item.get('parameter', '')
                canonical = item.get('parameter_canonical', '')
                
                if canonical and canonical != param_display:
                    st.markdown(f"**Parameter {item_idx + 1}:** `{param_display}` → **{canonical}** ✨")
                else:
                    st.markdown(f"**Parameter {item_idx + 1}:**")
                
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                
                with col1:
                    # Use canonical name if available
                    param_name = st.text_input(
                        "Parameter Name",
                        value=canonical if canonical else param_display,
                        key=f"param_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['parameter'] = param_name
                    if canonical:
                        item['parameter_canonical'] = param_name
                
                with col2:
                    direction = st.selectbox(
                        "Direction",
                        options=["increase", "decrease", "target", "stable", "double", "halve"],
                        index=["increase", "decrease", "target", "stable", "double", "halve"].index(item['direction']) if item['direction'] in ["increase", "decrease", "target", "stable", "double", "halve"] else 0,
                        key=f"dir_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['direction'] = direction
                
                with col3:
                    # Convert value to float if it's string
                    current_value = item.get('value', 0.0)
                    if isinstance(current_value, str):
                        try:
                            current_value = float(current_value) if current_value.strip() else 0.0
                        except:
                            current_value = 0.0
                    elif current_value is None:
                        current_value = 0.0
                    
                    value = st.number_input(
                        "Value",
                        min_value=0.0,
                        max_value=1000000.0,
                        value=float(current_value),
                        step=0.1,
                        key=f"val_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['value'] = value
                
                with col4:
                    unit = st.selectbox(
                        "Unit",
                        options=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"],
                        index=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"].index(item.get('unit', '')) if item.get('unit', '') in ["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"] else 0,
                        key=f"unit_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['unit'] = unit
                    item['value_type'] = 'percent' if unit == '%' else 'absolute'
                
                with col5:
                    if st.button("🗑️", key=f"del_{scenario_idx}_{item_idx}", help="Delete parameter"):
                        scenario['items'].pop(item_idx)
                        st.session_state.detected_scenarios[scenario_idx] = scenario
                        st.rerun()
                
                st.markdown("---")
        
        # Update in session state
        st.session_state.detected_scenarios[scenario_idx] = scenario
        
    else:
        st.info("No parameters detected. Click 'Add Parameter' to add manually.")
    
    # Add parameter button
    if st.button(f"➕ Add Parameter", key=f"add_param_{scenario_idx}"):
        new_item = {
            'parameter': 'New Parameter',
            'direction': 'increase',
            'value': 0.0,
            'unit': '',
            'value_type': 'absolute',
            'confidence': 0.5,
            'source_sentence': 'User-added'
        }
        scenario['items'].append(new_item)
        st.session_state.detected_scenarios[scenario_idx] = scenario
        st.rerun()
                
                with col1:
                    param_name = st.text_input(
                        "Parameter Name",
                        value=item['parameter'],
                        key=f"param_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['parameter'] = param_name
                
                with col2:
                    direction = st.selectbox(
                        "Direction",
                        options=["increase", "decrease", "target", "stable", "double", "halve"],
                        index=["increase", "decrease", "target", "stable", "double", "halve"].index(item['direction']) if item['direction'] in ["increase", "decrease", "target", "stable", "double", "halve"] else 0,
                        key=f"dir_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['direction'] = direction
                
                with col3:
                    # Convert value to float if it's string
                    current_value = item.get('value', 0.0)
                    if isinstance(current_value, str):
                        try:
                            current_value = float(current_value) if current_value.strip() else 0.0
                        except:
                            current_value = 0.0
                    elif current_value is None:
                        current_value = 0.0
                    
                    value = st.number_input(
                        "Value",
                        min_value=0.0,
                        max_value=1000000.0,
                        value=float(current_value),
                        step=0.1,
                        key=f"val_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['value'] = value
                
                with col4:
                    unit = st.selectbox(
                        "Unit",
                        options=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"],
                        index=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"].index(item.get('unit', '')) if item.get('unit', '') in ["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GW", "TWh"] else 0,
                        key=f"unit_{scenario_idx}_{item_idx}",
                        label_visibility="collapsed"
                    )
                    item['unit'] = unit
                    item['value_type'] = 'percent' if unit == '%' else 'absolute'
                
                with col5:
                    if st.button("🗑️", key=f"del_{scenario_idx}_{item_idx}", help="Delete parameter"):
                        scenario['items'].pop(item_idx)
                        st.session_state.detected_scenarios[scenario_idx] = scenario
                        st.rerun()
                
                st.markdown("---")
        
        # Update in session state
        st.session_state.detected_scenarios[scenario_idx] = scenario
        
    else:
        st.info("No parameters detected. Click 'Add Parameter' to add manually.")
    
    # Add parameter button
    if st.button(f"➕ Add Parameter", key=f"add_param_{scenario_idx}"):
        new_item = {
            'parameter': 'New Parameter',
            'direction': 'increase',
            'value': 0.0,
            'unit': '',
            'value_type': 'absolute',
            'confidence': 0.5,
            'source_sentence': 'User-added'
        }
        scenario['items'].append(new_item)
        st.session_state.detected_scenarios[scenario_idx] = scenario
        st.rerun()


def display_mapping_interface():
    """Display interface for mapping scenario parameters to dataset variables"""
    
    st.markdown("Map scenario parameters to your dataset variables:")
    
    # Get available variables
    df_long = st.session_state.df_long
    available_vars = df_long['variable'].unique().tolist()
    
    # Collect all unique parameters from all scenarios
    all_params = set()
    for scenario in st.session_state.detected_scenarios:
        for item in scenario['items']:
            all_params.add(item['parameter'])
    
    all_params = sorted(list(all_params))
    
    if not all_params:
        st.info("No parameters to map. Add parameters in Step 2.")
        return
    
    # Create mapping table
    st.markdown("#### 🔗 Parameter Mapping")
    
    mapping_data = []
    
    for param in all_params:
        # Get suggestions
        suggestions = suggest_variable_mapping(param, available_vars)
        
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown(f"**{param}**")
        
        with col2:
            # Dropdown with suggestions
            options = ['[None]'] + [s['variable'] for s in suggestions]
            
            # Check if already mapped
            current_mapping = st.session_state.scenario_mappings.get(param, '[None]')
            default_idx = options.index(current_mapping) if current_mapping in options else 0
            
            selected = st.selectbox(
                "Map to variable:",
                options=options,
                index=default_idx,
                key=f"map_{param}",
                label_visibility="collapsed"
            )
            
            # Store mapping
            if selected != '[None]':
                st.session_state.scenario_mappings[param] = selected
            elif param in st.session_state.scenario_mappings:
                del st.session_state.scenario_mappings[param]
        
        with col3:
            if suggestions and selected != '[None]':
                # Show similarity score
                matched = [s for s in suggestions if s['variable'] == selected]
                if matched:
                    similarity = matched[0]['similarity']
                    st.metric("Match", f"{similarity:.0%}")


def generate_cleaned_scenarios():
    """Generate cleaned, structured text for all scenarios"""
    
    cleaned = {}
    
    for scenario in st.session_state.detected_scenarios:
        cleaned_text = generate_cleaned_scenario_text(
            scenario,
            mappings=st.session_state.scenario_mappings
        )
        cleaned[scenario['title']] = cleaned_text
    
    st.session_state.cleaned_scenarios = cleaned
    st.success("✅ Cleaned scenarios generated!")


def display_cleaned_scenarios():
    """Display and export cleaned scenario text"""
    
    for title, text in st.session_state.cleaned_scenarios.items():
        with st.expander(f"📄 {title}", expanded=True):
            st.markdown(text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as text
                st.download_button(
                    label="📥 Download as Text",
                    data=text,
                    file_name=f"{title.replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"txt_{title}"
                )
            
            with col2:
                # Download as JSON
                import json
                scenario_data = next(s for s in st.session_state.detected_scenarios if s['title'] == title)
                json_data = json.dumps(scenario_data, indent=2)
                
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"{title.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"json_{title}"
                )


def display_scenario_comparison():
    """Display X-Y comparison plots between scenarios"""
    
    if len(st.session_state.detected_scenarios) < 2:
        st.info("ℹ️ Need at least 2 scenarios to create comparison plots.")
        return
    
    st.markdown("Compare parameter values across scenarios using X-Y plots.")
    
    # Collect all parameters with numeric values
    param_values = {}  # {scenario_title: {parameter: value}}
    
    for scenario in st.session_state.detected_scenarios:
        param_values[scenario['title']] = {}
        for item in scenario['items']:
            if item['value'] is not None:
                param_values[scenario['title']][item['parameter']] = item['value']
    
    # Get common parameters (present in at least 2 scenarios)
    all_params_by_scenario = [set(pv.keys()) for pv in param_values.values()]
    
    if len(all_params_by_scenario) < 2:
        st.warning("Not enough scenarios with parameters to compare.")
        return
    
    # Find parameters present in multiple scenarios
    common_params = set()
    for i, params_i in enumerate(all_params_by_scenario):
        for j, params_j in enumerate(all_params_by_scenario):
            if i < j:
                common_params.update(params_i.intersection(params_j))
    
    common_params = sorted(list(common_params))
    
    if not common_params:
        st.warning("No common parameters found across scenarios.")
        return
    
    # Select parameters for X and Y axes
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox(
            "X-axis parameter:",
            options=common_params,
            index=0
        )
    
    with col2:
        y_param = st.selectbox(
            "Y-axis parameter:",
            options=[p for p in common_params if p != x_param],
            index=0 if len(common_params) > 1 else 0
        )
    
    # Create comparison plot
    fig = go.Figure()
    
    for scenario_title, params in param_values.items():
        if x_param in params and y_param in params:
            fig.add_trace(go.Scatter(
                x=[params[x_param]],
                y=[params[y_param]],
                mode='markers+text',
                name=scenario_title,
                text=[scenario_title],
                textposition="top center",
                marker=dict(size=15),
                showlegend=True
            ))
    
    fig.update_layout(
        title=f"Scenario Comparison: {x_param} vs {y_param}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export plot
    with st.expander("💾 Export Comparison Plot"):
        from core.viz.export import quick_export_buttons
        quick_export_buttons(fig, f"scenario_comparison_{x_param}_vs_{y_param}", ['png', 'pdf', 'html'])


if __name__ == "__main__":
    main()
