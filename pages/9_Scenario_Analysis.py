"""
Page 9: Scenario Analysis (NLP)
Analyze scenario text, extract parameters, map to variables, and compare scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import io
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
    if "show_comparison_table" not in st.session_state:
        st.session_state.show_comparison_table = False


def display_editable_comparison_table():
    """
    Display comparison table in PIVOT format:
    - Parameters in ROWS (first column)
    - Scenarios in COLUMNS (header row)
    - Each cell shows: value, direction, unit
    - Click cell to edit all three at once
    """
    
    st.markdown("### 🔍 Parameter Comparison Table")
    st.caption("Compare parameters across scenarios. Click any cell to edit value, direction, and unit.")
    
    scenarios = st.session_state.detected_scenarios
    
    # Collect all unique parameters
    all_params = set()
    scenario_data = {}  # {scenario_id: {param: item}}
    
    for scenario in scenarios:
        scenario_id = scenario['id']
        scenario_data[scenario_id] = {}
        
        for item in scenario['items']:
            param_name = item.get('parameter_canonical', item.get('parameter', ''))
            all_params.add(param_name)
            scenario_data[scenario_id][param_name] = item
    
    all_params = sorted(list(all_params))
    
    if not all_params:
        st.info("No parameters to compare")
        return
    
    # === BUILD PIVOT TABLE ===
    st.markdown("---")
    
    # Direction symbols
    direction_symbols = {
        'increase': '↑',
        'decrease': '↓',
        'target': '→',
        'stable': '═',
        'double': '⇈',
        'halve': '⇊'
    }
    
    # Build table data for display
    table_data = []
    for param in all_params:
        row = {'Parameter': param}
        
        for scenario in scenarios:
            scenario_title = scenario['title']
            scenario_id = scenario['id']
            
            if param in scenario_data[scenario_id]:
                item = scenario_data[scenario_id][param]
                symbol = direction_symbols.get(item.get('direction', ''), '?')
                value = item.get('value')
                unit = item.get('unit', '')
                
                if value is not None:
                    if unit == '%':
                        cell_value = f"{symbol} {value:.1f}%"
                    elif unit:
                        cell_value = f"{symbol} {value:.2f} {unit}"
                    else:
                        cell_value = f"{symbol} {value:.2f}"
                else:
                    cell_value = f"{symbol} (no value)"
            else:
                cell_value = "—"
            
            row[scenario_title] = cell_value
        
        table_data.append(row)
    
    # Display table
    df_display = pd.DataFrame(table_data)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(all_params) * 40 + 50)
    )
    
    # === EDIT CELL INTERFACE ===
    st.markdown("---")
    st.markdown("### ✏️ Edit Cell")
    st.caption("Select a parameter and scenario to edit its value, direction, and unit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_param = st.selectbox(
            "Select Parameter:",
            options=all_params,
            key="edit_param"
        )
    
    with col2:
        selected_scenario = st.selectbox(
            "Select Scenario:",
            options=[s['title'] for s in scenarios],
            key="edit_scenario"
        )
    
    # Find the selected scenario
    selected_scenario_obj = None
    for scenario in scenarios:
        if scenario['title'] == selected_scenario:
            selected_scenario_obj = scenario
            break
    
    if selected_scenario_obj:
        scenario_id = selected_scenario_obj['id']
        
        # Check if parameter exists in this scenario
        if selected_param in scenario_data[scenario_id]:
            # Parameter exists - edit it
            item = scenario_data[scenario_id][selected_param]
            st.success(f"✅ **{selected_param}** exists in **{selected_scenario}**")
            
            col_val, col_dir, col_unit, col_del = st.columns([2, 2, 2, 1])
            
            with col_val:
                current_value = item.get('value', 0.0)
                if isinstance(current_value, str):
                    try:
                        current_value = float(current_value) if current_value.strip() else 0.0
                    except:
                        current_value = 0.0
                elif current_value is None:
                    current_value = 0.0
                
                new_value = st.number_input(
                    "Value:",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=float(current_value),
                    step=0.1,
                    key="edit_value"
                )
            
            with col_dir:
                new_direction = st.selectbox(
                    "Direction:",
                    options=["increase", "decrease", "target", "stable", "double", "halve"],
                    index=["increase", "decrease", "target", "stable", "double", "halve"].index(item['direction']) if item['direction'] in ["increase", "decrease", "target", "stable", "double", "halve"] else 0,
                    key="edit_direction"
                )
            
            with col_unit:
                new_unit = st.selectbox(
                    "Unit:",
                    options=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GtCO2", "GW", "MW", "TWh"],
                    index=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GtCO2", "GW", "MW", "TWh"].index(item.get('unit', '')) if item.get('unit', '') in ["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GtCO2", "GW", "MW", "TWh"] else 0,
                    key="edit_unit"
                )
            
            with col_del:
                st.markdown("")
                st.markdown("")
                if st.button("🗑️ Delete", key="delete_cell", use_container_width=True):
                    # Delete this parameter from this scenario
                    delete_parameter_from_scenario(selected_scenario_obj, selected_param)
                    st.success(f"✅ Deleted **{selected_param}** from **{selected_scenario}**")
                    st.rerun()
            
            # Update button
            if st.button("💾 Update Cell", type="primary", use_container_width=True):
                update_parameter_in_scenario(selected_scenario_obj, selected_param, new_value, new_direction, new_unit)
                st.success(f"✅ Updated **{selected_param}** in **{selected_scenario}**")
                st.rerun()
        
        else:
            # Parameter doesn't exist - offer to add it
            st.info(f"ℹ️ **{selected_param}** does not exist in **{selected_scenario}**")
            st.markdown("**Add this parameter to the scenario:**")
            
            col_val, col_dir, col_unit = st.columns([2, 2, 2])
            
            with col_val:
                new_value = st.number_input(
                    "Value:",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=0.0,
                    step=0.1,
                    key="add_value"
                )
            
            with col_dir:
                new_direction = st.selectbox(
                    "Direction:",
                    options=["increase", "decrease", "target", "stable", "double", "halve"],
                    index=0,
                    key="add_direction"
                )
            
            with col_unit:
                new_unit = st.selectbox(
                    "Unit:",
                    options=["", "%", "absolute", "billion", "million", "thousand", "MtCO2", "GtCO2", "GW", "MW", "TWh"],
                    index=0,
                    key="add_unit"
                )
            
            if st.button("➕ Add to Scenario", type="primary", use_container_width=True):
                add_parameter_to_scenario(selected_scenario_obj, selected_param, new_value, new_direction, new_unit)
                st.success(f"✅ Added **{selected_param}** to **{selected_scenario}**")
                st.rerun()
    
    # === DELETE PARAMETER FROM ALL SCENARIOS ===
    st.markdown("---")
    st.markdown("### 🗑️ Delete Parameter Completely")
    st.caption("Remove a parameter from ALL scenarios")
    
    col_delete1, col_delete2 = st.columns([3, 1])
    
    with col_delete1:
        param_to_delete = st.selectbox(
            "Select parameter to delete from all scenarios:",
            options=all_params,
            key="delete_all_param"
        )
    
    with col_delete2:
        st.markdown("")
        st.markdown("")
        if st.button("🗑️ Delete from All", key="delete_all_button"):
            delete_parameter_from_all_scenarios(param_to_delete)
            st.success(f"✅ Deleted **{param_to_delete}** from all scenarios")
            st.rerun()
    
    # === MERGE PARAMETERS ===
    st.markdown("---")
    st.markdown("### 🔗 Merge Parameters")
    st.caption("Combine two parameters into one across all scenarios")
    
    col_merge1, col_merge2, col_merge3 = st.columns([2, 2, 1])
    
    with col_merge1:
        merge_from = st.selectbox(
            "Merge this parameter:",
            options=all_params,
            key="merge_from_param"
        )
    
    with col_merge2:
        merge_to = st.selectbox(
            "Into this parameter:",
            options=all_params,
            key="merge_to_param"
        )
    
    with col_merge3:
        st.markdown("")
        st.markdown("")
        if st.button("🔗 Merge"):
            if merge_from == merge_to:
                st.error("Cannot merge a parameter into itself")
            else:
                merge_parameters_in_table(merge_from, merge_to)
                st.success(f"✅ Merged **{merge_from}** into **{merge_to}**")
                st.rerun()
    
    # === EXPORT ===
    st.markdown("---")
    st.markdown("### 📥 Export Table")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        csv_data = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        excel_buffer = io.BytesIO()
        df_display.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp3:
        json_data = df_display.to_json(orient='records')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # === CLOSE BUTTON ===
    st.markdown("---")
    if st.button("❌ Close Comparison Table", use_container_width=True):
        st.session_state.show_comparison_table = False
        st.rerun()


def update_parameter_in_scenario(scenario, param_name, value, direction, unit):
    """Update parameter in a specific scenario"""
    for item in scenario['items']:
        item_param = item.get('parameter_canonical', item.get('parameter', ''))
        if item_param == param_name:
            item['value'] = value
            item['direction'] = direction
            item['unit'] = unit
            item['value_type'] = 'percent' if unit == '%' else 'absolute'
            break
    
    # Update in session state
    for idx, s in enumerate(st.session_state.detected_scenarios):
        if s['id'] == scenario['id']:
            st.session_state.detected_scenarios[idx] = scenario
            break


def add_parameter_to_scenario(scenario, param_name, value, direction, unit):
    """Add new parameter to a scenario"""
    new_item = {
        'parameter': param_name,
        'parameter_canonical': param_name,
        'direction': direction,
        'value': value,
        'unit': unit,
        'value_type': 'percent' if unit == '%' else 'absolute',
        'confidence': 0.5,
        'source_sentence': 'User-added via comparison table',
        'extraction_method': 'user_added'
    }
    
    scenario['items'].append(new_item)
    
    # Update in session state
    for idx, s in enumerate(st.session_state.detected_scenarios):
        if s['id'] == scenario['id']:
            st.session_state.detected_scenarios[idx] = scenario
            break


def delete_parameter_from_scenario(scenario, param_name):
    """Delete parameter from a specific scenario"""
    items_to_keep = []
    
    for item in scenario['items']:
        item_param = item.get('parameter_canonical', item.get('parameter', ''))
        if item_param != param_name:
            items_to_keep.append(item)
    
    scenario['items'] = items_to_keep
    
    # Update in session state
    for idx, s in enumerate(st.session_state.detected_scenarios):
        if s['id'] == scenario['id']:
            st.session_state.detected_scenarios[idx] = scenario
            break


def delete_parameter_from_all_scenarios(param_name):
    """Delete parameter from all scenarios"""
    for scenario in st.session_state.detected_scenarios:
        items_to_keep = []
        
        for item in scenario['items']:
            item_param = item.get('parameter_canonical', item.get('parameter', ''))
            if item_param != param_name:
                items_to_keep.append(item)
        
        scenario['items'] = items_to_keep


def apply_comparison_edits(edited_df: pd.DataFrame):
    """
    Apply edits from comparison table back to original scenarios
    
    Parameters:
    -----------
    edited_df : pd.DataFrame
        Edited comparison table
    """
    
    scenarios = st.session_state.detected_scenarios
    
    # Group edits by scenario
    for _, row in edited_df.iterrows():
        scenario_id = row['scenario_id']
        item_idx = row['item_idx']
        
        # Find the scenario
        for scenario in scenarios:
            if scenario['id'] == scenario_id:
                # Check if item_idx still exists (user might have deleted rows)
                if item_idx < len(scenario['items']):
                    # Update the item
                    scenario['items'][item_idx]['parameter'] = row['Parameter']
                    scenario['items'][item_idx]['parameter_canonical'] = row['Parameter']
                    scenario['items'][item_idx]['direction'] = row['Direction']
                    scenario['items'][item_idx]['value'] = row['Value']
                    scenario['items'][item_idx]['unit'] = row['Unit']
                    scenario['items'][item_idx]['value_type'] = 'percent' if row['Unit'] == '%' else 'absolute'
                break
    
    # Handle deleted rows - if a row is missing from edited_df, remove from scenario
    original_row_ids = set(edited_df['row_id'].tolist())
    
    for scenario in scenarios:
        # Find which items to keep
        items_to_keep = []
        for item_idx, item in enumerate(scenario['items']):
            # Check if this item's row_id is in edited_df
            # We need to reconstruct row_id based on position
            # Actually, let's rebuild based on what's in edited_df
            pass  # This is complex, let's handle deletion differently
    
    # Update session state
    st.session_state.detected_scenarios = scenarios


def merge_parameters_in_table(from_param: str, to_param: str):
    """
    Merge parameters in the scenarios (not just table display)
    """
    
    scenarios = st.session_state.detected_scenarios
    
    for scenario in scenarios:
        # Find items with from_param
        items_to_merge = []
        target_item = None
        
        for item in scenario['items']:
            param_name = item.get('parameter_canonical', item.get('parameter', ''))
            
            if param_name == from_param:
                items_to_merge.append(item)
            elif param_name == to_param:
                target_item = item
        
        # Merge logic
        for item in items_to_merge:
            if target_item:
                # Both exist - merge values if target has no value
                if target_item.get('value') is None and item.get('value') is not None:
                    target_item['value'] = item['value']
                    target_item['unit'] = item['unit']
                    target_item['direction'] = item['direction']
                
                # Remove from_item
                scenario['items'].remove(item)
            else:
                # Only from_item exists - rename it
                item['parameter'] = to_param
                item['parameter_canonical'] = to_param
                target_item = item  # Now this becomes the target
    
    # Update session state
    st.session_state.detected_scenarios = scenarios


def display_parameter_comparison_table():
    """
    Display interactive comparison table across all scenarios
    Allows users to see all parameters side-by-side and merge duplicates
    """
    
    st.markdown("**Compare and align parameters across all scenarios:**")
    
    scenarios = st.session_state.detected_scenarios
    
    # Collect all unique parameters across all scenarios
    all_params = set()
    scenario_params = {}  # {scenario_id: {param_canonical: item}}
    
    for scenario in scenarios:
        scenario_id = scenario['id']
        scenario_params[scenario_id] = {}
        
        for item in scenario['items']:
            # Use canonical name if available, otherwise original
            param_name = item.get('parameter_canonical', item.get('parameter', ''))
            all_params.add(param_name)
            scenario_params[scenario_id][param_name] = item
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    if not all_params:
        st.info("No parameters extracted yet")
        return
    
    # Build comparison table data
    table_data = []
    
    for param in all_params:
        row = {'Parameter': param}
        
        # Add data from each scenario
        for scenario in scenarios:
            scenario_id = scenario['id']
            scenario_title = scenario['title']
            
            if param in scenario_params[scenario_id]:
                item = scenario_params[scenario_id][param]
                
                # Format: "↑ 20%" or "→ 5 MtCO2"
                direction_symbols = {
                    'increase': '↑',
                    'decrease': '↓',
                    'target': '→',
                    'stable': '═',
                    'double': '⇈',
                    'halve': '⇊'
                }
                
                symbol = direction_symbols.get(item.get('direction', ''), '?')
                value = item.get('value')
                unit = item.get('unit', '')
                
                if value is not None:
                    if unit == '%':
                        cell_value = f"{symbol} {value:.1f}%"
                    elif unit:
                        cell_value = f"{symbol} {value:.2f} {unit}"
                    else:
                        cell_value = f"{symbol} {value:.2f}"
                else:
                    cell_value = f"{symbol} (no value)"
                
                row[scenario_title] = cell_value
            else:
                row[scenario_title] = "—"  # Not in this scenario
        
        table_data.append(row)
    
    # Convert to DataFrame
    df_comparison = pd.DataFrame(table_data)
    
    # Display with expandable options
    with st.expander("🔧 Table Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            show_only_common = st.checkbox(
                "Show only parameters common to all scenarios",
                value=False,
                help="Filter to show only parameters that appear in every scenario"
            )
        
        with col2:
            highlight_differences = st.checkbox(
                "Highlight inconsistencies",
                value=True,
                help="Highlight parameters with different units or missing values"
            )
    
    # Filter if needed
    if show_only_common:
        # Keep only rows where all scenarios have data
        scenario_cols = [s['title'] for s in scenarios]
        df_filtered = df_comparison[
            df_comparison[scenario_cols].apply(
                lambda row: all(val != "—" for val in row), 
                axis=1
            )
        ]
    else:
        df_filtered = df_comparison
    
    # Display table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # === MERGE PARAMETERS FEATURE ===
    st.markdown("---")
    st.markdown("### 🔗 Merge Similar Parameters")
    st.caption("Combine parameters that refer to the same thing but have different names")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        param_to_merge_from = st.selectbox(
            "Merge this parameter:",
            options=all_params,
            key="merge_from",
            help="Select the parameter to merge (will be removed)"
        )
    
    with col2:
        param_to_merge_to = st.selectbox(
            "Into this parameter:",
            options=all_params,
            key="merge_to",
            help="Select the target parameter (will keep this one)"
        )
    
    with col3:
        st.markdown("")  # Spacing
        st.markdown("")  # Spacing
        if st.button("🔗 Merge"):
            if param_to_merge_from == param_to_merge_to:
                st.error("Cannot merge a parameter into itself!")
            else:
                merge_parameters(param_to_merge_from, param_to_merge_to)
                st.success(f"✅ Merged '{param_to_merge_from}' into '{param_to_merge_to}'")
                st.rerun()
    
    # === EXPORT COMPARISON TABLE ===
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export2:
        excel_buffer = io.BytesIO()
        df_filtered.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col_export3:
        json_data = df_filtered.to_json(orient='records')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def merge_parameters(from_param: str, to_param: str):
    """
    Merge one parameter into another across all scenarios
    
    Parameters:
    -----------
    from_param : str
        Parameter to merge (will be removed)
    to_param : str
        Target parameter (will be kept)
    """
    
    scenarios = st.session_state.detected_scenarios
    
    for scenario in scenarios:
        # Find items to merge
        from_item = None
        to_item = None
        from_idx = None
        
        for idx, item in enumerate(scenario['items']):
            param_name = item.get('parameter_canonical', item.get('parameter', ''))
            
            if param_name == from_param:
                from_item = item
                from_idx = idx
            elif param_name == to_param:
                to_item = item
        
        # Merge logic
        if from_item and to_item:
            # Both exist - merge values if needed
            if to_item.get('value') is None and from_item.get('value') is not None:
                to_item['value'] = from_item['value']
                to_item['unit'] = from_item['unit']
                to_item['direction'] = from_item['direction']
            
            # Remove from_item
            scenario['items'].pop(from_idx)
        
        elif from_item and not to_item:
            # Only from_item exists - rename it
            from_item['parameter'] = to_param
            from_item['parameter_canonical'] = to_param
    
    # Update session state
    st.session_state.detected_scenarios = scenarios


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
    
    # === GLiNER STATUS DISPLAY ===
    try:
        from core.nlp.ml_extractor import get_gliner_status
        gliner_status = get_gliner_status()
        
        col_status1, col_status2 = st.columns([3, 1])
        
        with col_status1:
            if gliner_status['available']:
                if gliner_status['model_loaded']:
                    st.success("✅ **Hybrid AI Mode Active:** GLiNER ML + Option B (Best Accuracy: ~85%)")
                else:
                    st.info("🔄 **Hybrid AI Mode Ready:** GLiNER will load automatically on first use")
            else:
                st.warning("⚠️ **Option B Mode Only:** Using Templates + spaCy + Regex (~70% accuracy)")
                
        with col_status2:
            if not gliner_status['available']:
                if st.button("📥 How to Install GLiNER"):
                    st.code("pip install gliner --break-system-packages", language="bash")
                    st.info("Run this command, then restart the app for AI-powered extraction!")
    
    except Exception:
        # Silently continue if status check fails
        pass
    
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
        
        # === NEW: COMPARISON TABLE (After User Approval) ===
        st.markdown("---")
        st.subheader("📊 Step 2b: Compare & Align Parameters Across Scenarios")
        
        if len(st.session_state.detected_scenarios) >= 2:
            st.markdown("Review all parameters from all scenarios in one table:")
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                if st.button("🔍 Compare Parameters Across Scenarios", type="primary", use_container_width=True):
                    st.session_state.show_comparison_table = True
                    st.rerun()
            
            with col2:
                st.caption("Click to see an editable comparison table of all parameters from all scenarios")
            
            # Show table only if button was clicked
            if st.session_state.get('show_comparison_table', False):
                display_editable_comparison_table()
        else:
            st.info("ℹ️ Need at least 2 scenarios to show comparison table")
        
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
        items_to_delete = []  # Track items to delete
        
        for item_idx, item in enumerate(scenario['items']):
            with st.container():
                # Show both original and canonical if different
                param_display = item.get('parameter', '')
                canonical = item.get('parameter_canonical', '')
                extraction_method = item.get('extraction_method', 'unknown')
                source_sentence = item.get('source_sentence', '')
                
                # Method badges
                method_badges = {
                    'hybrid_both': '🟢 AI+Templates',
                    'gliner_only': '🟡 AI Only',
                    'optionb_only': '🔵 Templates Only',
                    'optionb': '⚪ Templates',
                    'templates': '📋 Template',
                    'spacy': '🔤 spaCy',
                    'regex': '🔍 Regex'
                }
                
                method_badge = method_badges.get(extraction_method, f'❓ {extraction_method}')
                
                if canonical and canonical != param_display:
                    st.markdown(f"**Parameter {item_idx + 1}:** `{param_display}` → **{canonical}** ✨ | {method_badge}")
                else:
                    st.markdown(f"**Parameter {item_idx + 1}:** {method_badge}")
                
                # Show source sentence if available
                if source_sentence:
                    # Try to highlight the relevant part
                    # Look for the parameter name in the sentence
                    param_to_find = item.get('parameter_original', item.get('parameter', ''))
                    
                    # Find where parameter appears in sentence
                    if param_to_find.lower() in source_sentence.lower():
                        # Find position
                        idx = source_sentence.lower().find(param_to_find.lower())
                        
                        # Extract context around it (50 chars before and after)
                        start = max(0, idx - 50)
                        end = min(len(source_sentence), idx + len(param_to_find) + 50)
                        
                        context_snippet = source_sentence[start:end]
                        
                        # Add ellipsis if truncated
                        if start > 0:
                            context_snippet = "..." + context_snippet
                        if end < len(source_sentence):
                            context_snippet = context_snippet + "..."
                        
                        st.caption(f"📝 *\"{context_snippet}\"*")
                    else:
                        # Fallback: show truncated sentence
                        if len(source_sentence) > 150:
                            display_sentence = source_sentence[:150] + "..."
                        else:
                            display_sentence = source_sentence
                        
                        st.caption(f"📝 *\"{display_sentence}\"*")
                
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
                        items_to_delete.append(item_idx)
                
                st.markdown("---")
        
        # Delete items after iteration (to avoid index issues)
        if items_to_delete:
            # Sort in reverse order to delete from end to start
            for idx in sorted(items_to_delete, reverse=True):
                scenario['items'].pop(idx)
            st.session_state.detected_scenarios[scenario_idx] = scenario
            st.rerun()
        
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
