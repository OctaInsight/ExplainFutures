"""
Page 10: Scenario Analysis (NLP) - PRODUCTION v3.2
Enhanced with Phase 1-3 improvements + Full Database Integration
- Proper authentication and project checks (using Page 2 pattern)
- Load existing scenarios or create new
- Fully editable tables and fields
- Save to database with progress tracking and project statistics
- Update or append scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import io
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Scenario Analysis (NLP)",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Authentication check - using Page 2 pattern
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.database.supabase_manager import get_db_manager

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

# Get database manager - using Page 2 pattern
try:
    db = get_db_manager()
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False
    st.error("⚠️ Database not available")

# Render shared sidebar
render_app_sidebar()


def check_existing_scenarios(project_id: str) -> Tuple[bool, int, List[Dict]]:
    """
    Check if project has existing scenarios
    
    Returns:
    --------
    (has_scenarios, count, scenario_list)
    """
    if not DB_AVAILABLE:
        return False, 0, []
    
    try:
        result = db.client.table('scenarios')\
            .select('scenario_id, title, horizon, baseline_year, created_at')\
            .eq('project_id', project_id)\
            .execute()
        
        scenarios = result.data if result.data else []
        has_scenarios = len(scenarios) > 0
        
        return has_scenarios, len(scenarios), scenarios
        
    except Exception as e:
        st.error(f"Error checking scenarios: {str(e)}")
        return False, 0, []


def load_scenarios_from_database(project_id: str) -> List[Dict]:
    """Load complete scenarios with all Phase 2 fields from database"""
    if not DB_AVAILABLE:
        return []
    
    try:
        scenarios_result = db.client.table('scenarios')\
            .select('*')\
            .eq('project_id', project_id)\
            .order('created_at', desc=False)\
            .execute()
        
        if not scenarios_result.data:
            return []
        
        scenarios = []
        
        for scenario_row in scenarios_result.data:
            params_result = db.client.table('scenario_parameters')\
                .select('*')\
                .eq('scenario_id', scenario_row['scenario_id'])\
                .execute()
            
            items = []
            for param_row in params_result.data:
                item = {
                    'id': param_row['scenario_param_id'],
                    'parameter': param_row['parameter_name'],
                    'parameter_canonical': param_row.get('parameter_canonical', param_row['parameter_name']),
                    'parameter_original': param_row.get('parameter_original', param_row['parameter_name']),
                    'category': param_row.get('category', 'other'),
                    'value': param_row.get('value'),
                    'unit': param_row.get('unit', ''),
                    'direction': param_row.get('direction', 'target'),
                    'value_type': param_row.get('value_type', 'absolute_target'),
                    'baseline_year': param_row.get('baseline_year'),
                    'target_year': param_row.get('target_year'),
                    'time_expression': param_row.get('time_expression', ''),
                    'time_confidence': param_row.get('time_confidence'),
                    'value_min': param_row.get('value_min'),
                    'value_max': param_row.get('value_max'),
                    'base_value': param_row.get('base_value'),
                    'target_value': param_row.get('target_value'),
                    'is_range': param_row.get('is_range', False),
                    'is_rate': param_row.get('is_rate', False),
                    'rate_period': param_row.get('rate_period', ''),
                    'confidence': param_row.get('confidence'),
                    'extraction_method': param_row.get('extraction_method', 'legacy'),
                    'source_sentence': param_row.get('source_sentence', ''),
                    'unification_confidence': param_row.get('unification_confidence'),
                    'template': param_row.get('template_used', ''),
                    'horizon': param_row.get('target_year', scenario_row.get('horizon'))
                }
                items.append(item)
            
            scenario = {
                'id': scenario_row['scenario_id'],
                'title': scenario_row['title'],
                'text': scenario_row.get('source_text', scenario_row.get('description', '')),
                'horizon': scenario_row.get('horizon'),
                'baseline_year': scenario_row.get('baseline_year'),
                'time_confidence': scenario_row.get('time_confidence'),
                'items': items,
                'from_database': True
            }
            
            scenarios.append(scenario)
        
        return scenarios
        
    except Exception as e:
        st.error(f"Error loading scenarios: {str(e)}")
        return []


def save_scenarios_to_database(
    scenarios: List[Dict],
    project_id: str,
    user_id: str,
    mode: str = 'append'
) -> Tuple[bool, str]:
    """
    Save scenarios to database with full Phase 2 fields and project statistics
    
    Parameters:
    -----------
    scenarios : List[Dict]
        List of scenario dictionaries with items
    project_id : str
        Current project ID
    user_id : str
        Current user ID
    mode : str
        'append' (add to existing) or 'replace' (delete then insert)
    
    Returns:
    --------
    (success: bool, message: str)
    """
    if not DB_AVAILABLE:
        return False, "Database not available"
    
    try:
        if mode == 'replace':
            existing_scenarios = db.client.table('scenarios')\
                .select('scenario_id')\
                .eq('project_id', project_id)\
                .execute()
            
            for scenario in existing_scenarios.data:
                db.client.table('scenario_parameters')\
                    .delete()\
                    .eq('scenario_id', scenario['scenario_id'])\
                    .execute()
            
            db.client.table('scenarios')\
                .delete()\
                .eq('project_id', project_id)\
                .execute()
        
        saved_count = 0
        min_baseline = None
        max_target = None
        
        for scenario in scenarios:
            scenario_id = scenario.get('id') or str(uuid.uuid4())
            
            scenario_data = {
                'scenario_id': scenario_id,
                'project_id': project_id,
                'user_id': user_id,
                'title': scenario.get('title', 'Untitled Scenario'),
                'description': scenario.get('text', ''),
                'source_text': scenario.get('text', ''),
                'horizon': scenario.get('horizon'),
                'baseline_year': scenario.get('baseline_year'),
                'time_confidence': scenario.get('time_confidence'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            db.client.table('scenarios').insert(scenario_data).execute()
            
            if scenario.get('baseline_year'):
                if min_baseline is None or scenario['baseline_year'] < min_baseline:
                    min_baseline = scenario['baseline_year']
            
            if scenario.get('horizon'):
                if max_target is None or scenario['horizon'] > max_target:
                    max_target = scenario['horizon']
            
            for item in scenario.get('items', []):
                param_id = item.get('id') or str(uuid.uuid4())
                
                param_data = {
                    'scenario_param_id': param_id,
                    'scenario_id': scenario_id,
                    'parameter_name': item.get('parameter', 'Unknown'),
                    'parameter_canonical': item.get('parameter_canonical', item.get('parameter', 'Unknown')),
                    'parameter_original': item.get('parameter_original', item.get('parameter', 'Unknown')),
                    'category': item.get('category', 'other'),
                    'value': item.get('value'),
                    'unit': item.get('unit', ''),
                    'direction': item.get('direction', 'target'),
                    'value_type': item.get('value_type', 'absolute_target'),
                    'baseline_year': item.get('baseline_year'),
                    'target_year': item.get('target_year'),
                    'time_expression': item.get('time_expression', ''),
                    'time_confidence': item.get('time_confidence'),
                    'value_min': item.get('value_min'),
                    'value_max': item.get('value_max'),
                    'base_value': item.get('base_value'),
                    'target_value': item.get('target_value'),
                    'is_range': item.get('is_range', False),
                    'is_rate': item.get('is_rate', False),
                    'rate_period': item.get('rate_period', ''),
                    'confidence': item.get('confidence'),
                    'extraction_method': item.get('extraction_method', 'legacy'),
                    'source_sentence': item.get('source_sentence', ''),
                    'unification_confidence': item.get('unification_confidence'),
                    'template_used': item.get('template', ''),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                db.client.table('scenario_parameters').insert(param_data).execute()
            
            saved_count += 1
        
        project_updates = {}
        if min_baseline is not None:
            project_updates['baseline_year'] = min_baseline
        if max_target is not None:
            project_updates['target_year'] = max_target
        project_updates['total_scenarios'] = saved_count if mode == 'replace' else None
        
        if project_updates:
            if mode == 'append':
                existing_count_result = db.client.table('scenarios')\
                    .select('scenario_id', count='exact')\
                    .eq('project_id', project_id)\
                    .execute()
                project_updates['total_scenarios'] = existing_count_result.count
            
            db.client.table('projects')\
                .update(project_updates)\
                .eq('project_id', project_id)\
                .execute()
        
        db.upsert_progress_step(project_id, "scenarios_analyzed", 7)
        db.recompute_and_update_project_progress(project_id)
        
        mode_text = "replaced" if mode == 'replace' else "saved"
        return True, f"Successfully {mode_text} {saved_count} scenario(s)"
        
    except Exception as e:
        return False, f"Error saving scenarios: {str(e)}"


def initialize_scenario_state():
    """Initialize session state for scenario analysis"""
    if 'detected_scenarios' not in st.session_state:
        st.session_state.detected_scenarios = []
    if 'scenarios_processed' not in st.session_state:
        st.session_state.scenarios_processed = False
    if 'scenario_text_input' not in st.session_state:
        st.session_state.scenario_text_input = ""
    if 'enable_ml_extraction' not in st.session_state:
        st.session_state.enable_ml_extraction = False
    if 'ml_confidence_threshold' not in st.session_state:
        st.session_state.ml_confidence_threshold = 0.7


def render_scenario_table(scenario: Dict, scenario_idx: int):
    """Render editable table for a single scenario"""
    st.markdown(f"### {scenario['title']}")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.caption(f"📅 Baseline Year: {scenario.get('baseline_year', 'N/A')}")
    with col_info2:
        st.caption(f"🎯 Target Year: {scenario.get('horizon', 'N/A')}")
    with col_info3:
        if scenario.get('time_confidence'):
            st.caption(f"✓ Time Confidence: {scenario.get('time_confidence', 0):.2f}")
    
    if not scenario.get('items'):
        st.info("No parameters extracted for this scenario")
        return
    
    df = pd.DataFrame(scenario['items'])
    
    display_columns = [
        'parameter_canonical', 'value', 'unit', 'direction',
        'value_type', 'baseline_year', 'target_year',
        'confidence', 'category'
    ]
    
    available_cols = [col for col in display_columns if col in df.columns]
    df_display = df[available_cols].copy()
    
    rename_map = {
        'parameter_canonical': 'Parameter',
        'value': 'Value',
        'unit': 'Unit',
        'direction': 'Direction',
        'value_type': 'Value Type',
        'baseline_year': 'Baseline Year',
        'target_year': 'Target Year',
        'confidence': 'Confidence',
        'category': 'Category'
    }
    df_display = df_display.rename(columns=rename_map)
    
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        num_rows="dynamic",
        key=f"scenario_table_{scenario_idx}",
        column_config={
            "Confidence": st.column_config.NumberColumn(
                format="%.2f",
                min_value=0.0,
                max_value=1.0
            ),
            "Value": st.column_config.NumberColumn(
                format="%.2f"
            )
        }
    )
    
    reverse_rename = {v: k for k, v in rename_map.items()}
    edited_df = edited_df.rename(columns=reverse_rename)
    
    for idx, row in edited_df.iterrows():
        for col in edited_df.columns:
            if col in scenario['items'][idx]:
                scenario['items'][idx][col] = row[col]


# Main page logic
st.title("📊 Scenario Analysis (NLP)")
st.markdown("*Natural Language Processing for Scenario Parameter Extraction*")
st.markdown("---")

# Check if project is selected
if not st.session_state.get('current_project_id'):
    st.warning("⚠️ No project selected")
    st.info("Please select or create a project from the Home page")
    
    if st.button("← Go to Home"):
        st.switch_page("pages/01_Home.py")
    st.stop()

# Initialize state
initialize_scenario_state()

project_id = st.session_state.current_project_id
user_id = st.session_state.get('user_id')

# Check for existing scenarios
if DB_AVAILABLE:
    has_existing, existing_count, existing_list = check_existing_scenarios(project_id)
else:
    has_existing = False
    existing_count = 0
    existing_list = []

# Display existing scenarios info
if has_existing:
    st.info(f"📋 This project has {existing_count} existing scenario(s)")
    
    col_load, col_new = st.columns(2)
    
    with col_load:
        if st.button("📥 Load Existing Scenarios", use_container_width=True, type="primary"):
            with st.spinner("Loading scenarios from database..."):
                loaded_scenarios = load_scenarios_from_database(project_id)
                if loaded_scenarios:
                    st.session_state.detected_scenarios = loaded_scenarios
                    st.session_state.scenarios_processed = True
                    st.success(f"✅ Loaded {len(loaded_scenarios)} scenario(s)")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ No scenarios could be loaded")
    
    with col_new:
        if st.button("➕ Create New Scenarios", use_container_width=True):
            st.session_state.detected_scenarios = []
            st.session_state.scenarios_processed = False
            st.session_state.scenario_text_input = ""
            st.rerun()
    
    st.markdown("---")

# ML Settings
with st.expander("⚙️ ML Extraction Settings", expanded=False):
    st.markdown("### Machine Learning Configuration")
    
    ml_enabled = st.checkbox(
        "Enable ML-based extraction",
        value=st.session_state.enable_ml_extraction,
        help="Use machine learning models for parameter extraction (requires additional dependencies)"
    )
    if ml_enabled != st.session_state.enable_ml_extraction:
        st.session_state.enable_ml_extraction = ml_enabled
    
    if st.session_state.enable_ml_extraction:
        ml_threshold = st.slider(
            "ML Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=st.session_state.ml_confidence_threshold,
            step=0.05,
            help="Minimum confidence score for ML predictions"
        )
        if ml_threshold != st.session_state.ml_confidence_threshold:
            st.session_state.ml_confidence_threshold = ml_threshold

st.markdown("---")

st.markdown("### 📄 Enter Scenario Text")

scenario_text = st.text_area(
    "Paste your scenario text here",
    value=st.session_state.scenario_text_input,
    height=300,
    placeholder="""Example:

Scenario 1: Green Growth (by 2050)
GDP increases by 25%.
CO2 emissions decrease by 60%.
Renewable energy reaches 75%.

Scenario 2: Business as Usual (by 2050)
GDP increases by 15%.
CO2 emissions increase by 10%.
Renewable energy reaches 35%."""
)

if scenario_text != st.session_state.scenario_text_input:
    st.session_state.scenario_text_input = scenario_text

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("🔍 **Extract Parameters**", use_container_width=True, type="primary", disabled=not scenario_text.strip()):
        with st.spinner("Extracting parameters from scenarios..."):
            try:
                segments = segment_scenarios(scenario_text)
                
                extracted_scenarios = extract_parameters_from_scenarios(
                    segments,
                    enable_ml=st.session_state.enable_ml_extraction,
                    ml_confidence_threshold=st.session_state.ml_confidence_threshold
                )
                
                st.session_state.detected_scenarios = extracted_scenarios
                st.session_state.scenarios_processed = True
                
                st.success(f"✅ Extracted {len(extracted_scenarios)} scenarios")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during extraction: {str(e)}")
                st.exception(e)

with col_btn2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.scenario_text_input = ""
        st.session_state.detected_scenarios = []
        st.session_state.scenarios_processed = False
        st.rerun()

st.markdown("---")

# Display extracted/loaded scenarios
if st.session_state.scenarios_processed and st.session_state.detected_scenarios:
    st.markdown("## 📋 Extracted Scenarios")
    
    st.info(f"✅ {len(st.session_state.detected_scenarios)} scenario(s) ready")
    
    for idx, scenario in enumerate(st.session_state.detected_scenarios):
        with st.expander(f"**{scenario['title']}**", expanded=(idx == 0)):
            render_scenario_table(scenario, idx)
    
    st.markdown("---")
    
    st.markdown("### 💾 Save to Database")
    
    col_save1, col_save2 = st.columns([1, 1])
    
    with col_save1:
        save_mode = st.radio(
            "Save mode:",
            options=['append', 'replace'],
            format_func=lambda x: '➕ Append to existing scenarios' if x == 'append' else '🔄 Replace all existing scenarios',
            horizontal=False
        )
    
    with col_save2:
        st.markdown("")
        st.markdown("")
        if st.button("💾 **Save Scenarios**", use_container_width=True, type="primary"):
            if not user_id:
                st.error("❌ User ID not found. Please log in again.")
            else:
                with st.spinner(f"Saving scenarios ({save_mode})..."):
                    success, message = save_scenarios_to_database(
                        st.session_state.detected_scenarios,
                        project_id,
                        user_id,
                        mode=save_mode
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

else:
    st.info("👆 Enter scenario text above and click 'Extract Parameters' to begin")

# Debug sections
with st.expander("🔬 Phase 2 Enhanced Fields (Debug)", expanded=False):
    st.caption("View time extraction and value semantics details")
    
    if st.session_state.detected_scenarios:
        for scenario in st.session_state.detected_scenarios:
            st.markdown(f"**{scenario['title']}**")
            
            if scenario.get('baseline_year') or scenario.get('horizon'):
                st.text(f"⏰ Scenario time: {scenario.get('baseline_year', 'N/A')} → {scenario.get('horizon', 'N/A')}")
            
            for item in scenario.get('items', []):
                param = item.get('parameter_canonical', item.get('parameter', 'Unknown'))
                st.markdown(f"**{param}**")
                
                cols = st.columns(3)
                with cols[0]:
                    st.text(f"Value Type: {item.get('value_type', 'N/A')}")
                    if item.get('is_range'):
                        st.text(f"Range: {item.get('value_min', 'N/A')}-{item.get('value_max', 'N/A')}")
                    if item.get('is_rate'):
                        st.text(f"Rate: {item.get('rate_period', 'N/A')}")
                
                with cols[1]:
                    time_expr = item.get('time_expression', '')
                    if time_expr:
                        st.text(f"Time: {time_expr}")
                    if item.get('baseline_year'):
                        st.text(f"Baseline: {item['baseline_year']}")
                    if item.get('target_year'):
                        st.text(f"Target: {item['target_year']}")
                
                with cols[2]:
                    st.text(f"Confidence: {item.get('confidence', 0):.2f}")
                    st.text(f"Method: {item.get('extraction_method', 'N/A')}")
                
                st.markdown("---")
    else:
        st.info("No scenarios to display")

with st.expander("🧪 Phase 3 Evaluation Harness (Debug)", expanded=False):
    st.caption("Run regression tests and quality metrics on gold test cases")
    
    st.markdown("### Run Evaluation")
    st.info("⚙️ This will test the extraction pipeline against gold standard test cases")
    
    col_eval1, col_eval2, col_eval3 = st.columns([2, 2, 1])
    
    with col_eval1:
        eval_enable_ml = st.checkbox(
            "Enable ML for evaluation",
            value=st.session_state.enable_ml_extraction,
            key="eval_enable_ml",
            help="Run evaluation with ML extraction enabled"
        )
    
    with col_eval2:
        eval_threshold = st.slider(
            "ML Threshold for evaluation",
            min_value=0.3,
            max_value=0.9,
            value=st.session_state.ml_confidence_threshold,
            step=0.05,
            key="eval_threshold",
            help="ML confidence threshold for evaluation"
        )
    
    with col_eval3:
        if st.button("▶️ Run Tests", use_container_width=True):
            with st.spinner("Running evaluation..."):
                try:
                    tests_dir = Path(__file__).parent.parent / "tests"
                    if str(tests_dir) not in sys.path:
                        sys.path.insert(0, str(tests_dir))
                    
                    from evaluate_extraction import run_evaluation, generate_report
                    
                    metrics, results = run_evaluation(
                        enable_ml=eval_enable_ml,
                        ml_threshold=eval_threshold,
                        verbose=False
                    )
                    
                    st.markdown("### Evaluation Results")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Overall Pass Rate", f"{metrics.overall_pass_rate():.1%}")
                    with col_m2:
                        st.metric("Tests Passed", f"{metrics.passed_cases}/{metrics.total_cases}")
                    with col_m3:
                        failed = metrics.total_cases - metrics.passed_cases
                        st.metric("Tests Failed", failed, delta=-failed if failed > 0 else 0)
                    
                    failed_results = [r for r in results if not r['passed']]
                    if failed_results:
                        st.markdown("### ❌ Failed Cases")
                        for result in failed_results[:5]:
                            with st.expander(f"{result['case_id']}: {result['description']}", expanded=False):
                                st.markdown("**Errors:**")
                                for error in result['errors']:
                                    st.text(f"• {error}")
                    else:
                        st.success("✅ All tests passed!")
                    
                    cases_with_warnings = [r for r in results if r['warnings']]
                    if cases_with_warnings:
                        st.markdown("### ⚠️ Warnings")
                        with st.expander(f"{len(cases_with_warnings)} cases with warnings", expanded=False):
                            for result in cases_with_warnings[:5]:
                                st.markdown(f"**{result['case_id']}:**")
                                for warning in result['warnings']:
                                    st.text(f"• {warning}")
                    
                    report = generate_report(metrics, results)
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report,
                        file_name="evaluation_report.txt",
                        mime="text/plain"
                    )
                    
                except ImportError as e:
                    st.error(f"❌ Could not import evaluation module: {e}")
                    st.info("Make sure `tests/evaluate_extraction.py` exists in your project directory")
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.exception(e)

st.markdown("---")
st.caption("💡 **Tip**: All tables and fields are editable. Click any cell to modify values.")
st.caption("🔄 Project statistics (baseline year, target year, total scenarios) are automatically updated on save.")
st.caption("📊 Progress tracking: Each save adds 7% to 'scenarios_analyzed' step.")
