"""
Page 10: Scenario Analysis (NLP) - COMPLETE v3.1
Enhanced with Phase 1-3 improvements + Full Database Integration
- Proper project/user data loading
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

st.set_page_config(
    page_title="Scenario Analysis (NLP)",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

from core.nlp.lang_detect import detect_language
from core.nlp.scenario_segment import segment_scenarios
from core.nlp.parameter_extract import extract_parameters_from_scenarios
from core.nlp.value_parse import parse_value_string, normalize_unit, format_value
from core.nlp.mapping import suggest_variable_mapping, create_mapping
from core.nlp.clean_text import generate_cleaned_scenario_text
from core.nlp.schema import Scenario, ScenarioItem

initialize_session_state()
config = get_config()

render_app_sidebar()

if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()


def get_supabase_client():
    """Get Supabase client"""
    from supabase import create_client
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_KEY"]
    )


def load_project_and_user_data():
    """
    Load project and user data from database
    Populate session state with project_id, user_id, and project details
    """
    try:
        supabase = get_supabase_client()
        
        user_id = st.session_state.get('user_id')
        if not user_id:
            auth_user = supabase.auth.get_user()
            if auth_user and auth_user.user:
                user_id = auth_user.user.id
                st.session_state.user_id = user_id
        
        if not user_id:
            return False, "Could not determine user ID"
        
        user_result = supabase.table('users').select('*').eq('user_id', user_id).execute()
        if user_result.data:
            st.session_state.user_data = user_result.data[0]
        
        project_id = st.session_state.get('project_id')
        
        if not project_id:
            projects_result = supabase.table('projects')\
                .select('*')\
                .eq('owner_id', user_id)\
                .eq('status', 'active')\
                .order('last_accessed', desc=True)\
                .limit(1)\
                .execute()
            
            if projects_result.data:
                project_id = projects_result.data[0]['project_id']
                st.session_state.project_id = project_id
                st.session_state.project_data = projects_result.data[0]
            else:
                return False, "No active project found. Please create or select a project first."
        else:
            project_result = supabase.table('projects').select('*').eq('project_id', project_id).execute()
            if project_result.data:
                st.session_state.project_data = project_result.data[0]
            else:
                return False, "Project not found"
        
        supabase.table('projects')\
            .update({'last_accessed': datetime.now().isoformat()})\
            .eq('project_id', project_id)\
            .execute()
        
        return True, "Success"
        
    except Exception as e:
        return False, f"Error loading project data: {str(e)}"


def check_existing_scenarios(project_id: str) -> Tuple[bool, int, List[Dict]]:
    """
    Check if project has existing scenarios
    
    Returns:
    --------
    (has_scenarios, count, scenario_list)
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('scenarios')\
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
    try:
        supabase = get_supabase_client()
        
        scenarios_result = supabase.table('scenarios')\
            .select('*')\
            .eq('project_id', project_id)\
            .order('created_at', desc=False)\
            .execute()
        
        if not scenarios_result.data:
            return []
        
        scenarios = []
        
        for scenario_row in scenarios_result.data:
            params_result = supabase.table('scenario_parameters')\
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


def calculate_project_statistics(scenarios: List[Dict]) -> Dict:
    """
    Calculate project-level statistics from scenarios
    
    Returns:
    --------
    dict with baseline_year, scenario_target_year, total_scenarios
    """
    if not scenarios:
        return {
            'baseline_year': None,
            'scenario_target_year': None,
            'total_scenarios': 0
        }
    
    baseline_years = [s.get('baseline_year') for s in scenarios if s.get('baseline_year')]
    target_years = [s.get('horizon') for s in scenarios if s.get('horizon')]
    
    earliest_baseline = min(baseline_years) if baseline_years else None
    latest_target = max(target_years) if target_years else None
    
    return {
        'baseline_year': earliest_baseline,
        'scenario_target_year': latest_target,
        'total_scenarios': len(scenarios)
    }


def save_scenarios_to_database(scenarios: List[Dict], project_id: str, user_id: str, mode: str = 'update') -> bool:
    """
    Save scenarios to database with Phase 2 fields and update project statistics
    
    Parameters:
    -----------
    scenarios : list
        List of scenario dictionaries
    project_id : str
        Project UUID
    user_id : str
        User UUID
    mode : str
        'update' to replace existing or 'append' to add new
    
    Returns:
    --------
    success : bool
    """
    try:
        supabase = get_supabase_client()
        
        if mode == 'update':
            existing_result = supabase.table('scenarios')\
                .select('scenario_id')\
                .eq('project_id', project_id)\
                .execute()
            
            for old_scenario in existing_result.data:
                supabase.table('scenario_parameters')\
                    .delete()\
                    .eq('scenario_id', old_scenario['scenario_id'])\
                    .execute()
                
                supabase.table('scenarios')\
                    .delete()\
                    .eq('scenario_id', old_scenario['scenario_id'])\
                    .execute()
        
        for scenario in scenarios:
            scenario_id = scenario.get('id')
            if not scenario_id or not scenario.get('from_database'):
                scenario_id = str(uuid.uuid4())
            
            scenario_data = {
                'scenario_id': scenario_id,
                'project_id': project_id,
                'created_by': user_id,
                'title': scenario['title'],
                'description': scenario.get('text', '')[:500],
                'horizon': scenario.get('horizon'),
                'baseline_year': scenario.get('baseline_year'),
                'time_confidence': scenario.get('time_confidence'),
                'source_text': scenario.get('text', ''),
                'extraction_config': {
                    'ml_enabled': st.session_state.get('enable_ml_extraction', True),
                    'ml_threshold': st.session_state.get('ml_confidence_threshold', 0.5)
                },
                'ml_enabled': st.session_state.get('enable_ml_extraction', True),
                'ml_threshold': st.session_state.get('ml_confidence_threshold', 0.5),
                'total_items_extracted': len(scenario.get('items', [])),
                'high_confidence_items': sum(1 for i in scenario.get('items', []) if i.get('confidence', 0) > 0.8),
                'extraction_version': 'v3.0',
                'updated_at': datetime.now().isoformat()
            }
            
            supabase.table('scenarios').upsert(scenario_data).execute()
            
            for item in scenario.get('items', []):
                param_id = item.get('id')
                if not param_id:
                    param_id = str(uuid.uuid4())
                
                param_data = {
                    'scenario_param_id': param_id,
                    'scenario_id': scenario_id,
                    'project_id': project_id,
                    'parameter_name': item['parameter'],
                    'parameter_canonical': item.get('parameter_canonical', item['parameter']),
                    'parameter_original': item.get('parameter_original', item['parameter']),
                    'category': item.get('category'),
                    'value': float(item['value']) if item.get('value') is not None else None,
                    'unit': item.get('unit'),
                    'direction': item.get('direction'),
                    'value_type': item.get('value_type'),
                    'baseline_year': item.get('baseline_year'),
                    'target_year': item.get('target_year'),
                    'time_expression': item.get('time_expression'),
                    'time_confidence': item.get('time_confidence'),
                    'value_min': item.get('value_min'),
                    'value_max': item.get('value_max'),
                    'base_value': item.get('base_value'),
                    'target_value': item.get('target_value'),
                    'is_range': item.get('is_range', False),
                    'is_rate': item.get('is_rate', False),
                    'rate_period': item.get('rate_period'),
                    'confidence': item.get('confidence'),
                    'extraction_method': item.get('extraction_method'),
                    'source_sentence': item.get('source_sentence'),
                    'unification_confidence': item.get('unification_confidence'),
                    'template_used': item.get('template')
                }
                
                supabase.table('scenario_parameters').upsert(param_data).execute()
        
        stats = calculate_project_statistics(scenarios)
        
        project_updates = {
            'baseline_year': stats['baseline_year'],
            'scenario_target_year': stats['scenario_target_year'],
            'total_scenarios': stats['total_scenarios'],
            'updated_at': datetime.now().isoformat()
        }
        
        supabase.table('projects')\
            .update(project_updates)\
            .eq('project_id', project_id)\
            .execute()
        
        update_project_progress(project_id, 'scenarios_analyzed', 7)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        st.exception(e)
        return False


def update_project_progress(project_id: str, step_key: str, percent_increase: int):
    """Update project progress steps"""
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('project_progress_steps')\
            .select('step_percent')\
            .eq('project_id', project_id)\
            .eq('step_key', step_key)\
            .execute()
        
        if result.data:
            current_percent = result.data[0]['step_percent']
            new_percent = min(100, current_percent + percent_increase)
            
            supabase.table('project_progress_steps')\
                .update({'step_percent': new_percent, 'updated_at': datetime.now().isoformat()})\
                .eq('project_id', project_id)\
                .eq('step_key', step_key)\
                .execute()
        else:
            supabase.table('project_progress_steps')\
                .insert({
                    'project_id': project_id,
                    'step_key': step_key,
                    'step_percent': percent_increase
                })\
                .execute()
        
    except Exception as e:
        st.warning(f"Could not update progress: {str(e)}")


def format_value_with_semantics(item: Dict) -> str:
    """Format value display based on semantic type"""
    value_type = item.get('value_type', 'absolute')
    value = item.get('value')
    unit = item.get('unit', '')
    
    if item.get('is_range'):
        min_val = item.get('value_min', '')
        max_val = item.get('value_max', '')
        return f"{min_val}-{max_val} {unit}"
    elif item.get('is_rate'):
        rate_period = item.get('rate_period', '')
        return f"{value} {unit} {rate_period}"
    elif value_type == 'delta':
        sign = '+' if value and value > 0 else ''
        return f"Δ{sign}{value} {unit}"
    elif item.get('base_value') and item.get('target_value'):
        return f"{item['base_value']} → {item['target_value']} {unit}"
    else:
        return f"{value} {unit}" if value is not None else "N/A"


def format_time_display(item: Dict) -> str:
    """Format time information for display"""
    baseline = item.get('baseline_year')
    target = item.get('target_year')
    expr = item.get('time_expression', '')
    
    if baseline and target:
        return f"{baseline}→{target}"
    elif target:
        return f"by {target}"
    elif expr:
        return expr
    else:
        return "N/A"


def format_value_type(value_type: str) -> str:
    """Format value type with icon"""
    icons = {
        'delta': '📈',
        'absolute_target': '🎯',
        'range': '📊',
        'rate': '⏱️',
        'percent_point': '📍',
        'direction_only': '➡️'
    }
    icon = icons.get(value_type, '❓')
    return f"{icon} {value_type.replace('_', ' ').title()}"


def format_extraction_method(method: str) -> str:
    """Format extraction method with icon"""
    icons = {
        'hybrid_both': '🔀',
        'optionb_only': '📝',
        'gliner_only': '🤖',
        'optionb': '📝',
        'legacy': '🔙'
    }
    icon = icons.get(method, '❓')
    short_name = method.replace('_only', '').replace('hybrid_', '').replace('_both', '')
    return f"{icon} {short_name}"


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
    if "enable_ml_extraction" not in st.session_state:
        st.session_state.enable_ml_extraction = True
    if "ml_confidence_threshold" not in st.session_state:
        st.session_state.ml_confidence_threshold = 0.5
    if "load_mode_selected" not in st.session_state:
        st.session_state.load_mode_selected = False
    if "save_mode" not in st.session_state:
        st.session_state.save_mode = 'update'


if 'nlp_models_loaded' not in st.session_state:
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🤖 Initializing NLP Module")
            st.info("⏳ **Please wait** - Preparing language models...")
            st.caption("First-time setup in progress...")


@st.cache_resource(show_spinner=False)
def load_nlp_models():
    """Load all NLP models with progress tracking"""
    import spacy
    
    models = {}
    
    try:
        models['spacy'] = spacy.load('en_core_web_sm')
    except OSError:
        models['spacy'] = None
    
    try:
        from core.nlp.ml_extractor import load_gliner_model
        models['gliner'] = load_gliner_model()
    except Exception:
        models['gliner'] = None
    
    return models


if 'nlp_models_loaded' not in st.session_state:
    if 'loading_placeholder' in locals():
        loading_placeholder.empty()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🤖 Loading NLP Models")
        st.info("**First-time setup** - Loading language models and AI extractors")
        st.caption("⏱️ Estimated time: 20-30 seconds | Future page visits will be instant!")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        status_text.markdown("**Step 1/3:** Loading spaCy Language Model")
        detail_text.text("📦 Initializing English language processor...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        detail_text.text("📦 Loading linguistic patterns and rules...")
        progress_bar.progress(30)
        
        status_text.markdown("**Step 2/3:** Loading AI Extraction Models")
        detail_text.text("🤖 Loading transformer-based extractors (if available)...")
        progress_bar.progress(40)
        
        nlp_models = load_nlp_models()
        progress_bar.progress(70)
        
        status_text.markdown("**Step 3/3:** Initializing NLP Components")
        detail_text.text("⚙️ Setting up parameter extraction pipelines...")
        progress_bar.progress(85)
        time.sleep(0.2)
        
        detail_text.text("⚙️ Preparing ensemble extraction methods...")
        progress_bar.progress(95)
        time.sleep(0.2)
        
        progress_bar.progress(100)
        detail_text.text("✅ All components ready!")
        time.sleep(0.5)
        
        status_text.empty()
        detail_text.empty()
        progress_bar.empty()
        
        st.session_state.nlp_models = nlp_models
        st.session_state.nlp_models_loaded = True
        
        st.success("✅ **NLP Models Loaded Successfully**")
        st.info("💡 Models are now cached. Page refreshes will be instant.")
        time.sleep(1.0)
        st.rerun()
else:
    nlp_models = st.session_state.nlp_models


st.title("📝 Scenario Analysis (NLP)")
st.markdown("*Analyze scenario text and extract structured parameters*")
st.markdown("---")

initialize_scenario_state()


success, message = load_project_and_user_data()

if not success:
    st.error(f"⚠️ {message}")
    st.info("💡 Please ensure you have selected or created a project before accessing this page.")
    
    if st.button("🔄 Retry Loading Project"):
        st.rerun()
    
    st.stop()


project_id = st.session_state.get('project_id')
user_id = st.session_state.get('user_id')
project_data = st.session_state.get('project_data', {})
user_data = st.session_state.get('user_data', {})

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Project")
st.sidebar.markdown(f"**Project:** {project_data.get('project_name', 'Unknown')}")
st.sidebar.markdown(f"**Owner:** {user_data.get('full_name', user_data.get('username', 'Unknown'))}")
st.sidebar.markdown(f"**Status:** {project_data.get('status', 'Unknown')}")

if project_data.get('baseline_year'):
    st.sidebar.markdown(f"**Baseline Year:** {project_data['baseline_year']}")
if project_data.get('scenario_target_year'):
    st.sidebar.markdown(f"**Target Year:** {project_data['scenario_target_year']}")
if project_data.get('total_scenarios'):
    st.sidebar.markdown(f"**Total Scenarios:** {project_data['total_scenarios']}")


if not st.session_state.load_mode_selected:
    st.markdown("## 🔍 Check Existing Scenarios")
    
    with st.spinner("Checking for existing scenarios..."):
        has_scenarios, scenario_count, scenario_list = check_existing_scenarios(project_id)
    
    if has_scenarios:
        st.info(f"📊 **Found {scenario_count} existing scenario(s) for this project**")
        
        with st.expander("👀 View Existing Scenarios", expanded=True):
            for s in scenario_list:
                baseline_info = f" (Baseline: {s.get('baseline_year')})" if s.get('baseline_year') else ""
                st.markdown(f"- **{s['title']}** - Target: {s.get('horizon', 'N/A')}{baseline_info} - Created: {s['created_at'][:10]}")
        
        st.markdown("### What would you like to do?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 **Load & Update Existing**", use_container_width=True, type="primary"):
                with st.spinner("Loading scenarios from database..."):
                    loaded_scenarios = load_scenarios_from_database(project_id)
                    if loaded_scenarios:
                        st.session_state.detected_scenarios = loaded_scenarios
                        st.session_state.scenarios_processed = True
                        st.session_state.save_mode = 'update'
                        st.session_state.load_mode_selected = True
                        st.success(f"✅ Loaded {len(loaded_scenarios)} scenarios")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to load scenarios")
        
        with col2:
            if st.button("➕ **Create New Scenarios**", use_container_width=True):
                st.session_state.detected_scenarios = []
                st.session_state.scenarios_processed = False
                st.session_state.save_mode = 'append'
                st.session_state.load_mode_selected = True
                st.info("✅ Ready to create new scenarios")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        st.caption("💡 **Load & Update**: Replace existing scenarios with edited versions")
        st.caption("💡 **Create New**: Add new scenarios while keeping existing ones")
    
    else:
        st.success("✨ **No existing scenarios found.** Ready to create your first scenarios!")
        
        if st.button("🚀 **Start Creating Scenarios**", use_container_width=True, type="primary"):
            st.session_state.detected_scenarios = []
            st.session_state.scenarios_processed = False
            st.session_state.save_mode = 'append'
            st.session_state.load_mode_selected = True
            st.rerun()
    
    st.stop()


st.markdown("## 📊 Current Mode")
mode_display = "🔄 **Update Mode**" if st.session_state.save_mode == 'update' else "➕ **Append Mode**"
mode_desc = "Editing existing scenarios - will replace on save" if st.session_state.save_mode == 'update' else "Creating new scenarios - will add to existing on save"

col_mode1, col_mode2 = st.columns([3, 1])
with col_mode1:
    st.info(f"{mode_display}: {mode_desc}")
with col_mode2:
    if st.button("🔄 Switch Mode"):
        st.session_state.save_mode = 'append' if st.session_state.save_mode == 'update' else 'update'
        st.rerun()

st.markdown("---")


if st.session_state.scenarios_processed and st.session_state.detected_scenarios:
    st.markdown("## ✏️ Step 1: Edit Scenario Details")
    
    for scenario_idx, scenario in enumerate(st.session_state.detected_scenarios):
        with st.expander(f"📋 Scenario {scenario_idx + 1}: {scenario['title']}", expanded=True):
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                new_title = st.text_input(
                    "Scenario Title",
                    value=scenario['title'],
                    key=f"title_{scenario_idx}"
                )
                if new_title != scenario['title']:
                    st.session_state.detected_scenarios[scenario_idx]['title'] = new_title
            
            with col2:
                new_horizon = st.number_input(
                    "Target Year",
                    min_value=2020,
                    max_value=2100,
                    value=int(scenario.get('horizon', 2050)) if scenario.get('horizon') else 2050,
                    key=f"horizon_{scenario_idx}"
                )
                if new_horizon != scenario.get('horizon'):
                    st.session_state.detected_scenarios[scenario_idx]['horizon'] = new_horizon
            
            with col3:
                new_baseline = st.number_input(
                    "Baseline Year",
                    min_value=2000,
                    max_value=2100,
                    value=int(scenario.get('baseline_year', 2024)) if scenario.get('baseline_year') else 2024,
                    key=f"baseline_{scenario_idx}"
                )
                if new_baseline != scenario.get('baseline_year'):
                    st.session_state.detected_scenarios[scenario_idx]['baseline_year'] = new_baseline
            
            new_text = st.text_area(
                "Scenario Description",
                value=scenario.get('text', ''),
                height=100,
                key=f"text_{scenario_idx}"
            )
            if new_text != scenario.get('text'):
                st.session_state.detected_scenarios[scenario_idx]['text'] = new_text
    
    st.markdown("---")
    
    st.markdown("## 📊 Step 2: Review & Edit Parameters")
    
    for scenario_idx, scenario in enumerate(st.session_state.detected_scenarios):
        st.markdown(f"### {scenario['title']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Target Year", scenario.get('horizon', 'N/A'))
        with col2:
            st.metric("Baseline Year", scenario.get('baseline_year', 'N/A'))
        with col3:
            time_conf = scenario.get('time_confidence', 0)
            time_color = "🟢" if time_conf > 0.8 else "🟡" if time_conf > 0.6 else "🔴"
            st.metric("Time Confidence", f"{time_color} {time_conf:.0%}" if time_conf > 0 else "N/A")
        with col4:
            baseline = scenario.get('baseline_year')
            horizon = scenario.get('horizon')
            time_range = f"{baseline} → {horizon}" if baseline and horizon else "N/A"
            st.metric("Time Range", time_range)
        
        items = scenario.get('items', [])
        
        if not items:
            st.info("No parameters detected")
            
            if st.button(f"➕ Add Parameter Manually", key=f"add_param_{scenario_idx}"):
                new_item = {
                    'id': str(uuid.uuid4()),
                    'parameter': 'New Parameter',
                    'parameter_canonical': 'New Parameter',
                    'category': 'other',
                    'value': 0.0,
                    'unit': '%',
                    'direction': 'target',
                    'value_type': 'absolute_target',
                    'confidence': 1.0,
                    'extraction_method': 'manual'
                }
                st.session_state.detected_scenarios[scenario_idx]['items'].append(new_item)
                st.rerun()
        else:
            rows = []
            for item_idx, item in enumerate(items):
                conf = item.get('confidence', 0)
                conf_icon = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                
                value_display = format_value_with_semantics(item)
                time_display = format_time_display(item)
                
                row = {
                    'idx': item_idx,
                    'Status': conf_icon,
                    'Parameter': item.get('parameter_canonical', item.get('parameter', 'Unknown')),
                    'Category': item.get('category', 'N/A'),
                    'Direction': item.get('direction', 'N/A'),
                    'Value': item.get('value', 0),
                    'Unit': item.get('unit', ''),
                    'Value Type': item.get('value_type', 'N/A'),
                    'Time': time_display,
                    'Confidence': conf,
                    'Method': item.get('extraction_method', 'N/A')
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            edited_df = st.data_editor(
                df,
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "idx": None,
                    "Status": st.column_config.TextColumn("", width="small"),
                    "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                    "Category": st.column_config.SelectboxColumn(
                        "Category",
                        options=['economy', 'environment', 'social', 'other'],
                        width="small"
                    ),
                    "Direction": st.column_config.SelectboxColumn(
                        "Direction",
                        options=['increase', 'decrease', 'target', 'stable', 'double', 'halve'],
                        width="small"
                    ),
                    "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                    "Unit": st.column_config.TextColumn("Unit", width="small"),
                    "Value Type": st.column_config.SelectboxColumn(
                        "Type",
                        options=['absolute_target', 'delta', 'range', 'rate', 'percent_point'],
                        width="small"
                    ),
                    "Time": st.column_config.TextColumn("Time", width="medium"),
                    "Confidence": st.column_config.NumberColumn("Conf.", format="%.2f"),
                    "Method": st.column_config.TextColumn("Method", width="small")
                },
                key=f"param_table_{scenario_idx}"
            )
            
            if len(edited_df) != len(df):
                if len(edited_df) > len(df):
                    for new_idx in range(len(df), len(edited_df)):
                        new_row = edited_df.iloc[new_idx]
                        new_item = {
                            'id': str(uuid.uuid4()),
                            'parameter': new_row['Parameter'],
                            'parameter_canonical': new_row['Parameter'],
                            'category': new_row['Category'],
                            'direction': new_row['Direction'],
                            'value': new_row['Value'],
                            'unit': new_row['Unit'],
                            'value_type': new_row['Value Type'],
                            'confidence': new_row['Confidence'],
                            'extraction_method': 'manual'
                        }
                        st.session_state.detected_scenarios[scenario_idx]['items'].append(new_item)
                    st.rerun()
                
                elif len(edited_df) < len(df):
                    removed_indices = set(df['idx']) - set(edited_df['idx'])
                    for removed_idx in sorted(removed_indices, reverse=True):
                        del st.session_state.detected_scenarios[scenario_idx]['items'][removed_idx]
                    st.rerun()
            
            for edit_idx, edit_row in edited_df.iterrows():
                if edit_idx < len(items):
                    orig_idx = int(edit_row['idx'])
                    st.session_state.detected_scenarios[scenario_idx]['items'][orig_idx].update({
                        'parameter': edit_row['Parameter'],
                        'parameter_canonical': edit_row['Parameter'],
                        'category': edit_row['Category'],
                        'direction': edit_row['Direction'],
                        'value': edit_row['Value'],
                        'unit': edit_row['Unit'],
                        'value_type': edit_row['Value Type'],
                        'confidence': edit_row['Confidence']
                    })
            
            with st.expander("🔍 Advanced Parameter Details", expanded=False):
                for item_idx, item in enumerate(items):
                    st.markdown(f"**{item.get('parameter_canonical', item.get('parameter'))}**")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.text(f"Original: {item.get('parameter_original', 'N/A')}")
                        st.text(f"Category: {item.get('category', 'N/A')}")
                        
                        new_target_year = st.number_input(
                            "Target Year",
                            min_value=2000,
                            max_value=2100,
                            value=int(item.get('target_year', 2050)) if item.get('target_year') else 2050,
                            key=f"target_year_{scenario_idx}_{item_idx}"
                        )
                        if new_target_year != item.get('target_year'):
                            st.session_state.detected_scenarios[scenario_idx]['items'][item_idx]['target_year'] = new_target_year
                    
                    with col_b:
                        st.text(f"Value Type: {item.get('value_type', 'N/A')}")
                        
                        if item.get('is_range'):
                            new_min = st.number_input(
                                "Min Value",
                                value=float(item.get('value_min', 0)),
                                key=f"min_{scenario_idx}_{item_idx}"
                            )
                            new_max = st.number_input(
                                "Max Value",
                                value=float(item.get('value_max', 0)),
                                key=f"max_{scenario_idx}_{item_idx}"
                            )
                            if new_min != item.get('value_min'):
                                st.session_state.detected_scenarios[scenario_idx]['items'][item_idx]['value_min'] = new_min
                            if new_max != item.get('value_max'):
                                st.session_state.detected_scenarios[scenario_idx]['items'][item_idx]['value_max'] = new_max
                        
                        if item.get('is_rate'):
                            new_rate_period = st.text_input(
                                "Rate Period",
                                value=item.get('rate_period', 'per year'),
                                key=f"rate_period_{scenario_idx}_{item_idx}"
                            )
                            if new_rate_period != item.get('rate_period'):
                                st.session_state.detected_scenarios[scenario_idx]['items'][item_idx]['rate_period'] = new_rate_period
                    
                    with col_c:
                        st.text(f"Confidence: {item.get('confidence', 0):.2f}")
                        st.text(f"Method: {item.get('extraction_method', 'N/A')}")
                    
                    source_text = st.text_area(
                        "Source Sentence",
                        value=item.get('source_sentence', ''),
                        height=60,
                        key=f"source_{scenario_idx}_{item_idx}"
                    )
                    if source_text != item.get('source_sentence'):
                        st.session_state.detected_scenarios[scenario_idx]['items'][item_idx]['source_sentence'] = source_text
                    
                    st.markdown("---")
        
        st.markdown("---")
    
    st.markdown("---")
    st.markdown("## 💾 Step 3: Save to Database")
    
    stats = calculate_project_statistics(st.session_state.detected_scenarios)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Scenarios", stats['total_scenarios'])
    with col_stat2:
        st.metric("Baseline Year", stats['baseline_year'] if stats['baseline_year'] else "N/A")
    with col_stat3:
        st.metric("Target Year", stats['scenario_target_year'] if stats['scenario_target_year'] else "N/A")
    with col_stat4:
        total_params = sum(len(s.get('items', [])) for s in st.session_state.detected_scenarios)
        st.metric("Total Parameters", total_params)
    
    col_save1, col_save2, col_save3 = st.columns([2, 1, 1])
    
    with col_save1:
        st.info(f"**Save Mode**: {mode_display}")
    
    with col_save2:
        if st.button("💾 **SAVE TO DATABASE**", use_container_width=True, type="primary"):
            with st.spinner("Saving to database..."):
                success = save_scenarios_to_database(
                    st.session_state.detected_scenarios,
                    project_id,
                    user_id,
                    st.session_state.save_mode
                )
                
                if success:
                    st.success("✅ **Scenarios saved successfully!**")
                    st.balloons()
                    
                    for scenario in st.session_state.detected_scenarios:
                        scenario['from_database'] = True
                    
                    st.session_state.project_data['baseline_year'] = stats['baseline_year']
                    st.session_state.project_data['scenario_target_year'] = stats['scenario_target_year']
                    st.session_state.project_data['total_scenarios'] = stats['total_scenarios']
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to save scenarios")
    
    with col_save3:
        if st.button("🔄 Reset & Start Over", use_container_width=True):
            st.session_state.detected_scenarios = []
            st.session_state.scenarios_processed = False
            st.session_state.load_mode_selected = False
            st.rerun()

else:
    st.markdown("## 📝 Step 1: Input Scenario Text")
    
    st.markdown("### ⚙️ Extraction Settings")
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        enable_ml = st.checkbox(
            "🤖 Enable ML Extraction (GLiNER)",
            value=st.session_state.enable_ml_extraction,
            help="Use transformer-based model for parameter extraction"
        )
        if enable_ml != st.session_state.enable_ml_extraction:
            st.session_state.enable_ml_extraction = enable_ml
    
    with col_set2:
        ml_threshold = st.slider(
            "ML Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=st.session_state.ml_confidence_threshold,
            step=0.05,
            help="Minimum confidence to accept ML extractions"
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
