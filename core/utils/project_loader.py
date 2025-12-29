"""
Project Data Loader
Loads all project data from database into session state when project is opened
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


def load_complete_project_data(project_id: str, db) -> Dict[str, Any]:
    """
    Load ALL project data from database and populate session state
    This should be called when a project is opened from the home page
    
    Returns dict with status and loaded data
    """
    
    result = {
        'success': True,
        'message': '',
        'data_loaded': False,
        'errors': []
    }
    
    try:
        # 1. Load Project Info
        st.write("📂 Loading project information...")
        project_result = db.client.table('projects').select('*').eq('project_id', project_id).execute()
        
        if project_result.data:
            project = project_result.data[0]
            st.session_state.project_info = project
            st.session_state.workflow_state = project.get('workflow_state', 'setup')
            st.session_state.completion_percentage = project.get('completion_percentage', 0)
        else:
            result['errors'].append("Project not found")
            result['success'] = False
            return result
        
        # 2. Load Step Completion Status
        st.write("✅ Loading workflow progress...")
        step_completion = db.get_step_completion(project_id)
        
        # Update session state with all step completion flags
        for key, value in step_completion.items():
            st.session_state[key] = value
        
        # 3. Load Parameters
        st.write("📊 Loading parameters...")
        parameters = db.get_project_parameters(project_id)
        
        if parameters:
            st.session_state.project_parameters = parameters
            st.session_state.value_columns = [p['parameter_name'] for p in parameters]
            result['data_loaded'] = True
        
        # 4. Load Health Report
        st.write("🔍 Loading health report...")
        health_report = db.get_health_report(project_id)
        
        if health_report:
            st.session_state.health_report = health_report
            
            # Extract important fields to session state
            st.session_state.data_loaded = True
            
            # If we have time_metadata, extract it
            if 'time_metadata' in health_report and health_report['time_metadata']:
                time_meta = health_report['time_metadata']
                
                # Parse dates if they're strings
                if 'min_time' in time_meta and isinstance(time_meta['min_time'], str):
                    time_meta['min_time'] = pd.to_datetime(time_meta['min_time'])
                if 'max_time' in time_meta and isinstance(time_meta['max_time'], str):
                    time_meta['max_time'] = pd.to_datetime(time_meta['max_time'])
        
        # 5. Load Uploaded Files Info
        st.write("📁 Loading file history...")
        uploaded_files = db.get_uploaded_files(project_id)
        
        if uploaded_files:
            st.session_state.uploaded_files_history = uploaded_files
        
        # 6. Try to reconstruct df_long from parameters (if needed for visualizations)
        # This is a placeholder - actual data reconstruction would need the raw data
        # For now, we just mark that data exists
        if parameters:
            st.session_state.has_processed_data = True
        
        # 7. Set data_loaded flag if we have parameters
        if parameters:
            st.session_state.data_loaded = True
            result['data_loaded'] = True
            result['message'] = f"✅ Loaded {len(parameters)} parameters"
        else:
            st.session_state.data_loaded = False
            result['message'] = "ℹ️ No data found - please upload data"
        
        return result
        
    except Exception as e:
        result['success'] = False
        result['errors'].append(str(e))
        result['message'] = f"❌ Error loading project data: {str(e)}"
        return result


def load_project_on_open(project_id: str, db) -> bool:
    """
    Main function to call when opening a project
    Shows loading progress and populates session state
    
    Returns True if successful, False otherwise
    """
    
    # Store project ID
    st.session_state.current_project_id = project_id
    
    with st.spinner("Loading project data..."):
        result = load_complete_project_data(project_id, db)
    
    if result['success']:
        if result['data_loaded']:
            st.success(result['message'])
        else:
            st.info(result['message'])
        return True
    else:
        st.error(result['message'])
        for error in result['errors']:
            st.error(f"  • {error}")
        return False


def ensure_project_data_loaded(project_id: str, db) -> bool:
    """
    Check if project data is loaded, if not, load it
    Call this at the start of each page
    
    Returns True if data is available, False otherwise
    """
    
    # Check if we have the right project loaded
    if st.session_state.get('current_project_id') != project_id:
        st.session_state.current_project_id = project_id
        load_complete_project_data(project_id, db)
    
    # Check if we have any data loaded
    if not st.session_state.get('project_parameters') and not st.session_state.get('data_loaded'):
        # Try to load data silently
        load_complete_project_data(project_id, db)
    
    return st.session_state.get('data_loaded', False)
