"""
Page 2: Data Import & Diagnostics (Updated with Database Integration)
Handles file upload, time column selection, data validation, and health reporting
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import time
from pathlib import Path


# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Upload & Data Diagnostics",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)


# Authentication check
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import (
    display_error, display_success, display_warning, display_info,
    detect_datetime_columns, detect_numeric_columns, format_percentage,
    format_number
)
from core.shared_sidebar import render_app_sidebar  
from core.io.loaders import load_file_smart, convert_to_long_format, preview_dataframe, get_excel_sheet_names
from core.io.validators import parse_time_column, validate_time_series, check_numeric_columns, detect_and_report_issues
from core.database.supabase_manager import get_db_manager

# Initialize
initialize_session_state()
config = get_config()

# Render shared sidebar
render_app_sidebar()

# Get database manager
try:
    db = get_db_manager()
    DB_AVAILABLE = True
except:
    DB_AVAILABLE = False
    st.error("⚠️ Database not available")

st.title("📁 Upload & Data Diagnostics")
st.markdown("*Upload your time-series data and assess data quality*")
st.markdown("---")


def calculate_data_health_score(health_report):
    """Calculate overall data health score (0-100)"""
    score = 100
    issues = []
    
    # Missing values penalty
    if health_report.get("missing_values"):
        for var, info in health_report["missing_values"].items():
            missing_pct = info["percentage"]
            
            if missing_pct > 0.20:
                score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data (critical)")
            elif missing_pct > 0.05:
                score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data")
    
    # Outliers check
    if health_report.get("outliers"):
        outlier_count = sum(info["count"] for info in health_report["outliers"].values())
        if outlier_count > 0:
            score -= 5
            issues.append(f"📊 {outlier_count} outliers detected")
    
    # Time series issues
    if "time_metadata" in health_report:
        time_meta = health_report["time_metadata"]
        if time_meta.get("duplicate_count", 0) > 0:
            score -= 10
            issues.append(f"⏰ {time_meta['duplicate_count']} duplicate timestamps")
    
    # Warnings
    if health_report.get("warnings"):
        score -= len(health_report["warnings"]) * 5
        for warning in health_report["warnings"]:
            issues.append(f"⚠️ {warning}")
    
    score = max(0, min(100, score))
    
    if score >= 85:
        category = "excellent"
    elif score >= 70:
        category = "good"
    elif score >= 50:
        category = "fair"
    else:
        category = "poor"
    
    return score, category, issues


def update_project_progress(stage: str = "data_import", page: int = 2, percentage: int = 7):
    """Update project progress in database and session state"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return
    
    try:
        # Update in database
        db.update_project_progress(
            project_id=st.session_state.current_project_id,
            workflow_state=stage,
            current_page=page,
            completion_percentage=percentage
        )
        
        # Update session state for immediate UI feedback
        st.session_state.data_loaded = True
        
    except Exception as e:
        st.warning(f"Could not update progress: {str(e)}")


def main():
    """Main page function"""
    
    # Check if project is selected
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project first")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # Initialize active view if not set
    if 'active_view' not in st.session_state:
        st.session_state.active_view = 'upload'
    
    # Check if we should switch to health report
    if st.session_state.get('switch_to_health_report', False):
        st.session_state.active_view = 'health'
        st.session_state.switch_to_health_report = False
    
    # Create navigation tabs
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Upload Data", 
                    type="primary" if st.session_state.active_view == 'upload' else "secondary",
                    use_container_width=True,
                    key="nav_upload"):
            st.session_state.active_view = 'upload'
            st.rerun()
    
    with col2:
        if st.button("🔍 Data Health Report", 
                    type="primary" if st.session_state.active_view == 'health' else "secondary",
                    use_container_width=True,
                    key="nav_health"):
            st.session_state.active_view = 'health'
            st.rerun()
    
    st.markdown("---")
    
    # Show the active view
    if st.session_state.active_view == 'health':
        render_health_report_section()
    else:
        render_upload_section()


def render_upload_section():
    """Render the file upload and configuration section"""
    
    st.header("Step 1: Upload Your Data File")
    
    # Show previously uploaded files
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        uploaded_files = db.get_uploaded_files(st.session_state.current_project_id)
        if uploaded_files:
            with st.expander(f"📂 Previously Uploaded Files ({len(uploaded_files)})"):
                for file_info in uploaded_files[:5]:  # Show last 5
                    st.caption(f"📄 {file_info['filename']} - {file_info['file_size']/1024:.1f} KB - {file_info['uploaded_at'][:10]}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=config["accepted_file_types"],
        help=f"Supported formats: {', '.join(config['accepted_file_types'])}"
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a CSV, TXT, or Excel file to get started")
        
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your file should have:
            - **One time/date column** (any standard datetime format)
            - **One or more numeric columns** (variables to analyze)
            
            Example CSV:
            ```
            Date,Temperature,Humidity,Pressure
            2023-01-01,22.5,65.2,1013.2
            2023-01-02,23.1,68.5,1012.8
            2023-01-03,21.8,70.1,1014.3
            ```
            """)
        
        return
    
    # Store uploaded file info
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.upload_timestamp = datetime.now()
    
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")
    st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
    
    # Excel sheet selection
    sheet_name = None
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        st.markdown("### Excel File Options")
        sheet_names, error = get_excel_sheet_names(uploaded_file.getvalue())
        
        if error:
            display_error("Could not read Excel file", error)
            return
        
        if len(sheet_names) > 1:
            sheet_name = st.selectbox("Select sheet to load", sheet_names)
        else:
            sheet_name = sheet_names[0]
            st.info(f"Loading sheet: **{sheet_name}**")
    
    # CSV/TXT options
    delimiter = None
    decimal = "."
    
    if uploaded_file.name.endswith(('.csv', '.txt')):
        with st.expander("⚙️ Advanced Options (CSV/TXT)"):
            col1, col2 = st.columns(2)
            
            with col1:
                delimiter_option = st.selectbox(
                    "Delimiter",
                    ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab", "Pipe (|)"]
                )
                
                if delimiter_option != "Auto-detect":
                    delimiter_map = {
                        "Comma (,)": ",",
                        "Semicolon (;)": ";",
                        "Tab": "\t",
                        "Pipe (|)": "|"
                    }
                    delimiter = delimiter_map[delimiter_option]
            
            with col2:
                decimal = st.selectbox("Decimal separator", [".", ","])
    
    st.markdown("---")
    
    # Load file button
    if st.button("📊 Load File", type="primary", use_container_width=True):
        with st.spinner("Loading file..."):
            df_raw, error, metadata = load_file_smart(
                uploaded_file,
                delimiter=delimiter,
                decimal=decimal,
                sheet_name=sheet_name
            )
        
        if error:
            display_error("Failed to load file", error)
            return
        
        # Store raw data
        st.session_state.df_raw = df_raw
        st.session_state.file_metadata = metadata
        
        # Save to database
        if DB_AVAILABLE and st.session_state.get('current_project_id'):
            file_info = db.save_uploaded_file(
                project_id=st.session_state.current_project_id,
                filename=uploaded_file.name,
                file_size=uploaded_file.size,
                file_type=uploaded_file.type,
                metadata=metadata
            )
            
            if file_info:
                st.success(f"✅ File information saved to database")
        
        display_success(f"File loaded successfully! {metadata['rows']} rows × {metadata['columns']} columns")
        
        with st.expander("👀 Preview Raw Data"):
            preview_dataframe(df_raw)
    
    # If data is loaded, show configuration
    if st.session_state.df_raw is not None:
        st.markdown("---")
        render_data_configuration()


def render_data_configuration():
    """Render data configuration section"""
    
    st.header("Step 2: Configure Your Data")
    
    df = st.session_state.df_raw
    
    # Detect potential columns
    datetime_candidates = detect_datetime_columns(df)
    numeric_candidates = detect_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕐 Time Column")
        
        if datetime_candidates:
            default_time = datetime_candidates[0]
        else:
            default_time = df.columns[0]
        
        time_column = st.selectbox(
            "Select time column",
            df.columns.tolist(),
            index=df.columns.tolist().index(default_time) if default_time in df.columns else 0
        )
        
        sample_values = df[time_column].dropna().head(5)
        st.caption("Sample values:")
        for val in sample_values:
            st.caption(f"  • {val}")
        
        datetime_format = st.selectbox(
            "Datetime format",
            ["auto"] + config["datetime_formats"]
        )
    
    with col2:
        st.markdown("#### 📊 Variables (Value Columns)")
        
        if numeric_candidates:
            st.info(f"Detected {len(numeric_candidates)} numeric columns")
        
        default_vars = [col for col in numeric_candidates if col != time_column]
        
        value_columns = st.multiselect(
            "Select variables to analyze",
            [col for col in df.columns if col != time_column],
            default=default_vars[:10] if len(default_vars) > 10 else default_vars
        )
        
        if value_columns:
            st.success(f"✅ Selected {len(value_columns)} variables")
    
    st.markdown("---")
    
    if not value_columns:
        st.warning("⚠️ Please select at least one variable to continue")
        return
    
    if st.button("⚡ Process Data", type="primary", use_container_width=True):
        process_data(df, time_column, value_columns, datetime_format)


def process_data(df, time_column, value_columns, datetime_format):
    """Process and validate the data"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Step 1: Parse time column
    status.text("Parsing time column...")
    progress_bar.progress(0.20)
    
    if datetime_format == "auto":
        datetime_format = None
    
    parsed_time, error = parse_time_column(df, time_column, datetime_format)
    
    if error:
        display_error("Failed to parse time column", error)
        progress_bar.empty()
        status.empty()
        return
    
    is_valid, error, time_metadata = validate_time_series(parsed_time)
    
    if not is_valid:
        display_error("Time column validation failed", error)
        progress_bar.empty()
        status.empty()
        return
    
    df[time_column] = parsed_time
    
    # Step 2: Validate numeric columns
    status.text("Validating variables...")
    progress_bar.progress(0.40)
    
    valid_cols, invalid_cols, conversion_info = check_numeric_columns(df, value_columns)
    
    if invalid_cols:
        st.warning(f"⚠️ Some columns could not be used: {', '.join(invalid_cols)}")
        if not valid_cols:
            display_error("No valid numeric columns found")
            progress_bar.empty()
            status.empty()
            return
        value_columns = valid_cols
    
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 3: Convert to long format
    status.text("Converting to standard format...")
    progress_bar.progress(0.60)
    
    df_long, error = convert_to_long_format(df, time_column, value_columns)
    
    if error:
        display_error("Failed to convert data", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Step 4: Save parameters to database
    status.text("Saving parameters to database...")
    progress_bar.progress(0.75)
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        # Prepare parameter data
        parameters = []
        for col in value_columns:
            col_data = df[col].dropna()
            param_info = {
                'name': col,
                'data_type': 'numeric',
                'min_value': float(col_data.min()) if len(col_data) > 0 else None,
                'max_value': float(col_data.max()) if len(col_data) > 0 else None,
                'mean_value': float(col_data.mean()) if len(col_data) > 0 else None,
                'std_value': float(col_data.std()) if len(col_data) > 0 else None,
                'missing_count': int(df[col].isna().sum()),
                'total_count': len(df[col])
            }
            parameters.append(param_info)
        
        # Save to database
        success = db.save_parameters(st.session_state.current_project_id, parameters)
        if success:
            st.success("✅ Parameters saved to database")
    
    # Step 5: Generate health report
    status.text("Generating health report...")
    progress_bar.progress(0.90)
    
    health_report = detect_and_report_issues(df_long)
    health_report["time_metadata"] = time_metadata
    health_report["conversion_info"] = conversion_info
    
    # Store in session state
    st.session_state.df_long = df_long
    st.session_state.df_clean = df_long.copy()
    st.session_state.time_column = time_column
    st.session_state.value_columns = value_columns
    st.session_state.health_report = health_report
    st.session_state.data_loaded = True
    st.session_state.preprocessing_applied = False
    
    # Step 6: Save health report to database
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        status.text("Saving health report to database...")
        progress_bar.progress(0.95)
        
        # Calculate health score
        health_score, health_category, issues = calculate_data_health_score(health_report)
        
        # Prepare health report data for database
        health_data = {
            'health_score': health_score,
            'health_category': health_category,
            'total_parameters': len(value_columns),
            'total_data_points': len(df_long),
            'total_missing_values': sum(info['count'] for info in health_report.get('missing_values', {}).values()),
            'missing_percentage': sum(info['count'] for info in health_report.get('missing_values', {}).values()) / len(df_long) if len(df_long) > 0 else 0,
            'critical_issues': len([i for i in issues if 'critical' in i.lower()]),
            'warnings': len([i for i in issues if 'warning' in i.lower() or '⚠️' in i]),
            'duplicate_timestamps': time_metadata.get('duplicate_count', 0),
            'outlier_count': sum(info['count'] for info in health_report.get('outliers', {}).values()),
            'missing_values_detail': health_report.get('missing_values', {}),
            'outliers_detail': health_report.get('outliers', {}),
            'coverage_detail': health_report.get('coverage', {}),
            'issues_list': issues,
            'time_metadata': {
                'min_time': time_metadata['min_time'].isoformat() if 'min_time' in time_metadata else None,
                'max_time': time_metadata['max_time'].isoformat() if 'max_time' in time_metadata else None,
                'time_span': time_metadata.get('time_span', 0),
                'duplicate_count': time_metadata.get('duplicate_count', 0)
            },
            'parameters_analyzed': value_columns
        }
        
        # Save to database
        if db.save_health_report(st.session_state.current_project_id, health_data):
            st.success("✅ Health report saved to database")
    
    # Update project progress (7% for step 1 of 13)
    update_project_progress(stage="data_imported", page=2, percentage=7)
    
    # Mark step 1 as complete in database
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        db.update_step_completion(
            project_id=st.session_state.current_project_id,
            step_key='data_loaded',
            completed=True
        )
        # Also update session state for immediate UI feedback
        st.session_state.data_loaded = True
    
    progress_bar.progress(1.0)
    status.text("Complete!")
    
    # Celebration!
    st.balloons()
    
    # Success message
    display_success("🎉 Data processed and saved successfully!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(value_columns))
    col2.metric("Data Points", len(df_long))
    col3.metric("Time Range", f"{time_metadata['time_span']} days")
    
    st.markdown("---")
    
    # Navigation button to switch to health report tab
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    
    with col_nav2:
        if st.button("📊 View Data Health Report →", type="primary", use_container_width=True, key="goto_health_report"):
            # Force switch to health report tab by storing in session state
            # We'll check this in the main function to switch tabs
            st.session_state.switch_to_health_report = True
            st.rerun()


def render_duplicate_management(duplicates):
    """Render duplicate parameter management UI"""
    st.warning(f"⚠️ Found {len(duplicates)} duplicate parameter(s)")
    
    with st.expander("🔍 Manage Duplicate Parameters"):
        for param_name, param_list in duplicates.items():
            st.markdown(f"### Parameter: **{param_name}**")
            st.caption(f"Found {len(param_list)} instances")
            
            for i, param in enumerate(param_list):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.text(f"Instance {i+1}: {param['parameter_id'][:8]}...")
                    st.caption(f"Updated: {param['updated_at'][:10]}")
                
                with col2:
                    if st.button("Keep", key=f"keep_{param['parameter_id']}"):
                        if db.merge_parameters([p['parameter_id'] for p in param_list], param['parameter_id']):
                            st.success("Merged!")
                            time.sleep(1)
                            st.rerun()
                
                with col3:
                    new_name = st.text_input("Rename", key=f"rename_{param['parameter_id']}", placeholder="New name")
                    if new_name and st.button("✓", key=f"rename_btn_{param['parameter_id']}"):
                        if db.rename_parameter(param['parameter_id'], new_name):
                            st.success("Renamed!")
                            time.sleep(1)
                            st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{param['parameter_id']}"):
                        if db.delete_parameter(param['parameter_id']):
                            st.success("Deleted!")
                            time.sleep(1)
                            st.rerun()
            
            st.markdown("---")


def render_database_health_report_full(health_report):
    """Render complete health report from database with full metrics"""
    
    st.header("📊 Data Health Report")
    
    # Health Score Dashboard
    st.markdown("### Overall Data Health")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        health_score = health_report['health_score']
        health_category = health_report['health_category']
        
        if health_category == "excellent":
            score_color = "🟢"
            score_text = "Excellent"
        elif health_category == "good":
            score_color = "🟡"
            score_text = "Good"
        elif health_category == "fair":
            score_color = "🟠"
            score_text = "Fair"
        else:
            score_color = "🔴"
            score_text = "Poor"
        
        st.metric("Health Score", f"{health_score}/100")
        st.markdown(f"### {score_color} {score_text}")
    
    with col2:
        issues = health_report.get('issues_list', [])
        
        if issues:
            st.markdown("**Issues Detected:**")
            for issue in issues[:5]:
                st.caption(issue)
            if len(issues) > 5:
                st.caption(f"... and {len(issues) - 5} more issues")
        else:
            st.success("✅ No significant issues detected!")
    
    with col3:
        st.metric("Parameters", health_report['total_parameters'])
        st.metric("Data Points", health_report['total_data_points'])
    
    st.markdown("---")
    
    # Missing Values
    st.subheader("❓ Missing Values")
    
    missing_detail = health_report.get('missing_values_detail', {})
    
    if missing_detail:
        missing_data = []
        for var, info in missing_detail.items():
            missing_pct = info['percentage']
            status = "🔴 Critical" if missing_pct > 0.20 else "🟡 Warning" if missing_pct > 0.05 else "🟢 Good"
            
            missing_data.append({
                "Variable": var,
                "Missing Count": info['count'],
                "Total": info['total'],
                "Missing %": f"{missing_pct*100:.1f}%",
                "Status": status
            })
        
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
        # Overall summary
        overall_pct = health_report['missing_percentage']
        
        if overall_pct > 0.20:
            st.error(f"⚠️ Overall missing data: {overall_pct*100:.1f}% - Consider data cleaning")
        elif overall_pct > 0.05:
            st.warning(f"ℹ️ Overall missing data: {overall_pct*100:.1f}% - Monitor quality")
        else:
            st.success(f"✅ Overall missing data: {overall_pct*100:.1f}% - Good quality")
    else:
        st.success("✅ No missing values detected!")
    
    st.markdown("---")
    
    # Parameters table
    st.subheader("📋 Parameters Overview")
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        param_table = []
        for param in parameters:
            param_table.append({
                "Parameter": param['parameter_name'],
                "Type": param.get('data_type', 'numeric'),
                "Min": f"{param.get('min_value', 0):.2f}" if param.get('min_value') is not None else "N/A",
                "Max": f"{param.get('max_value', 0):.2f}" if param.get('max_value') is not None else "N/A",
                "Mean": f"{param.get('mean_value', 0):.2f}" if param.get('mean_value') is not None else "N/A",
                "Missing": f"{param.get('missing_count', 0):,}",
                "Total": f"{param.get('total_count', 0):,}"
            })
        
        param_df = pd.DataFrame(param_table)
        st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Check for duplicates
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        duplicates = db.check_duplicate_parameters(st.session_state.current_project_id)
        
        if duplicates:
            render_duplicate_management(duplicates)
    
    st.markdown("---")
    
    # Navigation
    st.header("🎯 What's Next?")
    
    # Recommendations based on health score
    if health_score >= 85:
        st.success("**Excellent Data Quality! ✨** Your data is ready for analysis.")
        primary_action = "visualize"
    elif health_score >= 70:
        st.info("**Good Data Quality 👍** Minor issues detected.")
        primary_action = "either"
    elif health_score >= 50:
        st.warning("**Fair Data Quality ⚠️** Consider cleaning the data first.")
        primary_action = "clean"
    else:
        st.error("**Poor Data Quality 🔴** Data cleaning strongly recommended.")
        primary_action = "clean"
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if health_score >= 70:
            st.success("✅ Recommended")
        
        if st.button("📊 Go to Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary",
                     key="nav_viz"):
            update_project_progress(stage="exploration", page=4, percentage=14)
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if health_score < 70:
            st.warning("⚠️ Recommended")
        
        if st.button("🧹 Go to Data Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary",
                     key="nav_clean"):
            update_project_progress(stage="preprocessing", page=3, percentage=7)
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📁 Upload New Data")
        
        if st.button("📁 Upload More Data", 
                     use_container_width=True,
                     key="nav_upload"):
            st.session_state.active_view = 'upload'
            st.rerun()


def render_database_health_report(project_data):
    """Render health report based on database data only"""
    
    st.header("Data Health Report (from Database)")
    
    parameters = project_data['parameters']
    
    # Summary metrics
    st.markdown("### 📊 Project Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Parameters", project_data['variable_count'])
    
    with col2:
        st.metric("Total Data Points", project_data['total_count'])
    
    with col3:
        missing_pct = project_data['overall_missing_pct'] * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # Parameters table
    st.subheader("📋 Parameters Overview")
    
    param_table = []
    for param in parameters:
        param_table.append({
            "Parameter": param['parameter_name'],
            "Type": param.get('data_type', 'numeric'),
            "Min": f"{param.get('min_value', 0):.2f}" if param.get('min_value') is not None else "N/A",
            "Max": f"{param.get('max_value', 0):.2f}" if param.get('max_value') is not None else "N/A",
            "Mean": f"{param.get('mean_value', 0):.2f}" if param.get('mean_value') is not None else "N/A",
            "Missing": param.get('missing_count', 0),
            "Total": param.get('total_count', 0),
            "Missing %": f"{(param.get('missing_count', 0) / param.get('total_count', 1) * 100):.1f}%"
        })
    
    param_df = pd.DataFrame(param_table)
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Check for duplicates
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        duplicates = db.check_duplicate_parameters(st.session_state.current_project_id)
        
        if duplicates:
            st.warning(f"⚠️ Found {len(duplicates)} duplicate parameter(s)")
            
            with st.expander("🔍 Manage Duplicate Parameters"):
                for param_name, param_list in duplicates.items():
                    st.markdown(f"### Parameter: **{param_name}**")
                    st.caption(f"Found {len(param_list)} instances")
                    
                    for i, param in enumerate(param_list):
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.text(f"Instance {i+1}: {param['parameter_id'][:8]}...")
                            st.caption(f"Updated: {param['updated_at'][:10]}")
                        
                        with col2:
                            if st.button("Keep", key=f"keep_{param['parameter_id']}"):
                                if db.merge_parameters([p['parameter_id'] for p in param_list], param['parameter_id']):
                                    st.success("Merged!")
                                    time.sleep(1)
                                    st.rerun()
                        
                        with col3:
                            new_name = st.text_input("Rename", key=f"rename_{param['parameter_id']}", placeholder="New name")
                            if new_name and st.button("✓", key=f"rename_btn_{param['parameter_id']}"):
                                if db.rename_parameter(param['parameter_id'], new_name):
                                    st.success("Renamed!")
                                    time.sleep(1)
                                    st.rerun()
                        
                        with col4:
                            if st.button("Delete", key=f"delete_{param['parameter_id']}"):
                                if db.delete_parameter(param['parameter_id']):
                                    st.success("Deleted!")
                                    time.sleep(1)
                                    st.rerun()
                    
                    st.markdown("---")
    
    st.markdown("---")
    
    # Navigation
    st.header("🎯 What's Next?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if st.button("📊 Go to Visualization", use_container_width=True, type="primary"):
            update_project_progress(stage="exploration", page=4, percentage=14)
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if st.button("🧹 Go to Data Cleaning", use_container_width=True):
            update_project_progress(stage="preprocessing", page=3, percentage=7)
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📁 Upload New Data")
        if st.button("📁 Upload More Data", use_container_width=True):
            st.session_state.active_view = 'upload'
            st.rerun()


def render_health_report_section():
    """Render the data health report section"""
    
    # First, try to load step completion from database
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        step_completion = db.get_step_completion(st.session_state.current_project_id)
        # Update session state with database values
        for key, value in step_completion.items():
            st.session_state[key] = value
    
    # Check if data is loaded in session state OR if we have data in database
    has_session_data = st.session_state.get('data_loaded', False)
    
    if not has_session_data and DB_AVAILABLE and st.session_state.get('current_project_id'):
        # Try to load or generate health report from database
        st.info("📊 Checking for existing health report...")
        
        # Check if we need to update the report
        needs_update = db.needs_health_report_update(st.session_state.current_project_id)
        
        if needs_update:
            with st.spinner("🔄 Generating health report from database parameters..."):
                health_report = db.generate_health_report_from_parameters(st.session_state.current_project_id)
            
            if not health_report.get('success'):
                st.warning("⚠️ " + health_report.get('message', 'Could not generate health report'))
                st.info("📤 Please upload and process data first")
                
                if st.button("📤 Go to Upload Data", type="primary", use_container_width=True):
                    st.session_state.active_view = 'upload'
                    st.rerun()
                return
            
            st.success("✅ Health report generated from database!")
        else:
            # Load existing report
            health_report = db.get_health_report(st.session_state.current_project_id)
            
            if not health_report:
                st.warning("⚠️ No data found")
                st.info("📤 Please upload and process data first")
                
                if st.button("📤 Go to Upload Data", type="primary", use_container_width=True):
                    st.session_state.active_view = 'upload'
                    st.rerun()
                return
            
            st.success("✅ Loaded existing health report from database")
        
        # Display the health report
        render_database_health_report_full(health_report)
        return
    
    # If we have session state data, use the original health report
    if not has_session_data:
        st.info("📤 Upload and process data first to see the health report")
        
        if st.button("📤 Go to Upload Data", type="primary", use_container_width=True):
            st.session_state.active_view = 'upload'
            st.rerun()
        return
    
    if st.session_state.health_report is None:
        st.warning("⚠️ No health report available")
        return
    
    st.header("Data Health Report")
    
    # Load parameters from database for duplicate checking
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        db_parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if db_parameters:
            st.info(f"📊 Total parameters in database: {len(db_parameters)}")
            
            # Check for duplicates
            duplicates = db.check_duplicate_parameters(st.session_state.current_project_id)
            
            if duplicates:
                render_duplicate_management(duplicates)
    
    # Rest of health report (existing code)
    health = st.session_state.health_report
    df_long = st.session_state.df_long
    
    health_score, health_category, issues = calculate_data_health_score(health)
    
    # Health Score Dashboard
    st.markdown("### 📊 Overall Data Health")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if health_category == "excellent":
            score_color = "🟢"
            score_text = "Excellent"
        elif health_category == "good":
            score_color = "🟡"
            score_text = "Good"
        elif health_category == "fair":
            score_color = "🟠"
            score_text = "Fair"
        else:
            score_color = "🔴"
            score_text = "Poor"
        
        st.metric("Health Score", f"{health_score}/100")
        st.markdown(f"### {score_color} {score_text}")
    
    with col2:
        if issues:
            st.markdown("**Issues Detected:**")
            for issue in issues[:5]:
                st.caption(issue)
            if len(issues) > 5:
                st.caption(f"... and {len(issues) - 5} more issues")
        else:
            st.success("✅ No significant issues detected!")
    
    with col3:
        st.metric("Variables", len(st.session_state.value_columns))
        st.metric("Data Points", len(df_long))
    
    st.markdown("---")
    
    # Time metadata
    if "time_metadata" in health:
        st.subheader("⏰ Time Series Information")
        
        time_meta = health["time_metadata"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Start Date", time_meta["min_time"].strftime("%Y-%m-%d"))
            st.metric("End Date", time_meta["max_time"].strftime("%Y-%m-%d"))
        
        with col2:
            st.metric("Time Span", f"{time_meta['time_span']} days")
            if "estimated_frequency" in time_meta:
                st.metric("Frequency", time_meta["estimated_frequency"])
        
        with col3:
            if "duplicate_count" in time_meta and time_meta["duplicate_count"] > 0:
                st.metric("Duplicate Timestamps", time_meta["duplicate_count"], delta="Warning", delta_color="inverse")
            else:
                st.metric("Duplicate Timestamps", "0", delta="Good", delta_color="normal")
    
    st.markdown("---")
    
    # Navigation Section
    st.header("🎯 What's Next?")
    
    if health_score >= 85:
        st.success("**Excellent Data Quality! ✨** Your data is ready for analysis.")
        primary_action = "visualize"
    elif health_score >= 70:
        st.info("**Good Data Quality 👍** Minor issues detected.")
        primary_action = "either"
    elif health_score >= 50:
        st.warning("**Fair Data Quality ⚠️** Consider cleaning the data first.")
        primary_action = "clean"
    else:
        st.error("**Poor Data Quality 🔴** Data cleaning strongly recommended.")
        primary_action = "clean"
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if health_score >= 70:
            st.success("✅ Recommended")
        
        if st.button("📊 Go to Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary"):
            update_project_progress(stage="exploration", page=4, percentage=14)
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if health_score < 70:
            st.warning("⚠️ Recommended")
        
        if st.button("🧹 Go to Data Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary"):
            update_project_progress(stage="preprocessing", page=3, percentage=7)
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📁 Upload New Data")
        
        if st.button("📁 Upload Different File", 
                     use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.df_raw = None
            st.session_state.df_long = None
            st.session_state.health_report = None
            st.rerun()


if __name__ == "__main__":
    main()
