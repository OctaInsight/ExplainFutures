"""
Page 2: Data Import & Diagnostics (FINAL FIXED VERSION)
Fixed: Data loading from database + Navigation buttons always show
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
    """Calculate overall data health score (0-100) - FIXED VERSION"""
    score = 100
    issues = []
    
    # Missing values penalty
    if health_report.get("missing_values"):
        for var, info in health_report["missing_values"].items():
            missing_pct = info.get("percentage", 0)
            
            if missing_pct > 0.20:
                score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data (critical)")
            elif missing_pct > 0.05:
                score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data")
    
    # Outliers check
    if health_report.get("outliers"):
        try:
            outlier_count = sum(info.get("count", 0) for info in health_report["outliers"].values())
            if outlier_count > 0:
                score -= 5
                issues.append(f"📊 {outlier_count} outliers detected")
        except:
            pass
    
    # Time series issues
    if "time_metadata" in health_report:
        time_meta = health_report["time_metadata"]
        if time_meta.get("duplicate_count", 0) > 0:
            score -= 10
            issues.append(f"⏰ {time_meta['duplicate_count']} duplicate timestamps")
    
    # Warnings - FIXED: Handle both list and int
    warnings_data = health_report.get("warnings")
    if warnings_data:
        if isinstance(warnings_data, list):
            score -= len(warnings_data) * 5
            for warning in warnings_data:
                issues.append(f"⚠️ {warning}")
        elif isinstance(warnings_data, (int, float)):
            score -= int(warnings_data) * 5
    
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
        db.update_project_progress(
            project_id=st.session_state.current_project_id,
            workflow_state=stage,
            current_page=page,
            completion_percentage=percentage
        )
        st.session_state.data_loaded = True
    except Exception as e:
        st.warning(f"Could not update progress: {str(e)}")


def show_next_steps_after_upload():
    """Show navigation options after data is successfully uploaded and processed"""
    
    st.markdown("---")
    st.header("🎯 What's Next?")
    
    # Get health score
    health_score = 100
    if st.session_state.get('health_report'):
        try:
            health_score, _, _ = calculate_data_health_score(st.session_state.health_report)
        except:
            health_score = 75
    
    # Recommendations
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
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if health_score >= 70:
            st.success("✅ Recommended")
        
        if st.button("📊 Go to Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary",
                     key="nav_viz_upload"):
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db.update_project_progress(
                    project_id=st.session_state.current_project_id,
                    workflow_state="exploration",
                    current_page=4,
                    completion_percentage=14
                )
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if health_score < 70:
            st.warning("⚠️ Recommended")
        
        if st.button("🧹 Go to Data Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary",
                     key="nav_clean_upload"):
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db.update_project_progress(
                    project_id=st.session_state.current_project_id,
                    workflow_state="preprocessing",
                    current_page=3,
                    completion_percentage=7
                )
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📋 Data Health Report")
        st.info("💡 Switch to **Data Health Report** tab above to see full analysis")


def load_data_from_database_if_available():
    """
    Load project data from database if available
    This populates session state with parameters and health report
    """
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return False
    
    try:
        # Get parameters
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if not parameters:
            return False
        
        # Store parameters in session state
        st.session_state.project_parameters = parameters
        st.session_state.value_columns = [p['parameter_name'] for p in parameters]
        
        # Try to get health report
        health_report = db.get_health_report(st.session_state.current_project_id)
        
        if health_report:
            st.session_state.health_report = health_report
            st.session_state.data_loaded = True
        
        return True
        
    except Exception as e:
        st.warning(f"Could not load data from database: {str(e)}")
        return False


def main():
    """Main page function"""
    
    # Check if project is selected
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project first")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # CRITICAL: Load data from database if not in session state
    if not st.session_state.get('data_loaded') and not st.session_state.get('df_long'):
        # Try to load from database
        data_loaded = load_data_from_database_if_available()
        
        if data_loaded:
            params_count = len(st.session_state.get('project_parameters', []))
            st.info(f"📊 Project data loaded from database: {params_count} parameters")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Data", "🔍 Data Health Report"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_health_report_section()


def render_upload_section():
    """Render the file upload section"""
    
    st.header("Step 1: Upload Your Data File")
    
    # Show data status if exists in database
    if st.session_state.get('data_loaded') and st.session_state.get('value_columns'):
        with st.expander("✅ Current Project Data", expanded=False):
            st.success(f"📊 {len(st.session_state.value_columns)} parameters loaded")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption("**Parameters:**")
                for var in st.session_state.value_columns[:10]:
                    st.text(f"  • {var}")
                if len(st.session_state.value_columns) > 10:
                    st.caption(f"  ... and {len(st.session_state.value_columns) - 10} more")
            
            with col2:
                if st.session_state.get('health_report'):
                    health = st.session_state.health_report
                    st.caption("**Health Status:**")
                    st.text(f"  • Score: {health.get('health_score', 'N/A')}/100")
                    st.text(f"  • Category: {health.get('health_category', 'N/A')}")
                    st.text(f"  • Data points: {health.get('total_data_points', 'N/A')}")
    
    # Show previously uploaded files
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            uploaded_files = db.get_uploaded_files(st.session_state.current_project_id)
            if uploaded_files:
                with st.expander(f"📂 Previously Uploaded Files ({len(uploaded_files)})"):
                    for file_info in uploaded_files[:5]:
                        st.caption(f"📄 {file_info['filename']} - {file_info['file_size']/1024:.1f} KB - {file_info['uploaded_at'][:10]}")
        except:
            pass
    
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
        
        # Show navigation even if no new upload (if data exists in database)
        if st.session_state.get('data_loaded') and st.session_state.get('value_columns'):
            st.markdown("---")
            show_next_steps_after_upload()
        
        return
    
    # Store file info
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
        
        st.session_state.df_raw = df_raw
        st.session_state.file_metadata = metadata
        
        # Save to database
        if DB_AVAILABLE and st.session_state.get('current_project_id'):
            try:
                db.save_uploaded_file(
                    project_id=st.session_state.current_project_id,
                    filename=uploaded_file.name,
                    file_size=uploaded_file.size,
                    file_type=uploaded_file.type,
                    metadata=metadata
                )
                st.success(f"✅ File information saved to database")
            except:
                pass
        
        display_success(f"File loaded successfully! {metadata['rows']} rows × {metadata['columns']} columns")
        
        with st.expander("👀 Preview Raw Data"):
            preview_dataframe(df_raw)
    
    # If data loaded, show configuration
    if st.session_state.get('df_raw') is not None:
        st.markdown("---")
        render_data_configuration()


def render_data_configuration():
    """Render data configuration section"""
    
    st.header("Step 2: Configure Your Data")
    
    df = st.session_state.df_raw
    
    # Detect columns
    datetime_candidates = detect_datetime_columns(df)
    numeric_candidates = detect_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕐 Time Column")
        default_time = datetime_candidates[0] if datetime_candidates else df.columns[0]
        time_column = st.selectbox(
            "Select time column",
            df.columns.tolist(),
            index=df.columns.tolist().index(default_time) if default_time in df.columns else 0
        )
        sample_values = df[time_column].dropna().head(5)
        st.caption("Sample values:")
        for val in sample_values:
            st.caption(f"  • {val}")
        datetime_format = st.selectbox("Datetime format", ["auto"] + config["datetime_formats"])
    
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
    """Process and validate data"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Parse time
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
    
    # Validate numeric columns
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
    
    # Convert to long format
    status.text("Converting to standard format...")
    progress_bar.progress(0.60)
    
    df_long, error = convert_to_long_format(df, time_column, value_columns)
    
    if error:
        display_error("Failed to convert data", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Save parameters
    status.text("Saving parameters...")
    progress_bar.progress(0.75)
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            parameters = []
            for col in value_columns:
                col_data = df[col].dropna()
                parameters.append({
                    'name': col,
                    'data_type': 'numeric',
                    'min_value': float(col_data.min()) if len(col_data) > 0 else None,
                    'max_value': float(col_data.max()) if len(col_data) > 0 else None,
                    'mean_value': float(col_data.mean()) if len(col_data) > 0 else None,
                    'std_value': float(col_data.std()) if len(col_data) > 0 else None,
                    'missing_count': int(df[col].isna().sum()),
                    'total_count': len(df[col])
                })
            db.save_parameters(st.session_state.current_project_id, parameters)
            st.success("✅ Parameters saved")
        except Exception as e:
            st.warning(f"Could not save parameters: {str(e)}")
    
    # Generate health report
    status.text("Generating health report...")
    progress_bar.progress(0.90)
    
    health_report = detect_and_report_issues(df_long)
    health_report["time_metadata"] = time_metadata
    health_report["conversion_info"] = conversion_info
    
    # Store in session
    st.session_state.df_long = df_long
    st.session_state.df_clean = df_long.copy()
    st.session_state.time_column = time_column
    st.session_state.value_columns = value_columns
    st.session_state.health_report = health_report
    st.session_state.data_loaded = True
    st.session_state.preprocessing_applied = False
    
    # Save health report
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            status.text("Saving health report...")
            progress_bar.progress(0.95)
            
            health_score, health_category, issues = calculate_data_health_score(health_report)
            
            health_data = {
                'health_score': health_score,
                'health_category': health_category,
                'total_parameters': len(value_columns),
                'total_data_points': len(df_long),
                'total_missing_values': sum(info.get('count', 0) for info in health_report.get('missing_values', {}).values()),
                'missing_percentage': sum(info.get('count', 0) for info in health_report.get('missing_values', {}).values()) / len(df_long) if len(df_long) > 0 else 0,
                'critical_issues': len([i for i in issues if 'critical' in i.lower()]),
                'warnings': len([i for i in issues if 'warning' in i.lower() or '⚠️' in i]),
                'duplicate_timestamps': time_metadata.get('duplicate_count', 0),
                'outlier_count': sum(info.get('count', 0) for info in health_report.get('outliers', {}).values()),
                'missing_values_detail': health_report.get('missing_values', {}),
                'outliers_detail': health_report.get('outliers', {}),
                'coverage_detail': health_report.get('coverage', {}),
                'issues_list': issues,
                'time_metadata': time_metadata,
                'conversion_info': health_report.get('conversion_info', {}),
                'warnings_list': health_report.get('warnings', []),
                'raw_health_report': health_report,
                'parameters_analyzed': value_columns
            }
            
            db.save_health_report(st.session_state.current_project_id, health_data)
            st.success("✅ Health report saved")
        except Exception as e:
            st.warning(f"Could not save health report: {str(e)}")
    
    # Update progress
    update_project_progress(stage="data_imported", page=2, percentage=7)
    
    # Mark complete
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            db.update_step_completion(
                project_id=st.session_state.current_project_id,
                step_key='data_loaded',
                completed=True
            )
        except:
            pass
    
    progress_bar.progress(1.0)
    status.text("Complete!")
    
    st.balloons()
    
    display_success("🎉 Data processed successfully!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(value_columns))
    col2.metric("Data Points", len(df_long))
    col3.metric("Time Range", f"{time_metadata['time_span']} days")
    
    # ALWAYS show navigation buttons after successful processing
    show_next_steps_after_upload()


def render_health_report_section():
    """Render health report section"""
    
    # Try to load from session state first
    if st.session_state.get('health_report') is not None:
        # Check if we have df_long in session (means fresh upload)
        if st.session_state.get('df_long') is not None:
            render_session_health_report()
        else:
            # Have health report but no df_long - from database
            render_database_health_report_full(st.session_state.health_report)
        return
    
    # Try database
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            st.info("📊 Checking for health report...")
            
            # Try to load existing report
            health_report = db.get_health_report(st.session_state.current_project_id)
            
            if not health_report:
                # No report - try to generate from parameters
                needs_update = db.needs_health_report_update(st.session_state.current_project_id)
                
                if needs_update:
                    with st.spinner("🔄 Generating from database..."):
                        health_report = db.generate_health_report_from_parameters(st.session_state.current_project_id)
                    
                    if not health_report.get('success'):
                        st.warning("⚠️ " + health_report.get('message', 'No health report'))
                        st.info("📤 Upload data first. Switch to **Upload Data** tab.")
                        return
                    
                    st.success("✅ Generated!")
                else:
                    st.warning("⚠️ No data found")
                    st.info("📤 Upload data first. Switch to **Upload Data** tab.")
                    return
            else:
                st.success("✅ Loaded from database")
            
            # Store and display
            st.session_state.health_report = health_report
            render_database_health_report_full(health_report)
            return
            
        except Exception as e:
            st.error(f"Error loading health report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    st.info("📤 Upload data first. Switch to **Upload Data** tab.")


def render_database_health_report_full(health_report):
    """Render complete health report from database"""
    
    st.header("📊 Data Health Report")
    
    # Health Score
    st.markdown("### Overall Data Health")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        health_score = health_report.get('health_score', 0)
        health_category = health_report.get('health_category', 'unknown')
        
        score_map = {
            "excellent": ("🟢", "Excellent"),
            "good": ("🟡", "Good"),
            "fair": ("🟠", "Fair"),
            "poor": ("🔴", "Poor")
        }
        score_color, score_text = score_map.get(health_category, ("⚪", "Unknown"))
        
        st.metric("Health Score", f"{health_score}/100")
        st.markdown(f"### {score_color} {score_text}")
    
    with col2:
        issues = health_report.get('issues_list', [])
        if issues:
            st.markdown("**Issues Detected:**")
            for issue in issues[:5]:
                st.caption(issue)
            if len(issues) > 5:
                st.caption(f"... and {len(issues) - 5} more")
        else:
            st.success("✅ No significant issues!")
    
    with col3:
        st.metric("Parameters", health_report.get('total_parameters', 0))
        st.metric("Data Points", health_report.get('total_data_points', 0))
    
    st.markdown("---")
    
    # Missing Values
    st.subheader("❓ Missing Values")
    
    missing_detail = health_report.get('missing_values_detail', {})
    
    if missing_detail:
        missing_data = []
        for var, info in missing_detail.items():
            missing_count = info.get('count', 0)
            total_count = info.get('total', info.get('total_count', 0))
            
            if total_count > 0:
                missing_pct = missing_count / total_count
            else:
                missing_pct = info.get('percentage', 0)
            
            status = "🔴 Critical" if missing_pct > 0.20 else "🟡 Warning" if missing_pct > 0.05 else "🟢 Good"
            
            missing_data.append({
                "Variable": var,
                "Missing Count": missing_count,
                "Total": total_count,
                "Missing %": f"{missing_pct*100:.1f}%",
                "Status": status
            })
        
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
        overall_pct = health_report.get('missing_percentage', 0)
        
        if overall_pct > 0.20:
            st.error(f"⚠️ Overall: {overall_pct*100:.1f}%")
        elif overall_pct > 0.05:
            st.warning(f"ℹ️ Overall: {overall_pct*100:.1f}%")
        else:
            st.success(f"✅ Overall: {overall_pct*100:.1f}%")
    else:
        st.success("✅ No missing values!")
    
    st.markdown("---")
    
    # Time metadata
    time_meta = health_report.get('time_metadata', {})
    if time_meta:
        st.subheader("⏰ Time Series Info")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'min_time' in time_meta:
                st.metric("Start Date", str(time_meta['min_time'])[:10])
            if 'max_time' in time_meta:
                st.metric("End Date", str(time_meta['max_time'])[:10])
        
        with col2:
            if 'time_span' in time_meta:
                st.metric("Time Span", f"{time_meta['time_span']} days")
            if 'estimated_frequency' in time_meta:
                st.metric("Frequency", time_meta['estimated_frequency'])
        
        with col3:
            dup_count = time_meta.get('duplicate_count', 0)
            if dup_count > 0:
                st.metric("Duplicates", dup_count, delta="Warning", delta_color="inverse")
            else:
                st.metric("Duplicates", "0", delta="Good")
    
    st.markdown("---")
    
    # Parameters table
    st.subheader("📋 Parameters Overview")
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            parameters = db.get_project_parameters(st.session_state.current_project_id)
            
            param_table = []
            for param in parameters:
                param_table.append({
                    "Parameter": param.get('parameter_name', 'Unknown'),
                    "Type": param.get('data_type', 'numeric'),
                    "Min": f"{param.get('min_value', 0):.2f}" if param.get('min_value') is not None else "N/A",
                    "Max": f"{param.get('max_value', 0):.2f}" if param.get('max_value') is not None else "N/A",
                    "Mean": f"{param.get('mean_value', 0):.2f}" if param.get('mean_value') is not None else "N/A",
                    "Missing": f"{param.get('missing_count', 0):,}",
                    "Total": f"{param.get('total_count', 0):,}"
                })
            
            param_df = pd.DataFrame(param_table)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load parameters: {str(e)}")
    
    st.markdown("---")
    
    # Navigation
    render_health_report_navigation(health_report.get('health_score', 100))


def render_session_health_report():
    """Render health report from session state (just after upload)"""
    
    try:
        health = st.session_state.health_report
        df_long = st.session_state.get('df_long')
        
        if df_long is None:
            st.warning("⚠️ Data not available")
            return
        
        health_score, health_category, issues = calculate_data_health_score(health)
        
        # Build report dict
        report_dict = {
            'health_score': health_score,
            'health_category': health_category,
            'total_parameters': len(st.session_state.get('value_columns', [])),
            'total_data_points': len(df_long),
            'total_missing_values': sum(info.get('count', 0) for info in health.get('missing_values', {}).values()),
            'missing_percentage': sum(info.get('count', 0) for info in health.get('missing_values', {}).values()) / len(df_long) if len(df_long) > 0 else 0,
            'critical_issues': len([i for i in issues if 'critical' in i.lower()]),
            'warnings': len([i for i in issues if 'warning' in i.lower()]),
            'duplicate_timestamps': health.get('time_metadata', {}).get('duplicate_count', 0),
            'outlier_count': sum(info.get('count', 0) for info in health.get('outliers', {}).values()),
            'missing_values_detail': health.get('missing_values', {}),
            'outliers_detail': health.get('outliers', {}),
            'coverage_detail': health.get('coverage', {}),
            'issues_list': issues,
            'time_metadata': health.get('time_metadata', {}),
            'parameters_analyzed': st.session_state.get('value_columns', [])
        }
        
        render_database_health_report_full(report_dict)
    
    except Exception as e:
        st.error(f"Error rendering health report: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_health_report_navigation(health_score):
    """Render navigation at bottom of health report"""
    
    st.header("🎯 What's Next?")
    
    if health_score >= 85:
        st.success("**Excellent Data Quality! ✨**")
        primary_action = "visualize"
    elif health_score >= 70:
        st.info("**Good Data Quality 👍**")
        primary_action = "either"
    elif health_score >= 50:
        st.warning("**Fair Data Quality ⚠️**")
        primary_action = "clean"
    else:
        st.error("**Poor Data Quality 🔴**")
        primary_action = "clean"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize")
        if health_score >= 70:
            st.success("✅ Recommended")
        
        if st.button("📊 Go to Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary",
                     key="nav_viz_health"):
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db.update_project_progress(
                    project_id=st.session_state.current_project_id,
                    workflow_state="exploration",
                    current_page=4,
                    completion_percentage=14
                )
            st.switch_page("pages/04_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if health_score < 70:
            st.warning("⚠️ Recommended")
        
        if st.button("🧹 Go to Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary",
                     key="nav_clean_health"):
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db.update_project_progress(
                    project_id=st.session_state.current_project_id,
                    workflow_state="preprocessing",
                    current_page=3,
                    completion_percentage=7
                )
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📁 Upload More")
        st.info("💡 Switch to **Upload Data** tab")


if __name__ == "__main__":
    if not DB_AVAILABLE:
        st.error("❌ Database not available")
        st.stop()
    
    main()
