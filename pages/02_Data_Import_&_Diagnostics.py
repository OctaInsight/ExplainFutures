"""
Page 2: Data Import & Diagnostics (FIXED - Boolean error corrected)
✅ Saves with data_source='raw'
✅ Fixed: DataFrame boolean check error
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import time
from pathlib import Path

st.set_page_config(
    page_title="Upload & Data Diagnostics",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

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

initialize_session_state()
config = get_config()
render_app_sidebar()

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
    """Calculate health score"""
    score = 100
    issues = []
    
    if health_report.get("missing_values"):
        for var, info in health_report["missing_values"].items():
            missing_pct = info.get("percentage", 0)
            if missing_pct > 0.20:
                score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing (critical)")
            elif missing_pct > 0.05:
                score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing")
    
    if health_report.get("outliers"):
        try:
            outlier_count = sum(info.get("count", 0) for info in health_report["outliers"].values())
            if outlier_count > 0:
                score -= 5
                issues.append(f"📊 {outlier_count} outliers")
        except:
            pass
    
    if "time_metadata" in health_report:
        time_meta = health_report["time_metadata"]
        if time_meta.get("duplicate_count", 0) > 0:
            score -= 10
            issues.append(f"⏰ {time_meta['duplicate_count']} duplicates")
    
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


def get_comprehensive_health_report():
    """Get comprehensive health report from database"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return None
    
    try:
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if not parameters:
            return None
        
        total_missing = 0
        total_count = 0
        missing_values_detail = {}
        
        for param in parameters:
            param_name = param['parameter_name']
            missing_count = param.get('missing_count', 0)
            param_total = param.get('total_count', 0)
            
            total_missing += missing_count
            total_count += param_total
            
            if param_total > 0:
                missing_pct = missing_count / param_total
                missing_values_detail[param_name] = {
                    'count': missing_count,
                    'percentage': missing_pct,
                    'total': param_total
                }
        
        health_score = 100
        issues = []
        
        for var, info in missing_values_detail.items():
            missing_pct = info['percentage']
            if missing_pct > 0.20:
                health_score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing (critical)")
            elif missing_pct > 0.05:
                health_score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing")
        
        health_score = max(0, min(100, health_score))
        
        if health_score >= 85:
            category = "excellent"
        elif health_score >= 70:
            category = "good"
        elif health_score >= 50:
            category = "fair"
        else:
            category = "poor"
        
        comprehensive_report = {
            'health_score': health_score,
            'health_category': category,
            'total_parameters': len(parameters),
            'total_data_points': total_count,
            'total_missing_values': total_missing,
            'missing_percentage': total_missing / total_count if total_count > 0 else 0,
            'critical_issues': len([i for i in issues if 'critical' in i.lower()]),
            'warnings': len([i for i in issues if 'warning' in i.lower()]),
            'missing_values_detail': missing_values_detail,
            'issues_list': issues,
            'parameters_analyzed': [p['parameter_name'] for p in parameters],
            'time_metadata': {},
            'outliers_detail': {},
            'coverage_detail': {},
            'duplicate_timestamps': 0,
            'outlier_count': 0
        }
        
        db_health_report = db.get_health_report(st.session_state.current_project_id)
        if db_health_report:
            comprehensive_report['time_metadata'] = db_health_report.get('time_metadata', {})
            comprehensive_report['duplicate_timestamps'] = db_health_report.get('duplicate_timestamps', 0)
            comprehensive_report['outliers_detail'] = db_health_report.get('outliers_detail', {})
            comprehensive_report['outlier_count'] = db_health_report.get('outlier_count', 0)
        
        return comprehensive_report
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def update_project_progress(stage: str = "data_import", page: int = 2, percentage: int = 7):
    """Update progress"""
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
        st.warning(f"Progress update failed: {str(e)}")


def show_next_steps_after_upload():
    """Show next steps"""
    st.markdown("---")
    st.header("🎯 What's Next?")
    
    health_score = 100
    comprehensive_report = get_comprehensive_health_report()
    
    if comprehensive_report:
        health_score = comprehensive_report['health_score']
    elif st.session_state.get('health_report'):
        try:
            health_score, _, _ = calculate_data_health_score(st.session_state.health_report)
        except:
            health_score = 75
    
    if health_score >= 85:
        st.success("**Excellent Data Quality! ✨**")
        primary_action = "visualize"
    elif health_score >= 70:
        st.info("**Good Data Quality 👍**")
        primary_action = "either"
    elif health_score >= 50:
        st.warning("**Fair Data Quality ⚠️** Consider cleaning")
        primary_action = "clean"
    else:
        st.error("**Poor Data Quality 🔴** Cleaning recommended")
        primary_action = "clean"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if health_score >= 70:
            st.success("✅ Recommended")
        
        if st.button("📊 Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary"):
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
        
        if st.button("🧹 Data Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary"):
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db.update_project_progress(
                    project_id=st.session_state.current_project_id,
                    workflow_state="preprocessing",
                    current_page=3,
                    completion_percentage=7
                )
            st.switch_page("pages/03_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📋 Health Report")
        st.info("💡 Switch to **Data Health Report** tab")


def load_data_from_database_if_available():
    """Load from database"""
    if not DB_AVAILABLE or not st.session_state.get('current_project_id'):
        return False
    
    try:
        parameters = db.get_project_parameters(st.session_state.current_project_id)
        
        if not parameters:
            return False
        
        st.session_state.project_parameters = parameters
        st.session_state.value_columns = [p['parameter_name'] for p in parameters]
        
        comprehensive_report = get_comprehensive_health_report()
        
        if comprehensive_report:
            st.session_state.health_report = comprehensive_report
            st.session_state.data_loaded = True
        
        return True
        
    except Exception as e:
        st.warning(f"Load failed: {str(e)}")
        return False


def main():
    """Main function"""
    
    if not st.session_state.get('current_project_id'):
        st.warning("⚠️ Please select a project")
        if st.button("← Go to Home"):
            st.switch_page("pages/01_Home.py")
        st.stop()
    
    # FIXED: Check if df_long exists properly
    df_long_exists = False
    if 'df_long' in st.session_state and st.session_state.df_long is not None:
        try:
            # Check if it's a DataFrame and not empty
            if isinstance(st.session_state.df_long, pd.DataFrame) and len(st.session_state.df_long) > 0:
                df_long_exists = True
        except:
            pass
    
    if not st.session_state.get('data_loaded') and not df_long_exists:
        data_loaded = load_data_from_database_if_available()
        
        if data_loaded:
            params_count = len(st.session_state.get('project_parameters', []))
            st.info(f"📊 Loaded from database: {params_count} parameters")
    
    tab1, tab2 = st.tabs(["📤 Upload Data", "🔍 Data Health Report"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_health_report_section()


def render_upload_section():
    """Upload section"""
    st.header("Step 1: Upload Your Data File")
    
    if st.session_state.get('data_loaded') and st.session_state.get('value_columns'):
        with st.expander("✅ Current Project Data", expanded=False):
            st.success(f"📊 {len(st.session_state.value_columns)} parameters loaded")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption("**Parameters:**")
                for var in st.session_state.value_columns[:10]:
                    st.text(f"  • {var}")
                if len(st.session_state.value_columns) > 10:
                    st.caption(f"  ... +{len(st.session_state.value_columns) - 10} more")
            
            with col2:
                if st.session_state.get('health_report'):
                    health = st.session_state.health_report
                    st.caption("**Health:**")
                    st.text(f"  Score: {health.get('health_score', 'N/A')}/100")
                    st.text(f"  Category: {health.get('health_category', 'N/A')}")
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            uploaded_files = db.get_uploaded_files(st.session_state.current_project_id)
            if uploaded_files:
                with st.expander(f"📂 Previous Uploads ({len(uploaded_files)})"):
                    for file_info in uploaded_files[:5]:
                        st.caption(f"📄 {file_info['filename']} - {file_info['uploaded_at'][:10]}")
        except:
            pass
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=config["accepted_file_types"]
    )
    
    if uploaded_file is None:
        st.info("👆 Upload CSV, TXT, or Excel")
        
        if st.session_state.get('data_loaded') and st.session_state.get('value_columns'):
            st.markdown("---")
            show_next_steps_after_upload()
        return
    
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.upload_timestamp = datetime.now()
    
    st.success(f"✅ File: **{uploaded_file.name}**")
    st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
    
    sheet_name = None
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        sheet_names, error = get_excel_sheet_names(uploaded_file.getvalue())
        if error:
            display_error("Excel read failed", error)
            return
        if len(sheet_names) > 1:
            sheet_name = st.selectbox("Sheet", sheet_names)
        else:
            sheet_name = sheet_names[0]
    
    delimiter = None
    decimal = "."
    
    if uploaded_file.name.endswith(('.csv', '.txt')):
        with st.expander("⚙️ Options"):
            col1, col2 = st.columns(2)
            with col1:
                delim_opt = st.selectbox("Delimiter", ["Auto", ",", ";", "Tab", "|"])
                if delim_opt != "Auto":
                    delimiter = "\t" if delim_opt == "Tab" else delim_opt
            with col2:
                decimal = st.selectbox("Decimal", [".", ","])
    
    st.markdown("---")
    
    if st.button("📊 Load File", type="primary", use_container_width=True):
        with st.spinner("Loading..."):
            df_raw, error, metadata = load_file_smart(
                uploaded_file,
                delimiter=delimiter,
                decimal=decimal,
                sheet_name=sheet_name
            )
        
        if error:
            display_error("Load failed", error)
            return
        
        st.session_state.df_raw = df_raw
        st.session_state.file_metadata = metadata
        
        if DB_AVAILABLE and st.session_state.get('current_project_id'):
            try:
                db.save_uploaded_file(
                    project_id=st.session_state.current_project_id,
                    filename=uploaded_file.name,
                    file_size=uploaded_file.size,
                    file_type=uploaded_file.type,
                    metadata=metadata
                )
            except:
                pass
        
        display_success(f"Loaded! {metadata['rows']} × {metadata['columns']}")
        
        with st.expander("👀 Preview"):
            preview_dataframe(df_raw)
    
    if st.session_state.get('df_raw') is not None:
        st.markdown("---")
        render_data_configuration()


def render_data_configuration():
    """Configure data"""
    st.header("Step 2: Configure Your Data")
    
    df = st.session_state.df_raw
    
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
        datetime_format = st.selectbox("Format", ["auto"] + config["datetime_formats"])
    
    with col2:
        st.markdown("#### 📊 Variables")
        default_vars = [col for col in numeric_candidates if col != time_column]
        value_columns = st.multiselect(
            "Select variables",
            [col for col in df.columns if col != time_column],
            default=default_vars[:10]
        )
        if value_columns:
            st.success(f"✅ {len(value_columns)} selected")
    
    st.markdown("---")
    
    if not value_columns:
        st.warning("⚠️ Select at least one variable")
        return
    
    if st.button("⚡ Process Data", type="primary", use_container_width=True):
        process_data(df, time_column, value_columns, datetime_format)


def process_data(df, time_column, value_columns, datetime_format):
    """Process data"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    status.text("Parsing time...")
    progress_bar.progress(0.20)
    
    if datetime_format == "auto":
        datetime_format = None
    
    parsed_time, error = parse_time_column(df, time_column, datetime_format)
    
    if error:
        display_error("Time parse failed", error)
        progress_bar.empty()
        status.empty()
        return
    
    is_valid, error, time_metadata = validate_time_series(parsed_time)
    
    if not is_valid:
        display_error("Time validation failed", error)
        progress_bar.empty()
        status.empty()
        return
    
    df[time_column] = parsed_time
    
    status.text("Validating variables...")
    progress_bar.progress(0.40)
    
    valid_cols, invalid_cols, conversion_info = check_numeric_columns(df, value_columns)
    
    if invalid_cols:
        st.warning(f"⚠️ Skipped: {', '.join(invalid_cols)}")
        if not valid_cols:
            display_error("No valid columns")
            progress_bar.empty()
            status.empty()
            return
        value_columns = valid_cols
    
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    status.text("Converting format...")
    progress_bar.progress(0.60)
    
    df_long, error = convert_to_long_format(df, time_column, value_columns)
    
    if error:
        display_error("Conversion failed", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Save with data_source='raw'
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        status.text("Saving to database...")
        progress_bar.progress(0.70)
        
        try:
            success = db.save_timeseries_data(
                project_id=st.session_state.current_project_id,
                df_long=df_long,
                data_source='raw',
                batch_size=1000
            )
            
            if success:
                summary = db.get_timeseries_summary(
                    project_id=st.session_state.current_project_id,
                    data_source='raw'
                )
                st.success(f"✅ Saved {summary['total_records']:,} data points (source='raw')")
            else:
                st.warning("⚠️ Save failed")
                
        except Exception as e:
            st.warning(f"Save error: {str(e)}")
    
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
            st.warning(f"Param save error: {str(e)}")
    
    status.text("Generating health report...")
    progress_bar.progress(0.90)
    
    health_report = detect_and_report_issues(df_long)
    health_report["time_metadata"] = time_metadata
    health_report["conversion_info"] = conversion_info
    
    st.session_state.df_long = df_long
    st.session_state.df_clean = df_long.copy()
    st.session_state.time_column = time_column
    st.session_state.value_columns = value_columns
    
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
                'warnings': len([i for i in issues if 'warning' in i.lower()]),
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
            st.warning(f"Health save error: {str(e)}")
    
    comprehensive_report = get_comprehensive_health_report()
    if comprehensive_report:
        st.session_state.health_report = comprehensive_report
        st.session_state.data_loaded = True
    
    update_project_progress(stage="data_imported", page=2, percentage=7)
    
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
    
    display_success("🎉 Data processed!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(value_columns))
    col2.metric("Data Points", len(df_long))
    col3.metric("Time Range", f"{time_metadata['time_span']} days")
    
    show_next_steps_after_upload()


def render_health_report_section():
    """Health report section"""
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        try:
            comprehensive_report = get_comprehensive_health_report()
            
            if not comprehensive_report:
                st.warning("⚠️ No data")
                st.info("📤 Upload data first")
                return
            
            render_comprehensive_health_report(comprehensive_report)
            return
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.info("📤 Upload data first")


def render_comprehensive_health_report(health_report):
    """Render health report"""
    
    st.header("📊 Data Health Report")
    
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
            st.markdown("**Issues:**")
            for issue in issues[:5]:
                st.caption(issue)
            if len(issues) > 5:
                st.caption(f"... +{len(issues) - 5}")
        else:
            st.success("✅ No issues!")
    
    with col3:
        st.metric("Parameters", health_report.get('total_parameters', 0))
        st.metric("Data Points", health_report.get('total_data_points', 0))
    
    st.markdown("---")
    
    st.subheader("❓ Missing Values")
    
    missing_detail = health_report.get('missing_values_detail', {})
    
    if missing_detail:
        missing_data = []
        for var, info in missing_detail.items():
            missing_count = info.get('count', 0)
            total_count = info.get('total', 0)
            
            if total_count > 0:
                missing_pct = missing_count / total_count
            else:
                missing_pct = info.get('percentage', 0)
            
            status = "🔴" if missing_pct > 0.20 else "🟡" if missing_pct > 0.05 else "🟢"
            
            missing_data.append({
                "Variable": var,
                "Missing": missing_count,
                "Total": total_count,
                "Missing %": f"{missing_pct*100:.1f}%",
                "Status": status
            })
        
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values!")
    
    st.markdown("---")
    
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        st.subheader("📋 Parameters")
        try:
            parameters = db.get_project_parameters(st.session_state.current_project_id)
            
            param_table = []
            for param in parameters:
                param_table.append({
                    "Parameter": param.get('parameter_name', 'Unknown'),
                    "Min": f"{param.get('min_value', 0):.2f}" if param.get('min_value') is not None else "N/A",
                    "Max": f"{param.get('max_value', 0):.2f}" if param.get('max_value') is not None else "N/A",
                    "Mean": f"{param.get('mean_value', 0):.2f}" if param.get('mean_value') is not None else "N/A",
                    "Missing": f"{param.get('missing_count', 0):,}",
                    "Total": f"{param.get('total_count', 0):,}"
                })
            
            st.dataframe(pd.DataFrame(param_table), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Load error: {str(e)}")


if __name__ == "__main__":
    if not DB_AVAILABLE:
        st.error("❌ Database unavailable")
        st.stop()
    
    main()
