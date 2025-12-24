"""
Page 1: Upload & Data Diagnostics
Handles file upload, time column selection, data validation, and health reporting
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

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

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Upload & Data Diagnostics", page_icon="📁", layout="wide")

# Render shared sidebar
render_app_sidebar()  

st.title("📁 Upload & Data Diagnostics")
st.markdown("*Upload your time-series data and assess data quality*")
st.markdown("---")


def calculate_data_health_score(health_report):
    """
    Calculate overall data health score (0-100)
    Returns: (score, category, issues)
    """
    score = 100
    issues = []
    
    # Missing values penalty
    if health_report.get("missing_values"):
        total_missing = 0
        total_values = 0
        for var, info in health_report["missing_values"].items():
            total_missing += info["count"]
            missing_pct = info["percentage"]
            
            if missing_pct > 0.20:  # >20% missing
                score -= 15
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data (critical)")
            elif missing_pct > 0.05:  # >5% missing
                score -= 5
                issues.append(f"⚠️ {var}: {missing_pct*100:.1f}% missing data")
    
    # Outliers check
    if health_report.get("outliers"):
        outlier_count = sum(info["count"] for info in health_report["outliers"].values())
        if outlier_count > 0:
            score -= 5
            issues.append(f"📊 {outlier_count} outliers detected across variables")
    
    # Time series issues
    if "time_metadata" in health_report:
        time_meta = health_report["time_metadata"]
        if time_meta.get("duplicate_count", 0) > 0:
            score -= 10
            issues.append(f"⏰ {time_meta['duplicate_count']} duplicate timestamps found")
    
    # Warnings
    if health_report.get("warnings"):
        score -= len(health_report["warnings"]) * 5
        for warning in health_report["warnings"]:
            issues.append(f"⚠️ {warning}")
    
    # Ensure score is between 0 and 100
    score = max(0, min(100, score))
    
    # Categorize
    if score >= 85:
        category = "excellent"
    elif score >= 70:
        category = "good"
    elif score >= 50:
        category = "fair"
    else:
        category = "poor"
    
    return score, category, issues


def main():
    """Main page function"""
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📤 Upload Data", "🔍 Data Health Report"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_health_report_section()


def render_upload_section():
    """Render the file upload and configuration section"""
    
    st.header("Step 1: Upload Your Data File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=config["accepted_file_types"],
        help=f"Supported formats: {', '.join(config['accepted_file_types'])}"
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a CSV, TXT, or Excel file to get started")
        
        # Show example format
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
    
    # Excel sheet selection (if applicable)
    sheet_name = None
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        st.markdown("### Excel File Options")
        
        # Get sheet names
        sheet_names, error = get_excel_sheet_names(uploaded_file.getvalue())
        
        if error:
            display_error("Could not read Excel file", error)
            return
        
        if len(sheet_names) > 1:
            sheet_name = st.selectbox(
                "Select sheet to load",
                sheet_names,
                help="Choose which sheet contains your data"
            )
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
                    ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab", "Pipe (|)"],
                    help="Character that separates columns"
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
                decimal = st.selectbox(
                    "Decimal separator",
                    [".", ","],
                    help="Character used for decimal points"
                )
    
    # Load file button
    st.markdown("---")
    
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
        
        display_success(f"File loaded successfully! {metadata['rows']} rows × {metadata['columns']} columns")
        
        # Show preview
        with st.expander("👀 Preview Raw Data"):
            preview_dataframe(df_raw)
    
    # If data is loaded, show configuration
    if st.session_state.df_raw is not None:
        st.markdown("---")
        render_data_configuration()


def render_data_configuration():
    """Render data configuration section (time column, variables)"""
    
    st.header("Step 2: Configure Your Data")
    
    df = st.session_state.df_raw
    
    # Detect potential columns
    datetime_candidates = detect_datetime_columns(df)
    numeric_candidates = detect_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕐 Time Column")
        
        # Time column selection
        if datetime_candidates:
            default_time = datetime_candidates[0]
        else:
            default_time = df.columns[0]
        
        time_column = st.selectbox(
            "Select time column",
            df.columns.tolist(),
            index=df.columns.tolist().index(default_time) if default_time in df.columns else 0,
            help="Column containing timestamps or dates"
        )
        
        # Show sample values
        sample_values = df[time_column].dropna().head(5)
        st.caption("Sample values:")
        for val in sample_values:
            st.caption(f"  • {val}")
        
        # Datetime format selection
        datetime_format = st.selectbox(
            "Datetime format",
            ["auto"] + config["datetime_formats"],
            help="Format of the time column. 'auto' tries to detect automatically"
        )
    
    with col2:
        st.markdown("#### 📊 Variables (Value Columns)")
        
        # Show suggested numeric columns
        if numeric_candidates:
            st.info(f"Detected {len(numeric_candidates)} numeric columns")
        
        # Variable selection
        default_vars = [col for col in numeric_candidates if col != time_column]
        
        value_columns = st.multiselect(
            "Select variables to analyze",
            [col for col in df.columns if col != time_column],
            default=default_vars[:10] if len(default_vars) > 10 else default_vars,
            help="Choose numeric columns to include in analysis"
        )
        
        if value_columns:
            st.success(f"✅ Selected {len(value_columns)} variables")
    
    # Process button
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
    progress_bar.progress(0.25)
    
    if datetime_format == "auto":
        datetime_format = None
    
    parsed_time, error = parse_time_column(df, time_column, datetime_format)
    
    if error:
        display_error("Failed to parse time column", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Validate time series
    is_valid, error, time_metadata = validate_time_series(parsed_time)
    
    if not is_valid:
        display_error("Time column validation failed", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Update dataframe with parsed time
    df[time_column] = parsed_time
    
    # Step 2: Validate numeric columns
    status.text("Validating variables...")
    progress_bar.progress(0.50)
    
    valid_cols, invalid_cols, conversion_info = check_numeric_columns(df, value_columns)
    
    if invalid_cols:
        st.warning(f"⚠️ Some columns could not be used as numeric: {', '.join(invalid_cols)}")
        for col in invalid_cols:
            st.caption(f"  • {col}: {conversion_info[col]}")
        
        if not valid_cols:
            display_error("No valid numeric columns found")
            progress_bar.empty()
            status.empty()
            return
        
        value_columns = valid_cols
    
    # Convert to numeric
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 3: Convert to long format
    status.text("Converting to standard format...")
    progress_bar.progress(0.75)
    
    df_long, error = convert_to_long_format(df, time_column, value_columns)
    
    if error:
        display_error("Failed to convert to long format", error)
        progress_bar.empty()
        status.empty()
        return
    
    # Step 4: Generate health report
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
    
    progress_bar.progress(1.0)
    status.text("Complete!")
    
    # Success message
    st.balloons()
    display_success("Data processed successfully!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables", len(value_columns))
    col2.metric("Data Points", len(df_long))
    col3.metric("Time Range", f"{time_metadata['time_span']} days")
    
    st.info("💡 Go to the **Data Health Report** tab to review data quality")


def render_health_report_section():
    """Render the data health report section"""
    
    if not st.session_state.data_loaded:
        st.info("📤 Upload and process data first to see the health report")
        return
    
    if st.session_state.health_report is None:
        st.warning("⚠️ No health report available. Please reprocess your data.")
        return
    
    st.header("Data Health Report")
    
    health = st.session_state.health_report
    df_long = st.session_state.df_long
    
    # Calculate health score
    health_score, health_category, issues = calculate_data_health_score(health)
    
    # Health Score Dashboard
    st.markdown("### 📊 Overall Data Health")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Health score with color
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
        # Health issues summary
        if issues:
            st.markdown("**Issues Detected:**")
            for issue in issues[:5]:  # Show top 5 issues
                st.caption(issue)
            if len(issues) > 5:
                st.caption(f"... and {len(issues) - 5} more issues")
        else:
            st.success("✅ No significant issues detected!")
    
    with col3:
        # Quick stats
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
    
    # Missing values
    st.subheader("❓ Missing Values")
    
    if health["missing_values"]:
        missing_data = []
        for var, info in health["missing_values"].items():
            missing_data.append({
                "Variable": var,
                "Missing Count": info["count"],
                "Missing %": f"{info['percentage']*100:.1f}%",
                "Status": "🔴 Critical" if info['percentage'] > 0.20 else "🟡 Warning" if info['percentage'] > 0.05 else "🟢 Good"
            })
        
        missing_df = pd.DataFrame(missing_data)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
        # Summary
        total_missing = sum(info["count"] for info in health["missing_values"].values())
        total_values = len(df_long)
        overall_pct = total_missing / total_values if total_values > 0 else 0
        
        if overall_pct > 0.20:
            display_warning(f"Overall missing data: {format_percentage(overall_pct)} - Consider data cleaning")
        elif overall_pct > 0.05:
            st.info(f"ℹ️ Overall missing data: {format_percentage(overall_pct)} - Monitor quality")
        else:
            display_success(f"Overall missing data: {format_percentage(overall_pct)} - Good quality")
    else:
        display_success("No missing values detected!")
    
    st.markdown("---")
    
    # Outliers
    st.subheader("📊 Outlier Detection")
    
    if health["outliers"]:
        outlier_data = []
        for var, info in health["outliers"].items():
            outlier_data.append({
                "Variable": var,
                "Outliers": info["count"],
                "Lower Bound": f"{info['lower_bound']:.2f}",
                "Upper Bound": f"{info['upper_bound']:.2f}"
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        st.caption("💡 Outliers detected using IQR method (values beyond 3×IQR from quartiles)")
    else:
        st.info("No outlier information available")
    
    st.markdown("---")
    
    # Coverage
    st.subheader("📅 Time Coverage by Variable")
    
    if health["coverage"]:
        coverage_data = []
        for var, info in health["coverage"].items():
            coverage_data.append({
                "Variable": var,
                "Start": info["start"].strftime("%Y-%m-%d"),
                "End": info["end"].strftime("%Y-%m-%d"),
                "Data Points": info["points"]
            })
        
        coverage_df = pd.DataFrame(coverage_data)
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Warnings
    if health.get("warnings"):
        st.subheader("⚠️ Warnings")
        for warning in health["warnings"]:
            st.warning(warning)
        st.markdown("---")
    
    # INTELLIGENT RECOMMENDATIONS & NEXT STEPS
    st.markdown("---")
    st.header("🎯 What's Next?")
    
    # Provide intelligent recommendations
    st.markdown("### 💡 Recommendations Based on Data Health")
    
    if health_score >= 85:
        st.success("""
        **Excellent Data Quality! ✨**
        
        Your data is in great shape and ready for analysis:
        - ✅ Minimal missing values
        - ✅ Clean time series
        - ✅ Good data coverage
        
        **Recommended:** Proceed directly to visualization and analysis.
        """)
        primary_action = "visualize"
        
    elif health_score >= 70:
        st.info("""
        **Good Data Quality 👍**
        
        Your data is generally good but has minor issues:
        - Some missing values or outliers detected
        - Overall structure is sound
        
        **Options:**
        - Proceed to visualization if issues are acceptable
        - Or clean the data first for optimal results
        """)
        primary_action = "either"
        
    elif health_score >= 50:
        st.warning("""
        **Fair Data Quality ⚠️**
        
        Your data has noticeable quality issues:
        - Significant missing values or outliers
        - May affect analysis quality
        
        **Recommended:** Clean and preprocess your data before analysis.
        """)
        primary_action = "clean"
        
    else:
        st.error("""
        **Poor Data Quality 🔴**
        
        Your data has serious quality issues:
        - High missing data percentages
        - Multiple structural problems
        
        **Strongly Recommended:** 
        1. Clean and preprocess the data, OR
        2. Upload a better quality dataset
        """)
        primary_action = "clean"
    
    st.markdown("---")
    
    # Action buttons
    st.markdown("### 🚀 Choose Your Next Step")
    st.markdown("*Select what you'd like to do next:*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Visualize Data")
        if health_score >= 70:
            st.success("✅ Recommended")
        st.markdown("Explore your data with interactive charts and plots.")
        
        if st.button("📊 Go to Visualization", 
                     use_container_width=True, 
                     type="primary" if primary_action == "visualize" else "secondary",
                     key="btn_visualize"):
            st.switch_page("pages/3_Data_Exploration_and_Visualization.py")
    
    with col2:
        st.markdown("#### 🧹 Clean Data")
        if health_score < 70:
            st.warning("⚠️ Recommended")
        st.markdown("Handle missing values, outliers, and preprocess data.")
        
        if st.button("🧹 Go to Data Cleaning", 
                     use_container_width=True,
                     type="primary" if primary_action == "clean" else "secondary",
                     key="btn_clean"):
            st.switch_page("pages/2_Data_Cleaning_and_Preprocessing.py")
    
    with col3:
        st.markdown("#### 📁 Upload New Data")
        st.markdown("Start over with a different dataset or re-upload.")
        
        if st.button("📁 Upload Different File", 
                     use_container_width=True,
                     key="btn_reupload"):
            # Clear current data
            st.session_state.data_loaded = False
            st.session_state.df_raw = None
            st.session_state.df_long = None
            st.session_state.health_report = None
            st.rerun()
    
    st.markdown("---")
    
    # Optional: Save to database (inactive for now)
    st.markdown("### 💾 Data Persistence")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("""
        **Database Integration** (Coming in Phase 4)
        
        Once activated, your data will be automatically saved to Supabase database 
        for persistent storage and easy access across sessions.
        """)
    
    with col2:
        if st.button("💾 Save to Database", 
                     disabled=True, 
                     use_container_width=True,
                     help="Feature available in Phase 4"):
            pass  # Placeholder for future implementation


if __name__ == "__main__":
    main()
