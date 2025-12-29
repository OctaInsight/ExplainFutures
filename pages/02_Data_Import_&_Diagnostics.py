# ADD THIS SECTION AFTER process_data() function completes successfully
# Insert this RIGHT AFTER the "Data processed successfully" message
# Around line 540 in your current file

def show_next_steps_after_upload():
    """Show navigation options after data is successfully uploaded and processed"""
    
    st.markdown("---")
    st.header("🎯 What's Next?")
    
    # Get health score if available
    health_score = 100
    if st.session_state.get('health_report'):
        health = st.session_state.health_report
        # Calculate health score
        missing_pct = sum(info.get('count', 0) for info in health.get('missing_values', {}).values()) / len(st.session_state.df_long) if len(st.session_state.df_long) > 0 else 0
        
        health_score = 100
        if missing_pct > 0.20:
            health_score -= 20
        elif missing_pct > 0.05:
            health_score -= 10
    
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
            # Update progress
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db = get_db_manager()
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
            # Update progress
            if DB_AVAILABLE and st.session_state.get('current_project_id'):
                db = get_db_manager()
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


# MODIFY YOUR process_data() FUNCTION
# Replace the end of process_data() with this:

def process_data(df, time_column, value_columns, datetime_format):
    """Process and validate the data"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # ... [All your existing processing code] ...
    
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
            'time_metadata': time_metadata,
            'conversion_info': health_report.get('conversion_info', {}),
            'warnings_list': health_report.get('warnings', []),
            'raw_health_report': health_report,
            'parameters_analyzed': value_columns
        }
        
        # Save to database
        if db.save_health_report(st.session_state.current_project_id, health_data):
            st.success("✅ Health report saved to database")
    
    # Update project progress (7% for step 1 of 13)
    if DB_AVAILABLE and st.session_state.get('current_project_id'):
        db.update_project_progress(
            project_id=st.session_state.current_project_id,
            workflow_state="data_imported",
            current_page=2,
            completion_percentage=7
        )
        
        # Mark step 1 as complete
        db.update_step_completion(
            project_id=st.session_state.current_project_id,
            step_key='data_loaded',
            completed=True
        )
    
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
    
    # ADD NAVIGATION BUTTONS HERE
    show_next_steps_after_upload()
