"""
Debug Page: Parameter Data Availability Checker
Shows which parameters have data in timeseries_data vs parameters table
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

st.set_page_config(
    page_title="Debug: Parameter Data Check",
    page_icon="🔍",
    layout="wide"
)

# Authentication check FIRST (before any other code)
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import initialize_session_state
from core.shared_sidebar import render_app_sidebar

initialize_session_state()
render_app_sidebar()

st.title("🔍 Parameter Data Availability Checker")
st.markdown("*Debug tool to check which parameters have actual data in timeseries_data table*")
st.markdown("---")

# Check project
if not st.session_state.get('current_project_id'):
    st.warning("⚠️ Please select a project from the sidebar")
    if st.button("← Go to Home"):
        st.switch_page("pages/01_Home.py")
    st.stop()

project_id = st.session_state.current_project_id

st.info(f"📊 Checking data for Project ID: **{project_id}**")

# Add button to run the check
if st.button("🔍 Run Parameter Data Check", type="primary", use_container_width=True):
    with st.spinner("Loading data..."):
        try:
            from core.database.supabase_manager import get_db_manager
            db = get_db_manager()
            
            # Load parameters from parameters table
            parameters = db.get_project_parameters(project_id)
            
            if not parameters:
                st.error("❌ No parameters found in parameters table")
                st.stop()
            
            st.success(f"✅ Found **{len(parameters)}** parameters in metadata")
            
            # Load timeseries data
            # Load RAW data
            df_raw = None
            for src_label in ('raw', 'original'):
                try:
                    df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                    if df_temp is not None and len(df_temp) > 0:
                        df_raw = df_temp
                        break
                except:
                    continue
            
            # Load CLEANED data
            df_cleaned = None
            try:
                df_cleaned = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            except:
                df_cleaned = None
            
            # Combine
            if df_cleaned is not None and len(df_cleaned) > 0:
                df_all = pd.concat([df_raw, df_cleaned], ignore_index=True)
            else:
                df_all = df_raw.copy() if df_raw is not None else pd.DataFrame()
            
            if len(df_all) == 0:
                st.error("❌ No data found in timeseries_data table")
                st.stop()
            
            # Count data points per variable
            variable_counts = df_all['variable'].value_counts().to_dict()
            
            # Build comparison table
            comparison_data = []
            
            for param in parameters:
                param_name = param['parameter_name']
                data_count = variable_counts.get(param_name, 0)
                has_data = data_count > 0
                
                # Get metadata
                total_count = param.get('total_count', 0)
                missing_count = param.get('missing_count', 0)
                min_val = param.get('min_value')
                max_val = param.get('max_value')
                
                comparison_data.append({
                    'Parameter': param_name,
                    'Has Data': '✅ YES' if has_data else '❌ NO',
                    'Data Points': data_count,
                    'Metadata Total': total_count,
                    'Metadata Missing': missing_count,
                    'Min Value': f"{min_val:.2f}" if min_val is not None else "N/A",
                    'Max Value': f"{max_val:.2f}" if max_val is not None else "N/A",
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Summary metrics
            st.markdown("---")
            st.subheader("📊 Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            params_with_data = len([p for p in comparison_data if '✅' in p['Has Data']])
            params_without_data = len([p for p in comparison_data if '❌' in p['Has Data']])
            
            col1.metric("Total Parameters", len(parameters))
            col2.metric("WITH Data", params_with_data)
            col3.metric("WITHOUT Data", params_without_data)
            col4.metric("Total Data Points", f"{len(df_all):,}")
            
            # Display main table
            st.markdown("---")
            st.subheader("📋 Detailed Comparison")
            
            # Sort by data points (descending)
            df_comparison_sorted = df_comparison.sort_values('Data Points', ascending=False)
            
            # Display dataframe
            st.dataframe(df_comparison_sorted, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df_comparison_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"parameter_data_check_{project_id}.csv",
                mime="text/csv"
            )
            
            # Highlight parameters without data
            st.markdown("---")
            
            if params_without_data > 0:
                st.error(f"⚠️ Found **{params_without_data}** parameters with NO data in timeseries_data table")
                
                st.markdown("#### Parameters WITHOUT Data:")
                
                no_data_params = [p for p in comparison_data if '❌' in p['Has Data']]
                
                for i, param in enumerate(no_data_params, 1):
                    st.warning(f"{i}. **{param['Parameter']}** - Metadata has {param['Metadata Total']} entries but NO data in timeseries_data")
                
                st.markdown("---")
                st.markdown("#### 🔍 Possible Reasons:")
                st.info("""
                **Why do these parameters exist in metadata but not in data?**
                
                1. **Upload Error:** Data wasn't properly uploaded for these parameters
                2. **Cleaning Removed All Data:** All values were filtered out during cleaning
                3. **Metadata Only:** These parameters were defined but never had data
                4. **Data Source Mismatch:** Data might be under a different variable name
                5. **Partial Upload:** Upload was interrupted or incomplete
                
                **Recommended Actions:**
                - ✅ Check Page 2 (Data Import) to verify upload
                - ✅ Review cleaning operations on Page 3
                - ✅ Re-upload data if necessary
                - ✅ Consider removing unused parameters from metadata
                """)
            else:
                st.success("✅ **All parameters have data!** All parameters in the metadata table have corresponding data in timeseries_data.")
            
            # Show data sources breakdown
            st.markdown("---")
            st.subheader("🗂️ Data Sources Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Raw Data")
                if df_raw is not None and len(df_raw) > 0:
                    st.success(f"✅ **{len(df_raw):,}** raw data points")
                    raw_vars = len(df_raw['variable'].unique())
                    st.info(f"📊 **{raw_vars}** unique variables in raw data")
                else:
                    st.warning("⚠️ No raw data found")
            
            with col2:
                st.markdown("#### Cleaned Data")
                if df_cleaned is not None and len(df_cleaned) > 0:
                    st.success(f"✅ **{len(df_cleaned):,}** cleaned data points")
                    cleaned_vars = len(df_cleaned['variable'].unique())
                    st.info(f"📊 **{cleaned_vars}** unique variables in cleaned data")
                else:
                    st.info("ℹ️ No cleaned data found (this is normal if you haven't cleaned data yet)")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("👆 Click the button above to check which parameters have data")
    
    st.markdown("---")
    st.markdown("### ℹ️ What This Tool Does")
    st.markdown("""
    This debug tool helps you identify discrepancies between:
    
    1. **Parameters Table** (metadata) - Shows ALL parameters defined in your project
    2. **Timeseries Data Table** (actual data) - Shows which parameters have actual data points
    
    **Common Issues:**
    - Parameters in metadata but no data → Upload issue or cleaning removed all data
    - Different counts on different pages → Using different data sources
    
    **Use This Tool To:**
    - ✅ Verify data upload completeness
    - ✅ Identify missing data
    - ✅ Debug parameter count discrepancies
    - ✅ Clean up unused parameters
    """)
