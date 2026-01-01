"""
INLINE DEBUG CODE SNIPPET
Copy and paste this into any Streamlit page to add a debug expander
"""

# ADD THIS CODE ANYWHERE IN YOUR PAGE (after loading data)

with st.expander("🔍 DEBUG: Check Parameter Data Availability"):
    st.markdown("**Quick check to see which parameters have data in timeseries_data**")
    
    if st.button("🔍 Run Check", key="debug_param_check"):
        try:
            from core.database.supabase_manager import get_db_manager
            db = get_db_manager()
            project_id = st.session_state.get('current_project_id')
            
            # Get parameters from metadata
            parameters = db.get_project_parameters(project_id)
            
            # Get data from timeseries
            df_raw = None
            for src in ('raw', 'original'):
                try:
                    df_raw = db.load_timeseries_data(project_id=project_id, data_source=src)
                    if df_raw is not None and len(df_raw) > 0:
                        break
                except:
                    pass
            
            df_cleaned = None
            try:
                df_cleaned = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            except:
                pass
            
            # Combine data
            if df_cleaned is not None and len(df_cleaned) > 0:
                df_all = pd.concat([df_raw, df_cleaned], ignore_index=True)
            else:
                df_all = df_raw.copy() if df_raw is not None else pd.DataFrame()
            
            # Count data per variable
            variable_counts = df_all['variable'].value_counts().to_dict()
            
            # Build comparison
            comparison = []
            for param in parameters:
                param_name = param['parameter_name']
                data_count = variable_counts.get(param_name, 0)
                comparison.append({
                    'Parameter': param_name,
                    'Has Data': '✅' if data_count > 0 else '❌',
                    'Data Points': data_count,
                    'Status': 'OK' if data_count > 0 else 'NO DATA'
                })
            
            df_check = pd.DataFrame(comparison)
            
            # Summary
            with_data = len([c for c in comparison if c['Data Points'] > 0])
            without_data = len([c for c in comparison if c['Data Points'] == 0])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Parameters", len(parameters))
            col2.metric("With Data", with_data)
            col3.metric("Without Data", without_data)
            
            # Show table
            st.dataframe(df_check, use_container_width=True, hide_index=True)
            
            # Highlight issues
            if without_data > 0:
                st.error(f"⚠️ Found {without_data} parameters with NO data:")
                no_data_params = [c['Parameter'] for c in comparison if c['Data Points'] == 0]
                for param in no_data_params:
                    st.text(f"  • {param}")
            else:
                st.success("✅ All parameters have data!")
                
        except Exception as e:
            st.error(f"Error: {e}")


# EXAMPLE: How to add to Page 3, 4, or 5
"""
In your main() function, after loading data, add:

def main():
    # ... your existing code ...
    
    # Load data
    load_project_data_from_database()
    
    # Add debug expander HERE
    with st.expander("🔍 DEBUG: Check Parameter Data Availability"):
        # ... paste the debug code above ...
    
    # ... rest of your page code ...
"""
