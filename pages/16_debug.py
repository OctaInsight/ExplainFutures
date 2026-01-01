"""
Debug Script: Check Parameter Data Availability
Shows which parameters have data in timeseries_data table vs parameters table
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database.supabase_manager import get_db_manager

def check_parameter_data_availability(project_id):
    """
    Check which parameters have data in timeseries_data
    """
    print("="*80)
    print("PARAMETER DATA AVAILABILITY CHECK")
    print("="*80)
    print(f"\nProject ID: {project_id}\n")
    
    try:
        db = get_db_manager()
        
        # Get all parameters from parameters table
        print("📊 Loading parameters from 'parameters' table...")
        parameters = db.get_project_parameters(project_id)
        
        if not parameters:
            print("❌ No parameters found in parameters table")
            return
        
        print(f"✅ Found {len(parameters)} parameters in metadata\n")
        
        # Get all data from timeseries_data table
        print("📊 Loading data from 'timeseries_data' table...")
        
        # Load RAW data
        df_raw = None
        for src_label in ('raw', 'original'):
            try:
                df_temp = db.load_timeseries_data(project_id=project_id, data_source=src_label)
                if df_temp is not None and len(df_temp) > 0:
                    df_raw = df_temp
                    print(f"✅ Found {len(df_raw)} raw data points")
                    break
            except Exception as e:
                continue
        
        # Load CLEANED data
        df_cleaned = None
        try:
            df_cleaned = db.load_timeseries_data(project_id=project_id, data_source='cleaned')
            if df_cleaned is not None and len(df_cleaned) > 0:
                print(f"✅ Found {len(df_cleaned)} cleaned data points")
        except:
            print("ℹ️  No cleaned data found")
        
        # Combine all data
        if df_cleaned is not None and len(df_cleaned) > 0:
            df_all = pd.concat([df_raw, df_cleaned], ignore_index=True)
        else:
            df_all = df_raw.copy() if df_raw is not None else pd.DataFrame()
        
        if len(df_all) == 0:
            print("❌ No data found in timeseries_data table")
            return
        
        print(f"✅ Total data points: {len(df_all)}\n")
        
        # Count data points per variable
        print("📊 Counting data points per variable...")
        variable_counts = df_all['variable'].value_counts().to_dict()
        
        # Build comparison table
        print("\n" + "="*80)
        print("PARAMETER DATA AVAILABILITY TABLE")
        print("="*80)
        
        comparison_data = []
        
        for param in parameters:
            param_name = param['parameter_name']
            data_count = variable_counts.get(param_name, 0)
            has_data = "✅ YES" if data_count > 0 else "❌ NO"
            
            # Get metadata
            total_count = param.get('total_count', 0)
            missing_count = param.get('missing_count', 0)
            
            comparison_data.append({
                'Parameter': param_name,
                'Has Data?': has_data,
                'Data Points': data_count,
                'Metadata Total': total_count,
                'Metadata Missing': missing_count,
            })
        
        # Convert to DataFrame for nice display
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by data points (descending)
        df_comparison = df_comparison.sort_values('Data Points', ascending=False)
        
        # Display table
        print(df_comparison.to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        params_with_data = len([p for p in comparison_data if p['Data Points'] > 0])
        params_without_data = len([p for p in comparison_data if p['Data Points'] == 0])
        
        print(f"Total Parameters in Metadata: {len(parameters)}")
        print(f"Parameters WITH data: {params_with_data}")
        print(f"Parameters WITHOUT data: {params_without_data}")
        
        if params_without_data > 0:
            print(f"\n⚠️  FOUND {params_without_data} PARAMETERS WITH NO DATA:")
            for p in comparison_data:
                if p['Data Points'] == 0:
                    print(f"   - {p['Parameter']}")
        
        print("\n" + "="*80)
        
        # Export to CSV
        output_file = f"parameter_data_check_{project_id}.csv"
        df_comparison.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return df_comparison
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PARAMETER DATA AVAILABILITY CHECKER")
    print("="*80)
    
    # Get project ID from user or use default
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    else:
        # Try to get from environment or prompt
        project_id = input("\nEnter Project ID: ").strip()
    
    if not project_id:
        print("❌ No project ID provided")
        print("\nUsage: python check_parameter_data.py <project_id>")
        sys.exit(1)
    
    # Run check
    df_result = check_parameter_data_availability(project_id)
    
    print("\n✅ Check complete!\n")
