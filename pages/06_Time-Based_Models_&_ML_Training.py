"""
Page 6: Time-Based Models & ML Training
Train mathematical, time-series, and ML models for each selected variable/component
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Time-Based Models & ML Training",
    page_icon="🔮",
    layout="wide"  # CRITICAL: Use full page width
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import model training functions
from core.models.data_preparation import get_available_series, get_series_data
from core.models.model_trainer import train_all_models_for_variable
from core.models.model_visualization import create_model_comparison_plot

# Initialize
initialize_session_state()
config = get_config()


# Render shared sidebar
render_app_sidebar()

st.title("🤖 Time-Based Models & ML Training")
st.markdown("*Train mathematical, time-series, and ML models on your selected variables*")
st.markdown("---")


def initialize_page_state():
    """Initialize session state for this page"""
    if "selected_variables_for_modeling" not in st.session_state:
        st.session_state.selected_variables_for_modeling = []
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False


def main():
    """Main page function"""
    
    # Initialize state
    initialize_page_state()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Diagnostics.py")
        return
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ How This Page Works", expanded=False):
        st.markdown("""
        ### Training Process
        
        **Step 1:** Select variables/components to model
        - Original variables
        - Cleaned variables  
        - PCA components
        - Factor scores
        - ICA components
        
        **Step 2:** Train three tiers of models
        
        **Tier 1 - Mathematical Models:**
        - Linear Trend
        - Polynomial (degree 2, 3)
        - Logarithmic
        - Exponential
        - Power
        - Piecewise Linear (optional)
        
        **Tier 2 - Time Series Models:**
        - ETS / Holt-Winters
        - ARIMA
        - SARIMA
        
        **Tier 3 - ML Models (with lag features):**
        - Gradient Boosting
        - Random Forest
        - SVR
        - kNN
        
        **Step 3:** Visualize and compare
        - Each variable gets its own tab
        - Compare up to 3 models at once
        - Export all figures
        
        **Training Split:** 80% training, 20% testing (time-aware)
        """)
    
    st.markdown("---")
    
    # === STEP 1: SELECT VARIABLES ===
    st.subheader("📋 Step 1: Select Variables/Components for Modeling")
    
    # Get available series
    available_series = get_available_series()
    
    # Flatten all available series
    all_series = []
    for category, series_list in available_series.items():
        all_series.extend(series_list)
    
    if len(all_series) == 0:
        st.warning("⚠️ No variables/components available for modeling!")
        st.info("💡 Please complete previous steps to load data and optionally run dimensionality reduction")
        return
    
    # Display available series by category
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Available Variables & Components:**")
        
        for category, series_list in available_series.items():
            if series_list:
                with st.expander(f"{category} ({len(series_list)})", expanded=(category == 'Original Variables')):
                    for series in series_list:
                        st.caption(f"• {series}")
        
        # Multi-select
        selected_series = st.multiselect(
            "Select variables/components to train models on",
            all_series,
            default=all_series[:3] if len(all_series) >= 3 else all_series,
            help="Choose which time series to model"
        )
    
    with col2:
        st.metric("Total Available", len(all_series))
        st.metric("Selected", len(selected_series))
        
        if len(selected_series) > 0:
            st.info(f"💡 **{len(selected_series)} tabs** will be created (one per variable)")
    
    if len(selected_series) == 0:
        st.warning("⚠️ Please select at least one variable/component to model")
        return
    
    # Store selected series
    st.session_state.selected_variables_for_modeling = selected_series
    
    st.markdown("---")
    
    # === STEP 2: TRAINING CONFIGURATION ===
    st.subheader("⚙️ Step 2: Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_split = st.slider(
            "Training set size",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Fraction of data to use for training (time-aware split)"
        )
    
    with col2:
        st.metric("Train %", f"{train_split*100:.0f}%")
        st.metric("Test %", f"{(1-train_split)*100:.0f}%")
    
    with col3:
        st.info("""
        **Time-Aware Split:**
        - First 80% → Training
        - Last 20% → Testing
        - No random shuffle!
        """)
    
    st.markdown("---")
    
    # === STEP 3: TRAIN MODELS ===
    st.subheader("🚀 Step 3: Train All Models")
    
    if st.button("🤖 Train Models for All Selected Variables", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_container = st.container()
        
        # Train for each selected series
        for idx, series_name in enumerate(selected_series):
            status_text.text(f"Training models for: {series_name}...")
            
            try:
                # Get REAL series data
                with st.spinner(f"Loading data for {series_name}..."):
                    series_data = get_series_data(series_name)
                    
                # Debug: Show data info
                st.info(f"✓ Loaded {series_name}: {len(series_data['values'])} data points")
                
                # Train all models using the real trainer
                with st.spinner(f"Training models for {series_name}..."):
                    results = train_all_models_for_variable(
                        variable_name=series_name,
                        series_data=series_data,
                        train_split=train_split,
                        detect_seasonality=True
                    )
                
                # Check if training was successful
                if results.get('success', False):
                    total_models = results.get('total_models', 0)
                    st.success(f"✓ Trained {total_models} models for {series_name}")
                    
                    # Store results
                    st.session_state.trained_models[series_name] = results
                else:
                    error_msg = results.get('error', 'Unknown error')
                    with error_container:
                        st.error(f"❌ Training failed for {series_name}: {error_msg}")
                
                # Update progress
                progress = (idx + 1) / len(selected_series)
                progress_bar.progress(progress)
                
            except Exception as e:
                with error_container:
                    st.error(f"❌ Error training {series_name}: {str(e)}")
                    with st.expander("Show full error details"):
                        import traceback
                        st.code(traceback.format_exc())
                continue
        
        # Complete
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.training_complete = True
        st.success(f"✅ Successfully trained models for {len(selected_series)} variables!")
        st.balloons()
    
    st.markdown("---")
    
    # === STEP 4: DISPLAY RESULTS IN TABS ===
    if st.session_state.training_complete and st.session_state.trained_models:
        st.subheader("📊 Step 4: Review Trained Models")
        
        # Create tabs - one per variable
        tabs = st.tabs([f"📈 {name}" for name in selected_series])
        
        for tab_idx, (tab, series_name) in enumerate(zip(tabs, selected_series)):
            with tab:
                display_variable_tab(series_name)


def display_variable_tab(series_name: str):
    """
    Display training results for a single variable in its tab
    
    Parameters:
    -----------
    series_name : str
        Name of the variable
    """
    
    st.markdown(f"### {series_name}")
    
    # Check if results exist
    if series_name not in st.session_state.trained_models:
        st.warning(f"No training results for {series_name}")
        return
    
    results = st.session_state.trained_models[series_name]
    
    # Get actual series data
    try:
        series_data = get_series_data(series_name)
    except Exception as e:
        st.error(f"Error loading series data: {str(e)}")
        return
    
    # Get split info
    split_data = results.get('train_test_split', {})
    train_values = split_data.get('train_values', np.array([]))
    test_values = split_data.get('test_values', np.array([]))
    train_timestamps = split_data.get('train_timestamps', pd.DatetimeIndex([]))
    test_timestamps = split_data.get('test_timestamps', pd.DatetimeIndex([]))
    split_index = split_data.get('split_index', 0)
    
    # === MODEL SELECTION FOR COMPARISON ===
    st.markdown("#### Select Models to Compare (up to 3)")
    
    # Get all available models from ACTUAL results
    all_models = []
    all_models.extend(list(results.get('tier1_math', {}).keys()))
    all_models.extend(list(results.get('tier2_timeseries', {}).keys()))
    all_models.extend(list(results.get('tier3_ml', {}).keys()))
    
    if not all_models:
        st.warning("No models were trained successfully for this variable")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model1 = st.selectbox(
            "Model 1",
            ["None"] + all_models,
            index=1 if len(all_models) > 0 else 0,
            key=f"model1_{series_name}"
        )
    
    with col2:
        model2 = st.selectbox(
            "Model 2",
            ["None"] + all_models,
            index=2 if len(all_models) > 1 else 0,
            key=f"model2_{series_name}"
        )
    
    with col3:
        model3 = st.selectbox(
            "Model 3",
            ["None"] + all_models,
            index=3 if len(all_models) > 2 else 0,
            key=f"model3_{series_name}"
        )
    
    # Get selected models (exclude "None")
    selected_models = [m for m in [model1, model2, model3] if m != "None"]
    
    st.markdown("---")
    
    # === SCATTER PLOT CUSTOMIZATION (Below dropdowns) ===
    st.markdown("#### 🎨 Customize Plot")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        marker_size = st.slider("Point Size", 1, 15, 6, key=f"size_{series_name}")
        marker_color = st.color_picker("Point Color", "#000000", key=f"color_{series_name}")
    
    with col_b:
        plot_mode = st.selectbox(
            "Plot Style",
            ["Points Only", "Points + Lines"],
            index=0,
            key=f"mode_{series_name}"
        )
        mode = 'markers' if plot_mode == "Points Only" else 'lines+markers'
    
    with col_c:
        y_scale = st.selectbox(
            "Y-axis Scale",
            ["Linear", "Logarithmic"],
            index=0,
            key=f"yscale_{series_name}"
        )
    
    with col_d:
        auto_range = st.checkbox("Auto Y-range", value=True, key=f"auto_{series_name}")
        if not auto_range:
            y_min = st.number_input("Y min", value=0.0, key=f"ymin_{series_name}")
            y_max = st.number_input("Y max", value=100.0, key=f"ymax_{series_name}")
    
    if len(selected_models) == 0:
        st.info("👆 Select at least one model to visualize")
    else:
        st.markdown("---")
        
        # === VISUALIZATION ===
        st.markdown("#### Model Comparison Plot")
        
        # Create comparison plot with REAL data
        fig = create_model_comparison_plot(
            variable_name=series_name,
            train_values=train_values,
            test_values=test_values,
            train_timestamps=train_timestamps,
            test_timestamps=test_timestamps,
            model_results=results,
            selected_models=selected_models,
            split_index=split_index
        )
        
        # Apply user customizations
        fig.data[0].mode = mode
        fig.data[0].marker.size = marker_size
        fig.data[0].marker.color = marker_color
        
        # Apply Y-axis scale
        if y_scale == "Logarithmic":
            fig.update_yaxes(type="log")
        
        # Apply Y-axis range
        if not auto_range:
            fig.update_yaxes(range=[y_min, y_max])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        with st.expander("💾 Export Figure"):
            quick_export_buttons(
                fig,
                f"model_comparison_{series_name}",
                ['png', 'pdf', 'html']
            )
    
    st.markdown("---")
    
    # === MODEL SUMMARY TABLE ===
    st.markdown("#### Trained Models Summary")
    
    summary_data = []
    
    # Tier 1 - Mathematical Models (REAL DATA)
    for model_name, model_data in results.get('tier1_math', {}).items():
        test_metrics = model_data.get('test_metrics', {})
        summary_data.append({
            'Tier': 'Tier 1 (Math)',
            'Model': model_name,
            'Type': 'Mathematical',
            'R²': f"{test_metrics.get('r2', 0):.3f}",
            'MAE': f"{test_metrics.get('mae', 0):.3f}",
            'RMSE': f"{test_metrics.get('rmse', 0):.3f}",
            'Equation': model_data.get('equation', 'N/A')
        })
    
    # Tier 2 - Time Series Models (REAL DATA)
    for model_name, model_data in results.get('tier2_timeseries', {}).items():
        test_metrics = model_data.get('test_metrics', {})
        summary_data.append({
            'Tier': 'Tier 2 (TS)',
            'Model': model_name,
            'Type': 'Time Series',
            'R²': f"{test_metrics.get('r2', 0):.3f}",
            'MAE': f"{test_metrics.get('mae', 0):.3f}",
            'RMSE': f"{test_metrics.get('rmse', 0):.3f}",
            'Equation': model_data.get('equation', 'N/A')
        })
    
    # Tier 3 - ML Models (REAL DATA)
    for model_name, model_data in results.get('tier3_ml', {}).items():
        test_metrics = model_data.get('test_metrics', {})
        summary_data.append({
            'Tier': 'Tier 3 (ML)',
            'Model': model_name,
            'Type': 'Machine Learning',
            'R²': f"{test_metrics.get('r2', 0):.3f}",
            'MAE': f"{test_metrics.get('mae', 0):.3f}",
            'RMSE': f"{test_metrics.get('rmse', 0):.3f}",
            'Equation': model_data.get('equation', 'N/A')
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show total count
        st.caption(f"**Total models trained: {len(summary_data)}**")
    else:
        st.warning("No model results available")


if __name__ == "__main__":
    main()
