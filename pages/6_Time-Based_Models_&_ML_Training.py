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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import model training functions (TO BE IMPLEMENTED IN core/models/)
# from core.models.data_preparation import get_available_series, get_series_data, prepare_series_for_modeling
# from core.models.tier1_math_models import train_tier1_models
# from core.models.tier2_timeseries_models import train_tier2_models
# from core.models.tier3_ml_models import train_tier3_models
# from core.models.model_visualization import create_model_comparison_plot

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Time-Based Models & ML Training", 
    page_icon="🤖", 
    layout="wide"
)

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


def get_available_series():
    """
    Get all available variables and components from previous steps
    TO BE IMPLEMENTED IN: core/models/data_preparation.py
    
    Returns:
    --------
    available_series : dict
        Dictionary organized by category:
        {
            'Original Variables': [...],
            'Cleaned Variables': [...],
            'PCA Components': [...],
            'Factor Scores': [...],
            'ICA Components': [...]
        }
    """
    # PLACEHOLDER - This will call the actual function from core/models/data_preparation.py
    available = {
        'Original Variables': [],
        'Cleaned Variables': [],
        'PCA Components': [],
        'Factor Scores': [],
        'ICA Components': []
    }
    
    # Get original variables
    if st.session_state.data_loaded and st.session_state.df_long is not None:
        original_vars = st.session_state.df_long['variable'].unique().tolist()
        available['Original Variables'] = sorted(original_vars)
    
    # Get cleaned variables
    if st.session_state.get('preprocessing_applied', False):
        if st.session_state.df_long is not None:
            all_vars = st.session_state.df_long['variable'].unique().tolist()
            cleaned_vars = [v for v in all_vars if v not in available['Original Variables']]
            available['Cleaned Variables'] = sorted(cleaned_vars)
    
    # Get PCA components
    if st.session_state.get('pca_accepted', False) and 'pca' in st.session_state.get('reduction_results', {}):
        n_components = st.session_state.reduction_results['pca']['n_components']
        available['PCA Components'] = [f"PC{i+1}" for i in range(n_components)]
    
    # Get Factor scores
    if 'factor_analysis' in st.session_state.get('reduction_results', {}):
        n_factors = st.session_state.reduction_results['factor_analysis']['n_factors']
        available['Factor Scores'] = [f"Factor{i+1}" for i in range(n_factors)]
    
    # Get ICA components
    if 'ica' in st.session_state.get('reduction_results', {}):
        n_components = st.session_state.reduction_results['ica']['n_components']
        available['ICA Components'] = [f"IC{i+1}" for i in range(n_components)]
    
    return available


def train_all_models_for_variable(variable_name: str, series_data: dict, train_split: float = 0.8):
    """
    Train all three tiers of models for a single variable
    TO BE IMPLEMENTED IN: core/models/model_trainer.py
    
    Parameters:
    -----------
    variable_name : str
        Name of the variable
    series_data : dict
        Dictionary containing 'values' and 'timestamps'
    train_split : float
        Fraction of data to use for training (default 0.8)
    
    Returns:
    --------
    results : dict
        Dictionary with trained models from all tiers:
        {
            'tier1_math': {...},
            'tier2_timeseries': {...},
            'tier3_ml': {...},
            'train_test_split': {...}
        }
    """
    # PLACEHOLDER - This will be implemented in core/models/model_trainer.py
    st.info(f"🔄 Training all models for: {variable_name}")
    
    results = {
        'tier1_math': {},
        'tier2_timeseries': {},
        'tier3_ml': {},
        'train_test_split': {}
    }
    
    return results


def create_model_comparison_plot(variable_name: str, 
                                 series_data: dict,
                                 model_results: dict,
                                 selected_models: list):
    """
    Create interactive plot comparing selected models
    TO BE IMPLEMENTED IN: core/models/model_visualization.py
    
    Parameters:
    -----------
    variable_name : str
        Name of variable
    series_data : dict
        Original series data
    model_results : dict
        Trained model results
    selected_models : list
        List of up to 3 model names to display
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive comparison plot
    """
    # PLACEHOLDER - This will be implemented in core/models/model_visualization.py
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=list(range(len(series_data.get('values', [])))),
        y=series_data.get('values', []),
        mode='lines+markers',
        name='Actual Data',
        line=dict(color='black', width=2)
    ))
    
    # Placeholder for model lines
    for model_name in selected_models[:3]:  # Max 3 models
        fig.add_trace(go.Scatter(
            x=list(range(len(series_data.get('values', [])))),
            y=series_data.get('values', []),  # PLACEHOLDER - will be model predictions
            mode='lines',
            name=model_name,
            line=dict(width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{variable_name} - Model Comparison",
        xaxis_title="Time",
        yaxis_title="Value",
        height=500,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


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
        
        # Train for each selected series
        for idx, series_name in enumerate(selected_series):
            status_text.text(f"Training models for: {series_name}...")
            
            try:
                # Get series data (PLACEHOLDER - will be implemented)
                series_data = {
                    'values': np.random.randn(100),  # PLACEHOLDER
                    'timestamps': pd.date_range(start='2020-01-01', periods=100, freq='D')  # PLACEHOLDER
                }
                
                # Train all models
                results = train_all_models_for_variable(series_name, series_data, train_split)
                
                # Store results
                st.session_state.trained_models[series_name] = results
                
                # Update progress
                progress = (idx + 1) / len(selected_series)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error training {series_name}: {str(e)}")
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
    
    # Get series data (PLACEHOLDER)
    series_data = {
        'values': np.random.randn(100),  # PLACEHOLDER
        'timestamps': pd.date_range(start='2020-01-01', periods=100, freq='D')  # PLACEHOLDER
    }
    
    # === MODEL SELECTION FOR COMPARISON ===
    st.markdown("#### Select Models to Compare (up to 3)")
    
    # Get all available models
    all_models = []
    all_models.extend(list(results.get('tier1_math', {}).keys()))
    all_models.extend(list(results.get('tier2_timeseries', {}).keys()))
    all_models.extend(list(results.get('tier3_ml', {}).keys()))
    
    # PLACEHOLDER - actual model names
    if not all_models:
        all_models = [
            "Linear Trend",
            "Polynomial (degree 2)",
            "Polynomial (degree 3)",
            "ETS/Holt-Winters",
            "ARIMA",
            "Gradient Boosting",
            "Random Forest"
        ]
    
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
    
    if len(selected_models) == 0:
        st.info("👆 Select at least one model to visualize")
        return
    
    st.markdown("---")
    
    # === VISUALIZATION ===
    st.markdown("#### Model Comparison Plot")
    
    # Create comparison plot
    fig = create_model_comparison_plot(
        series_name,
        series_data,
        results,
        selected_models
    )
    
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
    
    # Tier 1 - Mathematical Models
    for model_name in results.get('tier1_math', {}).keys():
        summary_data.append({
            'Tier': 'Tier 1 (Math)',
            'Model': model_name,
            'Type': 'Mathematical',
            'Status': '✅ Trained'
        })
    
    # Tier 2 - Time Series Models
    for model_name in results.get('tier2_timeseries', {}).keys():
        summary_data.append({
            'Tier': 'Tier 2 (TS)',
            'Model': model_name,
            'Type': 'Time Series',
            'Status': '✅ Trained'
        })
    
    # Tier 3 - ML Models
    for model_name in results.get('tier3_ml', {}).keys():
        summary_data.append({
            'Tier': 'Tier 3 (ML)',
            'Model': model_name,
            'Type': 'Machine Learning',
            'Status': '✅ Trained'
        })
    
    # PLACEHOLDER data if no models
    if not summary_data:
        summary_data = [
            {'Tier': 'Tier 1 (Math)', 'Model': 'Linear Trend', 'Type': 'Mathematical', 'Status': '✅ Trained'},
            {'Tier': 'Tier 1 (Math)', 'Model': 'Polynomial (2)', 'Type': 'Mathematical', 'Status': '✅ Trained'},
            {'Tier': 'Tier 2 (TS)', 'Model': 'ETS/Holt-Winters', 'Type': 'Time Series', 'Status': '✅ Trained'},
            {'Tier': 'Tier 3 (ML)', 'Model': 'Gradient Boosting', 'Type': 'Machine Learning', 'Status': '✅ Trained'},
        ]
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
