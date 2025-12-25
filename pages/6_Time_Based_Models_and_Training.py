"""
Page 6: Time-Based Models & Training
Train multiple models on time series data using 80/20 split
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

# Import modeling modules
from core.models.base_model import ModelArtifact
from core.models.regression_models import create_regression_models
from core.models.timeseries_models import create_timeseries_models
from core.models.ml_models import create_ml_models
from core.models.feature_engineering import (
    prepare_regression_data, 
    build_ml_features,
    detect_seasonality,
    check_stationarity
)
from core.models.validation import time_train_test_split, prepare_timeseries_data
from core.models.evaluation import compute_all_metrics, compute_reliability_score

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Time-Based Models", 
    page_icon="🤖", 
    layout="wide"
)

# Render shared sidebar
render_app_sidebar()

st.title("🤖 Time-Based Models & Training")
st.markdown("*Train multiple models to understand and predict time series behavior*")
st.markdown("---")


def initialize_modeling_state():
    """Initialize session state for modeling"""
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}
    if "model_training_complete" not in st.session_state:
        st.session_state.model_training_complete = False
    if "series_to_model" not in st.session_state:
        st.session_state.series_to_model = []


def get_available_series():
    """
    Get all available time series from previous steps
    
    Returns:
    --------
    available_series : dict
        Dictionary of series by category
    """
    available = {
        'Original Variables': [],
        'Cleaned Variables': [],
        'PCA Components': [],
        'Factor Scores': [],
        'ICA Components': []
    }
    
    # Original variables
    if st.session_state.data_loaded and st.session_state.df_long is not None:
        original_vars = st.session_state.df_long['variable'].unique().tolist()
        available['Original Variables'] = sorted(original_vars)
    
    # Cleaned variables
    if st.session_state.get('preprocessing_applied', False):
        if st.session_state.df_long is not None:
            all_vars = st.session_state.df_long['variable'].unique().tolist()
            original_vars = available['Original Variables']
            cleaned_vars = [v for v in all_vars if v not in original_vars]
            available['Cleaned Variables'] = sorted(cleaned_vars)
    
    # PCA components
    if st.session_state.get('pca_accepted', False) and 'pca' in st.session_state.get('reduction_results', {}):
        pca_results = st.session_state.reduction_results['pca']
        n_components = pca_results['n_components']
        available['PCA Components'] = [f"PC{i+1}" for i in range(n_components)]
    
    # Factor scores
    if 'factor_analysis' in st.session_state.get('reduction_results', {}):
        fa_results = st.session_state.reduction_results['factor_analysis']
        n_factors = fa_results['n_factors']
        available['Factor Scores'] = [f"Factor{i+1}" for i in range(n_factors)]
    
    # ICA components
    if 'ica' in st.session_state.get('reduction_results', {}):
        ica_results = st.session_state.reduction_results['ica']
        n_components = ica_results['n_components']
        available['ICA Components'] = [f"IC{i+1}" for i in range(n_components)]
    
    return available


def get_series_data(series_name: str):
    """
    Get time series data for a given series name
    
    Parameters:
    -----------
    series_name : str
        Name of the series
        
    Returns:
    --------
    series : pd.Series
        Time series values
    timestamps : pd.Series
        Datetime index
    """
    # Determine time column
    time_col = 'timestamp' if 'timestamp' in st.session_state.df_long.columns else 'time'
    
    # Check if it's a component
    if series_name.startswith('PC'):
        # PCA component
        if 'pca_components' in st.session_state:
            idx = int(series_name.replace('PC', '')) - 1
            series = pd.Series(st.session_state.pca_components[:, idx])
            
            # Get timestamps from original data
            if st.session_state.get('preprocessing_applied', False):
                df_long = st.session_state.df_clean
            else:
                df_long = st.session_state.df_long
            
            timestamps = df_long[time_col].unique()[:len(series)]
            timestamps = pd.Series(pd.to_datetime(timestamps))
            
            return series, timestamps
    
    elif series_name.startswith('Factor'):
        # Factor score
        if 'factor_analysis' in st.session_state.reduction_results:
            idx = int(series_name.replace('Factor', '')) - 1
            fa_results = st.session_state.reduction_results['factor_analysis']
            series = pd.Series(fa_results['factors'][:, idx])
            
            # Get timestamps
            if st.session_state.get('preprocessing_applied', False):
                df_long = st.session_state.df_clean
            else:
                df_long = st.session_state.df_long
            
            timestamps = df_long[time_col].unique()[:len(series)]
            timestamps = pd.Series(pd.to_datetime(timestamps))
            
            return series, timestamps
    
    elif series_name.startswith('IC'):
        # ICA component
        if 'ica' in st.session_state.reduction_results:
            idx = int(series_name.replace('IC', '')) - 1
            ica_results = st.session_state.reduction_results['ica']
            series = pd.Series(ica_results['components'][:, idx])
            
            # Get timestamps
            if st.session_state.get('preprocessing_applied', False):
                df_long = st.session_state.df_clean
            else:
                df_long = st.session_state.df_long
            
            timestamps = df_long[time_col].unique()[:len(series)]
            timestamps = pd.Series(pd.to_datetime(timestamps))
            
            return series, timestamps
    
    else:
        # Regular variable
        if st.session_state.get('preprocessing_applied', False):
            df_long = st.session_state.df_clean
        else:
            df_long = st.session_state.df_long
        
        series_data = df_long[df_long['variable'] == series_name].copy()
        series_data = series_data.sort_values(time_col)
        
        series = series_data['value'].reset_index(drop=True)
        timestamps = pd.Series(pd.to_datetime(series_data[time_col].values))
        
        return series, timestamps


def train_all_models_for_series(series_name: str, train_size: float = 0.8):
    """
    Train all models for a single time series
    
    Parameters:
    -----------
    series_name : str
        Name of the series
    train_size : float
        Fraction for training (default 0.8)
        
    Returns:
    --------
    results : dict
        Dictionary of ModelArtifact objects
    """
    # Get data
    series, timestamps = get_series_data(series_name)
    
    # Prepare data
    prepared = prepare_timeseries_data(series, timestamps, validate=True)
    series_clean = prepared['series']
    timestamps_clean = prepared['timestamps']
    
    # Detect characteristics
    seasonality_info = detect_seasonality(series_clean, timestamps_clean)
    stationarity_info = check_stationarity(series_clean)
    
    results = {}
    
    # ===== REGRESSION MODELS =====
    try:
        X_reg, y_reg = prepare_regression_data(series_clean, timestamps_clean)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, split_info = time_train_test_split(
            X_reg, y_reg, train_size=train_size, timestamps=timestamps_clean
        )
        
        regression_models = create_regression_models()
        
        for model in regression_models:
            try:
                # Fit
                model.fit(X_train_reg, y_train_reg)
                
                # Predict
                y_pred_train = model.predict(X_train_reg)
                y_pred_test = model.predict(X_test_reg)
                
                # Metrics
                train_metrics = compute_all_metrics(y_train_reg, y_pred_train)
                test_metrics = compute_all_metrics(y_test_reg, y_pred_test)
                reliability = compute_reliability_score(train_metrics, test_metrics)
                test_metrics['reliability'] = reliability
                
                # Create artifact
                artifact = ModelArtifact(
                    model=model,
                    series_name=series_name,
                    train_data={
                        'X': X_train_reg,
                        'y_true': y_train_reg,
                        'y_pred': y_pred_train,
                        'metrics': train_metrics
                    },
                    test_data={
                        'X': X_test_reg,
                        'y_true': y_test_reg,
                        'y_pred': y_pred_test,
                        'metrics': test_metrics
                    },
                    config={'train_size': train_size, 'split_info': split_info}
                )
                
                results[model.name] = artifact
                
            except Exception as e:
                st.warning(f"Failed to train {model.name}: {str(e)}")
                continue
    
    except Exception as e:
        st.warning(f"Regression models failed: {str(e)}")
    
    # ===== TIME SERIES MODELS =====
    try:
        # Prepare for time series models
        y_ts = series_clean.values
        
        # Split
        split_idx = int(len(y_ts) * train_size)
        y_train_ts = y_ts[:split_idx]
        y_test_ts = y_ts[split_idx:]
        
        # Seasonal period (if detected)
        seasonal_period = seasonality_info.get('primary_period') if seasonality_info['has_seasonality'] else None
        
        ts_models = create_timeseries_models(seasonal_period=seasonal_period)
        
        for model in ts_models:
            try:
                # Fit on training data
                model.fit(None, y_train_ts)
                
                # Predict
                y_pred_train = model.predict(np.arange(len(y_train_ts)))
                y_pred_test = model.predict(np.arange(len(y_test_ts)))
                
                # Metrics
                train_metrics = compute_all_metrics(y_train_ts, y_pred_train)
                test_metrics = compute_all_metrics(y_test_ts, y_pred_test)
                reliability = compute_reliability_score(train_metrics, test_metrics)
                test_metrics['reliability'] = reliability
                
                # Create artifact
                artifact = ModelArtifact(
                    model=model,
                    series_name=series_name,
                    train_data={
                        'X': None,
                        'y_true': y_train_ts,
                        'y_pred': y_pred_train,
                        'metrics': train_metrics
                    },
                    test_data={
                        'X': None,
                        'y_true': y_test_ts,
                        'y_pred': y_pred_test,
                        'metrics': test_metrics
                    },
                    config={'train_size': train_size, 'seasonal_period': seasonal_period}
                )
                
                results[model.name] = artifact
                
            except Exception as e:
                st.warning(f"Failed to train {model.name}: {str(e)}")
                continue
    
    except Exception as e:
        st.warning(f"Time series models failed: {str(e)}")
    
    # ===== ML MODELS =====
    try:
        # Build features
        X_ml, y_ml = build_ml_features(
            series=series_clean,
            timestamps=timestamps_clean,
            lag_list=[1, 2, 3, 7],
            rolling_windows=[7, 14],
            include_time=True,
            include_seasonality=True
        )
        
        # Split
        X_train_ml, X_test_ml, y_train_ml, y_test_ml, split_info_ml = time_train_test_split(
            X_ml.values, y_ml.values, train_size=train_size
        )
        
        ml_models = create_ml_models(include_xgboost=True)
        
        for model in ml_models:
            try:
                # Fit
                model.fit(X_train_ml, y_train_ml)
                
                # Predict
                y_pred_train = model.predict(X_train_ml)
                y_pred_test = model.predict(X_test_ml)
                
                # Metrics
                train_metrics = compute_all_metrics(y_train_ml, y_pred_train)
                test_metrics = compute_all_metrics(y_test_ml, y_pred_test)
                reliability = compute_reliability_score(train_metrics, test_metrics)
                test_metrics['reliability'] = reliability
                
                # Create artifact
                artifact = ModelArtifact(
                    model=model,
                    series_name=series_name,
                    train_data={
                        'X': X_train_ml,
                        'y_true': y_train_ml,
                        'y_pred': y_pred_train,
                        'metrics': train_metrics
                    },
                    test_data={
                        'X': X_test_ml,
                        'y_true': y_test_ml,
                        'y_pred': y_pred_test,
                        'metrics': test_metrics
                    },
                    config={'train_size': train_size, 'feature_names': X_ml.columns.tolist()}
                )
                
                results[model.name] = artifact
                
            except Exception as e:
                st.warning(f"Failed to train {model.name}: {str(e)}")
                continue
    
    except Exception as e:
        st.warning(f"ML models failed: {str(e)}")
    
    return results


def main():
    """Main page function"""
    
    # Initialize state
    initialize_modeling_state()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Diagnostics.py")
        return
    
    # Get available series
    available_series = get_available_series()
    
    # Flatten for selection
    all_series = []
    for category, series_list in available_series.items():
        all_series.extend(series_list)
    
    if len(all_series) == 0:
        st.warning("⚠️ No time series available for modeling!")
        st.info("💡 Go to previous steps to load data and optionally run dimensionality reduction")
        return
    
    # === STEP 1: SELECT SERIES ===
    st.subheader("📋 Step 1: Select Time Series to Model")
    
    with st.expander("ℹ️ What are we doing here?", expanded=False):
        st.markdown("""
        ### Time-Based Modeling
        
        For each selected time series, we will train **13-15 different models**:
        
        **Regression Models (6):**
        - Linear, Polynomial (2,3), Exponential, Logarithmic, Power
        - Best for: Interpretable trends
        
        **Time Series Models (3):**
        - Holt-Winters (ETS), ARIMA, Auto-ARIMA
        - Best for: Seasonality and autocorrelation
        
        **Machine Learning (4-5):**
        - Random Forest, Gradient Boosting, SVR, kNN, XGBoost
        - Best for: Complex patterns
        
        ### Train/Test Split
        - **Training:** First 80% of data
        - **Testing:** Last 20% of data
        - Time-aware (no random shuffling!)
        
        ### Next Steps
        After training, go to **Model Evaluation** to:
        - Compare all models
        - Select the best model for each series
        - Use selected models for forecasting
        """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show available series by category
        for category, series_list in available_series.items():
            if series_list:
                with st.expander(f"{category} ({len(series_list)})", expanded=(category == 'Original Variables')):
                    for s in series_list:
                        st.caption(f"• {s}")
        
        # Multi-select
        selected_series = st.multiselect(
            "Select time series to model",
            all_series,
            default=all_series[:3] if len(all_series) >= 3 else all_series,
            help="Choose which time series to train models on"
        )
    
    with col2:
        st.metric("Total Available", len(all_series))
        st.metric("Selected", len(selected_series))
        
        if len(selected_series) > 0:
            st.metric("Models per Series", "13-15")
            st.metric("Total Models", len(selected_series) * 14)
    
    if len(selected_series) == 0:
        st.warning("⚠️ Please select at least one time series to model")
        return
    
    st.markdown("---")
    
    # === STEP 2: TRAINING CONFIGURATION ===
    st.subheader("⚙️ Step 2: Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_size = st.slider(
            "Training set size",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Fraction of data to use for training"
        )
    
    with col2:
        st.metric("Train %", f"{train_size*100:.0f}%")
        st.metric("Test %", f"{(1-train_size)*100:.0f}%")
    
    with col3:
        if st.button("ℹ️ Why 80/20?", help="Learn about train/test split"):
            st.info("""
            **80/20 Split:**
            - Standard in ML practice
            - Enough data for training
            - Enough data for reliable testing
            - Can adjust if needed
            
            **Time-Aware:**
            - First 80% = training
            - Last 20% = testing
            - No random shuffling!
            - Respects temporal order
            """)
    
    st.markdown("---")
    
    # === STEP 3: TRAIN MODELS ===
    st.subheader("🚀 Step 3: Train Models")
    
    if st.button("🤖 Train All Models", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train for each series
        for idx, series_name in enumerate(selected_series):
            status_text.text(f"Training models for: {series_name}...")
            
            try:
                # Train all models
                results = train_all_models_for_series(series_name, train_size=train_size)
                
                # Store results
                st.session_state.model_results[series_name] = results
                
                # Update progress
                progress = (idx + 1) / len(selected_series)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error training {series_name}: {str(e)}")
                continue
        
        # Complete
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.model_training_complete = True
        st.session_state.series_to_model = selected_series
        
        st.success(f"✅ Successfully trained models for {len(selected_series)} time series!")
        st.balloons()
        
        st.info("👉 Go to **Model Evaluation & Selection** to compare models and select the best ones")
    
    # === SHOW RESULTS IF AVAILABLE ===
    if st.session_state.model_training_complete and st.session_state.model_results:
        st.markdown("---")
        st.subheader("📊 Training Summary")
        
        # Summary table
        summary_data = []
        
        for series_name, models in st.session_state.model_results.items():
            n_models = len(models)
            
            # Get best model by reliability
            best_model = None
            best_reliability = 0
            
            for model_name, artifact in models.items():
                reliability = artifact.test_data['metrics'].get('reliability', 0)
                if reliability > best_reliability:
                    best_reliability = reliability
                    best_model = model_name
            
            # Get average metrics
            avg_r2 = np.mean([a.test_data['metrics'].get('r2', 0) for a in models.values()])
            avg_mae = np.mean([a.test_data['metrics'].get('mae', 0) for a in models.values()])
            
            summary_data.append({
                'Series': series_name,
                'Models Trained': n_models,
                'Best Model': best_model,
                'Best Reliability': f"{best_reliability:.1f}",
                'Avg R²': f"{avg_r2:.3f}",
                'Avg MAE': f"{avg_mae:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Next step button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Go to Model Evaluation →", type="primary", use_container_width=True):
                st.switch_page("pages/7_Model_Evaluation_and_Selection.py")


if __name__ == "__main__":
    main()
