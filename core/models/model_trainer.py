"""
Model Trainer (Orchestrator)
Coordinates training of all three tiers of models
"""

import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime

# Import tier modules
from core.models.data_preparation import prepare_series_for_modeling, split_train_test
from core.models.tier1_math_models import train_tier1_models
from core.models.tier2_timeseries_models import train_tier2_models
from core.models.tier3_ml_models import train_tier3_models


def train_all_models_for_variable(variable_name: str,
                                  series_data: Dict,
                                  train_split: float = 0.8,
                                  detect_seasonality: bool = True) -> Dict:
    """
    Train all three tiers of models for a single variable
    
    This is the main orchestrator function that coordinates training across all tiers.
    
    Parameters:
    -----------
    variable_name : str
        Name of the variable/component
    series_data : dict
        Dictionary containing 'values', 'timestamps', 'name', 'source'
        (from get_series_data())
    train_split : float
        Fraction of data to use for training (default 0.8)
    detect_seasonality : bool
        Whether to detect and use seasonal period (default True)
        
    Returns:
    --------
    results : dict
        Complete training results:
        {
            'variable_name': str,
            'source': str,
            'tier1_math': dict,
            'tier2_timeseries': dict,
            'tier3_ml': dict,
            'train_test_split': dict,
            'training_timestamp': datetime,
            'success': bool,
            'error': str (if failed)
        }
    """
    
    try:
        # Prepare data
        values, timestamps = prepare_series_for_modeling(series_data)
        
        # Split into train/test (time-aware)
        split_data = split_train_test(values, timestamps, train_split)
        
        train_values = split_data['train_values']
        test_values = split_data['test_values']
        train_timestamps = split_data['train_timestamps']
        test_timestamps = split_data['test_timestamps']
        
        # Detect seasonal period (if requested)
        seasonal_period = None
        if detect_seasonality:
            seasonal_period = detect_seasonal_period(values, timestamps)
        
        # === TRAIN TIER 1: MATHEMATICAL MODELS ===
        tier1_results = train_tier1_models(
            train_values=train_values,
            test_values=test_values,
            train_timestamps=train_timestamps,
            test_timestamps=test_timestamps
        )
        
        # === TRAIN TIER 2: TIME SERIES MODELS ===
        tier2_results = train_tier2_models(
            train_values=train_values,
            test_values=test_values,
            seasonal_period=seasonal_period
        )
        
        # === TRAIN TIER 3: ML MODELS WITH LAG FEATURES ===
        tier3_results = train_tier3_models(
            train_values=train_values,
            test_values=test_values,
            train_timestamps=train_timestamps,
            test_timestamps=test_timestamps,
            lag_features=[1, 2, 3, 7],
            rolling_windows=[7, 14]
        )
        
        # Compile results
        results = {
            'variable_name': variable_name,
            'source': series_data.get('source', 'unknown'),
            'tier1_math': tier1_results,
            'tier2_timeseries': tier2_results,
            'tier3_ml': tier3_results,
            'train_test_split': split_data,
            'seasonal_period': seasonal_period,
            'training_timestamp': datetime.now(),
            'success': True,
            'n_tier1_models': len(tier1_results),
            'n_tier2_models': len(tier2_results),
            'n_tier3_models': len(tier3_results),
            'total_models': len(tier1_results) + len(tier2_results) + len(tier3_results)
        }
        
        return results
    
    except Exception as e:
        # Return error result
        return {
            'variable_name': variable_name,
            'source': series_data.get('source', 'unknown'),
            'tier1_math': {},
            'tier2_timeseries': {},
            'tier3_ml': {},
            'train_test_split': {},
            'training_timestamp': datetime.now(),
            'success': False,
            'error': str(e),
            'total_models': 0
        }


def detect_seasonal_period(values: np.ndarray, 
                          timestamps: pd.DatetimeIndex) -> int:
    """
    Detect seasonal period from data
    
    Parameters:
    -----------
    values : np.ndarray
        Time series values
    timestamps : pd.DatetimeIndex
        Timestamps
        
    Returns:
    --------
    period : int or None
        Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        None if no clear seasonality detected
    """
    
    try:
        # Calculate median time delta
        if len(timestamps) < 2:
            return None
        
        deltas = timestamps[1:] - timestamps[:-1]
        median_delta = deltas.median()
        
        # Determine frequency
        if median_delta <= pd.Timedelta(days=1):
            # Daily data - try weekly (7) or monthly (30) seasonality
            if len(values) >= 60:
                return 30  # Monthly
            elif len(values) >= 14:
                return 7   # Weekly
        
        elif median_delta <= pd.Timedelta(days=7):
            # Weekly data - try yearly (52) seasonality
            if len(values) >= 104:
                return 52
        
        elif median_delta <= pd.Timedelta(days=31):
            # Monthly data - try yearly (12) seasonality
            if len(values) >= 24:
                return 12
        
        elif median_delta <= pd.Timedelta(days=92):
            # Quarterly data - try yearly (4) seasonality
            if len(values) >= 8:
                return 4
        
        return None
    
    except:
        return None


def get_best_model(results: Dict, metric: str = 'r2') -> str:
    """
    Get the best performing model based on a metric
    
    Parameters:
    -----------
    results : dict
        Training results from train_all_models_for_variable()
    metric : str
        Metric to use for selection ('r2', 'mae', 'rmse', 'mape')
        
    Returns:
    --------
    best_model_name : str
        Name of the best model
    """
    
    best_score = -np.inf if metric == 'r2' else np.inf
    best_model = None
    
    # Check all tiers
    for tier_name in ['tier1_math', 'tier2_timeseries', 'tier3_ml']:
        tier_results = results.get(tier_name, {})
        
        for model_name, model_data in tier_results.items():
            test_metrics = model_data.get('test_metrics', {})
            score = test_metrics.get(metric, np.nan)
            
            if np.isnan(score):
                continue
            
            # Update best
            if metric == 'r2':
                if score > best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score < best_score:
                    best_score = score
                    best_model = model_name
    
    return best_model if best_model else "Linear Trend"


def summarize_training_results(results: Dict) -> Dict:
    """
    Create a summary of training results
    
    Parameters:
    -----------
    results : dict
        Training results from train_all_models_for_variable()
        
    Returns:
    --------
    summary : dict
        Summary statistics and best models
    """
    
    summary = {
        'variable_name': results.get('variable_name', 'Unknown'),
        'total_models_trained': results.get('total_models', 0),
        'tier1_count': results.get('n_tier1_models', 0),
        'tier2_count': results.get('n_tier2_models', 0),
        'tier3_count': results.get('n_tier3_models', 0),
        'best_by_r2': get_best_model(results, 'r2'),
        'best_by_mae': get_best_model(results, 'mae'),
        'training_successful': results.get('success', False),
        'training_timestamp': results.get('training_timestamp')
    }
    
    return summary
