"""
Time-Aware Validation
Train/test splits and cross-validation that respect temporal ordering
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta


def time_train_test_split(X: np.ndarray, 
                          y: np.ndarray,
                          train_size: float = 0.8,
                          timestamps: Optional[pd.Series] = None) -> Tuple:
    """
    Split data into train/test respecting temporal order
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    train_size : float
        Fraction for training (default 0.8)
    timestamps : pd.Series, optional
        Datetime index (for reference)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data
    split_info : dict
        Information about the split
    """
    n = len(y)
    split_idx = int(n * train_size)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    split_info = {
        'train_size': split_idx,
        'test_size': n - split_idx,
        'train_fraction': train_size,
        'split_index': split_idx
    }
    
    if timestamps is not None:
        split_info['train_end'] = timestamps.iloc[split_idx-1]
        split_info['test_start'] = timestamps.iloc[split_idx]
        split_info['test_end'] = timestamps.iloc[-1]
    
    return X_train, X_test, y_train, y_test, split_info


def rolling_origin_cv(X: np.ndarray,
                     y: np.ndarray,
                     min_train_size: int = 50,
                     test_size: int = 10,
                     step_size: int = 5) -> List[Tuple]:
    """
    Rolling origin cross-validation for time series
    
    Also called "walk-forward validation"
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    min_train_size : int
        Minimum training window size
    test_size : int
        Size of test window
    step_size : int
        How many steps to move forward each iteration
        
    Returns:
    --------
    splits : list of tuples
        [(train_indices, test_indices), ...]
    """
    n = len(y)
    splits = []
    
    current_train_end = min_train_size
    
    while current_train_end + test_size <= n:
        # Train indices: from start to current_train_end
        train_indices = np.arange(0, current_train_end)
        
        # Test indices: next test_size points
        test_indices = np.arange(current_train_end, 
                                current_train_end + test_size)
        
        splits.append((train_indices, test_indices))
        
        # Move forward
        current_train_end += step_size
    
    return splits


def expanding_window_cv(X: np.ndarray,
                       y: np.ndarray,
                       min_train_size: int = 50,
                       test_size: int = 10,
                       n_splits: int = 5) -> List[Tuple]:
    """
    Expanding window cross-validation
    
    Training window grows, test window slides
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    min_train_size : int
        Initial training window size
    test_size : int
        Size of test window
    n_splits : int
        Number of splits to create
        
    Returns:
    --------
    splits : list of tuples
        [(train_indices, test_indices), ...]
    """
    n = len(y)
    splits = []
    
    # Calculate step size to create n_splits
    total_testable = n - min_train_size
    step_size = max(1, total_testable // n_splits)
    
    current_train_end = min_train_size
    
    for _ in range(n_splits):
        if current_train_end + test_size > n:
            break
        
        # Train indices: from start to current_train_end (expanding)
        train_indices = np.arange(0, current_train_end)
        
        # Test indices
        test_indices = np.arange(current_train_end,
                                min(current_train_end + test_size, n))
        
        splits.append((train_indices, test_indices))
        
        # Expand window
        current_train_end += step_size
    
    return splits


def validate_temporal_ordering(timestamps: pd.Series) -> Dict:
    """
    Validate that timestamps are properly ordered
    
    Parameters:
    -----------
    timestamps : pd.Series
        Datetime series
        
    Returns:
    --------
    validation : dict
        Validation results
    """
    # Check for NaN
    has_nan = timestamps.isna().any()
    
    # Check ordering
    is_sorted = timestamps.is_monotonic_increasing
    
    # Check for duplicates
    has_duplicates = timestamps.duplicated().any()
    
    # Compute frequency
    if len(timestamps) > 1:
        deltas = timestamps.diff().dropna()
        median_delta = deltas.median()
        
        # Detect frequency
        if median_delta <= pd.Timedelta(days=1):
            frequency = 'daily or higher'
        elif median_delta <= pd.Timedelta(days=7):
            frequency = 'weekly'
        elif median_delta <= pd.Timedelta(days=31):
            frequency = 'monthly'
        elif median_delta <= pd.Timedelta(days=92):
            frequency = 'quarterly'
        else:
            frequency = 'yearly or irregular'
    else:
        frequency = 'unknown'
        median_delta = None
    
    # Check for gaps
    if median_delta is not None:
        expected_delta = median_delta
        gaps = deltas[deltas > expected_delta * 1.5]
        has_gaps = len(gaps) > 0
    else:
        has_gaps = None
    
    return {
        'is_valid': is_sorted and not has_nan and not has_duplicates,
        'is_sorted': is_sorted,
        'has_nan': has_nan,
        'has_duplicates': has_duplicates,
        'frequency': frequency,
        'median_interval': median_delta,
        'has_gaps': has_gaps,
        'n_points': len(timestamps)
    }


def prepare_timeseries_data(series: pd.Series,
                           timestamps: pd.Series,
                           validate: bool = True) -> Dict:
    """
    Prepare and validate time series data for modeling
    
    Parameters:
    -----------
    series : pd.Series
        Time series values
    timestamps : pd.Series
        Datetime index
    validate : bool
        Whether to run validation checks
        
    Returns:
    --------
    prepared_data : dict
        Prepared data with metadata
    """
    # Remove NaN values
    valid_idx = series.notna() & timestamps.notna()
    series_clean = series[valid_idx]
    timestamps_clean = timestamps[valid_idx]
    
    # Sort by time
    sort_idx = timestamps_clean.argsort()
    series_clean = series_clean.iloc[sort_idx]
    timestamps_clean = timestamps_clean.iloc[sort_idx]
    
    result = {
        'series': series_clean,
        'timestamps': timestamps_clean,
        'n_points': len(series_clean),
        'start_date': timestamps_clean.iloc[0],
        'end_date': timestamps_clean.iloc[-1],
        'duration': timestamps_clean.iloc[-1] - timestamps_clean.iloc[0]
    }
    
    # Validation
    if validate:
        validation = validate_temporal_ordering(timestamps_clean)
        result['validation'] = validation
        
        if not validation['is_valid']:
            result['warnings'] = []
            if not validation['is_sorted']:
                result['warnings'].append('Timestamps not in order (sorted automatically)')
            if validation['has_nan']:
                result['warnings'].append('Missing timestamps detected')
            if validation['has_duplicates']:
                result['warnings'].append('Duplicate timestamps detected')
    
    return result


def create_forecast_horizon(last_timestamp: pd.Timestamp,
                           target_date: pd.Timestamp,
                           frequency: str = 'D') -> pd.DatetimeIndex:
    """
    Create future timestamps for forecasting
    
    Parameters:
    -----------
    last_timestamp : pd.Timestamp
        Last observed timestamp
    target_date : pd.Timestamp
        Target forecast date
    frequency : str
        Frequency ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
    --------
    future_dates : pd.DatetimeIndex
        Future timestamps
    """
    return pd.date_range(
        start=last_timestamp + pd.Timedelta(days=1),
        end=target_date,
        freq=frequency
    )


def compute_cv_metrics(cv_results: List[Dict]) -> Dict:
    """
    Aggregate metrics across cross-validation folds
    
    Parameters:
    -----------
    cv_results : list of dict
        Results from each CV fold
        
    Returns:
    --------
    aggregated : dict
        Mean and std of metrics
    """
    metrics = ['mae', 'rmse', 'r2', 'mape']
    
    aggregated = {}
    
    for metric in metrics:
        values = [fold[metric] for fold in cv_results if metric in fold]
        
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
    
    return aggregated
