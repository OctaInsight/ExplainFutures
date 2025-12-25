"""
Feature Engineering for Time Series
Create lag features, rolling statistics, and seasonality indicators
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def create_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """
    Create time-based features from timestamps
    
    Parameters:
    -----------
    timestamps : pd.Series
        Datetime series
        
    Returns:
    --------
    features : pd.DataFrame
        DataFrame with time features
    """
    features = pd.DataFrame(index=timestamps.index)
    
    # Numeric time (days since start)
    features['time_numeric'] = (timestamps - timestamps.min()).dt.total_seconds() / 86400
    
    # Calendar features
    features['year'] = timestamps.dt.year
    features['month'] = timestamps.dt.month
    features['day_of_year'] = timestamps.dt.dayofyear
    features['day_of_week'] = timestamps.dt.dayofweek
    features['quarter'] = timestamps.dt.quarter
    
    # Cyclical encoding (for seasonality)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    return features


def create_lag_features(series: pd.Series, 
                       lags: List[int],
                       include_original: bool = True) -> pd.DataFrame:
    """
    Create lagged features from time series
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    lags : list of int
        Lag periods to create (e.g., [1, 2, 3, 7, 14])
    include_original : bool
        Whether to include original series
        
    Returns:
    --------
    features : pd.DataFrame
        DataFrame with lagged features
    """
    features = pd.DataFrame(index=series.index)
    
    if include_original:
        features['value'] = series
    
    for lag in lags:
        features[f'lag_{lag}'] = series.shift(lag)
    
    return features


def create_rolling_features(series: pd.Series,
                           windows: List[int],
                           stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    windows : list of int
        Window sizes (e.g., [7, 14, 30])
    stats : list of str
        Statistics to compute ('mean', 'std', 'min', 'max', 'median')
        
    Returns:
    --------
    features : pd.DataFrame
        DataFrame with rolling features
    """
    features = pd.DataFrame(index=series.index)
    
    for window in windows:
        for stat in stats:
            if stat == 'mean':
                features[f'rolling_mean_{window}'] = series.rolling(window).mean()
            elif stat == 'std':
                features[f'rolling_std_{window}'] = series.rolling(window).std()
            elif stat == 'min':
                features[f'rolling_min_{window}'] = series.rolling(window).min()
            elif stat == 'max':
                features[f'rolling_max_{window}'] = series.rolling(window).max()
            elif stat == 'median':
                features[f'rolling_median_{window}'] = series.rolling(window).median()
    
    return features


def create_diff_features(series: pd.Series,
                        periods: List[int] = [1]) -> pd.DataFrame:
    """
    Create differencing features (for non-stationarity)
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    periods : list of int
        Differencing periods
        
    Returns:
    --------
    features : pd.DataFrame
        DataFrame with differenced features
    """
    features = pd.DataFrame(index=series.index)
    
    for period in periods:
        features[f'diff_{period}'] = series.diff(period)
        features[f'pct_change_{period}'] = series.pct_change(period)
    
    return features


def build_ml_features(series: pd.Series,
                     timestamps: pd.Series,
                     lag_list: List[int] = [1, 2, 3, 7],
                     rolling_windows: List[int] = [7, 14],
                     include_time: bool = True,
                     include_seasonality: bool = True) -> pd.DataFrame:
    """
    Build complete feature set for ML models
    
    Parameters:
    -----------
    series : pd.Series
        Time series values
    timestamps : pd.Series
        Datetime index
    lag_list : list of int
        Lags to create
    rolling_windows : list of int
        Rolling window sizes
    include_time : bool
        Include time-based features
    include_seasonality : bool
        Include seasonal features
        
    Returns:
    --------
    features : pd.DataFrame
        Complete feature set
    """
    all_features = []
    
    # Lag features
    lag_features = create_lag_features(series, lags=lag_list, include_original=False)
    all_features.append(lag_features)
    
    # Rolling features
    rolling_features = create_rolling_features(series, windows=rolling_windows)
    all_features.append(rolling_features)
    
    # Differencing
    diff_features = create_diff_features(series, periods=[1])
    all_features.append(diff_features)
    
    # Time features (if timestamps available)
    if include_time and timestamps is not None:
        time_features = create_time_features(timestamps)
        
        if include_seasonality:
            # Keep only seasonality features
            seasonal_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'time_numeric']
            time_features = time_features[seasonal_cols]
        else:
            # Keep only time trend
            time_features = time_features[['time_numeric']]
        
        all_features.append(time_features)
    
    # Combine all features
    X = pd.concat(all_features, axis=1)
    
    # Target variable
    y = series.copy()
    
    # Drop rows with NaN (due to lags and rolling windows)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def prepare_regression_data(series: pd.Series,
                           timestamps: pd.Series,
                           use_log_time: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for regression models (trend models)
    
    Parameters:
    -----------
    series : pd.Series
        Time series values
    timestamps : pd.Series
        Datetime index
    use_log_time : bool
        Use log-transformed time
        
    Returns:
    --------
    X : np.ndarray
        Time features (Nx1 array)
    y : np.ndarray
        Target values
    """
    # Numeric time (days since start)
    time_numeric = (timestamps - timestamps.min()).dt.total_seconds() / 86400
    
    if use_log_time:
        # Avoid log(0)
        time_numeric = np.log1p(time_numeric)
    
    X = time_numeric.values.reshape(-1, 1)
    y = series.values
    
    # Remove any NaN
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def detect_seasonality(series: pd.Series, 
                      timestamps: pd.Series,
                      max_period: int = 365) -> dict:
    """
    Detect seasonality in time series using autocorrelation
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    timestamps : pd.Series
        Datetime index
    max_period : int
        Maximum period to check
        
    Returns:
    --------
    info : dict
        Seasonality information
    """
    from scipy import signal
    
    # Remove trend (simple detrending)
    series_detrended = series - series.rolling(window=min(30, len(series)//4), center=True).mean()
    series_detrended = series_detrended.fillna(0)
    
    # Autocorrelation
    autocorr = np.correlate(series_detrended, series_detrended, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find peaks
    peaks, properties = signal.find_peaks(autocorr[:max_period], height=0.3)
    
    if len(peaks) > 0:
        # Primary period
        primary_period = peaks[0]
        strength = properties['peak_heights'][0]
        
        return {
            'has_seasonality': True,
            'primary_period': primary_period,
            'strength': strength,
            'peaks': peaks[:3].tolist()  # Top 3 periods
        }
    else:
        return {
            'has_seasonality': False,
            'primary_period': None,
            'strength': 0,
            'peaks': []
        }


def check_stationarity(series: pd.Series) -> dict:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
        
    Returns:
    --------
    result : dict
        Stationarity test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Remove NaN
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        return {
            'is_stationary': None,
            'p_value': None,
            'note': 'Insufficient data for test'
        }
    
    try:
        result = adfuller(series_clean, autolag='AIC')
        
        is_stationary = result[1] < 0.05  # p-value < 0.05
        
        return {
            'is_stationary': is_stationary,
            'p_value': result[1],
            'adf_statistic': result[0],
            'critical_values': result[4],
            'note': 'Stationary' if is_stationary else 'Non-stationary (may need differencing)'
        }
    except Exception as e:
        return {
            'is_stationary': None,
            'p_value': None,
            'note': f'Test failed: {str(e)}'
        }
