"""
Data Preparation Module
Extract and prepare time series data from various sources
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional


def get_available_series() -> Dict[str, list]:
    """
    Get all available variables and components from previous steps
    
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
    available = {
        'Original Variables': [],
        'Cleaned Variables': [],
        'PCA Components': [],
        'Factor Scores': [],
        'ICA Components': []
    }
    
    # Get original variables
    if st.session_state.get('data_loaded', False) and st.session_state.df_long is not None:
        original_vars = st.session_state.df_long['variable'].unique().tolist()
        available['Original Variables'] = sorted(original_vars)
    
    # Get cleaned variables (those not in original)
    if st.session_state.get('preprocessing_applied', False):
        if st.session_state.df_long is not None:
            all_vars = st.session_state.df_long['variable'].unique().tolist()
            original_vars = available['Original Variables']
            cleaned_vars = [v for v in all_vars if v not in original_vars]
            available['Cleaned Variables'] = sorted(cleaned_vars)
    
    # Get PCA components
    if st.session_state.get('pca_accepted', False) and 'pca' in st.session_state.get('reduction_results', {}):
        pca_results = st.session_state.reduction_results['pca']
        n_components = pca_results['n_components']
        available['PCA Components'] = [f"PC{i+1}" for i in range(n_components)]
    
    # Get Factor scores
    if 'factor_analysis' in st.session_state.get('reduction_results', {}):
        fa_results = st.session_state.reduction_results['factor_analysis']
        n_factors = fa_results['n_factors']
        available['Factor Scores'] = [f"Factor{i+1}" for i in range(n_factors)]
    
    # Get ICA components
    if 'ica' in st.session_state.get('reduction_results', {}):
        ica_results = st.session_state.reduction_results['ica']
        n_components = ica_results['n_components']
        available['ICA Components'] = [f"IC{i+1}" for i in range(n_components)]
    
    return available


def get_series_data(series_name: str) -> Dict:
    """
    Get time series data for a given series name
    
    Parameters:
    -----------
    series_name : str
        Name of the series (e.g., "GDP", "PC1", "Factor1", "IC1")
        
    Returns:
    --------
    series_data : dict
        Dictionary containing:
        {
            'values': pd.Series or np.ndarray,
            'timestamps': pd.Series or pd.DatetimeIndex,
            'name': str,
            'source': str  # 'original', 'cleaned', 'pca', 'factor', 'ica'
        }
    """
    # Determine time column
    time_col = 'timestamp' if 'timestamp' in st.session_state.df_long.columns else 'time'
    
    # Get appropriate data source
    if st.session_state.get('preprocessing_applied', False) and st.session_state.get('df_clean') is not None:
        df_long = st.session_state.df_clean
    else:
        df_long = st.session_state.df_long
    
    # Check if it's a component
    if series_name.startswith('PC'):
        # PCA component
        return _get_pca_component(series_name, df_long, time_col)
    
    elif series_name.startswith('Factor'):
        # Factor score
        return _get_factor_score(series_name, df_long, time_col)
    
    elif series_name.startswith('IC'):
        # ICA component
        return _get_ica_component(series_name, df_long, time_col)
    
    else:
        # Regular variable (original or cleaned)
        return _get_regular_variable(series_name, df_long, time_col)


def _get_pca_component(series_name: str, df_long: pd.DataFrame, time_col: str) -> Dict:
    """Get PCA component data"""
    
    if 'pca_components' not in st.session_state:
        raise ValueError("PCA components not found in session state")
    
    # Extract component index
    idx = int(series_name.replace('PC', '')) - 1
    
    # Get component values
    values = pd.Series(st.session_state.pca_components[:, idx])
    
    # Get timestamps from original data
    timestamps = df_long[time_col].unique()[:len(values)]
    timestamps = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    
    return {
        'values': values,
        'timestamps': timestamps,
        'name': series_name,
        'source': 'pca'
    }


def _get_factor_score(series_name: str, df_long: pd.DataFrame, time_col: str) -> Dict:
    """Get Factor Analysis score data"""
    
    if 'factor_analysis' not in st.session_state.reduction_results:
        raise ValueError("Factor analysis results not found")
    
    # Extract factor index
    idx = int(series_name.replace('Factor', '')) - 1
    
    # Get factor scores
    fa_results = st.session_state.reduction_results['factor_analysis']
    values = pd.Series(fa_results['factors'][:, idx])
    
    # Get timestamps
    timestamps = df_long[time_col].unique()[:len(values)]
    timestamps = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    
    return {
        'values': values,
        'timestamps': timestamps,
        'name': series_name,
        'source': 'factor'
    }


def _get_ica_component(series_name: str, df_long: pd.DataFrame, time_col: str) -> Dict:
    """Get ICA component data"""
    
    if 'ica' not in st.session_state.reduction_results:
        raise ValueError("ICA results not found")
    
    # Extract component index
    idx = int(series_name.replace('IC', '')) - 1
    
    # Get component values
    ica_results = st.session_state.reduction_results['ica']
    values = pd.Series(ica_results['components'][:, idx])
    
    # Get timestamps
    timestamps = df_long[time_col].unique()[:len(values)]
    timestamps = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    
    return {
        'values': values,
        'timestamps': timestamps,
        'name': series_name,
        'source': 'ica'
    }


def _get_regular_variable(series_name: str, df_long: pd.DataFrame, time_col: str) -> Dict:
    """Get regular variable data (original or cleaned)"""
    
    # Filter for this variable
    series_data = df_long[df_long['variable'] == series_name].copy()
    
    if len(series_data) == 0:
        raise ValueError(f"Variable '{series_name}' not found in data")
    
    # Sort by time
    series_data = series_data.sort_values(time_col).reset_index(drop=True)
    
    # Extract values and timestamps
    values = series_data['value']
    timestamps = pd.to_datetime(series_data[time_col])
    
    # Determine source
    original_vars = st.session_state.df_long['variable'].unique().tolist() if st.session_state.df_long is not None else []
    source = 'original' if series_name in original_vars else 'cleaned'
    
    return {
        'values': values,
        'timestamps': timestamps,
        'name': series_name,
        'source': source
    }


def prepare_series_for_modeling(series_data: Dict) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Prepare series data for modeling (clean, validate, sort)
    
    Parameters:
    -----------
    series_data : dict
        Dictionary from get_series_data()
        
    Returns:
    --------
    values : np.ndarray
        Clean array of values
    timestamps : pd.DatetimeIndex
        Clean datetime index
    """
    values = series_data['values']
    timestamps = series_data['timestamps']
    
    # Convert to arrays
    if isinstance(values, pd.Series):
        values = values.values
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    
    # Remove NaN values
    valid_idx = ~np.isnan(values)
    values = values[valid_idx]
    timestamps = timestamps[valid_idx]
    
    # Sort by time (should already be sorted, but double-check)
    sort_idx = np.argsort(timestamps)
    values = values[sort_idx]
    timestamps = timestamps[sort_idx]
    
    return values, timestamps


def split_train_test(values: np.ndarray, 
                     timestamps: pd.DatetimeIndex, 
                     train_split: float = 0.8) -> Dict:
    """
    Split data into train and test sets (time-aware)
    
    Parameters:
    -----------
    values : np.ndarray
        Time series values
    timestamps : pd.DatetimeIndex
        Timestamps
    train_split : float
        Fraction for training (default 0.8)
        
    Returns:
    --------
    split_data : dict
        {
            'train_values': np.ndarray,
            'test_values': np.ndarray,
            'train_timestamps': pd.DatetimeIndex,
            'test_timestamps': pd.DatetimeIndex,
            'split_index': int,
            'train_size': int,
            'test_size': int
        }
    """
    n = len(values)
    split_idx = int(n * train_split)
    
    return {
        'train_values': values[:split_idx],
        'test_values': values[split_idx:],
        'train_timestamps': timestamps[:split_idx],
        'test_timestamps': timestamps[split_idx:],
        'split_index': split_idx,
        'train_size': split_idx,
        'test_size': n - split_idx
    }
