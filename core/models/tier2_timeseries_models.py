"""
Tier 2: Time Series Forecasting Models
ETS/Holt-Winters, ARIMA, and SARIMA
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_tier2_models(train_values: np.ndarray,
                      test_values: np.ndarray,
                      seasonal_period: int = None) -> Dict:
    """
    Train all Tier 2 time series models
    
    Parameters:
    -----------
    train_values : np.ndarray
        Training data values
    test_values : np.ndarray
        Testing data values
    seasonal_period : int, optional
        Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        
    Returns:
    --------
    results : dict
        Dictionary of trained models and their predictions
    """
    results = {}
    
    # 1. ETS / Holt-Winters
    results['ETS/Holt-Winters'] = train_ets_model(train_values, test_values, seasonal_period)
    
    # 2. ARIMA
    results['ARIMA'] = train_arima_model(train_values, test_values)
    
    # 3. SARIMA (if seasonal period specified)
    if seasonal_period and seasonal_period > 1:
        results['SARIMA'] = train_sarima_model(train_values, test_values, seasonal_period)
    
    return results


def train_ets_model(train_values: np.ndarray,
                   test_values: np.ndarray,
                   seasonal_period: int = None) -> Dict:
    """Train ETS/Holt-Winters model"""
    
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Determine seasonal component
        if seasonal_period and seasonal_period > 1 and len(train_values) >= 2 * seasonal_period:
            seasonal = 'add'
        else:
            seasonal = None
            seasonal_period = None
        
        # Fit model
        model = ExponentialSmoothing(
            train_values,
            trend='add',
            seasonal=seasonal,
            seasonal_periods=seasonal_period
        ).fit(optimized=True)
        
        # Predictions
        y_pred_train = model.fittedvalues
        y_pred_test = model.forecast(steps=len(test_values))
        
        # Ensure arrays
        y_pred_train = np.array(y_pred_train)
        y_pred_test = np.array(y_pred_test)
        
        # Handle NaN in fitted values (first few values)
        if np.any(np.isnan(y_pred_train)):
            # Fill NaN with actual values
            nan_idx = np.isnan(y_pred_train)
            y_pred_train[nan_idx] = train_values[nan_idx]
        
        equation = f"ETS(A,A,{'A' if seasonal else 'N'})" + (f" with period={seasonal_period}" if seasonal_period else "")
        
        return {
            'model': model,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_metrics': compute_metrics(train_values, y_pred_train),
            'test_metrics': compute_metrics(test_values, y_pred_test),
            'equation': equation,
            'type': 'ets',
            'seasonal_period': seasonal_period
        }
    
    except Exception as e:
        # Fallback: simple moving average
        return train_simple_ets_fallback(train_values, test_values, str(e))


def train_arima_model(train_values: np.ndarray,
                     test_values: np.ndarray,
                     order: tuple = (1, 1, 1)) -> Dict:
    """Train ARIMA model"""
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Fit model
        model = ARIMA(train_values, order=order).fit()
        
        # Predictions
        y_pred_train = model.fittedvalues
        y_pred_test = model.forecast(steps=len(test_values))
        
        # Ensure arrays
        y_pred_train = np.array(y_pred_train)
        y_pred_test = np.array(y_pred_test)
        
        # Handle NaN in fitted values
        if np.any(np.isnan(y_pred_train)):
            nan_idx = np.isnan(y_pred_train)
            y_pred_train[nan_idx] = train_values[nan_idx]
        
        equation = f"ARIMA{order}"
        
        return {
            'model': model,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_metrics': compute_metrics(train_values, y_pred_train),
            'test_metrics': compute_metrics(test_values, y_pred_test),
            'equation': equation,
            'type': 'arima',
            'order': order
        }
    
    except Exception as e:
        # Fallback: use simpler order
        try:
            return train_arima_model(train_values, test_values, order=(1, 0, 0))
        except:
            return train_simple_ets_fallback(train_values, test_values, str(e))


def train_sarima_model(train_values: np.ndarray,
                      test_values: np.ndarray,
                      seasonal_period: int,
                      order: tuple = (1, 1, 1),
                      seasonal_order: tuple = (1, 1, 1)) -> Dict:
    """Train SARIMA model"""
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Seasonal order format: (P, D, Q, s)
        seasonal_order_full = seasonal_order + (seasonal_period,)
        
        # Fit model
        model = SARIMAX(
            train_values,
            order=order,
            seasonal_order=seasonal_order_full
        ).fit(disp=False)
        
        # Predictions
        y_pred_train = model.fittedvalues
        y_pred_test = model.forecast(steps=len(test_values))
        
        # Ensure arrays
        y_pred_train = np.array(y_pred_train)
        y_pred_test = np.array(y_pred_test)
        
        # Handle NaN
        if np.any(np.isnan(y_pred_train)):
            nan_idx = np.isnan(y_pred_train)
            y_pred_train[nan_idx] = train_values[nan_idx]
        
        equation = f"SARIMA{order}×{seasonal_order}[{seasonal_period}]"
        
        return {
            'model': model,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_metrics': compute_metrics(train_values, y_pred_train),
            'test_metrics': compute_metrics(test_values, y_pred_test),
            'equation': equation,
            'type': 'sarima',
            'order': order,
            'seasonal_order': seasonal_order,
            'seasonal_period': seasonal_period
        }
    
    except Exception as e:
        # Fallback to regular ARIMA
        return train_arima_model(train_values, test_values)


def train_simple_ets_fallback(train_values: np.ndarray,
                              test_values: np.ndarray,
                              error_msg: str) -> Dict:
    """Fallback to simple exponential smoothing"""
    
    # Simple exponential smoothing (alpha = 0.3)
    alpha = 0.3
    y_pred_train = np.zeros_like(train_values)
    y_pred_train[0] = train_values[0]
    
    for i in range(1, len(train_values)):
        y_pred_train[i] = alpha * train_values[i-1] + (1 - alpha) * y_pred_train[i-1]
    
    # Forecast test (use last smoothed value)
    y_pred_test = np.full(len(test_values), y_pred_train[-1])
    
    return {
        'model': None,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(train_values, y_pred_train),
        'test_metrics': compute_metrics(test_values, y_pred_test),
        'equation': f"Simple ETS (fallback, alpha={alpha})",
        'type': 'ets_fallback',
        'error': error_msg
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute evaluation metrics"""
    
    # Handle NaN
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    
    if len(y_true_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'mape': np.nan
        }
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE (only if no zeros)
    if np.all(y_true_clean != 0):
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
