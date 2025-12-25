"""
Tier 1: Mathematical Models
Linear, Polynomial, Logarithmic, Exponential, Power, and Piecewise Linear
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
from scipy.optimize import curve_fit


def train_tier1_models(train_values: np.ndarray,
                      test_values: np.ndarray,
                      train_timestamps: pd.DatetimeIndex,
                      test_timestamps: pd.DatetimeIndex) -> Dict:
    """
    Train all Tier 1 mathematical models
    
    Parameters:
    -----------
    train_values : np.ndarray
        Training data values
    test_values : np.ndarray
        Testing data values
    train_timestamps : pd.DatetimeIndex
        Training timestamps
    test_timestamps : pd.DatetimeIndex
        Testing timestamps
        
    Returns:
    --------
    results : dict
        Dictionary of trained models and their predictions
    """
    results = {}
    
    # Convert timestamps to numeric (days since start)
    all_timestamps = pd.DatetimeIndex(list(train_timestamps) + list(test_timestamps))
    time_numeric = (all_timestamps - all_timestamps[0]).total_seconds() / 86400  # Days
    
    X_train = time_numeric[:len(train_values)].reshape(-1, 1)
    X_test = time_numeric[len(train_values):].reshape(-1, 1)
    
    # 1. Linear Trend
    results['Linear Trend'] = train_linear_model(X_train, train_values, X_test, test_values)
    
    # 2. Polynomial Degree 2
    results['Polynomial (degree 2)'] = train_polynomial_model(X_train, train_values, X_test, test_values, degree=2)
    
    # 3. Polynomial Degree 3
    results['Polynomial (degree 3)'] = train_polynomial_model(X_train, train_values, X_test, test_values, degree=3)
    
    # 4. Logarithmic (if all values positive)
    if np.all(train_values > 0):
        results['Logarithmic'] = train_logarithmic_model(X_train, train_values, X_test, test_values)
    
    # 5. Exponential (if all values positive)
    if np.all(train_values > 0):
        results['Exponential'] = train_exponential_model(X_train, train_values, X_test, test_values)
    
    # 6. Power (if all values positive)
    if np.all(train_values > 0) and np.all(X_train > 0):
        results['Power'] = train_power_model(X_train, train_values, X_test, test_values)
    
    # 7. Piecewise Linear (optional - 2 segments)
    results['Piecewise Linear'] = train_piecewise_linear_model(X_train, train_values, X_test, test_values, n_segments=2)
    
    return results


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train linear trend model: y = a + b*t"""
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Equation
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {intercept:.4f} + {slope:.4f}*t" if slope >= 0 else f"y = {intercept:.4f} - {abs(slope):.4f}*t"
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'linear'
    }


def train_polynomial_model(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           degree: int = 2) -> Dict:
    """Train polynomial model"""
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    # Equation
    terms = [f"{model.intercept_:.4f}"]
    for i, coef in enumerate(model.coef_):
        power = i + 1
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.4f}*t^{power}")
    equation = "y = " + " ".join(terms)
    
    return {
        'model': model,
        'poly_transformer': poly,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'polynomial',
        'degree': degree
    }


def train_logarithmic_model(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train logarithmic model: y = a + b*log(t)"""
    
    # Shift X to be positive if needed
    X_min = X_train.min()
    if X_min <= 0:
        X_train = X_train - X_min + 1
        X_test = X_test - X_min + 1
    
    X_train_log = np.log(X_train)
    X_test_log = np.log(X_test)
    
    model = LinearRegression()
    model.fit(X_train_log, y_train)
    
    y_pred_train = model.predict(X_train_log)
    y_pred_test = model.predict(X_test_log)
    
    # Equation
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {intercept:.4f} + {slope:.4f}*log(t)" if slope >= 0 else f"y = {intercept:.4f} - {abs(slope):.4f}*log(t)"
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'logarithmic'
    }


def train_exponential_model(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train exponential model: y = a * exp(b*t)"""
    
    # Use log transform: log(y) = log(a) + b*t
    y_train_log = np.log(y_train)
    
    model = LinearRegression()
    model.fit(X_train, y_train_log)
    
    # Predictions (transform back)
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    y_pred_train = np.exp(y_pred_train_log)
    y_pred_test = np.exp(y_pred_test_log)
    
    # Equation
    a = np.exp(model.intercept_)
    b = model.coef_[0]
    equation = f"y = {a:.4f} * exp({b:.4f}*t)" if b >= 0 else f"y = {a:.4f} * exp(-{abs(b):.4f}*t)"
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'exponential'
    }


def train_power_model(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train power model: y = a * t^b"""
    
    # Use log transform: log(y) = log(a) + b*log(t)
    X_train_log = np.log(X_train)
    X_test_log = np.log(X_test)
    y_train_log = np.log(y_train)
    
    model = LinearRegression()
    model.fit(X_train_log, y_train_log)
    
    # Predictions (transform back)
    y_pred_train_log = model.predict(X_train_log)
    y_pred_test_log = model.predict(X_test_log)
    
    y_pred_train = np.exp(y_pred_train_log)
    y_pred_test = np.exp(y_pred_test_log)
    
    # Equation
    a = np.exp(model.intercept_)
    b = model.coef_[0]
    equation = f"y = {a:.4f} * t^{b:.4f}"
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'power'
    }


def train_piecewise_linear_model(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 n_segments: int = 2) -> Dict:
    """Train piecewise linear model (useful for regime changes)"""
    
    # Find breakpoint(s)
    n = len(X_train)
    breakpoints = [int(n * (i+1) / (n_segments+1)) for i in range(n_segments-1)]
    
    # Fit each segment
    models = []
    equations = []
    
    for i in range(n_segments):
        if i == 0:
            start = 0
            end = breakpoints[0] if breakpoints else n
        elif i == n_segments - 1:
            start = breakpoints[-1]
            end = n
        else:
            start = breakpoints[i-1]
            end = breakpoints[i]
        
        X_seg = X_train[start:end]
        y_seg = y_train[start:end]
        
        model_seg = LinearRegression()
        model_seg.fit(X_seg, y_seg)
        models.append(model_seg)
        
        slope = model_seg.coef_[0]
        intercept = model_seg.intercept_
        equations.append(f"Segment {i+1}: y = {intercept:.4f} + {slope:.4f}*t")
    
    # Predictions
    y_pred_train = np.zeros_like(y_train)
    for i in range(n_segments):
        if i == 0:
            start = 0
            end = breakpoints[0] if breakpoints else n
        elif i == n_segments - 1:
            start = breakpoints[-1]
            end = n
        else:
            start = breakpoints[i-1]
            end = breakpoints[i]
        
        y_pred_train[start:end] = models[i].predict(X_train[start:end])
    
    # Test predictions (use last segment)
    y_pred_test = models[-1].predict(X_test)
    
    equation = "; ".join(equations)
    
    return {
        'models': models,
        'breakpoints': breakpoints,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': equation,
        'type': 'piecewise_linear',
        'n_segments': n_segments
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute evaluation metrics"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (only if no zeros)
    if np.all(y_true != 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
