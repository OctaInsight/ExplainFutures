"""
Time Series Forecasting Models
ETS (Exponential Smoothing), Holt-Winters, and ARIMA
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.base_model import BaseTimeSeriesModel


class HoltWintersModel(BaseTimeSeriesModel):
    """
    Holt-Winters Exponential Smoothing (ETS)
    Handles trend and seasonality
    """
    
    def __init__(self, seasonal_periods: Optional[int] = None, trend: str = 'add'):
        """
        Initialize Holt-Winters model
        
        Parameters:
        -----------
        seasonal_periods : int, optional
            Number of periods in season (e.g., 12 for monthly data with yearly seasonality)
        trend : str
            'add' for additive, 'mul' for multiplicative
        """
        super().__init__(name="Holt-Winters (ETS)", model_type="timeseries")
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.level = None
        self.trend_component = None
        self.seasonal = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Holt-Winters model
        
        For time-series models, X can be ignored (we work with y directly)
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Determine if seasonal
        if self.seasonal_periods is not None and self.seasonal_periods > 1:
            seasonal = 'add'
        else:
            seasonal = None
        
        # Fit model
        try:
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=seasonal,
                seasonal_periods=self.seasonal_periods
            ).fit(optimized=True)
            
            self.is_fitted = True
            
        except Exception as e:
            # Fallback: simpler model without seasonality
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=None
            ).fit(optimized=True)
            
            self.is_fitted = True
            print(f"Warning: Seasonal fit failed, using trend-only model. Error: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        For Holt-Winters, X represents number of steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # X is the number of steps to forecast
        n_steps = len(X) if hasattr(X, '__len__') else int(X)
        
        # Forecast
        forecast = self.model.forecast(steps=n_steps)
        
        return np.array(forecast)
    
    def get_equation(self) -> str:
        """Get model description"""
        if not self.is_fitted:
            return "Model not fitted"
        
        if self.seasonal_periods:
            return f"Holt-Winters (trend={self.trend}, seasonal_period={self.seasonal_periods})"
        else:
            return f"Holt's Linear Trend (trend={self.trend})"


class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA - AutoRegressive Integrated Moving Average
    Classical time series forecasting model
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize ARIMA model
        
        Parameters:
        -----------
        order : tuple (p, d, q)
            p: AR order (autoregressive)
            d: Differencing order (integration)
            q: MA order (moving average)
        seasonal_order : tuple (P, D, Q, s), optional
            Seasonal ARIMA parameters
        """
        model_name = f"ARIMA({order[0]},{order[1]},{order[2]})"
        if seasonal_order:
            model_name = f"SARIMA{order}×{seasonal_order}"
        
        super().__init__(name=model_name, model_type="timeseries")
        self.order = order
        self.seasonal_order = seasonal_order
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ARIMA model"""
        from statsmodels.tsa.arima.model import ARIMA
        
        try:
            self.model = ARIMA(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit()
            
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to simpler ARIMA(1,1,1)
            print(f"Warning: ARIMA{self.order} failed, trying ARIMA(1,1,1). Error: {str(e)}")
            self.order = (1, 1, 1)
            self.seasonal_order = None
            
            self.model = ARIMA(
                y,
                order=self.order
            ).fit()
            
            self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # X represents number of steps ahead
        n_steps = len(X) if hasattr(X, '__len__') else int(X)
        
        # Forecast
        forecast = self.model.forecast(steps=n_steps)
        
        return np.array(forecast)
    
    def get_equation(self) -> str:
        """Get model description"""
        if not self.is_fitted:
            return "Model not fitted"
        
        p, d, q = self.order
        
        if self.seasonal_order:
            P, D, Q, s = self.seasonal_order
            return f"SARIMA({p},{d},{q})×({P},{D},{Q},{s})"
        else:
            return f"ARIMA({p},{d},{q})"


class AutoARIMAModel(BaseTimeSeriesModel):
    """
    Auto ARIMA - Automatically finds best ARIMA parameters
    Uses information criteria (AIC/BIC) to select model
    """
    
    def __init__(self, seasonal: bool = False, m: int = 12):
        """
        Initialize Auto ARIMA
        
        Parameters:
        -----------
        seasonal : bool
            Whether to use seasonal ARIMA
        m : int
            Seasonal period (e.g., 12 for monthly data)
        """
        super().__init__(name="Auto ARIMA", model_type="timeseries")
        self.seasonal = seasonal
        self.m = m
        self.best_order = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Auto ARIMA model"""
        try:
            from pmdarima import auto_arima
            
            self.model = auto_arima(
                y,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=self.seasonal,
                m=self.m if self.seasonal else 1,
                d=None,  # Let auto_arima determine differencing
                D=None,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_fits=50
            )
            
            self.best_order = self.model.order
            self.is_fitted = True
            
        except ImportError:
            # pmdarima not available, fall back to manual ARIMA
            print("Warning: pmdarima not available, using ARIMA(1,1,1)")
            from statsmodels.tsa.arima.model import ARIMA
            
            self.model = ARIMA(y, order=(1, 1, 1)).fit()
            self.best_order = (1, 1, 1)
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to simple ARIMA
            print(f"Warning: Auto ARIMA failed, using ARIMA(1,1,1). Error: {str(e)}")
            from statsmodels.tsa.arima.model import ARIMA
            
            self.model = ARIMA(y, order=(1, 1, 1)).fit()
            self.best_order = (1, 1, 1)
            self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_steps = len(X) if hasattr(X, '__len__') else int(X)
        
        forecast = self.model.predict(n_periods=n_steps)
        
        return np.array(forecast)
    
    def get_equation(self) -> str:
        """Get model description"""
        if not self.is_fitted:
            return "Model not fitted"
        
        if self.best_order:
            p, d, q = self.best_order
            return f"Auto-selected: ARIMA({p},{d},{q})"
        else:
            return "Auto ARIMA"


def create_timeseries_models(seasonal_period: Optional[int] = None) -> list:
    """
    Create time series model instances
    
    Parameters:
    -----------
    seasonal_period : int, optional
        Seasonal period (e.g., 12 for monthly data)
        
    Returns:
    --------
    models : list
        List of time series model instances
    """
    models = [
        HoltWintersModel(seasonal_periods=None, trend='add'),  # Simple Holt
        ARIMAModel(order=(1, 1, 1)),  # ARIMA(1,1,1)
        AutoARIMAModel(seasonal=False)  # Auto ARIMA
    ]
    
    # Add seasonal models if period specified
    if seasonal_period and seasonal_period > 1:
        models.extend([
            HoltWintersModel(seasonal_periods=seasonal_period, trend='add'),
            AutoARIMAModel(seasonal=True, m=seasonal_period)
        ])
    
    return models
