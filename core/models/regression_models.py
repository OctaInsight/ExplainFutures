"""
Regression Models for Time Series
Linear, polynomial, logarithmic, exponential, and power models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.base_model import BaseTimeSeriesModel


class LinearTrendModel(BaseTimeSeriesModel):
    """
    Simple linear trend: y = a + b*t
    """
    
    def __init__(self):
        super().__init__(name="Linear Trend", model_type="regression")
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model"""
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_equation(self) -> str:
        """Get equation string"""
        if not self.is_fitted:
            return "Model not fitted"
        
        slope = self.coefficients[0]
        intercept = self.intercept
        
        if slope >= 0:
            return f"y = {intercept:.4f} + {slope:.4f}*t"
        else:
            return f"y = {intercept:.4f} - {abs(slope):.4f}*t"


class PolynomialTrendModel(BaseTimeSeriesModel):
    """
    Polynomial trend: y = a + b*t + c*t^2 + ...
    """
    
    def __init__(self, degree: int = 2):
        super().__init__(
            name=f"Polynomial Trend (degree {degree})", 
            model_type="regression"
        )
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit polynomial model"""
        # Transform features
        X_poly = self.poly_features.fit_transform(X)
        
        # Fit linear model on polynomial features
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)
    
    def get_equation(self) -> str:
        """Get equation string"""
        if not self.is_fitted:
            return "Model not fitted"
        
        terms = [f"{self.intercept:.4f}"]
        
        for i, coef in enumerate(self.coefficients):
            power = i + 1
            if coef >= 0:
                terms.append(f"+ {coef:.4f}*t^{power}")
            else:
                terms.append(f"- {abs(coef):.4f}*t^{power}")
        
        return "y = " + " ".join(terms)


class ExponentialTrendModel(BaseTimeSeriesModel):
    """
    Exponential trend: y = a * exp(b*t)
    Fitted as: log(y) = log(a) + b*t
    """
    
    def __init__(self):
        super().__init__(name="Exponential Trend", model_type="regression")
        self.coefficients = None
        self.intercept = None
        self.a = None  # Multiplicative constant
        self.b = None  # Growth rate
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit exponential model"""
        # Check for non-positive values
        if np.any(y <= 0):
            raise ValueError("Exponential model requires all positive values. Use log transform on y.")
        
        # Transform: log(y) = log(a) + b*t
        y_log = np.log(y)
        
        self.model = LinearRegression()
        self.model.fit(X, y_log)
        
        self.b = self.model.coef_[0]  # Growth rate
        self.a = np.exp(self.model.intercept_)  # Multiplicative constant
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Predict log(y)
        y_log_pred = self.model.predict(X)
        
        # Transform back: y = exp(log(y))
        return np.exp(y_log_pred)
    
    def get_equation(self) -> str:
        """Get equation string"""
        if not self.is_fitted:
            return "Model not fitted"
        
        if self.b >= 0:
            return f"y = {self.a:.4f} * exp({self.b:.4f}*t)"
        else:
            return f"y = {self.a:.4f} * exp(-{abs(self.b):.4f}*t)"


class LogarithmicTrendModel(BaseTimeSeriesModel):
    """
    Logarithmic trend: y = a + b*log(t)
    """
    
    def __init__(self):
        super().__init__(name="Logarithmic Trend", model_type="regression")
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit logarithmic model"""
        # Check for non-positive time values
        if np.any(X <= 0):
            # Shift time to be positive
            X = X - X.min() + 1
        
        # Transform: log(t)
        X_log = np.log(X)
        
        self.model = LinearRegression()
        self.model.fit(X_log, y)
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Shift time (same as in fit)
        if np.any(X <= 0):
            X = X - X.min() + 1
        
        X_log = np.log(X)
        return self.model.predict(X_log)
    
    def get_equation(self) -> str:
        """Get equation string"""
        if not self.is_fitted:
            return "Model not fitted"
        
        slope = self.coefficients[0]
        intercept = self.intercept
        
        if slope >= 0:
            return f"y = {intercept:.4f} + {slope:.4f}*log(t)"
        else:
            return f"y = {intercept:.4f} - {abs(slope):.4f}*log(t)"


class PowerTrendModel(BaseTimeSeriesModel):
    """
    Power trend: y = a * t^b
    Fitted as: log(y) = log(a) + b*log(t)
    """
    
    def __init__(self):
        super().__init__(name="Power Trend", model_type="regression")
        self.coefficients = None
        self.intercept = None
        self.a = None  # Multiplicative constant
        self.b = None  # Power exponent
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit power model"""
        # Check for non-positive values
        if np.any(y <= 0):
            raise ValueError("Power model requires all positive y values")
        
        if np.any(X <= 0):
            # Shift time to be positive
            X = X - X.min() + 1
        
        # Transform: log(y) = log(a) + b*log(t)
        X_log = np.log(X)
        y_log = np.log(y)
        
        self.model = LinearRegression()
        self.model.fit(X_log, y_log)
        
        self.b = self.model.coef_[0]  # Power exponent
        self.a = np.exp(self.model.intercept_)  # Multiplicative constant
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Shift time (same as in fit)
        if np.any(X <= 0):
            X = X - X.min() + 1
        
        # Predict log(y)
        X_log = np.log(X)
        y_log_pred = self.model.predict(X_log)
        
        # Transform back
        return np.exp(y_log_pred)
    
    def get_equation(self) -> str:
        """Get equation string"""
        if not self.is_fitted:
            return "Model not fitted"
        
        return f"y = {self.a:.4f} * t^{self.b:.4f}"


def create_regression_models() -> list:
    """
    Create all regression model instances
    
    Returns:
    --------
    models : list
        List of regression model instances
    """
    return [
        LinearTrendModel(),
        PolynomialTrendModel(degree=2),
        PolynomialTrendModel(degree=3),
        ExponentialTrendModel(),
        LogarithmicTrendModel(),
        PowerTrendModel()
    ]
