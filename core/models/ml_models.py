"""
Machine Learning Models for Time Series
Random Forest, Gradient Boosting, SVR, kNN with lag features
"""

import numpy as np
import pandas as pd
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.base_model import BaseTimeSeriesModel


class RandomForestTSModel(BaseTimeSeriesModel):
    """
    Random Forest for Time Series
    Uses lag features and rolling statistics
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        super().__init__(name="Random Forest", model_type="ml")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_equation(self) -> str:
        """Get model description"""
        return f"Random Forest (n_trees={self.n_estimators}, depth={self.max_depth})"


class GradientBoostingTSModel(BaseTimeSeriesModel):
    """
    Gradient Boosting Regression for Time Series
    Generally the best ML model for time series
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3):
        super().__init__(name="Gradient Boosting", model_type="ml")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Gradient Boosting model"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_equation(self) -> str:
        """Get model description"""
        return f"Gradient Boosting (n_trees={self.n_estimators}, lr={self.learning_rate}, depth={self.max_depth})"


class SVRTSModel(BaseTimeSeriesModel):
    """
    Support Vector Regression for Time Series
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__(name="Support Vector Regression", model_type="ml")
        self.kernel = kernel
        self.C = C
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVR model"""
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        
        # SVR benefits from feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma='scale'
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_equation(self) -> str:
        """Get model description"""
        return f"SVR (kernel={self.kernel}, C={self.C})"


class KNNTSModel(BaseTimeSeriesModel):
    """
    k-Nearest Neighbors for Time Series
    """
    
    def __init__(self, n_neighbors: int = 5):
        super().__init__(name="k-Nearest Neighbors", model_type="ml")
        self.n_neighbors = n_neighbors
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit kNN model"""
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        
        # kNN benefits from feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights='distance'  # Weight by inverse distance
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_equation(self) -> str:
        """Get model description"""
        return f"kNN (k={self.n_neighbors}, weighted by distance)"


class XGBoostTSModel(BaseTimeSeriesModel):
    """
    XGBoost for Time Series (if available)
    Often the best performing ML model
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        super().__init__(name="XGBoost", model_type="ml")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_importances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost model"""
        try:
            from xgboost import XGBRegressor
            
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X, y, verbose=False)
            self.feature_importances_ = self.model.feature_importances_
            self.is_fitted = True
            
        except ImportError:
            # XGBoost not available, use GradientBoosting instead
            print("Warning: XGBoost not available, using GradientBoosting")
            from sklearn.ensemble import GradientBoostingRegressor
            
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42
            )
            
            self.model.fit(X, y)
            self.feature_importances_ = self.model.feature_importances_
            self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_equation(self) -> str:
        """Get model description"""
        return f"XGBoost (n_trees={self.n_estimators}, lr={self.learning_rate}, depth={self.max_depth})"


def create_ml_models(include_xgboost: bool = True) -> list:
    """
    Create ML model instances
    
    Parameters:
    -----------
    include_xgboost : bool
        Whether to include XGBoost (requires xgboost package)
        
    Returns:
    --------
    models : list
        List of ML model instances
    """
    models = [
        GradientBoostingTSModel(n_estimators=100, learning_rate=0.1, max_depth=3),
        RandomForestTSModel(n_estimators=100, max_depth=10),
        SVRTSModel(kernel='rbf', C=1.0),
        KNNTSModel(n_neighbors=5)
    ]
    
    if include_xgboost:
        try:
            import xgboost
            models.append(XGBoostTSModel(n_estimators=100, learning_rate=0.1, max_depth=3))
        except ImportError:
            pass  # Skip if not available
    
    return models
