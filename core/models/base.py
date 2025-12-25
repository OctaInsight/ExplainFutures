"""
Base Model Class
All time-series models inherit from this base class
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for all time-series models
    
    All models must implement:
    - fit(X, y)
    - predict(X)
    - get_equation() [optional, for interpretable models]
    """
    
    def __init__(self, name: str, model_type: str):
        """
        Initialize base model
        
        Parameters:
        -----------
        name : str
            Model name (e.g., "Linear Regression", "Random Forest")
        model_type : str
            Model category: 'regression', 'timeseries', 'ml'
        """
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.fit_time = None
        self.train_metrics = {}
        self.test_metrics = {}
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseTimeSeriesModel':
        """
        Fit the model to training data
        
        Parameters:
        -----------
        X : np.ndarray
            Features (could be time, lag features, etc.)
        y : np.ndarray
            Target values
            
        Returns:
        --------
        self : BaseTimeSeriesModel
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Features for prediction
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        pass
    
    def get_equation(self) -> Optional[str]:
        """
        Get model equation (for interpretable models)
        
        Returns:
        --------
        equation : str or None
            String representation of model equation
        """
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
        --------
        params : dict
            Dictionary of model parameters
        """
        return {
            'name': self.name,
            'type': self.model_type,
            'fitted': self.is_fitted,
            'fit_time': self.fit_time
        }
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        metrics : dict
            Dictionary of metric values
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (only if no zeros)
        if np.all(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            mape = np.nan
        
        # Residuals
        residuals = y_true - y_pred
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_error': np.max(np.abs(residuals))
        }
    
    def compute_reliability_score(self, train_metrics: Dict, test_metrics: Dict) -> float:
        """
        Compute reliability score (0-100)
        
        Based on:
        - Test set accuracy (primary)
        - Train-test consistency (no overfitting)
        - Residual stability
        
        Parameters:
        -----------
        train_metrics : dict
            Metrics on training set
        test_metrics : dict
            Metrics on test set
            
        Returns:
        --------
        score : float
            Reliability score between 0 and 100
        """
        # Normalize RMSE to 0-100 scale (lower is better)
        # Assume RMSE in range [0, 2*std(y)]
        test_rmse = test_metrics['rmse']
        
        # Base score from R² (if available)
        if 'r2' in test_metrics and not np.isnan(test_metrics['r2']):
            r2 = max(0, test_metrics['r2'])  # Clip negative R²
            base_score = r2 * 60  # R² contributes up to 60 points
        else:
            base_score = 30  # Default if R² not available
        
        # Penalty for overfitting (train much better than test)
        train_r2 = train_metrics.get('r2', 0)
        test_r2 = test_metrics.get('r2', 0)
        
        if not np.isnan(train_r2) and not np.isnan(test_r2):
            overfit_gap = train_r2 - test_r2
            overfit_penalty = max(0, overfit_gap * 30)  # Up to 30 point penalty
        else:
            overfit_penalty = 0
        
        # Bonus for low residual variance (stable predictions)
        residual_std = test_metrics.get('std_residual', np.inf)
        residual_mean = test_metrics.get('mean_residual', 0)
        
        if abs(residual_mean) < 0.1 * residual_std:  # Unbiased predictions
            stability_bonus = 10
        else:
            stability_bonus = 0
        
        # Final score
        score = base_score - overfit_penalty + stability_bonus
        score = np.clip(score, 0, 100)
        
        return score
    
    def save_results(self, train_results: Dict, test_results: Dict):
        """
        Save training and test results
        
        Parameters:
        -----------
        train_results : dict
            Training set predictions and metrics
        test_results : dict
            Test set predictions and metrics
        """
        self.train_metrics = train_results['metrics']
        self.test_metrics = test_results['metrics']
        
        # Compute reliability
        reliability = self.compute_reliability_score(
            self.train_metrics, 
            self.test_metrics
        )
        
        self.test_metrics['reliability'] = reliability
    
    def __repr__(self):
        return f"{self.name} ({self.model_type})"


class ModelArtifact:
    """
    Container for trained model + metadata + results
    """
    
    def __init__(self, 
                 model: BaseTimeSeriesModel,
                 series_name: str,
                 train_data: Dict,
                 test_data: Dict,
                 config: Dict):
        """
        Initialize model artifact
        
        Parameters:
        -----------
        model : BaseTimeSeriesModel
            Trained model instance
        series_name : str
            Name of time series
        train_data : dict
            Training data and predictions
        test_data : dict
            Test data and predictions
        config : dict
            Model configuration
        """
        self.model = model
        self.series_name = series_name
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.timestamp = datetime.now()
        
    def get_summary(self) -> Dict:
        """Get summary of model performance"""
        return {
            'series': self.series_name,
            'model_name': self.model.name,
            'model_type': self.model.model_type,
            'train_r2': self.train_data['metrics'].get('r2', np.nan),
            'test_r2': self.test_data['metrics'].get('r2', np.nan),
            'test_mae': self.test_data['metrics'].get('mae', np.nan),
            'test_rmse': self.test_data['metrics'].get('rmse', np.nan),
            'test_mape': self.test_data['metrics'].get('mape', np.nan),
            'reliability': self.test_data['metrics'].get('reliability', 0),
            'equation': self.model.get_equation(),
            'trained_at': self.timestamp
        }
    
    def __repr__(self):
        return f"ModelArtifact({self.series_name} - {self.model.name})"
