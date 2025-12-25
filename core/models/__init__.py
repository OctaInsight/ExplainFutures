"""
Time Series Models Module
Complete modeling framework for ExplainFutures
"""

from .base_model import BaseTimeSeriesModel, ModelArtifact
from .feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    build_ml_features,
    prepare_regression_data,
    detect_seasonality,
    check_stationarity
)
from .validation import (
    time_train_test_split,
    rolling_origin_cv,
    expanding_window_cv,
    validate_temporal_ordering,
    prepare_timeseries_data,
    create_forecast_horizon
)
from .regression_models import (
    LinearTrendModel,
    PolynomialTrendModel,
    ExponentialTrendModel,
    LogarithmicTrendModel,
    PowerTrendModel,
    create_regression_models
)
from .timeseries_models import (
    HoltWintersModel,
    ARIMAModel,
    AutoARIMAModel,
    create_timeseries_models
)
from .ml_models import (
    RandomForestTSModel,
    GradientBoostingTSModel,
    SVRTSModel,
    KNNTSModel,
    XGBoostTSModel,
    create_ml_models
)
from .evaluation import (
    compute_all_metrics,
    compute_reliability_score,
    get_metric_color,
    create_actual_vs_predicted_plot,
    create_residual_plot,
    create_time_series_plot,
    create_metrics_comparison_plot
)

__all__ = [
    # Base classes
    'BaseTimeSeriesModel',
    'ModelArtifact',
    
    # Feature engineering
    'create_time_features',
    'create_lag_features',
    'create_rolling_features',
    'build_ml_features',
    'prepare_regression_data',
    'detect_seasonality',
    'check_stationarity',
    
    # Validation
    'time_train_test_split',
    'rolling_origin_cv',
    'expanding_window_cv',
    'validate_temporal_ordering',
    'prepare_timeseries_data',
    'create_forecast_horizon',
    
    # Regression models
    'LinearTrendModel',
    'PolynomialTrendModel',
    'ExponentialTrendModel',
    'LogarithmicTrendModel',
    'PowerTrendModel',
    'create_regression_models',
    
    # Time series models
    'HoltWintersModel',
    'ARIMAModel',
    'AutoARIMAModel',
    'create_timeseries_models',
    
    # ML models
    'RandomForestTSModel',
    'GradientBoostingTSModel',
    'SVRTSModel',
    'KNNTSModel',
    'XGBoostTSModel',
    'create_ml_models',
    
    # Evaluation
    'compute_all_metrics',
    'compute_reliability_score',
    'get_metric_color',
    'create_actual_vs_predicted_plot',
    'create_residual_plot',
    'create_time_series_plot',
    'create_metrics_comparison_plot'
]
