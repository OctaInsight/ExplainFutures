"""
Tier 3: Machine Learning Models with Lag Features
Gradient Boosting, Random Forest, SVR, kNN with time-derived features
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_tier3_models(train_values: np.ndarray,
                      test_values: np.ndarray,
                      train_timestamps: pd.DatetimeIndex,
                      test_timestamps: pd.DatetimeIndex,
                      lag_features: list = [1, 2, 3, 7],
                      rolling_windows: list = [7, 14]) -> Dict:
    """
    Train all Tier 3 ML models with lag features
    
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
    lag_features : list
        Lag periods to create [1, 2, 3, 7] means y(t-1), y(t-2), etc.
    rolling_windows : list
        Window sizes for rolling statistics
        
    Returns:
    --------
    results : dict
        Dictionary of trained models and their predictions
    """
    results = {}
    
    # Build features
    feature_data = build_lag_features(
        train_values, test_values,
        train_timestamps, test_timestamps,
        lag_features, rolling_windows
    )
    
    if feature_data is None:
        return {}  # Not enough data for lag features
    
    X_train = feature_data['X_train']
    X_test = feature_data['X_test']
    y_train = feature_data['y_train']
    y_test = feature_data['y_test']
    
    # 1. Gradient Boosting (best default)
    results['Gradient Boosting'] = train_gbr_model(X_train, y_train, X_test, y_test)
    
    # 2. Random Forest
    results['Random Forest'] = train_rf_model(X_train, y_train, X_test, y_test)
    
    # 3. SVR
    results['SVR'] = train_svr_model(X_train, y_train, X_test, y_test)
    
    # 4. kNN
    results['kNN'] = train_knn_model(X_train, y_train, X_test, y_test)
    
    return results


def build_lag_features(train_values: np.ndarray,
                      test_values: np.ndarray,
                      train_timestamps: pd.DatetimeIndex,
                      test_timestamps: pd.DatetimeIndex,
                      lag_features: list,
                      rolling_windows: list) -> Dict:
    """
    Build lag features and rolling statistics
    
    Returns:
    --------
    feature_data : dict or None
        {
            'X_train': np.ndarray,
            'y_train': np.ndarray,
            'X_test': np.ndarray,
            'y_test': np.ndarray,
            'feature_names': list
        }
    """
    # Combine train and test for feature engineering
    all_values = np.concatenate([train_values, test_values])
    all_timestamps = pd.DatetimeIndex(list(train_timestamps) + list(test_timestamps))
    
    # Create DataFrame
    df = pd.DataFrame({
        'value': all_values,
        'timestamp': all_timestamps
    })
    
    feature_names = []
    
    # 1. Lag features
    for lag in lag_features:
        df[f'lag_{lag}'] = df['value'].shift(lag)
        feature_names.append(f'lag_{lag}')
    
    # 2. Rolling statistics
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        feature_names.extend([f'rolling_mean_{window}', f'rolling_std_{window}'])
    
    # 3. Calendar features (if available)
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    feature_names.extend(['month', 'day_of_week', 'day_of_year'])
    
    # 4. Time numeric (days since start)
    df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 86400
    feature_names.append('time_numeric')
    
    # Drop rows with NaN (due to lag features)
    df_clean = df.dropna()
    
    if len(df_clean) < 10:
        return None  # Not enough data after creating lag features
    
    # Split back into train and test
    train_size = len(train_values)
    
    # Find where clean data starts and ends
    clean_indices = df_clean.index.tolist()
    
    # Separate train and test from clean data
    train_clean = df_clean[df_clean.index < train_size]
    test_clean = df_clean[df_clean.index >= train_size]
    
    if len(train_clean) < 5 or len(test_clean) < 1:
        return None  # Not enough clean data
    
    # Extract features and target
    X_train = train_clean[feature_names].values
    y_train = train_clean['value'].values
    X_test = test_clean[feature_names].values
    y_test = test_clean['value'].values
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }


def train_gbr_model(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train Gradient Boosting Regressor"""
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': f"GBR (n_trees=100, depth=3, lr=0.1)",
        'type': 'gradient_boosting',
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }


def train_rf_model(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train Random Forest"""
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': f"Random Forest (n_trees=100, depth=10)",
        'type': 'random_forest',
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }


def train_svr_model(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train Support Vector Regression"""
    
    # SVR benefits from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale'
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': f"SVR (kernel=rbf, C=1.0)",
        'type': 'svr'
    }


def train_knn_model(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train k-Nearest Neighbors"""
    
    # kNN benefits from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance'
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test,
        'train_metrics': compute_metrics(y_train, y_pred_train),
        'test_metrics': compute_metrics(y_test, y_pred_test),
        'equation': f"kNN (k=5, weighted)",
        'type': 'knn'
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
