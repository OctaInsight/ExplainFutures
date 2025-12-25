"""
Model Evaluation Module
Compute metrics, reliability scores, and generate evaluation visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    metrics : dict
        Dictionary of all metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (only if no zeros or near-zeros)
    if np.all(np.abs(y_true) > 1e-10):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mape = np.nan
    
    # Residuals
    residuals = y_true - y_pred
    
    # Additional metrics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_error = np.max(np.abs(residuals))
    median_absolute_error = np.median(np.abs(residuals))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_error': max_error,
        'median_ae': median_absolute_error
    }


def compute_reliability_score(train_metrics: Dict, 
                             test_metrics: Dict,
                             penalize_overfitting: bool = True) -> float:
    """
    Compute reliability score (0-100)
    
    Higher score = more reliable model
    
    Parameters:
    -----------
    train_metrics : dict
        Metrics on training set
    test_metrics : dict
        Metrics on test set
    penalize_overfitting : bool
        Whether to penalize train-test gap
        
    Returns:
    --------
    score : float
        Reliability score between 0 and 100
    """
    # Base score from test R² (0-60 points)
    test_r2 = test_metrics.get('r2', 0)
    test_r2 = max(0, min(1, test_r2))  # Clip to [0, 1]
    base_score = test_r2 * 60
    
    # Penalty for overfitting (0-30 points penalty)
    if penalize_overfitting:
        train_r2 = train_metrics.get('r2', 0)
        train_r2 = max(0, min(1, train_r2))
        
        overfit_gap = max(0, train_r2 - test_r2)
        overfit_penalty = overfit_gap * 30
    else:
        overfit_penalty = 0
    
    # Bonus for low residual bias (0-10 points)
    mean_residual = test_metrics.get('mean_residual', 0)
    std_residual = test_metrics.get('std_residual', 1)
    
    if abs(mean_residual) < 0.1 * std_residual:
        bias_bonus = 10
    elif abs(mean_residual) < 0.3 * std_residual:
        bias_bonus = 5
    else:
        bias_bonus = 0
    
    # Bonus for low error variability (0-10 points)
    if not np.isnan(test_metrics.get('mape', np.nan)):
        mape = test_metrics.get('mape', 100)
        if mape < 5:
            error_bonus = 10
        elif mape < 10:
            error_bonus = 7
        elif mape < 20:
            error_bonus = 4
        else:
            error_bonus = 0
    else:
        # Use RMSE if MAPE not available
        rmse = test_metrics.get('rmse', np.inf)
        mae = test_metrics.get('mae', np.inf)
        if rmse < mae * 1.1:  # Consistent errors
            error_bonus = 5
        else:
            error_bonus = 0
    
    # Final score
    score = base_score - overfit_penalty + bias_bonus + error_bonus
    score = np.clip(score, 0, 100)
    
    return score


def get_metric_color(value: float, metric_name: str, 
                    all_values: List[float]) -> str:
    """
    Get color for metric value (for heatmap)
    
    Parameters:
    -----------
    value : float
        Metric value
    metric_name : str
        Name of metric
    all_values : list
        All values for this metric (for context)
        
    Returns:
    --------
    color : str
        Color code
    """
    # For R², higher is better
    if metric_name in ['r2', 'reliability']:
        if value >= 0.9:
            return '#10b981'  # Green
        elif value >= 0.7:
            return '#fbbf24'  # Yellow
        else:
            return '#ef4444'  # Red
    
    # For errors (MAE, RMSE, MAPE), lower is better
    elif metric_name in ['mae', 'rmse', 'mape', 'max_error']:
        # Normalize to percentile
        if len(all_values) > 1:
            percentile = (sorted(all_values).index(value) + 1) / len(all_values)
            
            if percentile <= 0.25:  # Best 25%
                return '#10b981'  # Green
            elif percentile <= 0.75:  # Middle 50%
                return '#fbbf24'  # Yellow
            else:  # Worst 25%
                return '#ef4444'  # Red
        else:
            return '#9ca3af'  # Gray
    
    else:
        return '#9ca3af'  # Gray for unknown metrics


def create_actual_vs_predicted_plot(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   title: str = "Actual vs Predicted") -> go.Figure:
    """
    Create actual vs predicted scatter plot
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color='steelblue',
            opacity=0.6
        ),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def create_residual_plot(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        title: str = "Residual Plot") -> go.Figure:
    """
    Create residual plot
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    # Residual scatter
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            color='steelblue',
            opacity=0.6
        ),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_time_series_plot(timestamps: pd.Series,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           split_idx: Optional[int] = None,
                           title: str = "Time Series: Actual vs Predicted") -> go.Figure:
    """
    Create time series plot with train/test split
    
    Parameters:
    -----------
    timestamps : pd.Series
        Time index
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    split_idx : int, optional
        Index where train/test split occurs
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_true,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4)
    ))
    
    # Add vertical line at split
    if split_idx is not None and split_idx < len(timestamps):
        fig.add_vline(
            x=timestamps.iloc[split_idx],
            line_dash="dash",
            line_color="red",
            annotation_text="Train/Test Split",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=500,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_metrics_comparison_plot(metrics_df: pd.DataFrame,
                                   metric_name: str = 'mae',
                                   title: Optional[str] = None) -> go.Figure:
    """
    Create bar plot comparing metrics across models
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with models as rows, metrics as columns
    metric_name : str
        Which metric to plot
    title : str, optional
        Plot title
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    if metric_name not in metrics_df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in dataframe")
    
    # Sort by metric (lower is better for errors, higher for R²)
    if metric_name in ['r2', 'reliability']:
        ascending = False
    else:
        ascending = True
    
    data = metrics_df.sort_values(metric_name, ascending=ascending)
    
    # Color bars
    colors = []
    for val in data[metric_name]:
        if metric_name in ['r2', 'reliability']:
            if val >= 0.9:
                colors.append('#10b981')
            elif val >= 0.7:
                colors.append('#fbbf24')
            else:
                colors.append('#ef4444')
        else:
            # Use rank-based coloring
            rank = list(data[metric_name]).index(val)
            if rank < len(data) * 0.25:
                colors.append('#10b981')
            elif rank < len(data) * 0.75:
                colors.append('#fbbf24')
            else:
                colors.append('#ef4444')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data[metric_name],
        marker_color=colors,
        text=data[metric_name].round(3),
        textposition='outside'
    ))
    
    if title is None:
        title = f"{metric_name.upper()} Comparison Across Models"
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric_name.upper(),
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig
