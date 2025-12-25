"""
Model Visualization Module
Create comparison plots for trained models
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List


def create_model_comparison_plot(variable_name: str,
                                 train_values: np.ndarray,
                                 test_values: np.ndarray,
                                 train_timestamps: pd.DatetimeIndex,
                                 test_timestamps: pd.DatetimeIndex,
                                 model_results: Dict,
                                 selected_models: List[str],
                                 split_index: int) -> go.Figure:
    """
    Create interactive plot comparing selected models
    
    Parameters:
    -----------
    variable_name : str
        Name of variable
    train_values : np.ndarray
        Training data values
    test_values : np.ndarray
        Testing data values
    train_timestamps : pd.DatetimeIndex
        Training timestamps
    test_timestamps : pd.DatetimeIndex
        Testing timestamps
    model_results : dict
        Dictionary containing all trained models (tier1, tier2, tier3)
    selected_models : list
        List of up to 3 model names to display
    split_index : int
        Index where train/test split occurs
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive comparison plot
    """
    
    fig = go.Figure()
    
    # Combine data
    all_values = np.concatenate([train_values, test_values])
    all_timestamps = pd.DatetimeIndex(list(train_timestamps) + list(test_timestamps))
    
    # 1. Plot actual data (markers only by default)
    fig.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_values,
        mode='markers',  # Changed from 'lines+markers' to just 'markers'
        name='Actual Data',
        marker=dict(
            size=6,
            color='black'
        ),
        showlegend=True
    ))
    
    # 2. Add train/test split line
    if split_index > 0 and split_index < len(all_timestamps):
        # Get the timestamp at split point
        split_time = all_timestamps[split_index]
        
        # Add vertical line using shape instead of add_vline to avoid timestamp issues
        fig.add_shape(
            type="line",
            x0=split_time,
            x1=split_time,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add annotation
        fig.add_annotation(
            x=split_time,
            y=1.0,
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )
    
    # 3. Plot selected models
    colors = ['steelblue', 'orange', 'green']
    
    for idx, model_name in enumerate(selected_models[:3]):  # Max 3 models
        
        # Find model in results
        model_data = find_model_in_results(model_results, model_name)
        
        if model_data is None:
            continue
        
        # Get predictions
        train_pred = model_data.get('train_predictions', [])
        test_pred = model_data.get('test_predictions', [])
        
        # Handle different lengths (e.g., ML models might have fewer points due to lag features)
        if len(train_pred) < len(train_values):
            # Pad with NaN at the beginning
            train_pred_full = np.full(len(train_values), np.nan)
            train_pred_full[-len(train_pred):] = train_pred
            train_pred = train_pred_full
        
        # Combine predictions
        all_pred = np.concatenate([train_pred, test_pred])
        
        # Plot
        fig.add_trace(go.Scatter(
            x=all_timestamps,
            y=all_pred,
            mode='lines',
            name=model_name,
            line=dict(
                color=colors[idx],
                width=2,
                dash='dash'
            ),
            showlegend=True
        ))
    
    # Layout
    fig.update_layout(
        title=f"{variable_name} - Model Comparison",
        xaxis_title="Time",
        yaxis_title=f"{variable_name}",  # Use variable name as Y-axis label
        height=600,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',  # White background with slight transparency
            bordercolor='black',  # Black border
            borderwidth=1,
            font=dict(
                color='black',  # Black text for contrast
                size=11
            )
        )
    )
    
    return fig


def find_model_in_results(model_results: Dict, model_name: str) -> Dict:
    """
    Find model data in the results dictionary
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with tier1_math, tier2_timeseries, tier3_ml
    model_name : str
        Name of the model to find
        
    Returns:
    --------
    model_data : dict or None
        Model data if found, None otherwise
    """
    
    # Check Tier 1
    if model_name in model_results.get('tier1_math', {}):
        return model_results['tier1_math'][model_name]
    
    # Check Tier 2
    if model_name in model_results.get('tier2_timeseries', {}):
        return model_results['tier2_timeseries'][model_name]
    
    # Check Tier 3
    if model_name in model_results.get('tier3_ml', {}):
        return model_results['tier3_ml'][model_name]
    
    return None


def create_metrics_comparison_chart(model_results: Dict, metric: str = 'r2') -> go.Figure:
    """
    Create bar chart comparing all models by a specific metric
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with all model results
    metric : str
        Metric to compare ('mae', 'rmse', 'r2', 'mape')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart figure
    """
    
    model_names = []
    metric_values = []
    colors = []
    
    color_map = {
        'tier1_math': 'lightblue',
        'tier2_timeseries': 'lightgreen',
        'tier3_ml': 'lightsalmon'
    }
    
    # Collect metrics from all tiers
    for tier_name, tier_models in model_results.items():
        for model_name, model_data in tier_models.items():
            test_metrics = model_data.get('test_metrics', {})
            value = test_metrics.get(metric, np.nan)
            
            if not np.isnan(value):
                model_names.append(model_name)
                metric_values.append(value)
                colors.append(color_map.get(tier_name, 'gray'))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=metric_values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in metric_values],
        textposition='outside'
    ))
    
    # Layout
    metric_title = {
        'mae': 'Mean Absolute Error',
        'rmse': 'Root Mean Squared Error',
        'r2': 'R² Score',
        'mape': 'Mean Absolute Percentage Error'
    }.get(metric, metric.upper())
    
    fig.update_layout(
        title=f"Model Comparison: {metric_title}",
        xaxis_title="Model",
        yaxis_title=metric_title,
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig
