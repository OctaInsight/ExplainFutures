"""
Multi-Variable Multi-Axis Plotting Module
Creates interactive Plotly plots with multiple variables on independent Y-axes
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional, Any


def plot_multi_variable(
    df_long: pd.DataFrame,
    variables: list,
    axis_config: Dict[str, Dict[str, Any]],
    show_legend: bool = True,
    show_grid: bool = True,
    height: int = 600,
    title: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create a multi-variable plot with independent Y-axes
    
    Args:
        df_long: DataFrame in long format (timestamp, variable, value)
        variables: List of variable names to plot
        axis_config: Configuration dict for each variable
            Format: {
                'variable_name': {
                    'color': '#hex',
                    'axis_side': 'left' or 'right',
                    'scale': 'linear' or 'log',
                    'line_width': int,
                    'y_min': float (optional),
                    'y_max': float (optional)
                }
            }
        show_legend: Whether to show legend
        show_grid: Whether to show gridlines
        height: Plot height in pixels
        title: Optional plot title
    
    Returns:
        Plotly Figure object or None if error
    """
    try:
        # Validate inputs
        if df_long is None or len(df_long) == 0:
            return None
        
        if not variables or len(variables) == 0:
            return None
        
        # Determine axis layout
        left_vars = [v for v in variables if axis_config.get(v, {}).get('axis_side', 'left') == 'left']
        right_vars = [v for v in variables if axis_config.get(v, {}).get('axis_side', 'left') == 'right']
        
        # Create figure with secondary y-axis if needed
        if right_vars:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()
        
        # Plot each variable
        for idx, var in enumerate(variables):
            # Get variable data
            var_data = df_long[df_long['variable'] == var].copy()
            var_data = var_data.sort_values('timestamp')
            var_data = var_data.dropna(subset=['timestamp', 'value'])
            
            if len(var_data) == 0:
                continue
            
            # Get configuration
            config = axis_config.get(var, {})
            color = config.get('color', f'#{idx:02x}{(idx*50)%256:02x}{(idx*100)%256:02x}')
            line_width = config.get('line_width', 2)
            axis_side = config.get('axis_side', 'left')
            
            # Create trace
            trace = go.Scatter(
                x=var_data['timestamp'],
                y=var_data['value'],
                mode='lines',
                name=var,
                line=dict(color=color, width=line_width),
                hovertemplate=f'<b>{var}</b><br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            )
            
            # Add trace to appropriate axis
            if right_vars and axis_side == 'right':
                fig.add_trace(trace, secondary_y=True)
            else:
                if right_vars:
                    fig.add_trace(trace, secondary_y=False)
                else:
                    fig.add_trace(trace)
        
        # Configure axes
        # Left Y-axis
        left_config = _get_axis_config_for_side(left_vars, axis_config, 'left')
        
        yaxis_config = dict(
            title=", ".join(left_vars) if len(left_vars) <= 3 else f"{len(left_vars)} variables",
            showgrid=show_grid,
            gridcolor='lightgray',
            zeroline=False,
            type=left_config.get('scale', 'linear')
        )
        
        if left_config.get('y_min') is not None and left_config.get('y_max') is not None:
            yaxis_config['range'] = [left_config['y_min'], left_config['y_max']]
        
        if right_vars:
            fig.update_yaxes(yaxis_config, secondary_y=False)
        else:
            fig.update_layout(yaxis=yaxis_config)
        
        # Right Y-axis (if applicable)
        if right_vars:
            right_config = _get_axis_config_for_side(right_vars, axis_config, 'right')
            
            yaxis2_config = dict(
                title=", ".join(right_vars) if len(right_vars) <= 3 else f"{len(right_vars)} variables",
                showgrid=False,  # Avoid grid overlap
                zeroline=False,
                type=right_config.get('scale', 'linear')
            )
            
            if right_config.get('y_min') is not None and right_config.get('y_max') is not None:
                yaxis2_config['range'] = [right_config['y_min'], right_config['y_max']]
            
            fig.update_yaxes(yaxis2_config, secondary_y=True)
        
        # X-axis configuration
        fig.update_xaxes(
            title="Time",
            showgrid=show_grid,
            gridcolor='lightgray',
            zeroline=False
        )
        
        # Overall layout
        plot_title = title if title else f"Multi-Variable Comparison ({len(variables)} variables)"
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=height,
            hovermode='x unified',
            template='plotly_white',
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating multi-variable plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_axis_config_for_side(
    variables: list,
    axis_config: Dict[str, Dict[str, Any]],
    side: str
) -> Dict[str, Any]:
    """
    Get unified axis configuration for variables on one side
    
    Args:
        variables: List of variables on this side
        axis_config: Full axis configuration dict
        side: 'left' or 'right'
    
    Returns:
        Unified config dict
    """
    if not variables:
        return {}
    
    # Use config from first variable as default
    first_var_config = axis_config.get(variables[0], {})
    
    # Check if all variables have same scale
    scales = [axis_config.get(v, {}).get('scale', 'linear') for v in variables]
    scale = scales[0] if len(set(scales)) == 1 else 'linear'
    
    # Check if all have custom ranges
    y_mins = [axis_config.get(v, {}).get('y_min') for v in variables]
    y_maxs = [axis_config.get(v, {}).get('y_max') for v in variables]
    
    # Only use custom range if all variables on this side have it
    if all(v is not None for v in y_mins) and all(v is not None for v in y_maxs):
        y_min = min(y_mins)
        y_max = max(y_maxs)
    else:
        y_min, y_max = None, None
    
    return {
        'scale': scale,
        'y_min': y_min,
        'y_max': y_max
    }


def create_stacked_comparison(
    df_long: pd.DataFrame,
    variables: list,
    normalize: bool = False,
    colors: Optional[list] = None,
    height: int = 600
) -> Optional[go.Figure]:
    """
    Create a stacked area chart for multiple variables
    
    Args:
        df_long: DataFrame in long format
        variables: List of variables to stack
        normalize: Whether to normalize to percentages
        colors: Optional list of colors
        height: Plot height
    
    Returns:
        Plotly Figure or None
    """
    try:
        from core.utils import get_color_for_index
        
        # Pivot to wide format for stacking
        df_wide = df_long.pivot(index='timestamp', columns='variable', values='value')
        df_wide = df_wide[variables].sort_index()
        
        # Normalize if requested
        if normalize:
            df_wide = df_wide.div(df_wide.sum(axis=1), axis=0) * 100
        
        # Create figure
        fig = go.Figure()
        
        # Default colors
        if colors is None:
            colors = [get_color_for_index(i) for i in range(len(variables))]
        
        # Add traces (bottom to top)
        for idx, var in enumerate(variables):
            fig.add_trace(go.Scatter(
                x=df_wide.index,
                y=df_wide[var],
                name=var,
                mode='lines',
                stackgroup='one',
                fillcolor=colors[idx],
                line=dict(width=0.5, color=colors[idx]),
                hovertemplate=f'<b>{var}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>'
            ))
        
        # Layout
        y_title = "Percentage (%)" if normalize else "Value"
        
        fig.update_layout(
            title="Stacked Variable Comparison",
            xaxis_title="Time",
            yaxis_title=y_title,
            height=height,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating stacked comparison: {e}")
        return None