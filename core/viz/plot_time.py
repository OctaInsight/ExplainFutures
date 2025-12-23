"""
Single Variable Time Plotting Module
Creates interactive Plotly time-series plots for individual variables
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def plot_single_variable(
    df: pd.DataFrame,
    variable: str,
    color: str = "#1f77b4",
    show_markers: bool = False,
    line_width: int = 2,
    y_scale: str = "linear",
    show_grid: bool = True,
    height: int = 500,
    title: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create an interactive time-series plot for a single variable
    
    Args:
        df: DataFrame with 'timestamp' and 'value' columns
        variable: Name of the variable (for labeling)
        color: Line color (hex or named)
        show_markers: Whether to show data point markers
        line_width: Width of the line
        y_scale: 'linear' or 'log'
        show_grid: Whether to show gridlines
        height: Plot height in pixels
        title: Optional plot title
    
    Returns:
        Plotly Figure object or None if error
    """
    try:
        # Validate input
        if df is None or len(df) == 0:
            return None
        
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            return None
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove NaN values for plotting
        df_clean = df.dropna(subset=['timestamp', 'value'])
        
        if len(df_clean) == 0:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Determine marker mode
        mode = 'lines+markers' if show_markers else 'lines'
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=df_clean['timestamp'],
            y=df_clean['value'],
            mode=mode,
            name=variable,
            line=dict(color=color, width=line_width),
            marker=dict(size=6, color=color) if show_markers else None,
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        plot_title = title if title else f"{variable} over Time"
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Time",
                showgrid=show_grid,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                title=variable,
                type=y_scale,
                showgrid=show_grid,
                gridcolor='lightgray',
                zeroline=False
            ),
            height=height,
            hovermode='x unified',
            template='plotly_white',
            showlegend=False  # Single variable doesn't need legend
        )
        
        # Add range slider for time axis
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None


def plot_single_variable_with_stats(
    df: pd.DataFrame,
    variable: str,
    show_mean: bool = True,
    show_std_bands: bool = False,
    **kwargs
) -> Optional[go.Figure]:
    """
    Create a time-series plot with statistical overlays
    
    Args:
        df: DataFrame with 'timestamp' and 'value' columns
        variable: Name of the variable
        show_mean: Whether to show mean line
        show_std_bands: Whether to show ±1 std dev bands
        **kwargs: Additional arguments passed to plot_single_variable
    
    Returns:
        Plotly Figure object or None if error
    """
    try:
        # Create base plot
        fig = plot_single_variable(df, variable, **kwargs)
        
        if fig is None:
            return None
        
        # Calculate statistics
        values = df['value'].dropna()
        mean_val = values.mean()
        std_val = values.std()
        
        # Add mean line
        if show_mean:
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="right"
            )
        
        # Add std dev bands
        if show_std_bands:
            fig.add_hrect(
                y0=mean_val - std_val,
                y1=mean_val + std_val,
                fillcolor="gray",
                opacity=0.1,
                line_width=0,
                annotation_text="±1 σ",
                annotation_position="right"
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating plot with stats: {e}")
        return None


def create_subplot_comparison(
    df_long: pd.DataFrame,
    variables: list,
    colors: Optional[list] = None,
    height: int = 800
) -> Optional[go.Figure]:
    """
    Create a subplot grid comparing multiple variables
    
    Args:
        df_long: DataFrame in long format
        variables: List of variable names to plot
        colors: Optional list of colors (one per variable)
        height: Total plot height
    
    Returns:
        Plotly Figure with subplots
    """
    try:
        from plotly.subplots import make_subplots
        
        n_vars = len(variables)
        
        if n_vars == 0:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=n_vars,
            cols=1,
            subplot_titles=variables,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Default colors if not provided
        if colors is None:
            from core.utils import get_color_for_index
            colors = [get_color_for_index(i) for i in range(n_vars)]
        
        # Add traces
        for idx, var in enumerate(variables):
            var_data = df_long[df_long['variable'] == var].sort_values('timestamp')
            
            fig.add_trace(
                go.Scatter(
                    x=var_data['timestamp'],
                    y=var_data['value'],
                    name=var,
                    line=dict(color=colors[idx], width=2),
                    hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                ),
                row=idx+1,
                col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text=var, row=idx+1, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=n_vars, col=1)
        
        fig.update_layout(
            height=height,
            showlegend=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating subplot comparison: {e}")
        return None