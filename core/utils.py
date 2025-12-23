"""
Core Utilities Module
Common utility functions used across the application
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


def setup_page_config(
    title: str = "ExplainFutures",
    icon: str = "🔮",
    layout: str = "wide"
):
    """
    Configure Streamlit page settings
    
    Args:
        title: Page title
        icon: Page icon/favicon
        layout: Layout mode ('centered' or 'wide')
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousands separators
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        str: Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format number as percentage
    
    Args:
        value: Value to format (0.15 = 15%)
        decimals: Number of decimal places
    
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default


def get_color_for_index(index: int, colors: Optional[List[str]] = None) -> str:
    """
    Get color from palette by index (with wrapping)
    
    Args:
        index: Color index
        colors: Optional color palette
    
    Returns:
        str: Color hex code
    """
    if colors is None:
        from core.config import get_config
        colors = get_config()["default_color_palette"]
    
    return colors[index % len(colors)]


def display_error(message: str, details: Optional[str] = None):
    """
    Display error message with optional details
    
    Args:
        message: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("🔍 Error Details"):
            st.code(details)


def display_warning(message: str):
    """
    Display warning message
    
    Args:
        message: Warning message
    """
    st.warning(f"⚠️ {message}")


def display_success(message: str):
    """
    Display success message
    
    Args:
        message: Success message
    """
    st.success(f"✅ {message}")


def display_info(message: str):
    """
    Display info message
    
    Args:
        message: Info message
    """
    st.info(f"ℹ️ {message}")


def create_download_button(
    data: Any,
    filename: str,
    label: str = "📥 Download",
    mime_type: str = "text/csv",
    key: Optional[str] = None
):
    """
    Create a download button for data
    
    Args:
        data: Data to download (bytes, string, or DataFrame)
        filename: Suggested filename
        label: Button label
        mime_type: MIME type of data
        key: Optional unique key for button
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False).encode('utf-8')
        mime_type = "text/csv"
    
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type,
        key=key
    )


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> Tuple[bool, Optional[str]]:
    """
    Validate a DataFrame meets basic requirements
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if len(df) < min_rows:
        return False, f"DataFrame has only {len(df)} rows, need at least {min_rows}"
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
    
    return True, None


def sanitize_column_name(name: str) -> str:
    """
    Sanitize a column name for consistency
    
    Args:
        name: Original column name
    
    Returns:
        str: Sanitized column name
    """
    import re
    
    # Strip whitespace
    name = str(name).strip()
    
    # Replace multiple spaces with single underscore
    name = re.sub(r'\s+', '_', name)
    
    # Remove special characters except underscore and hyphen
    name = re.sub(r'[^\w\s-]', '', name)
    
    # Replace hyphens with underscores
    name = name.replace('-', '_')
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name


def get_datetime_format_suggestions() -> List[str]:
    """
    Get list of common datetime format strings
    
    Returns:
        list: List of datetime format strings
    """
    from core.config import get_config
    return get_config()["datetime_formats"]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        str: Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_variable_summary(df: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Get summary statistics for a variable in long format
    
    Args:
        df: DataFrame in long format with 'variable' and 'value' columns
        variable: Variable name
    
    Returns:
        dict: Summary statistics
    """
    var_data = df[df['variable'] == variable]['value']
    
    return {
        'count': len(var_data),
        'mean': var_data.mean(),
        'std': var_data.std(),
        'min': var_data.min(),
        'max': var_data.max(),
        'median': var_data.median(),
        'missing': var_data.isna().sum(),
        'missing_pct': var_data.isna().sum() / len(var_data) if len(var_data) > 0 else 0
    }


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect numeric columns in a DataFrame
    
    Args:
        df: Input DataFrame
    
    Returns:
        list: List of numeric column names
    """
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                pass
    
    return numeric_cols


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect potential datetime columns in a DataFrame
    
    Args:
        df: Input DataFrame
    
    Returns:
        list: List of potential datetime column names
    """
    datetime_cols = []
    
    for col in df.columns:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        
        # Check column name for time-related keywords
        col_lower = str(col).lower()
        time_keywords = ['date', 'time', 'timestamp', 'datetime', 'period', 'year', 'month', 'day']
        if any(keyword in col_lower for keyword in time_keywords):
            datetime_cols.append(col)
            continue
        
        # Try to parse as datetime
        if len(df) > 0:
            try:
                pd.to_datetime(df[col].dropna().iloc[:5], errors='raise')
                datetime_cols.append(col)
            except:
                pass
    
    return datetime_cols


class Timer:
    """Simple context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.duration = (datetime.now() - self.start_time).total_seconds()


def create_progress_tracker(total_steps: int, description: str = "Processing"):
    """
    Create a Streamlit progress tracker
    
    Args:
        total_steps: Total number of steps
        description: Description text
    
    Returns:
        tuple: (progress_bar, status_text)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"{description}... 0%")
    
    return progress_bar, status_text


def update_progress(progress_bar, status_text, current_step: int, total_steps: int, description: str = "Processing"):
    """
    Update progress tracker
    
    Args:
        progress_bar: Streamlit progress bar object
        status_text: Streamlit text object
        current_step: Current step number
        total_steps: Total number of steps
        description: Description text
    """
    progress = current_step / total_steps
    progress_bar.progress(progress)
    status_text.text(f"{description}... {int(progress * 100)}%")
