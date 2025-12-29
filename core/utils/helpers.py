"""
Utility helper functions for ExplainFutures
Display functions, data detection, and formatting utilities
"""

import streamlit as st
import pandas as pd
from typing import List


# ============================================================================
# Display Functions
# ============================================================================

def display_error(title: str, message: str = None):
    """Display error message"""
    if message:
        st.error(f"**{title}**\n\n{message}")
    else:
        st.error(title)


def display_success(message: str):
    """Display success message"""
    st.success(message)


def display_warning(message: str):
    """Display warning message"""
    st.warning(message)


def display_info(message: str):
    """Display info message"""
    st.info(message)


# ============================================================================
# Data Detection Functions
# ============================================================================

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that might contain datetime data
    
    Returns list of column names that are likely datetime columns
    """
    datetime_candidates = []
    
    for col in df.columns:
        # Check if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_candidates.append(col)
            continue
        
        # Check if column name suggests it's a date/time
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'day', 'month', 'year']):
            datetime_candidates.append(col)
            continue
        
        # Try to parse a sample as datetime
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors='coerce')
                    if pd.to_datetime(sample, errors='coerce').notna().sum() > len(sample) * 0.5:
                        datetime_candidates.append(col)
                except:
                    pass
    
    return datetime_candidates


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that contain numeric data
    
    Returns list of column names that are numeric or can be converted to numeric
    """
    numeric_candidates = []
    
    for col in df.columns:
        # Check if column is already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)
            continue
        
        # Try to convert to numeric
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    converted = pd.to_numeric(sample, errors='coerce')
                    if converted.notna().sum() > len(sample) * 0.5:
                        numeric_candidates.append(col)
                except:
                    pass
    
    return numeric_candidates


# ============================================================================
# Formatting Functions
# ============================================================================

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a float as a percentage string
    
    Args:
        value: Float value (0.0 to 1.0)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string (e.g., "45.2%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2, thousands_sep: bool = True) -> str:
    """
    Format a number with optional thousands separator
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        thousands_sep: Whether to include thousands separator
    
    Returns:
        Formatted number string
    """
    if thousands_sep:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"
