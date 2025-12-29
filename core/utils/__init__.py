"""
Utility functions for ExplainFutures
"""

from .helpers import (
    display_error,
    display_success,
    display_warning,
    display_info,
    detect_datetime_columns,
    detect_numeric_columns,
    format_percentage,
    format_number,
    get_color_for_index  # ADDED
)

from .project_loader import (
    load_complete_project_data,
    load_project_on_open,
    ensure_project_data_loaded
)

__all__ = [
    'display_error',
    'display_success',
    'display_warning',
    'display_info',
    'detect_datetime_columns',
    'detect_numeric_columns',
    'format_percentage',
    'format_number',
    'get_color_for_index',  # ADDED
    'load_complete_project_data',
    'load_project_on_open',
    'ensure_project_data_loaded',
]
