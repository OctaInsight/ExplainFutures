"""
core.utils package

Central export point for all utility helpers and project-loading logic.
All application code should import utilities ONLY from core.utils
and never directly from helpers.py or project_loader.py.
"""

# ================================
# UI / Display helpers
# ================================
from .helpers import (
    display_error,
    display_success,
    display_warning,
    display_info,
)

# ================================
# Formatting helpers
# ================================
from .helpers import (
    format_number,
    format_percentage,
    format_duration,
)

# ================================
# Detection & validation helpers
# ================================
from .helpers import (
    detect_datetime_columns,
    detect_numeric_columns,
    validate_dataframe,
    sanitize_column_name,
)

# ================================
# General utilities
# ================================
from .helpers import (
    safe_divide,
    get_color_for_index,
    create_download_button,
    get_variable_summary,
    setup_page_config,
)

# ================================
# Project loading helpers
# ================================
from .project_loader import (
    load_project_on_open,
    ensure_project_data_loaded,
)

# ================================
# Public API
# ================================
__all__ = [
    # Display
    "display_error",
    "display_success",
    "display_warning",
    "display_info",

    # Formatting
    "format_number",
    "format_percentage",
    "format_duration",

    # Detection / validation
    "detect_datetime_columns",
    "detect_numeric_columns",
    "validate_dataframe",
    "sanitize_column_name",

    # Utilities
    "safe_divide",
    "get_color_for_index",
    "create_download_button",
    "get_variable_summary",
    "setup_page_config",

    # Project loading
    "load_project_on_open",
    "ensure_project_data_loaded",
]
