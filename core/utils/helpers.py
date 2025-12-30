# core/utils/helpers.py

"""
Utility helper functions for ExplainFutures
Merged helper utilities (display, formatting, validation, detection, misc).
This file is intended to replace the old core/utils.py module content.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st


# ============================================================================
# Section 1: Page / UI Helpers
# ============================================================================

def setup_page_config(
    title: str = "ExplainFutures",
    icon: str = "🔮",
    layout: str = "wide",
):
    """
    Configure Streamlit page settings.

    Args:
        title: Page title
        icon: Page icon/favicon
        layout: Layout mode ('centered' or 'wide')
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )


# ============================================================================
# Section 2: Display Functions
# ============================================================================

def display_error(message: str, details: Optional[str] = None):
    """
    Display error message with optional details.

    Notes:
    - Backward compatible with older usage patterns:
      - display_error("Title", "Message") will still work because the second arg is treated as details.
      - display_error("Message") is the primary signature.
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("🔍 Error Details"):
            st.code(str(details))


def display_success(message: str):
    """Display success message."""
    st.success(f"✅ {message}")


def display_warning(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")


def display_info(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")


# ============================================================================
# Section 3: Formatting Utilities
# ============================================================================

def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: Number to format
        decimals: Decimal places

    Returns:
        Formatted string
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def format_percentage(value: Union[int, float], decimals: int = 1) -> str:
    """
    Format a fractional value as percentage (0.15 -> 15.0%).

    Args:
        value: Fraction (0-1)
        decimals: Decimal places

    Returns:
        Formatted percentage string
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    """
    try:
        seconds = float(seconds)
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}min"
        return f"{seconds / 3600:.1f}h"
    except Exception:
        return "N/A"


# ============================================================================
# Section 4: Math / Safety
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero / invalid.
    """
    try:
        if denominator == 0 or denominator is None or pd.isna(denominator):
            return default
        return float(numerator) / float(denominator)
    except Exception:
        return default


# ============================================================================
# Section 5: Colors
# ============================================================================

def get_color_for_index(index: int, colors: Optional[List[str]] = None) -> str:
    """
    Get color from palette by index (with wrapping).
    If colors not provided, attempts to read from core.config.get_config().
    Falls back to a local palette if core.config is unavailable.

    Args:
        index: Color index
        colors: Optional palette list

    Returns:
        Hex color string
    """
    if colors is None:
        try:
            from core.config import get_config  # type: ignore
            colors = get_config().get("default_color_palette", None)
        except Exception:
            colors = None

    if not colors:
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        ]

    return colors[index % len(colors)]


# ============================================================================
# Section 6: Download / Validation Utilities
# ============================================================================

def create_download_button(
    data: Any,
    filename: str,
    label: str = "📥 Download",
    mime_type: str = "text/csv",
    key: Optional[str] = None,
):
    """
    Create a download button for bytes/string/DataFrame.
    """
    payload = data
    mt = mime_type

    if isinstance(data, pd.DataFrame):
        payload = data.to_csv(index=False).encode("utf-8")
        mt = "text/csv"

    st.download_button(
        label=label,
        data=payload,
        file_name=filename,
        mime=mt,
        key=key,
    )


def validate_dataframe(
    df: Optional[pd.DataFrame],
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a DataFrame meets basic requirements.

    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"

    if len(df) < min_rows:
        return False, f"DataFrame has only {len(df)} rows, need at least {min_rows}"

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"

    return True, None


def sanitize_column_name(name: str) -> str:
    """
    Sanitize a column name for consistency.
    """
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\s-]", "", name)
    name = name.replace("-", "_")
    name = name.strip("_")
    return name


def get_datetime_format_suggestions() -> List[str]:
    """
    Get list of common datetime format strings from config if available.
    """
    try:
        from core.config import get_config  # type: ignore
        return list(get_config().get("datetime_formats", []))
    except Exception:
        return [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]


# ============================================================================
# Section 7: Data Detection Utilities
# ============================================================================

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that might contain datetime data.

    Heuristics:
    - dtype is datetime
    - column name contains datetime keywords
    - object column parses to datetime for >50% of a small sample
    """
    datetime_candidates: List[str] = []

    if df is None or df.empty:
        return datetime_candidates

    for col in df.columns:
        try:
            # Already datetime dtype
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_candidates.append(col)
                continue

            col_lower = str(col).lower()
            if any(k in col_lower for k in ["date", "time", "timestamp", "day", "month", "year"]):
                datetime_candidates.append(col)
                continue

            # Try parse sample for object columns
            if df[col].dtype == "object":
                sample = df[col].dropna().head(20)
                if len(sample) == 0:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", utc=False)
                if parsed.notna().sum() >= max(1, int(0.5 * len(sample))):
                    datetime_candidates.append(col)

        except Exception:
            continue

    # De-duplicate preserving order
    seen = set()
    out = []
    for c in datetime_candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that contain numeric data or can be converted to numeric.
    """
    numeric_candidates: List[str] = []

    if df is None or df.empty:
        return numeric_candidates

    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_candidates.append(col)
                continue

            if df[col].dtype == "object":
                sample = df[col].dropna().head(50)
                if len(sample) == 0:
                    continue
                converted = pd.to_numeric(sample, errors="coerce")
                if converted.notna().sum() >= max(1, int(0.6 * len(sample))):
                    numeric_candidates.append(col)

        except Exception:
            continue

    # De-duplicate preserving order
    seen = set()
    out = []
    for c in numeric_candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_variable_summary(df_long: pd.DataFrame, variable: str) -> Dict[str, Any]:
    """
    Get summary statistics for a variable in long format (expects columns: 'variable', 'value').
    """
    if df_long is None or df_long.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "missing": 0,
            "missing_pct": 0.0,
        }

    var_data = df_long[df_long["variable"] == variable]["value"]

    count = len(var_data)
    missing = int(var_data.isna().sum())

    return {
        "count": count,
        "mean": var_data.mean(),
        "std": var_data.std(),
        "min": var_data.min(),
        "max": var_data.max(),
        "median": var_data.median(),
        "missing": missing,
        "missing_pct": (missing / count) if count > 0 else 0.0,
    }
