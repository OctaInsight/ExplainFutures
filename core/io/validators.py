"""
Data Validators Module
Validates and parses time columns and data quality
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime


def parse_time_column(
    df: pd.DataFrame,
    time_column: str,
    datetime_format: Optional[str] = None
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Parse time column to datetime
    
    Args:
        df: Input DataFrame
        time_column: Name of time column
        datetime_format: Optional datetime format string
    
    Returns:
        tuple: (parsed Series, error_message)
    """
    try:
        if time_column not in df.columns:
            return None, f"Column '{time_column}' not found"
        
        time_series = df[time_column]
        
        # If already datetime, return it
        if pd.api.types.is_datetime64_any_dtype(time_series):
            return time_series, None
        
        # Try parsing with specified format
        if datetime_format and datetime_format != "auto":
            try:
                parsed = pd.to_datetime(time_series, format=datetime_format, errors='raise')
                return parsed, None
            except Exception as e:
                return None, f"Failed to parse with format '{datetime_format}': {str(e)}"
        
        # Try automatic parsing
        try:
            parsed = pd.to_datetime(time_series, errors='raise', infer_datetime_format=True)
            return parsed, None
        except Exception as e:
            # Try common formats one by one
            from core.config import get_config
            formats = get_config()["datetime_formats"]
            
            for fmt in formats:
                if fmt == "ISO8601":
                    continue
                try:
                    parsed = pd.to_datetime(time_series, format=fmt, errors='raise')
                    return parsed, None
                except:
                    continue
            
            return None, f"Could not parse time column. Please specify format manually."
        
    except Exception as e:
        return None, f"Error parsing time column: {str(e)}"


def validate_time_series(time_series: pd.Series) -> Tuple[bool, Optional[str], dict]:
    """
    Validate time series for common issues
    
    Args:
        time_series: Parsed datetime series
    
    Returns:
        tuple: (is_valid, error_message, metadata_dict)
    """
    metadata = {}
    
    try:
        # Check for nulls
        null_count = time_series.isna().sum()
        metadata["null_count"] = int(null_count)
        metadata["null_pct"] = float(null_count / len(time_series))
        
        if null_count > 0:
            return False, f"Time column contains {null_count} null values", metadata
        
        # Get time range
        metadata["min_time"] = time_series.min()
        metadata["max_time"] = time_series.max()
        metadata["time_span"] = (metadata["max_time"] - metadata["min_time"]).days
        
        # Check for duplicates
        duplicates = time_series.duplicated().sum()
        metadata["duplicate_count"] = int(duplicates)
        
        if duplicates > 0:
            metadata["warning"] = f"Time column has {duplicates} duplicate timestamps"
        
        # Estimate sampling frequency
        time_diffs = time_series.sort_values().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            metadata["median_interval_days"] = float(median_diff.total_seconds() / 86400)
            
            # Try to identify frequency
            days = median_diff.total_seconds() / 86400
            if days < 0.5:
                metadata["estimated_frequency"] = "Hourly or sub-hourly"
            elif days < 1.5:
                metadata["estimated_frequency"] = "Daily"
            elif days < 8:
                metadata["estimated_frequency"] = "Weekly"
            elif days < 35:
                metadata["estimated_frequency"] = "Monthly"
            elif days < 100:
                metadata["estimated_frequency"] = "Quarterly"
            else:
                metadata["estimated_frequency"] = "Yearly or irregular"
        
        return True, None, metadata
        
    except Exception as e:
        return False, f"Error validating time series: {str(e)}", metadata


def check_numeric_columns(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[List[str], List[str], dict]:
    """
    Check if columns are numeric or can be converted
    
    Args:
        df: Input DataFrame
        columns: List of column names to check
    
    Returns:
        tuple: (valid_columns, invalid_columns, conversion_info)
    """
    valid = []
    invalid = []
    conversion_info = {}
    
    for col in columns:
        if col not in df.columns:
            invalid.append(col)
            conversion_info[col] = "Column not found"
            continue
        
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            valid.append(col)
            conversion_info[col] = "Already numeric"
            continue
        
        # Try to convert
        try:
            pd.to_numeric(df[col], errors='raise')
            valid.append(col)
            conversion_info[col] = "Convertible to numeric"
        except:
            invalid.append(col)
            # Count non-numeric values
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            conversion_info[col] = f"Contains {non_numeric} non-numeric values"
    
    return valid, invalid, conversion_info


def detect_and_report_issues(df_long: pd.DataFrame) -> dict:
    """
    Detect various data quality issues in long-format DataFrame
    
    Args:
        df_long: DataFrame in long format (timestamp, variable, value)
    
    Returns:
        dict: Dictionary of detected issues
    """
    issues = {
        "missing_values": {},
        "outliers": {},
        "duplicates": {},
        "coverage": {},
        "warnings": []
    }
    
    try:
        # Check each variable
        for var in df_long['variable'].unique():
            var_data = df_long[df_long['variable'] == var]
            
            # Missing values
            missing_count = var_data['value'].isna().sum()
            missing_pct = missing_count / len(var_data) if len(var_data) > 0 else 0
            issues["missing_values"][var] = {
                "count": int(missing_count),
                "percentage": float(missing_pct)
            }
            
            # Outliers (simple IQR method)
            values = var_data['value'].dropna()
            if len(values) > 4:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outlier_count = ((values < lower_bound) | (values > upper_bound)).sum()
                issues["outliers"][var] = {
                    "count": int(outlier_count),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
            
            # Time coverage
            timestamps = var_data['timestamp'].dropna()
            if len(timestamps) > 0:
                issues["coverage"][var] = {
                    "start": timestamps.min(),
                    "end": timestamps.max(),
                    "points": len(timestamps)
                }
            
            # Duplicate timestamps
            dup_count = var_data['timestamp'].duplicated().sum()
            if dup_count > 0:
                issues["duplicates"][var] = int(dup_count)
        
        # Global warnings
        total_missing = sum(v["count"] for v in issues["missing_values"].values())
        if total_missing > 0:
            issues["warnings"].append(f"Total missing values: {total_missing}")
        
        total_outliers = sum(v["count"] for v in issues["outliers"].values())
        if total_outliers > 0:
            issues["warnings"].append(f"Total outliers detected: {total_outliers}")
        
    except Exception as e:
        issues["error"] = str(e)
    
    return issues


def suggest_datetime_format(sample_values: pd.Series, max_samples: int = 10) -> Optional[str]:
    """
    Suggest datetime format based on sample values
    
    Args:
        sample_values: Sample of datetime strings
        max_samples: Maximum number of samples to examine
    
    Returns:
        str: Suggested format or None
    """
    from core.config import get_config
    formats = get_config()["datetime_formats"]
    
    # Get non-null samples
    samples = sample_values.dropna().head(max_samples)
    
    if len(samples) == 0:
        return None
    
    # Try each format
    for fmt in formats:
        if fmt == "ISO8601":
            continue
        try:
            # Try to parse all samples
            for sample in samples:
                datetime.strptime(str(sample), fmt)
            return fmt  # If all succeeded, return this format
        except:
            continue
    
    return None
