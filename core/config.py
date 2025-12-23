"""
Core Configuration Module
Handles application configuration and session state initialization
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load application configuration
    
    Returns:
        dict: Configuration dictionary with all app settings
    """
    config = {
        # Application metadata
        "app_name": "ExplainFutures",
        "version": "1.0.0-phase1",
        "phase": "Phase 1: MVP - Data & Visualization",
        
        # File upload settings
        "max_file_size_mb": 200,
        "accepted_file_types": ["csv", "txt", "xlsx", "xls"],
        
        # Data parsing settings
        "datetime_formats": [
            "%Y-%m-%d",
            "%Y/%m/%d", 
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y",
            "%d.%m.%Y",
            "%Y%m%d",
            "ISO8601"  # pandas default
        ],
        "common_delimiters": [",", ";", "\t", "|"],
        "decimal_separators": [".", ","],
        
        # Visualization settings
        "default_plot_height": 500,
        "default_color_palette": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "#393b79", "#637939"
        ],
        "plot_template": "plotly_white",
        
        # Data health thresholds
        "missing_threshold_warn": 0.05,  # 5% - warning level
        "missing_threshold_critical": 0.20,  # 20% - critical level
        "outlier_std_threshold": 3.0,  # Standard deviations for outlier detection
        "min_data_points": 10,  # Minimum points required per variable
        
        # Preprocessing defaults
        "default_interpolation": "linear",
        "default_outlier_method": "clip",
        "resample_methods": ["mean", "median", "sum", "min", "max", "first", "last"],
        
        # Long format standard columns
        "long_format_columns": {
            "timestamp": "timestamp",
            "variable": "variable", 
            "value": "value"
        }
    }
    
    return config


def initialize_session_state():
    """
    Initialize Streamlit session state with default values
    
    Ensures all required session state keys exist with sensible defaults.
    This is called once at app startup.
    """
    
    # === PROJECT CONTEXT ===
    if "project_id" not in st.session_state:
        st.session_state.project_id = None
    
    if "project_name" not in st.session_state:
        st.session_state.project_name = "Unnamed Project"
    
    # === DATA STATE ===
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    
    if "df_long" not in st.session_state:
        st.session_state.df_long = None
    
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    
    # === METADATA ===
    if "time_column" not in st.session_state:
        st.session_state.time_column = None
    
    if "value_columns" not in st.session_state:
        st.session_state.value_columns = []
    
    if "variable_dtypes" not in st.session_state:
        st.session_state.variable_dtypes = {}
    
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = {}
    
    # === DATA HEALTH ===
    if "health_report" not in st.session_state:
        st.session_state.health_report = None
    
    # === PREPROCESSING STATE ===
    if "preprocessing_applied" not in st.session_state:
        st.session_state.preprocessing_applied = False
    
    if "preprocessing_config" not in st.session_state:
        st.session_state.preprocessing_config = {}
    
    # === UPLOAD STATE ===
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    if "upload_timestamp" not in st.session_state:
        st.session_state.upload_timestamp = None
    
    # === DATASET TRACKING (for future Supabase integration) ===
    if "dataset_id" not in st.session_state:
        st.session_state.dataset_id = None
    
    # === NAVIGATION ===
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"


def get_temp_dir() -> Path:
    """
    Get or create temporary directory for file processing
    
    Returns:
        Path: Path to temporary directory
    """
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_exports_dir() -> Path:
    """
    Get or create exports directory for user downloads
    
    Returns:
        Path: Path to exports directory
    """
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    return exports_dir


# Configuration singleton
_config = None

def get_config() -> Dict[str, Any]:
    """
    Get cached configuration (singleton pattern)
    
    Returns:
        dict: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
