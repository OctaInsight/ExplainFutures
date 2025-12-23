"""
Data Loaders Module
Handles loading of various file formats (CSV, TXT, Excel)
"""

import pandas as pd
import streamlit as st
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import io


def load_csv_file(
    file_content: bytes,
    delimiter: str = ",",
    decimal: str = ".",
    encoding: str = "utf-8"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load CSV file from bytes
    
    Args:
        file_content: File content as bytes
        delimiter: Column delimiter
        decimal: Decimal separator
        encoding: File encoding
    
    Returns:
        tuple: (DataFrame, error_message)
    """
    try:
        # Try to read with specified parameters
        df = pd.read_csv(
            io.BytesIO(file_content),
            delimiter=delimiter,
            decimal=decimal,
            encoding=encoding,
            low_memory=False
        )
        
        if df.empty:
            return None, "File is empty"
        
        return df, None
        
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"


def load_txt_file(
    file_content: bytes,
    delimiter: str = "\t",
    decimal: str = ".",
    encoding: str = "utf-8"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load TXT file from bytes (similar to CSV with different defaults)
    
    Args:
        file_content: File content as bytes
        delimiter: Column delimiter
        decimal: Decimal separator
        encoding: File encoding
    
    Returns:
        tuple: (DataFrame, error_message)
    """
    return load_csv_file(file_content, delimiter, decimal, encoding)


def load_excel_file(
    file_content: bytes,
    sheet_name: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Excel file from bytes
    
    Args:
        file_content: File content as bytes
        sheet_name: Sheet name to load (None = first sheet)
    
    Returns:
        tuple: (DataFrame, error_message)
    """
    try:
        # If no sheet specified, load first sheet
        if sheet_name is None or sheet_name == "":
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)
        else:
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
        
        if df.empty:
            return None, "Sheet is empty"
        
        return df, None
        
    except Exception as e:
        return None, f"Error reading Excel: {str(e)}"


def get_excel_sheet_names(file_content: bytes) -> Tuple[Optional[list], Optional[str]]:
    """
    Get list of sheet names from Excel file
    
    Args:
        file_content: File content as bytes
    
    Returns:
        tuple: (list of sheet names, error_message)
    """
    try:
        xl_file = pd.ExcelFile(io.BytesIO(file_content))
        return xl_file.sheet_names, None
    except Exception as e:
        return None, f"Error reading Excel file: {str(e)}"


def detect_delimiter(file_content: bytes, encoding: str = "utf-8") -> str:
    """
    Auto-detect delimiter in CSV/TXT file
    
    Args:
        file_content: File content as bytes
        encoding: File encoding
    
    Returns:
        str: Detected delimiter
    """
    from core.config import get_config
    
    try:
        # Read first few lines
        text = file_content.decode(encoding)
        lines = text.split('\n')[:5]
        
        # Count occurrences of common delimiters
        delimiters = get_config()["common_delimiters"]
        counts = {delim: 0 for delim in delimiters}
        
        for line in lines:
            for delim in delimiters:
                counts[delim] += line.count(delim)
        
        # Return delimiter with highest count
        if max(counts.values()) > 0:
            return max(counts, key=counts.get)
        
        return ","  # Default fallback
        
    except:
        return ","  # Default fallback


def load_file_smart(
    uploaded_file,
    delimiter: Optional[str] = None,
    decimal: str = ".",
    sheet_name: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, Any]]:
    """
    Smart file loader that handles CSV, TXT, and Excel
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        delimiter: Optional delimiter for CSV/TXT
        decimal: Decimal separator
        sheet_name: Optional sheet name for Excel
    
    Returns:
        tuple: (DataFrame, error_message, metadata_dict)
    """
    metadata = {
        "filename": uploaded_file.name,
        "file_type": None,
        "size_bytes": uploaded_file.size,
        "delimiter": delimiter,
        "decimal": decimal,
        "sheet_name": sheet_name
    }
    
    # Read file content
    file_content = uploaded_file.read()
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Determine file type and load
    if file_extension == ".csv":
        metadata["file_type"] = "CSV"
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = detect_delimiter(file_content)
            metadata["delimiter"] = delimiter
        
        df, error = load_csv_file(file_content, delimiter, decimal)
        
    elif file_extension == ".txt":
        metadata["file_type"] = "TXT"
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = detect_delimiter(file_content)
            metadata["delimiter"] = delimiter
        
        df, error = load_txt_file(file_content, delimiter, decimal)
        
    elif file_extension in [".xlsx", ".xls"]:
        metadata["file_type"] = "Excel"
        df, error = load_excel_file(file_content, sheet_name)
        
    else:
        return None, f"Unsupported file type: {file_extension}", metadata
    
    if error:
        return None, error, metadata
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    metadata["rows"] = len(df)
    metadata["columns"] = len(df.columns)
    metadata["column_names"] = list(df.columns)
    
    return df, None, metadata


def convert_to_long_format(
    df: pd.DataFrame,
    time_column: str,
    value_columns: list
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Convert wide-format DataFrame to standardized long format
    
    Standard format: timestamp, variable, value
    
    Args:
        df: Input DataFrame in wide format
        time_column: Name of the time column
        value_columns: List of value column names
    
    Returns:
        tuple: (long-format DataFrame, error_message)
    """
    try:
        # Verify columns exist
        if time_column not in df.columns:
            return None, f"Time column '{time_column}' not found"
        
        missing_cols = [col for col in value_columns if col not in df.columns]
        if missing_cols:
            return None, f"Value columns not found: {', '.join(missing_cols)}"
        
        # Select only needed columns
        df_subset = df[[time_column] + value_columns].copy()
        
        # Melt to long format
        df_long = pd.melt(
            df_subset,
            id_vars=[time_column],
            value_vars=value_columns,
            var_name='variable',
            value_name='value'
        )
        
        # Rename time column to standard name
        df_long = df_long.rename(columns={time_column: 'timestamp'})
        
        # Sort by timestamp and variable
        df_long = df_long.sort_values(['timestamp', 'variable']).reset_index(drop=True)
        
        return df_long, None
        
    except Exception as e:
        return None, f"Error converting to long format: {str(e)}"


def preview_dataframe(df: pd.DataFrame, max_rows: int = 10) -> None:
    """
    Display a preview of the DataFrame in Streamlit
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum number of rows to show
    """
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First rows:**")
        st.dataframe(df.head(max_rows), use_container_width=True)
    
    with col2:
        st.write("**Column types:**")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dtype) for dtype in df.dtypes]
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
