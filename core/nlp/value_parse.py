"""
Value parsing module
Parse and normalize numeric values, percentages, and units
"""

import re
from typing import Optional, Tuple, Dict


def parse_value_string(value_str: str) -> Dict:
    """
    Parse a value string and extract number, unit, type
    
    Parameters:
    -----------
    value_str : str
        String like "20%", "3.2 billion", "5 MtCO2"
        
    Returns:
    --------
    parsed : dict
        {
            'value': float,
            'unit': str,
            'value_type': str ('percent', 'absolute')
        }
    """
    value_str = value_str.strip()
    
    # Try percent
    percent_match = re.match(r'([+-]?\d+\.?\d*)\s*(%|percent|pct)', value_str, re.IGNORECASE)
    if percent_match:
        return {
            'value': float(percent_match.group(1)),
            'unit': '%',
            'value_type': 'percent'
        }
    
    # Try number with unit
    unit_match = re.match(r'(\d+\.?\d*)\s*(billion|million|thousand|trillion|mtco2|gw|twh|kt|mt|gt)', 
                          value_str, re.IGNORECASE)
    if unit_match:
        return {
            'value': float(unit_match.group(1)),
            'unit': unit_match.group(2).lower(),
            'value_type': 'absolute'
        }
    
    # Try plain number
    number_match = re.match(r'([+-]?\d+\.?\d*)', value_str)
    if number_match:
        return {
            'value': float(number_match.group(1)),
            'unit': 'absolute',
            'value_type': 'absolute'
        }
    
    # Could not parse
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown'
    }


def normalize_unit(unit: str) -> str:
    """
    Normalize unit string to standard form
    
    Parameters:
    -----------
    unit : str
        Unit string (may be non-standard)
        
    Returns:
    --------
    normalized : str
        Standardized unit
    """
    unit = unit.lower().strip()
    
    # Percent variations
    if unit in ['%', 'percent', 'pct', 'percentage']:
        return '%'
    
    # Large numbers
    if unit in ['bn', 'b']:
        return 'billion'
    if unit in ['mn', 'm']:
        return 'million'
    if unit in ['k', 'th']:
        return 'thousand'
    if unit in ['tn', 't']:
        return 'trillion'
    
    # Emissions
    if unit in ['mtco2', 'mtco₂', 'mt co2']:
        return 'MtCO2'
    if unit in ['gtco2', 'gtco₂', 'gt co2']:
        return 'GtCO2'
    if unit in ['ktco2', 'ktco₂', 'kt co2']:
        return 'ktCO2'
    
    # Energy
    if unit in ['gw', 'gigawatt', 'gigawatts']:
        return 'GW'
    if unit in ['mw', 'megawatt', 'megawatts']:
        return 'MW'
    if unit in ['twh', 'terawatt-hour', 'terawatt-hours']:
        return 'TWh'
    if unit in ['gwh', 'gigawatt-hour', 'gigawatt-hours']:
        return 'GWh'
    
    return unit


def convert_value(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert value from one unit to another
    
    Parameters:
    -----------
    value : float
        Value to convert
    from_unit : str
        Source unit
    to_unit : str
        Target unit
        
    Returns:
    --------
    converted : float or None
        Converted value, or None if conversion not possible
    """
    # Normalize units
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)
    
    # If same unit, no conversion needed
    if from_unit == to_unit:
        return value
    
    # Define conversion factors (relative to base unit)
    # For numbers: base is 1
    number_conversions = {
        'thousand': 1e3,
        'million': 1e6,
        'billion': 1e9,
        'trillion': 1e12
    }
    
    # For emissions: base is tons
    emission_conversions = {
        'ktCO2': 1e3,
        'MtCO2': 1e6,
        'GtCO2': 1e9
    }
    
    # For energy: base is Wh
    energy_conversions = {
        'GWh': 1e9,
        'TWh': 1e12
    }
    
    # Try number conversions
    if from_unit in number_conversions and to_unit in number_conversions:
        return value * (number_conversions[from_unit] / number_conversions[to_unit])
    
    # Try emission conversions
    if from_unit in emission_conversions and to_unit in emission_conversions:
        return value * (emission_conversions[from_unit] / emission_conversions[to_unit])
    
    # Try energy conversions
    if from_unit in energy_conversions and to_unit in energy_conversions:
        return value * (energy_conversions[from_unit] / energy_conversions[to_unit])
    
    # Conversion not possible
    return None


def format_value(value: float, unit: str) -> str:
    """
    Format value with unit for display
    
    Parameters:
    -----------
    value : float
        Numeric value
    unit : str
        Unit string
        
    Returns:
    --------
    formatted : str
        Formatted string like "3.2 billion" or "20%"
    """
    if unit == '%':
        return f"{value:.1f}%"
    elif unit in ['billion', 'million', 'thousand', 'trillion']:
        return f"{value:.2f} {unit}"
    elif unit in ['MtCO2', 'GtCO2', 'ktCO2']:
        return f"{value:.2f} {unit}"
    elif unit in ['GW', 'MW', 'TWh', 'GWh']:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.2f}"


def parse_range(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse range strings like "10-20%", "5 to 10 billion"
    
    Parameters:
    -----------
    range_str : str
        Range string
        
    Returns:
    --------
    (min_value, max_value) : Tuple[float, float] or (None, None)
    """
    # Pattern: "10-20" or "10 to 20"
    pattern = r'(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)'
    
    match = re.search(pattern, range_str)
    
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        return (min_val, max_val)
    
    return (None, None)
