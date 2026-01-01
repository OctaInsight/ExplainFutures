"""
Value parsing module
Parse and normalize numeric values, percentages, and units

PHASE 2 ENHANCEMENTS:
- Detect value semantics: absolute_target, delta, range, rate, percent_point
- Parse ranges (10-15%, between X and Y)
- Parse rates (CAGR 2%, 5% per year)
- Parse "from A to B" expressions
- Enhanced time extraction
"""

import re
from typing import Optional, Tuple, Dict, List


def parse_value_string(value_str: str, context: Optional[Dict] = None) -> Dict:
    """
    Parse a value string and extract number, unit, type, and semantics
    
    PHASE 2: Enhanced with semantic classification
    
    Parameters:
    -----------
    value_str : str
        String like "20%", "3.2 billion", "5 MtCO2", "10-15%", "2% per year"
    context : dict, optional
        Context about surrounding text (direction words, etc.)
        
    Returns:
    --------
    parsed : dict
        {
            'value': float,
            'unit': str,
            'value_type': str ('absolute_target', 'delta', 'range', 'rate', 'percent_point'),
            'value_min': float (for ranges),
            'value_max': float (for ranges),
            'is_range': bool,
            'is_rate': bool,
            'rate_period': str
        }
    """
    value_str = value_str.strip()
    
    # PHASE 2: Try range patterns first
    range_result = parse_range(value_str)
    if range_result['is_range']:
        return range_result
    
    # PHASE 2: Try rate patterns
    rate_result = parse_rate(value_str)
    if rate_result['is_rate']:
        return rate_result
    
    # PHASE 2: Try "from X to Y" pattern
    from_to_result = parse_from_to(value_str)
    if from_to_result['is_from_to']:
        return from_to_result
    
    # PHASE 2: Try percentage point pattern
    pp_result = parse_percentage_point(value_str)
    if pp_result['is_percent_point']:
        return pp_result
    
    # Try percent
    percent_match = re.match(r'([+-]?\d+\.?\d*)\s*(%|percent|pct)', value_str, re.IGNORECASE)
    if percent_match:
        # Determine if it's a delta or target based on context
        value_type = 'delta' if context and context.get('has_change_word') else 'absolute_target'
        
        return {
            'value': float(percent_match.group(1)),
            'unit': '%',
            'value_type': value_type,
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': False,
            'rate_period': ''
        }
    
    # Try number with unit
    unit_match = re.match(r'(\d+\.?\d*)\s*(billion|million|thousand|trillion|mtco2|gtco2|gw|twh|kt|mt|gt)', 
                          value_str, re.IGNORECASE)
    if unit_match:
        value_type = 'delta' if context and context.get('has_change_word') else 'absolute_target'
        
        return {
            'value': float(unit_match.group(1)),
            'unit': unit_match.group(2).lower(),
            'value_type': value_type,
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': False,
            'rate_period': ''
        }
    
    # Try plain number
    number_match = re.match(r'([+-]?\d+\.?\d*)', value_str)
    if number_match:
        return {
            'value': float(number_match.group(1)),
            'unit': 'absolute',
            'value_type': 'absolute_target',
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': False,
            'rate_period': ''
        }
    
    # Could not parse
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown',
        'value_min': None,
        'value_max': None,
        'is_range': False,
        'is_rate': False,
        'rate_period': ''
    }


def parse_range(range_str: str) -> Dict:
    """
    PHASE 2: Parse range strings like "10-20%", "5 to 10 billion", "between 10 and 15%"
    
    Parameters:
    -----------
    range_str : str
        Range string
        
    Returns:
    --------
    parsed : dict with is_range=True if valid range
    """
    # Pattern 1: "10-20" or "10 to 20" with optional unit
    pattern1 = r'(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)\s*(%|percent|billion|million|thousand|mtco2|gw|twh)?'
    match = re.search(pattern1, range_str, re.IGNORECASE)
    
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        unit = match.group(3).lower() if match.group(3) else 'absolute'
        unit = '%' if unit in ['%', 'percent'] else unit
        
        return {
            'value': (min_val + max_val) / 2,  # Midpoint as primary value
            'unit': unit,
            'value_type': 'range',
            'value_min': min_val,
            'value_max': max_val,
            'is_range': True,
            'is_rate': False,
            'rate_period': ''
        }
    
    # Pattern 2: "between X and Y"
    pattern2 = r'between\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)\s*(%|percent|billion|million)?'
    match = re.search(pattern2, range_str, re.IGNORECASE)
    
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        unit = match.group(3).lower() if match.group(3) else 'absolute'
        unit = '%' if unit in ['%', 'percent'] else unit
        
        return {
            'value': (min_val + max_val) / 2,
            'unit': unit,
            'value_type': 'range',
            'value_min': min_val,
            'value_max': max_val,
            'is_range': True,
            'is_rate': False,
            'rate_period': ''
        }
    
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown',
        'value_min': None,
        'value_max': None,
        'is_range': False,
        'is_rate': False,
        'rate_period': ''
    }


def parse_rate(rate_str: str) -> Dict:
    """
    PHASE 2: Parse rate strings like "2% per year", "CAGR 3%", "5% annually"
    
    Parameters:
    -----------
    rate_str : str
        Rate string
        
    Returns:
    --------
    parsed : dict with is_rate=True if valid rate
    """
    # Pattern 1: "X% per year/annually"
    pattern1 = r'(\d+\.?\d*)\s*%?\s*(?:per year|per annum|annually|yearly|/year|/yr)'
    match = re.search(pattern1, rate_str, re.IGNORECASE)
    
    if match:
        return {
            'value': float(match.group(1)),
            'unit': '%',
            'value_type': 'rate',
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': True,
            'rate_period': 'per year'
        }
    
    # Pattern 2: "CAGR X%"
    pattern2 = r'cagr\s+(\d+\.?\d*)\s*%?'
    match = re.search(pattern2, rate_str, re.IGNORECASE)
    
    if match:
        return {
            'value': float(match.group(1)),
            'unit': '%',
            'value_type': 'rate',
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': True,
            'rate_period': 'CAGR'
        }
    
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown',
        'value_min': None,
        'value_max': None,
        'is_range': False,
        'is_rate': False,
        'rate_period': ''
    }


def parse_from_to(from_to_str: str) -> Dict:
    """
    PHASE 2: Parse "from X to Y" expressions
    
    Parameters:
    -----------
    from_to_str : str
        String like "from 10% to 25%", "from 5 to 10 billion"
        
    Returns:
    --------
    parsed : dict with is_from_to=True and base_value/target_value
    """
    # Pattern: "from X to Y"
    pattern = r'from\s+(\d+\.?\d*)\s*(%|percent|billion|million)?\s+to\s+(\d+\.?\d*)\s*(%|percent|billion|million)?'
    match = re.search(pattern, from_to_str, re.IGNORECASE)
    
    if match:
        base_val = float(match.group(1))
        target_val = float(match.group(3))
        unit = match.group(2) or match.group(4)
        unit = '%' if unit and unit.lower() in ['%', 'percent'] else (unit.lower() if unit else 'absolute')
        
        # Calculate delta
        delta = target_val - base_val
        
        return {
            'value': delta,
            'unit': unit,
            'value_type': 'delta',
            'value_min': None,
            'value_max': None,
            'base_value': base_val,
            'target_value': target_val,
            'is_range': False,
            'is_rate': False,
            'rate_period': '',
            'is_from_to': True
        }
    
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown',
        'value_min': None,
        'value_max': None,
        'is_range': False,
        'is_rate': False,
        'rate_period': '',
        'is_from_to': False
    }


def parse_percentage_point(pp_str: str) -> Dict:
    """
    PHASE 2: Parse percentage point expressions like "10 percentage points", "5 pp"
    
    Parameters:
    -----------
    pp_str : str
        String with percentage point notation
        
    Returns:
    --------
    parsed : dict with value_type='percent_point'
    """
    # Pattern: "X percentage points" or "X pp"
    pattern = r'(\d+\.?\d*)\s*(?:percentage points?|pp)'
    match = re.search(pattern, pp_str, re.IGNORECASE)
    
    if match:
        return {
            'value': float(match.group(1)),
            'unit': 'pp',
            'value_type': 'percent_point',
            'value_min': None,
            'value_max': None,
            'is_range': False,
            'is_rate': False,
            'rate_period': '',
            'is_percent_point': True
        }
    
    return {
        'value': None,
        'unit': '',
        'value_type': 'unknown',
        'value_min': None,
        'value_max': None,
        'is_range': False,
        'is_rate': False,
        'rate_period': '',
        'is_percent_point': False
    }


def extract_time_expressions(text: str) -> List[Dict]:
    """
    PHASE 2: Extract time/horizon expressions from text
    
    Parameters:
    -----------
    text : str
        Text to extract from
        
    Returns:
    --------
    time_expressions : List[Dict]
        List of extracted time expressions with:
        {
            'expression': str (raw text),
            'baseline_year': int,
            'target_year': int,
            'confidence': float
        }
    """
    expressions = []
    
    # Pattern 1: "by YEAR" or "in YEAR"
    pattern1 = r'\b(by|in|until|to)\s+(20\d{2}|21\d{2})\b'
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        year = int(match.group(2))
        expressions.append({
            'expression': match.group(0),
            'baseline_year': None,
            'target_year': year,
            'confidence': 0.9,
            'position': match.start()
        })
    
    # Pattern 2: "from YEAR to YEAR" or "YEAR-YEAR"
    pattern2 = r'\b(from\s+)?(20\d{2}|21\d{2})\s*(?:to|-|–)\s*(20\d{2}|21\d{2})\b'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        start_year = int(match.group(2))
        end_year = int(match.group(3))
        expressions.append({
            'expression': match.group(0),
            'baseline_year': start_year,
            'target_year': end_year,
            'confidence': 0.95,
            'position': match.start()
        })
    
    # Pattern 3: "mid-century", "end of century"
    pattern3 = r'\b(mid-century|end of (?:the )?century)\b'
    for match in re.finditer(pattern3, text, re.IGNORECASE):
        expr_lower = match.group(0).lower()
        if 'mid' in expr_lower:
            year = 2050
        else:
            year = 2100
        
        expressions.append({
            'expression': match.group(0),
            'baseline_year': None,
            'target_year': year,
            'confidence': 0.7,
            'position': match.start()
        })
    
    # Pattern 4: Standalone year (lower confidence)
    pattern4 = r'\b(20[2-9][0-9]|21[0-9][0-9])\b'
    for match in re.finditer(pattern4, text):
        year = int(match.group(0))
        # Only add if not already captured by other patterns
        if not any(abs(e['position'] - match.start()) < 20 for e in expressions):
            expressions.append({
                'expression': match.group(0),
                'baseline_year': None,
                'target_year': year,
                'confidence': 0.6,
                'position': match.start()
            })
    
    return expressions


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
    """
    unit = unit.lower().strip()
    
    # Percent variations
    if unit in ['%', 'percent', 'pct', 'percentage']:
        return '%'
    
    # PHASE 2: Percentage points
    if unit in ['pp', 'percentage point', 'percentage points']:
        return 'pp'
    
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


def format_value(value: float, unit: str, value_type: Optional[str] = None) -> str:
    """
    Format value with unit for display
    
    PHASE 2: Enhanced with value_type awareness
    
    Parameters:
    -----------
    value : float
        Numeric value
    unit : str
        Unit string
    value_type : str, optional
        Type of value (for context)
        
    Returns:
    --------
    formatted : str
        Formatted string like "3.2 billion" or "20%"
    """
    # PHASE 2: Add type prefix for clarity
    prefix = ""
    if value_type == 'rate':
        prefix = "CAGR "
    elif value_type == 'delta':
        prefix = "Δ"
    
    if unit == '%':
        return f"{prefix}{value:.1f}%"
    elif unit == 'pp':
        return f"{value:.1f} pp"
    elif unit in ['billion', 'million', 'thousand', 'trillion']:
        return f"{prefix}{value:.2f} {unit}"
    elif unit in ['MtCO2', 'GtCO2', 'ktCO2']:
        return f"{prefix}{value:.2f} {unit}"
    elif unit in ['GW', 'MW', 'TWh', 'GWh']:
        return f"{prefix}{value:.2f} {unit}"
    else:
        return f"{prefix}{value:.2f}"
