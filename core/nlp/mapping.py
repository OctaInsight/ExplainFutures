"""
Mapping module
Map scenario parameters to dataset variables using similarity matching
"""

from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re


def suggest_variable_mapping(parameter: str, available_variables: List[str], top_n: int = 5) -> List[Dict]:
    """
    Suggest dataset variables that match a scenario parameter
    
    Uses:
    - String similarity
    - Synonym matching
    - Common abbreviations
    
    Parameters:
    -----------
    parameter : str
        Scenario parameter name
    available_variables : List[str]
        List of available variable names in dataset
    top_n : int
        Number of top suggestions to return
        
    Returns:
    --------
    suggestions : List[Dict]
        List of suggestions with:
        {
            'variable': str,
            'similarity': float (0-1)
        }
    """
    suggestions = []
    
    parameter_lower = parameter.lower().strip()
    
    for var in available_variables:
        var_lower = var.lower().strip()
        
        # Calculate base similarity
        similarity = calculate_similarity(parameter_lower, var_lower)
        
        # Boost for synonyms
        if are_synonyms(parameter_lower, var_lower):
            similarity += 0.2
        
        # Boost for abbreviations
        if is_abbreviation(parameter_lower, var_lower):
            similarity += 0.15
        
        # Boost for exact match
        if parameter_lower == var_lower:
            similarity = 1.0
        
        # Cap at 1.0
        similarity = min(similarity, 1.0)
        
        suggestions.append({
            'variable': var,
            'similarity': similarity
        })
    
    # Sort by similarity
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    
    return suggestions[:top_n]


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate string similarity using SequenceMatcher
    
    Parameters:
    -----------
    str1, str2 : str
        Strings to compare
        
    Returns:
    --------
    similarity : float (0-1)
    """
    return SequenceMatcher(None, str1, str2).ratio()


def are_synonyms(str1: str, str2: str) -> bool:
    """
    Check if two strings are synonyms based on predefined dictionary
    
    Parameters:
    -----------
    str1, str2 : str
        Strings to check
        
    Returns:
    --------
    is_synonym : bool
    """
    # Define synonym groups
    synonym_groups = [
        {'gdp', 'gross domestic product', 'economic output', 'economy'},
        {'co2', 'carbon dioxide', 'emissions', 'carbon emissions', 'co2 emissions'},
        {'renewable', 'renewables', 'renewable energy', 'clean energy', 're'},
        {'population', 'pop', 'inhabitants'},
        {'temperature', 'temp', 'warming', 'global temperature'},
        {'energy', 'power', 'electricity'},
        {'consumption', 'demand', 'usage'},
        {'production', 'output', 'generation'},
    ]
    
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    for group in synonym_groups:
        if str1 in group and str2 in group:
            return True
    
    return False


def is_abbreviation(abbr: str, full: str) -> bool:
    """
    Check if one string is an abbreviation of another
    
    Parameters:
    -----------
    abbr : str
        Potential abbreviation
    full : str
        Full string
        
    Returns:
    --------
    is_abbr : bool
    """
    abbr = abbr.lower().strip()
    full = full.lower().strip()
    
    # Check if abbr is in full
    if abbr in full:
        return True
    
    # Check if abbr matches first letters of words in full
    words = re.findall(r'\b\w', full)
    first_letters = ''.join(words)
    
    if abbr == first_letters:
        return True
    
    return False


def create_mapping(parameter: str, variable: str, mapping_type: str = 'direct') -> Dict:
    """
    Create a mapping entry
    
    Parameters:
    -----------
    parameter : str
        Scenario parameter name
    variable : str
        Dataset variable name
    mapping_type : str
        Type of mapping: 'direct', 'derived', 'related'
        
    Returns:
    --------
    mapping : dict
        {
            'parameter': str,
            'variable': str,
            'type': str,
            'equation': str (optional, for derived)
        }
    """
    mapping = {
        'parameter': parameter,
        'variable': variable,
        'type': mapping_type
    }
    
    if mapping_type == 'derived':
        mapping['equation'] = ''  # To be filled by user
    
    return mapping


def validate_mapping(mapping: Dict, available_variables: List[str]) -> bool:
    """
    Validate a mapping
    
    Parameters:
    -----------
    mapping : dict
        Mapping to validate
    available_variables : List[str]
        Available variables in dataset
        
    Returns:
    --------
    is_valid : bool
    """
    # Check if variable exists
    if mapping['variable'] not in available_variables:
        return False
    
    # Check if type is valid
    if mapping['type'] not in ['direct', 'derived', 'related']:
        return False
    
    return True


def apply_mapping(scenario_data: Dict, mappings: Dict[str, str]) -> Dict:
    """
    Apply mappings to scenario data
    
    Parameters:
    -----------
    scenario_data : dict
        Scenario with items
    mappings : dict
        {parameter_name: variable_name}
        
    Returns:
    --------
    mapped_scenario : dict
        Scenario data with added 'mapped_variable' field in items
    """
    mapped_scenario = scenario_data.copy()
    
    for item in mapped_scenario['items']:
        param_name = item['parameter']
        if param_name in mappings:
            item['mapped_variable'] = mappings[param_name]
        else:
            item['mapped_variable'] = None
    
    return mapped_scenario


def get_unmapped_parameters(scenario_data: Dict, mappings: Dict[str, str]) -> List[str]:
    """
    Get list of parameters that haven't been mapped
    
    Parameters:
    -----------
    scenario_data : dict
        Scenario with items
    mappings : dict
        Current mappings
        
    Returns:
    --------
    unmapped : List[str]
        List of parameter names without mappings
    """
    unmapped = []
    
    for item in scenario_data['items']:
        param_name = item['parameter']
        if param_name not in mappings:
            unmapped.append(param_name)
    
    return list(set(unmapped))  # Remove duplicates
