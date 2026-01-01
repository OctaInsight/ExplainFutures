"""
Mapping module
Map scenario parameters to dataset variables using similarity matching

PHASE 2 ENHANCEMENTS:
- Token-based similarity (lemmatized tokens)
- Category-aware mapping
- User synonym feedback integration
"""

from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re


def suggest_variable_mapping(
    parameter: str, 
    available_variables: List[str], 
    top_n: int = 5,
    parameter_category: Optional[str] = None,
    variable_categories: Optional[Dict[str, str]] = None,
    user_synonyms: Optional[Dict[str, List[str]]] = None
) -> List[Dict]:
    """
    Suggest dataset variables that match a scenario parameter
    
    PHASE 2: Enhanced with token-based similarity and category awareness
    
    Uses:
    - Token-based similarity (lemmatized)
    - String similarity
    - Synonym matching
    - Common abbreviations
    - Category matching
    
    Parameters:
    -----------
    parameter : str
        Scenario parameter name
    available_variables : List[str]
        List of available variable names in dataset
    top_n : int
        Number of top suggestions to return
    parameter_category : str, optional
        Category of the parameter (economy, environment, social)
    variable_categories : dict, optional
        Mapping of variable name to category
    user_synonyms : dict, optional
        User-provided synonym dictionary from unification module
        
    Returns:
    --------
    suggestions : List[Dict]
        List of suggestions with:
        {
            'variable': str,
            'similarity': float (0-1),
            'match_reason': str
        }
    """
    suggestions = []
    
    parameter_lower = parameter.lower().strip()
    parameter_tokens = tokenize_and_lemmatize(parameter_lower)
    
    for var in available_variables:
        var_lower = var.lower().strip()
        var_tokens = tokenize_and_lemmatize(var_lower)
        
        # Calculate base similarity
        similarity = calculate_similarity(parameter_lower, var_lower)
        match_reason = "string similarity"
        
        # PHASE 2: Token-based similarity (often better for multi-word terms)
        token_sim = token_similarity(parameter_tokens, var_tokens)
        if token_sim > similarity:
            similarity = token_sim
            match_reason = "token match"
        
        # PHASE 2: Category boost
        if parameter_category and variable_categories:
            var_category = variable_categories.get(var)
            if var_category and var_category == parameter_category:
                similarity += 0.15
                match_reason = f"category match ({parameter_category})"
        
        # Boost for synonyms (system)
        if are_synonyms(parameter_lower, var_lower):
            similarity += 0.2
            match_reason = "synonym match"
        
        # PHASE 2: User synonym boost
        if user_synonyms and parameter_lower in user_synonyms:
            if var_lower in [s.lower() for s in user_synonyms[parameter_lower]]:
                similarity += 0.25
                match_reason = "user synonym"
        
        # Boost for abbreviations
        if is_abbreviation(parameter_lower, var_lower):
            similarity += 0.15
            match_reason = "abbreviation match"
        
        # Boost for exact match
        if parameter_lower == var_lower:
            similarity = 1.0
            match_reason = "exact match"
        
        # Cap at 1.0
        similarity = min(similarity, 1.0)
        
        suggestions.append({
            'variable': var,
            'similarity': similarity,
            'match_reason': match_reason
        })
    
    # Sort by similarity
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    
    return suggestions[:top_n]


def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    PHASE 2: Tokenize and lemmatize text for better matching
    
    Parameters:
    -----------
    text : str
        Text to tokenize
        
    Returns:
    --------
    tokens : List[str]
        List of lemmatized tokens
    """
    # Simple lemmatization rules (without requiring NLTK)
    lemma_rules = {
        'emissions': 'emission',
        'countries': 'country',
        'energies': 'energy',
        'economies': 'economy',
        'populations': 'population',
        'investments': 'investment',
        'technologies': 'technology',
        'policies': 'policy',
        'rates': 'rate',
        'levels': 'level',
        'shares': 'share',
        'values': 'value',
        'changes': 'change',
        'increases': 'increase',
        'decreases': 'decrease'
    }
    
    # Tokenize (split on whitespace and special chars)
    tokens = re.findall(r'\w+', text.lower())
    
    # Apply simple lemmatization
    lemmatized = []
    for token in tokens:
        # Check if plural form exists in rules
        lemmatized.append(lemma_rules.get(token, token))
    
    return lemmatized


def token_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    PHASE 2: Calculate similarity based on token overlap
    
    Parameters:
    -----------
    tokens1, tokens2 : List[str]
        Lists of tokens to compare
        
    Returns:
    --------
    similarity : float (0-1)
    """
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


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
        {'gdp', 'gross domestic product', 'economic output', 'economy', 'national income'},
        {'co2', 'carbon dioxide', 'emissions', 'carbon emissions', 'co2 emissions', 'ghg', 'greenhouse gas'},
        {'renewable', 'renewables', 'renewable energy', 'clean energy', 're', 'green energy'},
        {'population', 'pop', 'inhabitants', 'people', 'demographic'},
        {'temperature', 'temp', 'warming', 'global temperature', 'climate'},
        {'energy', 'power', 'electricity'},
        {'consumption', 'demand', 'usage', 'use'},
        {'production', 'output', 'generation', 'supply'},
        {'investment', 'capital', 'funding', 'finance'},
        {'employment', 'jobs', 'labor', 'labour', 'workforce'},
        {'inequality', 'gini', 'disparity', 'income inequality'},
        {'productivity', 'efficiency', 'output per worker'}
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


def get_category_for_parameter(parameter: str) -> str:
    """
    PHASE 2: Infer category for a parameter based on keywords
    
    Parameters:
    -----------
    parameter : str
        Parameter name
        
    Returns:
    --------
    category : str
        'economy', 'environment', 'social', or 'other'
    """
    param_lower = parameter.lower()
    
    # Economic indicators
    economic_keywords = ['gdp', 'income', 'economic', 'investment', 'capital', 'finance', 
                        'productivity', 'employment', 'job', 'wage', 'salary', 'trade', 'export', 'import']
    
    # Environmental indicators
    environmental_keywords = ['emission', 'co2', 'carbon', 'climate', 'temperature', 'energy',
                             'renewable', 'pollution', 'waste', 'water', 'air', 'environmental',
                             'forest', 'biodiversity', 'ecosystem', 'ghg', 'methane']
    
    # Social indicators
    social_keywords = ['population', 'health', 'education', 'inequality', 'poverty', 'social',
                      'welfare', 'cohesion', 'access', 'literacy', 'mortality', 'life expectancy',
                      'demographic', 'urban', 'rural', 'housing']
    
    # Check each category
    if any(kw in param_lower for kw in economic_keywords):
        return 'economy'
    elif any(kw in param_lower for kw in environmental_keywords):
        return 'environment'
    elif any(kw in param_lower for kw in social_keywords):
        return 'social'
    else:
        return 'other'
