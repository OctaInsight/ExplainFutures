"""
Parameter Unification Module
Normalizes and unifies parameter names across different variations
"""

import re
from typing import Dict, List, Optional
from difflib import SequenceMatcher


# Comprehensive synonym groups for parameter unification
PARAMETER_SYNONYMS = {
    'gdp': {
        'canonical': 'GDP',
        'variations': [
            'gdp', 'GDP', 'Gdp',
            'gross domestic product',
            'Gross Domestic Product',
            'economic output',
            'economic growth',
            'economy',
            'national income'
        ]
    },
    
    'emissions': {
        'canonical': 'CO2 emissions',
        'variations': [
            'co2', 'CO2', 'Co2',
            'co2 emissions', 'CO2 emissions',
            'carbon dioxide', 'Carbon dioxide',
            'carbon dioxide emissions',
            'Carbon dioxide emissions',
            'carbon emissions',
            'Carbon emissions',
            'emissions', 'Emissions',
            'greenhouse gas', 'GHG',
            'carbon'
        ]
    },
    
    'renewable_energy': {
        'canonical': 'Renewable energy',
        'variations': [
            'renewable', 'renewables',
            'Renewable', 'Renewables',
            'renewable energy', 'Renewable energy',
            'renewable energy share',
            'clean energy', 'Clean energy',
            'green energy', 'Green energy',
            'sustainable energy',
            'solar and wind',
            're share'
        ]
    },
    
    'employment': {
        'canonical': 'Employment',
        'variations': [
            'employment', 'Employment',
            'jobs', 'Jobs',
            'labor', 'labour',
            'Labor', 'Labour',
            'workforce', 'Workforce',
            'labor force',
            'employment rate',
            'job creation'
        ]
    },
    
    'investment': {
        'canonical': 'Investment',
        'variations': [
            'investment', 'Investment',
            'investments', 'Investments',
            'capital', 'Capital',
            'capital investment',
            'public investment',
            'private investment',
            'funding'
        ]
    },
    
    'energy': {
        'canonical': 'Energy',
        'variations': [
            'energy', 'Energy',
            'power', 'Power',
            'electricity', 'Electricity',
            'energy consumption',
            'energy demand',
            'power generation'
        ]
    },
    
    'population': {
        'canonical': 'Population',
        'variations': [
            'population', 'Population',
            'people', 'People',
            'inhabitants',
            'demographic',
            'demographics'
        ]
    },
    
    'productivity': {
        'canonical': 'Productivity',
        'variations': [
            'productivity', 'Productivity',
            'efficiency', 'Efficiency',
            'output per worker',
            'labor productivity',
            'labour productivity'
        ]
    },
    
    'income': {
        'canonical': 'Income',
        'variations': [
            'income', 'Income',
            'earnings', 'Earnings',
            'wages', 'Wages',
            'salary', 'Salary',
            'household income',
            'disposable income',
            'per capita income'
        ]
    },
    
    'innovation': {
        'canonical': 'Innovation',
        'variations': [
            'innovation', 'Innovation',
            'r&d', 'R&D',
            'research', 'Research',
            'research and development',
            'technology',
            'technological progress'
        ]
    }
}


def unify_parameter(parameter: str) -> Dict[str, str]:
    """
    Unify parameter name to canonical form
    
    Parameters:
    -----------
    parameter : str
        Raw parameter name from extraction
        
    Returns:
    --------
    unified : dict
        {
            'original': str,
            'canonical': str,
            'category': str,
            'confidence': float
        }
    """
    
    param_lower = parameter.lower().strip()
    
    # Direct match check
    for category, data in PARAMETER_SYNONYMS.items():
        for variation in data['variations']:
            if param_lower == variation.lower():
                return {
                    'original': parameter,
                    'canonical': data['canonical'],
                    'category': category,
                    'confidence': 1.0
                }
    
    # Partial match check (contains)
    best_match = None
    best_score = 0.0
    
    for category, data in PARAMETER_SYNONYMS.items():
        for variation in data['variations']:
            # Check if variation is in parameter or parameter is in variation
            if variation.lower() in param_lower or param_lower in variation.lower():
                score = len(variation) / max(len(param_lower), len(variation))
                if score > best_score:
                    best_score = score
                    best_match = {
                        'original': parameter,
                        'canonical': data['canonical'],
                        'category': category,
                        'confidence': score
                    }
    
    if best_match and best_score > 0.6:
        return best_match
    
    # Fuzzy match check
    for category, data in PARAMETER_SYNONYMS.items():
        for variation in data['variations']:
            similarity = SequenceMatcher(None, param_lower, variation.lower()).ratio()
            if similarity > 0.85 and similarity > best_score:
                best_score = similarity
                best_match = {
                    'original': parameter,
                    'canonical': data['canonical'],
                    'category': category,
                    'confidence': similarity
                }
    
    if best_match:
        return best_match
    
    # No match found - return original with low confidence
    return {
        'original': parameter,
        'canonical': parameter,
        'category': 'other',
        'confidence': 0.3
    }


def unify_extracted_items(items: List[Dict]) -> List[Dict]:
    """
    Unify all parameters in extracted items
    
    Parameters:
    -----------
    items : List[Dict]
        List of extracted items with 'parameter' field
        
    Returns:
    --------
    unified_items : List[Dict]
        Items with added 'parameter_canonical' field
    """
    
    for item in items:
        if 'parameter' in item:
            unified = unify_parameter(item['parameter'])
            item['parameter_original'] = item['parameter']
            item['parameter_canonical'] = unified['canonical']
            item['parameter_category'] = unified['category']
            item['unification_confidence'] = unified['confidence']
    
    return items


def deduplicate_items(items: List[Dict]) -> List[Dict]:
    """
    Remove duplicate parameters, keeping best match
    
    Parameters:
    -----------
    items : List[Dict]
        List of extracted items
        
    Returns:
    --------
    deduped : List[Dict]
        Deduplicated items
    """
    
    # Group by canonical parameter
    grouped = {}
    
    for item in items:
        canonical = item.get('parameter_canonical', item.get('parameter', ''))
        
        if canonical not in grouped:
            grouped[canonical] = item
        else:
            # Keep the one with higher confidence
            existing = grouped[canonical]
            
            # Priority: 1) has value, 2) higher confidence, 3) longer source
            item_priority = (
                item.get('value') is not None,
                item.get('confidence', 0),
                len(item.get('source_sentence', ''))
            )
            
            existing_priority = (
                existing.get('value') is not None,
                existing.get('confidence', 0),
                len(existing.get('source_sentence', ''))
            )
            
            if item_priority > existing_priority:
                grouped[canonical] = item
    
    return list(grouped.values())


def merge_similar_parameters(items: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    Merge parameters that are very similar
    
    Parameters:
    -----------
    items : List[Dict]
        Extracted items
    similarity_threshold : float
        Minimum similarity to merge (0-1)
        
    Returns:
    --------
    merged : List[Dict]
        Merged items
    """
    
    if not items:
        return items
    
    merged = []
    used_indices = set()
    
    for i, item1 in enumerate(items):
        if i in used_indices:
            continue
        
        # Find similar items
        similar_group = [item1]
        
        for j, item2 in enumerate(items[i+1:], start=i+1):
            if j in used_indices:
                continue
            
            param1 = item1.get('parameter_canonical', item1.get('parameter', '')).lower()
            param2 = item2.get('parameter_canonical', item2.get('parameter', '')).lower()
            
            similarity = SequenceMatcher(None, param1, param2).ratio()
            
            if similarity >= similarity_threshold:
                similar_group.append(item2)
                used_indices.add(j)
        
        # Merge similar group
        if len(similar_group) == 1:
            merged.append(item1)
        else:
            # Take the best item from group
            best = max(similar_group, key=lambda x: (
                x.get('value') is not None,
                x.get('confidence', 0),
                x.get('unification_confidence', 0)
            ))
            merged.append(best)
        
        used_indices.add(i)
    
    return merged


def add_user_synonym(canonical_name: str, new_variation: str):
    """
    Add user-defined synonym to the system
    
    Parameters:
    -----------
    canonical_name : str
        The canonical parameter name
    new_variation : str
        New variation to add
    """
    
    # Find category
    for category, data in PARAMETER_SYNONYMS.items():
        if data['canonical'] == canonical_name:
            if new_variation not in data['variations']:
                data['variations'].append(new_variation)
            return True
    
    # Create new category if not exists
    category_key = canonical_name.lower().replace(' ', '_')
    PARAMETER_SYNONYMS[category_key] = {
        'canonical': canonical_name,
        'variations': [canonical_name, new_variation]
    }
    
    return True


def get_all_canonical_parameters() -> List[str]:
    """Get list of all canonical parameter names"""
    return [data['canonical'] for data in PARAMETER_SYNONYMS.values()]


def get_variations(canonical_name: str) -> List[str]:
    """Get all variations of a canonical parameter"""
    for data in PARAMETER_SYNONYMS.values():
        if data['canonical'] == canonical_name:
            return data['variations']
    return []
