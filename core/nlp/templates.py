"""
Template-based Extraction Module
Uses predefined templates to extract parameters from common sentence structures
"""

import re
from typing import List, Dict, Optional, Callable


def classify_direction(verb: str) -> str:
    """Classify verb into direction"""
    verb_lower = verb.lower()
    
    if any(word in verb_lower for word in ['increase', 'grow', 'rise', 'expand', 'gain']):
        return 'increase'
    elif any(word in verb_lower for word in ['decrease', 'fall', 'decline', 'drop', 'reduce', 'contract']):
        return 'decrease'
    elif any(word in verb_lower for word in ['reach', 'hit', 'attain', 'achieve']):
        return 'target'
    elif any(word in verb_lower for word in ['remain', 'stable', 'constant', 'unchanged']):
        return 'stable'
    elif 'double' in verb_lower:
        return 'double'
    elif 'halve' in verb_lower or 'half' in verb_lower:
        return 'halve'
    elif 'triple' in verb_lower:
        return 'triple'
    else:
        return 'increase'  # default


# Define extraction templates
TEMPLATES = [
    # Template 1: "X increases/decreases by Y%"
    {
        'name': 'simple_percent_change',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(\w+ing|\w+es?|\w+ed)\s+(?:by\s+)?(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.9
        }
    },
    
    # Template 2: "By YEAR, X has VERB, VERBing by VALUE%"
    {
        'name': 'narrative_with_year',
        'pattern': r'[Bb]y\s+\d{4},\s+([A-Z][A-Za-z0-9\s]+?)\s+(?:has|have)\s+\w+(?:ed|en)?,?\s+(\w+ing)\s+by\s+(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.95
        }
    },
    
    # Template 3: "X reaches/hits Y UNIT"
    {
        'name': 'target_value',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:reaches?|hits?|attains?)\s+(?:around\s+|about\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gtco2|gw|mw|twh|gwh)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': m.group(3).lower(),
            'value_type': 'absolute',
            'confidence': 0.9
        }
    },
    
    # Template 4: "VALUE% increase/decrease in X"
    {
        'name': 'reverse_percent',
        'pattern': r'(?:around\s+|about\s+|approximately\s+|only\s+)?(\d+\.?\d*)\s*(%|percent)\s+(increase|decrease|reduction|growth|decline)\s+(?:in|of)\s+([A-Z][A-Za-z0-9\s]+?)(?:\s|,|\.)',
        'extract': lambda m: {
            'parameter': m.group(4).strip(),
            'direction': classify_direction(m.group(3)),
            'value': float(m.group(1)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.9
        }
    },
    
    # Template 5: "X doubles/halves/triples"
    {
        'name': 'multiplier',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:doubles?|doubled|halves?|halved|triples?|tripled)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': 'double' if 'double' in m.group(0).lower() else 'halve' if 'halve' in m.group(0).lower() else 'triple',
            'value': 100.0 if 'double' in m.group(0).lower() else -50.0 if 'halve' in m.group(0).lower() else 200.0,
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.95
        }
    },
    
    # Template 6: "X of Y UNIT"
    {
        'name': 'of_format',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+of\s+(?:around\s+|about\s+|only\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gtco2|%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': '%' if m.group(3).lower() in ['%', 'percent'] else m.group(3).lower(),
            'value_type': 'percent' if m.group(3).lower() in ['%', 'percent'] else 'absolute',
            'confidence': 0.85
        }
    },
    
    # Template 7: "X: Y%"
    {
        'name': 'colon_format',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?):\s*(\d+\.?\d*)\s*(%|percent|trillion|billion|million|mtco2|gw)?',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': '%' if m.group(3) and m.group(3).lower() in ['%', 'percent'] else m.group(3).lower() if m.group(3) else 'absolute',
            'value_type': 'percent' if m.group(3) and m.group(3).lower() in ['%', 'percent'] else 'absolute',
            'confidence': 0.8
        }
    },
    
    # Template 8: "VERBing by AROUND/APPROXIMATELY VALUE%"
    {
        'name': 'participle_with_qualifier',
        'pattern': r'(\w+ing)\s+by\s+(around|about|approximately|only)\s+(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': 'Value',  # Will be filled by context
            'direction': classify_direction(m.group(1)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.85,
            'needs_context': True
        }
    },
    
    # Template 9: "reduction/increase in X of Y%"
    {
        'name': 'change_in_param',
        'pattern': r'(?:increase|decrease|reduction|growth|decline)\s+in\s+([A-Za-z0-9\s]+?)\s+of\s+(?:only\s+|around\s+|about\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(0)),
            'value': float(m.group(2)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.9
        }
    },
    
    # Template 10: "X continues to VERB, VERBing by Y%"
    {
        'name': 'continues_to_verb',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:continues?|continued)\s+to\s+\w+(?:,|\s+)\s*(\w+ing)\s+by\s+(?:around\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.95
        }
    },
    
    # Template 11: "X modestly/significantly increases by Y%"
    {
        'name': 'adverb_modifier',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:modestly|significantly|steadily|sharply|slightly|dramatically)\s+(\w+s?)\s+(?:by\s+)?(?:around\s+|about\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.9
        }
    },
    
    # Template 12: "as X increases/decreases"
    {
        'name': 'as_clause',
        'pattern': r'as\s+([A-Z][A-Za-z0-9\s]+?)\s+(\w+s?)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': None,
            'unit': '',
            'value_type': 'direction_only',
            'confidence': 0.7
        }
    },
    
    # Template 13: "leading to X of Y%"
    {
        'name': 'leading_to',
        'pattern': r'leading\s+to\s+(?:a\s+)?(?:increase|decrease|reduction|growth)\s+in\s+([A-Za-z0-9\s]+?)\s+of\s+(?:only\s+)?(?:around\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(0)),
            'value': float(m.group(2)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.9
        }
    },
    
    # Template 14: "with X VERBing"
    {
        'name': 'with_participle',
        'pattern': r'with\s+([A-Z][A-Za-z0-9\s]+?)\s+(\w+ing)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': None,
            'unit': '',
            'value_type': 'direction_only',
            'confidence': 0.7
        }
    },
    
    # Template 15: "X is expected to VERB by Y%"
    {
        'name': 'expected_to',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:is|are)\s+expected\s+to\s+(\w+)\s+by\s+(?:around\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.85
        }
    }
]


def extract_with_templates(sentence: str, context: Optional[Dict] = None) -> List[Dict]:
    """
    Extract parameters using template matching
    
    Parameters:
    -----------
    sentence : str
        Sentence to extract from
    context : dict, optional
        Context from previous extractions (for filling needs_context items)
        
    Returns:
    --------
    items : List[Dict]
        Extracted items
    """
    
    items = []
    
    for template in TEMPLATES:
        pattern = template['pattern']
        extract_fn = template['extract']
        
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        
        for match in matches:
            try:
                item = extract_fn(match)
                item['source_sentence'] = sentence
                item['template'] = template['name']
                
                # Handle context-dependent extractions
                if item.get('needs_context') and context:
                    # Try to get parameter from context
                    if 'last_subject' in context and context['last_subject']:
                        item['parameter'] = context['last_subject']
                        del item['needs_context']
                    else:
                        continue  # Skip if no context available
                
                items.append(item)
                
            except Exception as e:
                # Skip malformed matches
                continue
    
    return items


def get_template_names() -> List[str]:
    """Get list of all template names"""
    return [t['name'] for t in TEMPLATES]


def get_template_count() -> int:
    """Get total number of templates"""
    return len(TEMPLATES)
