"""
Template-based Extraction Module
Uses predefined templates to extract parameters from common sentence structures

PHASE 2 ENHANCEMENTS:
- Detect time expressions (by/to/from/range)
- Detect value semantics (target, delta, range, rate, pp)
- Extract baseline and target years
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


def extract_time_from_match(text: str, match_position: int = 0) -> Dict:
    """
    PHASE 2: Extract time information from text near a match
    
    Parameters:
    -----------
    text : str
        Full text to search
    match_position : int
        Position of the main match (to search nearby)
        
    Returns:
    --------
    time_info : dict
        {
            'target_year': int,
            'baseline_year': int,
            'time_expression': str,
            'time_confidence': float
        }
    """
    # Search in a window around the match
    window_start = max(0, match_position - 100)
    window_end = min(len(text), match_position + 100)
    window = text[window_start:window_end]
    
    time_info = {
        'target_year': None,
        'baseline_year': None,
        'time_expression': '',
        'time_confidence': 0.0
    }
    
    # Pattern 1: "by YEAR" or "in YEAR" (highest priority)
    pattern1 = r'\b(by|in|until|to)\s+(20\d{2}|21\d{2})\b'
    match = re.search(pattern1, window, re.IGNORECASE)
    if match:
        time_info['target_year'] = int(match.group(2))
        time_info['time_expression'] = match.group(0)
        time_info['time_confidence'] = 0.9
        return time_info
    
    # Pattern 2: "from YEAR to YEAR"
    pattern2 = r'\b(?:from\s+)?(20\d{2}|21\d{2})\s*(?:to|-|–)\s*(20\d{2}|21\d{2})\b'
    match = re.search(pattern2, window, re.IGNORECASE)
    if match:
        time_info['baseline_year'] = int(match.group(1))
        time_info['target_year'] = int(match.group(2))
        time_info['time_expression'] = match.group(0)
        time_info['time_confidence'] = 0.95
        return time_info
    
    # Pattern 3: Standalone year
    pattern3 = r'\b(20[2-9][0-9]|21[0-9][0-9])\b'
    match = re.search(pattern3, window)
    if match:
        time_info['target_year'] = int(match.group(1))
        time_info['time_expression'] = match.group(0)
        time_info['time_confidence'] = 0.6
        return time_info
    
    return time_info


# Define extraction templates
TEMPLATES = [
    # PHASE 2: Template 1a: "X increases by Y% by/in YEAR"
    {
        'name': 'percent_change_with_year',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(\w+ing|\w+es?|\w+ed)\s+(?:by\s+)?(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)\s+(by|in|until)\s+(20\d{2}|21\d{2})',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.95,
            'target_year': int(m.group(6)),
            'time_expression': f"{m.group(5)} {m.group(6)}",
            'time_confidence': 0.9
        }
    },
    
    # Template 1: "X increases/decreases by Y%"
    {
        'name': 'simple_percent_change',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(\w+ing|\w+es?|\w+ed)\s+(?:by\s+)?(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # PHASE 2: Template 1b: "X from Y% to Z%"
    {
        'name': 'from_to_percent',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+from\s+(\d+\.?\d*)\s*%\s+to\s+(\d+\.?\d*)\s*%',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'target' if float(m.group(3)) > float(m.group(2)) else 'decrease',
            'value': float(m.group(3)) - float(m.group(2)),
            'unit': '%',
            'value_type': 'delta',
            'base_value': float(m.group(2)),
            'target_value': float(m.group(3)),
            'confidence': 0.95,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # PHASE 2: Template 1c: Range "X between Y% and Z%"
    {
        'name': 'range_pattern',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+between\s+(\d+\.?\d*)\s*(?:and|-)\s*(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': (float(m.group(2)) + float(m.group(3))) / 2,
            'unit': '%',
            'value_type': 'range',
            'value_min': float(m.group(2)),
            'value_max': float(m.group(3)),
            'is_range': True,
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # PHASE 2: Template 1d: Rate "X grows at Y% per year"
    {
        'name': 'rate_pattern',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:grows?|increases?)\s+(?:at\s+|by\s+)?(\d+\.?\d*)\s*%\s*(?:per year|annually|/year)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'increase',
            'value': float(m.group(2)),
            'unit': '%',
            'value_type': 'rate',
            'is_rate': True,
            'rate_period': 'per year',
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # PHASE 2: Template 1e: Percentage points
    {
        'name': 'percentage_points',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:increases?|decreases?|changes?)\s+(?:by\s+)?(\d+\.?\d*)\s*(?:percentage points?|pp)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'increase' if 'increase' in m.group(0).lower() else 'decrease',
            'value': float(m.group(2)),
            'unit': 'pp',
            'value_type': 'percent_point',
            'confidence': 0.95,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 2: "By YEAR, X has VERB, VERBing by VALUE%"
    {
        'name': 'narrative_with_year',
        'pattern': r'[Bb]y\s+(20\d{2}|21\d{2}),\s+([A-Z][A-Za-z0-9\s]+?)\s+(?:has|have)\s+\w+(?:ed|en)?,?\s+(\w+ing)\s+by\s+(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(2).strip(),
            'direction': classify_direction(m.group(3)),
            'value': float(m.group(4)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.95,
            'target_year': int(m.group(1)),
            'time_expression': f"by {m.group(1)}",
            'time_confidence': 0.95
        }
    },
    
    # Template 3: "X reaches/hits Y UNIT"
    {
        'name': 'target_value',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:reaches?|hits?|attains?)\s+(?:around\s+|about\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gtco2|gw|mw|twh|gwh|%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': '%' if m.group(3).lower() in ['%', 'percent'] else m.group(3).lower(),
            'value_type': 'absolute_target',
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 4: "VALUE% increase/decrease in X"
    {
        'name': 'reverse_percent',
        'pattern': r'(?:around\s+|about\s+|approximately\s+|only\s+)?(\d+\.?\d*)\s*(%|percent)\s+(increase|decrease|reduction|growth|decline)\s+(?:in|of)\s+([A-Z][A-Za-z0-9\s]+?)(?:\s|,|\.)',
        'extract': lambda m, text: {
            'parameter': m.group(4).strip(),
            'direction': classify_direction(m.group(3)),
            'value': float(m.group(1)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 5: "X doubles/halves/triples"
    {
        'name': 'multiplier',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:doubles?|doubled|halves?|halved|triples?|tripled)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'double' if 'double' in m.group(0).lower() else 'halve' if 'halve' in m.group(0).lower() else 'triple',
            'value': 100.0 if 'double' in m.group(0).lower() else -50.0 if 'halve' in m.group(0).lower() else 200.0,
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.95,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 6: "X of Y UNIT"
    {
        'name': 'of_format',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+of\s+(?:around\s+|about\s+|only\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gtco2|%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': '%' if m.group(3).lower() in ['%', 'percent'] else m.group(3).lower(),
            'value_type': 'absolute_target',
            'confidence': 0.85,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 7: "X: Y%"
    {
        'name': 'colon_format',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?):\s*(\d+\.?\d*)\s*(%|percent|trillion|billion|million|mtco2|gw)?',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': 'target',
            'value': float(m.group(2)),
            'unit': '%' if m.group(3) and m.group(3).lower() in ['%', 'percent'] else m.group(3).lower() if m.group(3) else 'absolute',
            'value_type': 'absolute_target',
            'confidence': 0.8,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 8: "VERBing by AROUND/APPROXIMATELY VALUE%"
    {
        'name': 'participle_with_qualifier',
        'pattern': r'(\w+ing)\s+by\s+(around|about|approximately|only)\s+(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m, text: {
            'parameter': 'Value',  # Will be filled by context
            'direction': classify_direction(m.group(1)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.85,
            'needs_context': True,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 9: "reduction/increase in X of Y%"
    {
        'name': 'change_in_param',
        'pattern': r'(?:increase|decrease|reduction|growth|decline)\s+in\s+([A-Za-z0-9\s]+?)\s+of\s+(?:only\s+|around\s+|about\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(0)),
            'value': float(m.group(2)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.9,
            **extract_time_from_match(text, m.start())
        }
    },
    
    # Template 10: "X continues to VERB, VERBing by Y%"
    {
        'name': 'continues_to_verb',
        'pattern': r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:continues?|continued)\s+to\s+\w+(?:,|\s+)\s*(\w+ing)\s+by\s+(?:around\s+)?(\d+\.?\d*)\s*(%|percent)',
        'extract': lambda m, text: {
            'parameter': m.group(1).strip(),
            'direction': classify_direction(m.group(2)),
            'value': float(m.group(3)),
            'unit': '%',
            'value_type': 'delta',
            'confidence': 0.95,
            **extract_time_from_match(text, m.start())
        }
    }
]


def extract_with_templates(sentence: str, context: Optional[Dict] = None) -> List[Dict]:
    """
    Extract parameters using template matching
    
    PHASE 2: Enhanced with time and value semantics
    
    Parameters:
    -----------
    sentence : str
        Sentence to extract from
    context : dict, optional
        Context from previous extractions (for filling needs_context items)
        
    Returns:
    --------
    items : List[Dict]
        Extracted items with Phase 2 enhancements
    """
    
    items = []
    
    for template in TEMPLATES:
        pattern = template['pattern']
        extract_fn = template['extract']
        
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        
        for match in matches:
            try:
                # PHASE 2: Pass full text to extract function for time extraction
                item = extract_fn(match, sentence)
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
                
                # PHASE 2: Ensure backward compatibility fields exist
                if 'target_year' not in item:
                    item['target_year'] = None
                if 'baseline_year' not in item:
                    item['baseline_year'] = None
                if 'time_expression' not in item:
                    item['time_expression'] = ''
                if 'time_confidence' not in item:
                    item['time_confidence'] = 0.0
                
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
