"""
ML-based Extraction Module using GLiNER
Zero-shot Named Entity Recognition for scenario analysis
"""

import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

# Try to import GLiNER
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
    
    # Load model (lazy loading - only when first used)
    _gliner_model = None
    
    def get_gliner_model():
        """Lazy load GLiNER model"""
        global _gliner_model
        if _gliner_model is None:
            try:
                _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            except Exception as e:
                print(f"Warning: Could not load GLiNER model: {e}")
                return None
        return _gliner_model
    
except ImportError:
    GLINER_AVAILABLE = False
    
    def get_gliner_model():
        return None


def extract_with_gliner(text: str) -> List[Dict]:
    """
    Extract parameters using GLiNER zero-shot NER
    
    Parameters:
    -----------
    text : str
        Text to extract from
        
    Returns:
    --------
    items : List[Dict]
        Extracted parameter items
    """
    
    if not GLINER_AVAILABLE:
        return []
    
    model = get_gliner_model()
    if model is None:
        return []
    
    # Define entity labels for GLiNER
    labels = [
        "economic indicator",      # GDP, income, investment
        "environmental indicator", # emissions, energy, renewable
        "social indicator",        # employment, population, jobs
        "percentage value",        # 20%, 25 percent
        "numeric value",           # 3.2, 40, 100
        "unit",                    # trillion, billion, MtCO2, GW
        "year",                    # 2040, 2050
        "change direction",        # increase, decrease, growth
        "qualifier"                # around, approximately, about
    ]
    
    try:
        # Extract entities with threshold
        entities = model.predict_entities(
            text, 
            labels=labels, 
            threshold=0.4  # Lower threshold to catch more
        )
    except Exception as e:
        print(f"GLiNER extraction error: {e}")
        return []
    
    # Group entities into parameter items
    items = []
    
    # Strategy: Find indicator + value pairs
    indicators = [e for e in entities if e['label'] in ['economic indicator', 'environmental indicator', 'social indicator']]
    percentages = [e for e in entities if e['label'] == 'percentage value']
    numbers = [e for e in entities if e['label'] == 'numeric value']
    units = [e for e in entities if e['label'] == 'unit']
    directions = [e for e in entities if e['label'] == 'change direction']
    
    # Match indicators with values
    for indicator in indicators:
        param_name = indicator['text']
        param_start = indicator['start']
        param_end = indicator['end']
        
        # Find closest percentage or number after this indicator
        value = None
        unit = ''
        direction = None
        value_type = 'direction_only'
        
        # Look for percentage value near this indicator (within 100 chars)
        for pct in percentages:
            if abs(pct['start'] - param_end) < 100:  # Within 100 chars
                # Extract numeric value from percentage text
                pct_text = pct['text']
                match = re.search(r'(\d+\.?\d*)', pct_text)
                if match:
                    value = float(match.group(1))
                    unit = '%'
                    value_type = 'percent'
                    break
        
        # If no percentage, look for numeric value + unit
        if value is None:
            for num in numbers:
                if abs(num['start'] - param_end) < 100:
                    num_text = num['text']
                    match = re.search(r'(\d+\.?\d*)', num_text)
                    if match:
                        value = float(match.group(1))
                        value_type = 'absolute'
                        
                        # Find associated unit
                        for u in units:
                            if abs(u['start'] - num['end']) < 20:  # Unit right after number
                                unit = u['text'].lower()
                                break
                        break
        
        # Find direction
        for d in directions:
            if abs(d['start'] - param_start) < 50 or abs(d['start'] - param_end) < 50:
                direction = classify_gliner_direction(d['text'])
                break
        
        # If no explicit direction found, infer from context
        if direction is None:
            direction = infer_direction(text, param_start, param_end)
        
        # Create item
        items.append({
            'parameter': param_name,
            'direction': direction or 'target',
            'value': value,
            'unit': unit,
            'value_type': value_type,
            'confidence': indicator['score'],
            'source_sentence': text,
            'extraction_method': 'gliner'
        })
    
    return items


def classify_gliner_direction(direction_text: str) -> str:
    """Classify direction from GLiNER entity"""
    text_lower = direction_text.lower()
    
    if any(word in text_lower for word in ['increase', 'grow', 'rise', 'expand', 'gain', 'up']):
        return 'increase'
    elif any(word in text_lower for word in ['decrease', 'fall', 'decline', 'drop', 'reduce', 'down', 'contract']):
        return 'decrease'
    elif any(word in text_lower for word in ['reach', 'hit', 'attain', 'achieve', 'target']):
        return 'target'
    elif any(word in text_lower for word in ['stable', 'constant', 'unchanged', 'remain']):
        return 'stable'
    elif 'double' in text_lower:
        return 'double'
    elif 'halve' in text_lower or 'half' in text_lower:
        return 'halve'
    else:
        return 'increase'  # default


def infer_direction(text: str, start: int, end: int) -> Optional[str]:
    """Infer direction from surrounding context"""
    
    # Get text around the parameter (50 chars before and after)
    context_start = max(0, start - 50)
    context_end = min(len(text), end + 50)
    context = text[context_start:context_end].lower()
    
    # Check for direction keywords
    if any(word in context for word in ['increase', 'grow', 'rise', 'expand', 'gain', 'up']):
        return 'increase'
    elif any(word in context for word in ['decrease', 'fall', 'decline', 'drop', 'reduce', 'down']):
        return 'decrease'
    elif any(word in context for word in ['reach', 'hit', 'attain', 'target']):
        return 'target'
    elif any(word in context for word in ['stable', 'constant', 'remain']):
        return 'stable'
    
    return None


def hybrid_extract(text: str, optionb_items: List[Dict]) -> List[Dict]:
    """
    Hybrid extraction combining Option B + GLiNER
    
    Parameters:
    -----------
    text : str
        Text to extract from
    optionb_items : List[Dict]
        Items already extracted by Option B (templates + spaCy + regex)
        
    Returns:
    --------
    merged_items : List[Dict]
        Merged and deduplicated items with confidence scores
    """
    
    # Extract with GLiNER
    gliner_items = extract_with_gliner(text)
    
    if not gliner_items:
        # GLiNER not available or failed - just return Option B results
        for item in optionb_items:
            item['extraction_method'] = 'optionb'
        return optionb_items
    
    # Merge results using smart strategy
    merged = merge_extractions(optionb_items, gliner_items)
    
    return merged


def merge_extractions(optionb_items: List[Dict], gliner_items: List[Dict]) -> List[Dict]:
    """
    Intelligently merge Option B and GLiNER extractions
    
    Strategy:
    - Items both found → HIGH confidence (0.95)
    - Items only Option B found → MEDIUM confidence (0.70)
    - Items only GLiNER found → MEDIUM confidence (0.75)
    - Prefer Option B for direction, GLiNER for parameter names
    """
    
    merged = []
    matched_gliner = set()
    matched_optionb = set()
    
    # Phase 1: Find items both methods agree on (HIGH CONFIDENCE)
    for i, item_b in enumerate(optionb_items):
        for j, item_g in enumerate(gliner_items):
            if parameters_match(item_b['parameter'], item_g['parameter']):
                # Both found same parameter - merge with high confidence
                merged_item = {
                    'parameter': item_g['parameter'],  # GLiNER usually has better full names
                    'direction': item_b.get('direction') or item_g.get('direction'),  # Prefer Option B direction
                    'value': item_b.get('value') if item_b.get('value') is not None else item_g.get('value'),
                    'unit': item_b.get('unit') or item_g.get('unit'),
                    'value_type': item_b.get('value_type', 'absolute'),
                    'confidence': 0.95,  # HIGH - both methods agree
                    'source_sentence': item_b.get('source_sentence', ''),
                    'extraction_method': 'hybrid_both',
                    'parameter_original': item_b['parameter']
                }
                
                merged.append(merged_item)
                matched_optionb.add(i)
                matched_gliner.add(j)
                break
    
    # Phase 2: Add Option B unique items (MEDIUM CONFIDENCE)
    for i, item_b in enumerate(optionb_items):
        if i not in matched_optionb:
            item_b['confidence'] = item_b.get('confidence', 0.70)
            item_b['extraction_method'] = 'optionb_only'
            merged.append(item_b)
    
    # Phase 3: Add GLiNER unique items (MEDIUM CONFIDENCE)
    for j, item_g in enumerate(gliner_items):
        if j not in matched_gliner:
            item_g['confidence'] = item_g.get('confidence', 0.75)
            item_g['extraction_method'] = 'gliner_only'
            merged.append(item_g)
    
    # Sort by confidence (highest first)
    merged.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    return merged


def parameters_match(param1: str, param2: str, threshold: float = 0.7) -> bool:
    """
    Check if two parameter names refer to the same thing
    
    Uses fuzzy matching to handle variations like:
    - "GDP" vs "gross domestic product"
    - "CO2 emissions" vs "carbon dioxide emissions"
    """
    
    if not param1 or not param2:
        return False
    
    p1 = param1.lower().strip()
    p2 = param2.lower().strip()
    
    # Exact match
    if p1 == p2:
        return True
    
    # One contains the other
    if p1 in p2 or p2 in p1:
        return True
    
    # Fuzzy similarity
    similarity = SequenceMatcher(None, p1, p2).ratio()
    
    return similarity >= threshold


def is_gliner_available() -> bool:
    """Check if GLiNER is available"""
    return GLINER_AVAILABLE


def get_gliner_status() -> Dict[str, any]:
    """Get GLiNER status information"""
    return {
        'available': GLINER_AVAILABLE,
        'model_loaded': _gliner_model is not None if GLINER_AVAILABLE else False,
        'model_name': 'urchade/gliner_medium-v2.1' if GLINER_AVAILABLE else None
    }
