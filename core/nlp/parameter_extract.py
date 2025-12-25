"""
Parameter extraction module
Extracts parameters, values, and directions from scenario text
"""

import re
from typing import List, Dict, Optional, Tuple
import spacy

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    nlp = None


def extract_parameters_from_scenarios(scenarios: List[Dict]) -> List[Dict]:
    """
    Extract parameters from each scenario
    
    Parameters:
    -----------
    scenarios : List[Dict]
        List of scenario dictionaries with 'id', 'title', 'text'
        
    Returns:
    --------
    scenarios_with_params : List[Dict]
        Scenarios with added 'items' and 'horizon' fields
    """
    scenarios_with_params = []
    
    for scenario in scenarios:
        # Extract horizon (year)
        horizon = extract_horizon(scenario['text'])
        
        # Extract items (parameters)
        items = extract_items_from_text(scenario['text'])
        
        scenarios_with_params.append({
            'id': scenario['id'],
            'title': scenario['title'],
            'text': scenario['text'],
            'horizon': horizon,
            'items': items
        })
    
    return scenarios_with_params


def extract_horizon(text: str) -> Optional[int]:
    """
    Extract target year/horizon from text
    
    Looks for patterns like:
    - "by 2040"
    - "in 2050"
    - "until 2030"
    """
    # Pattern for years (2020-2100)
    pattern = r'(?:by|in|until|to)\s+(20[2-9][0-9]|21[0-9][0-9])'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if matches:
        # Return the first (or most common) year
        years = [int(y) for y in matches]
        return max(set(years), key=years.count)  # Most common year
    
    return None


def extract_items_from_text(text: str) -> List[Dict]:
    """
    Extract parameter items from text
    
    Returns list of items with:
    - parameter name
    - direction (increase/decrease/target/stable)
    - value (if specified)
    - unit
    - confidence score
    """
    items = []
    
    # Split into sentences
    sentences = split_sentences(text)
    
    for sentence in sentences:
        # Try spaCy extraction first (if available)
        if SPACY_AVAILABLE:
            extracted = extract_with_spacy(sentence)
            if extracted:
                items.extend(extracted)
                continue
        
        # Fallback to regex extraction
        extracted = extract_with_regex(sentence)
        if extracted:
            items.extend(extracted)
    
    return items


def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_with_spacy(sentence: str) -> List[Dict]:
    """
    Extract parameters using spaCy NLP
    """
    if not SPACY_AVAILABLE or not nlp:
        return []
    
    doc = nlp(sentence)
    items = []
    
    # Look for change verbs
    change_verbs = {
        'increase': 'increase',
        'decrease': 'decrease',
        'grow': 'increase',
        'fall': 'decrease',
        'rise': 'increase',
        'decline': 'decrease',
        'reach': 'target',
        'hit': 'target',
        'attain': 'target',
        'remain': 'stable',
        'stable': 'stable',
        'double': 'double',
        'halve': 'halve',
        'triple': 'triple'
    }
    
    for token in doc:
        if token.lemma_ in change_verbs:
            direction = change_verbs[token.lemma_]
            
            # Find subject (parameter)
            subject = None
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    # Get the full noun phrase
                    subject = get_noun_phrase(child)
                    break
            
            if not subject:
                continue
            
            # Find numeric value
            value, unit = extract_value_from_token(token, doc)
            
            # Calculate confidence
            confidence = calculate_confidence(subject, direction, value, unit)
            
            items.append({
                'parameter': subject,
                'direction': direction,
                'value': value,
                'unit': unit,
                'value_type': 'percent' if unit == '%' else 'absolute' if value is not None else 'direction_only',
                'confidence': confidence,
                'source_sentence': sentence
            })
    
    return items


def get_noun_phrase(token) -> str:
    """Extract full noun phrase from spaCy token"""
    # Get all tokens in the noun phrase
    phrase_tokens = []
    
    # Add determiners, adjectives, compounds
    for child in token.children:
        if child.dep_ in ['det', 'amod', 'compound']:
            phrase_tokens.append(child)
    
    phrase_tokens.append(token)
    
    # Sort by position
    phrase_tokens.sort(key=lambda t: t.i)
    
    return ' '.join([t.text for t in phrase_tokens])


def extract_value_from_token(token, doc) -> Tuple[Optional[float], str]:
    """Extract numeric value and unit from token context"""
    
    # Look for numbers in children
    for child in token.children:
        if child.like_num or child.pos_ == 'NUM':
            value_str = child.text.replace(',', '')
            
            # Check for percent
            for c in child.children:
                if c.text in ['%', 'percent', 'pct']:
                    return float(value_str), '%'
            
            # Check next token for percent
            if child.i + 1 < len(doc):
                next_token = doc[child.i + 1]
                if next_token.text in ['%', 'percent', 'pct']:
                    return float(value_str), '%'
            
            # Check for unit words
            for c in child.children:
                if c.text.lower() in ['billion', 'million', 'thousand', 'trillion']:
                    return float(value_str), c.text.lower()
            
            # Check next tokens for units
            if child.i + 1 < len(doc):
                next_token = doc[child.i + 1]
                if next_token.text.lower() in ['billion', 'million', 'thousand', 'trillion', 'mtco2', 'gw', 'twh']:
                    return float(value_str), next_token.text.lower()
            
            return float(value_str), 'absolute'
    
    return None, ''


def extract_with_regex(sentence: str) -> List[Dict]:
    """
    Extract parameters using regex patterns (fallback)
    """
    items = []
    
    # Pattern 1: "GDP increases by 20%"
    pattern1 = r'(\w+(?:\s+\w+)?)\s+(increases?|decreases?|grows?|falls?|rises?|declines?)\s+by\s+(\d+\.?\d*)\s*(%|percent|pct)'
    matches = re.finditer(pattern1, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        direction = 'increase' if match.group(2).lower() in ['increase', 'increases', 'grow', 'grows', 'rise', 'rises'] else 'decrease'
        value = float(match.group(3))
        unit = '%'
        
        items.append({
            'parameter': parameter,
            'direction': direction,
            'value': value,
            'unit': unit,
            'value_type': 'percent',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 2: "GDP reaches 3.2 trillion"
    pattern2 = r'(\w+(?:\s+\w+)?)\s+(reaches?|hits?|attains?)\s+(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gw|twh)'
    matches = re.finditer(pattern2, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(3))
        unit = match.group(4).lower()
        
        items.append({
            'parameter': parameter,
            'direction': 'target',
            'value': value,
            'unit': unit,
            'value_type': 'absolute',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 3: "20% increase in GDP"
    pattern3 = r'(\d+\.?\d*)\s*(%|percent|pct)\s+(increase|decrease)\s+in\s+(\w+(?:\s+\w+)?)'
    matches = re.finditer(pattern3, sentence, re.IGNORECASE)
    
    for match in matches:
        value = float(match.group(1))
        direction = 'increase' if match.group(3).lower() == 'increase' else 'decrease'
        parameter = match.group(4).strip()
        
        items.append({
            'parameter': parameter,
            'direction': direction,
            'value': value,
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 4: "GDP doubles"
    pattern4 = r'(\w+(?:\s+\w+)?)\s+(doubles?|halves?|triples?)'
    matches = re.finditer(pattern4, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        verb = match.group(2).lower()
        
        if 'double' in verb:
            direction = 'double'
            value = 100.0
            unit = '%'
        elif 'halve' in verb:
            direction = 'halve'
            value = -50.0
            unit = '%'
        elif 'triple' in verb:
            direction = 'triple'
            value = 200.0
            unit = '%'
        
        items.append({
            'parameter': parameter,
            'direction': direction,
            'value': value,
            'unit': unit,
            'value_type': 'percent',
            'confidence': 0.9,
            'source_sentence': sentence
        })
    
    # Pattern 5: "GDP increases" (direction only)
    pattern5 = r'(\w+(?:\s+\w+)?)\s+(increases?|decreases?|grows?|falls?|rises?|declines?|remains?\s+stable)'
    
    # Only match if no value was already extracted
    if not items:
        matches = re.finditer(pattern5, sentence, re.IGNORECASE)
        
        for match in matches:
            parameter = match.group(1).strip()
            verb = match.group(2).lower()
            
            if 'stable' in verb or 'remain' in verb:
                direction = 'stable'
            elif verb in ['increase', 'increases', 'grow', 'grows', 'rise', 'rises']:
                direction = 'increase'
            else:
                direction = 'decrease'
            
            items.append({
                'parameter': parameter,
                'direction': direction,
                'value': None,
                'unit': '',
                'value_type': 'direction_only',
                'confidence': 0.6,
                'source_sentence': sentence
            })
    
    return items


def calculate_confidence(parameter: str, direction: str, value: Optional[float], unit: str) -> float:
    """
    Calculate confidence score for extraction
    
    High (0.85-1.0): parameter + value + unit + direction
    Medium (0.6-0.85): parameter + value + direction
    Low (0.3-0.6): parameter + direction only
    """
    score = 0.3  # Base score
    
    if parameter:
        score += 0.2
    
    if direction:
        score += 0.2
    
    if value is not None:
        score += 0.3
    
    if unit:
        score += 0.15
    
    return min(score, 1.0)
