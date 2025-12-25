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
    Extract parameter items from text using HYBRID approach
    
    HYBRID SYSTEM (Option B + GLiNER):
    - Pass 1: Template matching
    - Pass 2: spaCy extraction  
    - Pass 3: Regex extraction
    - Pass 4: Unify parameter names
    - Pass 5: Deduplicate
    - Pass 6: GLiNER ML extraction (NEW!)
    - Pass 7: Merge Option B + GLiNER (BEST ACCURACY!)
    
    Returns list of items with:
    - parameter name (original and canonical)
    - direction (increase/decrease/target/stable)
    - value (if specified)
    - unit
    - confidence score
    - extraction_method (hybrid_both, optionb_only, gliner_only)
    """
    
    # Split into sentences
    sentences = split_sentences(text)
    
    all_items = []
    context = {'last_subject': None}
    
    # ===== OPTION B EXTRACTION =====
    
    # PASS 1: Template-based extraction (highest priority)
    from .templates import extract_with_templates
    
    for sentence in sentences:
        template_items = extract_with_templates(sentence, context)
        if template_items:
            all_items.extend(template_items)
            # Update context with last extracted parameter
            if template_items:
                context['last_subject'] = template_items[-1].get('parameter')
    
    # PASS 2: spaCy extraction (if available)
    if SPACY_AVAILABLE:
        for sentence in sentences:
            spacy_items = extract_with_spacy(sentence)
            if spacy_items:
                all_items.extend(spacy_items)
    
    # PASS 3: Regex extraction (fallback)
    for sentence in sentences:
        regex_items = extract_with_regex(sentence)
        if regex_items:
            all_items.extend(regex_items)
    
    # PASS 4: Unify parameter names
    from .unification import unify_extracted_items, deduplicate_items, merge_similar_parameters
    
    all_items = unify_extracted_items(all_items)
    
    # PASS 5: Deduplicate and merge
    all_items = deduplicate_items(all_items)
    all_items = merge_similar_parameters(all_items, similarity_threshold=0.85)
    
    # ===== HYBRID WITH GLINER ML =====
    
    # PASS 6 & 7: Use hybrid extraction (Option B + GLiNER)
    try:
        from .ml_extractor import hybrid_extract, is_gliner_available
        
        if is_gliner_available():
            # Combine Option B results with GLiNER
            all_items = hybrid_extract(text, all_items)
        else:
            # GLiNER not available - mark items as Option B only
            for item in all_items:
                if 'extraction_method' not in item:
                    item['extraction_method'] = 'optionb'
    
    except ImportError:
        # ml_extractor not available - just use Option B
        for item in all_items:
            if 'extraction_method' not in item:
                item['extraction_method'] = 'optionb'
    
    # Sort by confidence
    all_items.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    return all_items


def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # More sophisticated sentence splitting
    # First, handle abbreviations to avoid false splits
    text = text.replace('e.g.', 'eg').replace('i.e.', 'ie').replace('etc.', 'etc')
    
    # Split on period, exclamation, or question mark followed by space and capital letter
    sentences = re.split(r'[.!?]+(?=\s+[A-Z])', text)
    
    # Also split on newlines (for list-style input)
    all_sentences = []
    for sent in sentences:
        # Further split on newlines
        subsents = sent.split('\n')
        all_sentences.extend(subsents)
    
    # Clean and filter
    cleaned = [s.strip() for s in all_sentences if s.strip() and len(s.strip()) > 10]
    
    return cleaned


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
    """
    Extract full noun phrase from spaCy token
    Gets complete parameter names including compounds and modifiers
    """
    if not token:
        return ""
    
    # Collect all parts of the noun phrase
    phrase_tokens = []
    
    # Add left children (determiners, adjectives, compounds, numerals)
    left_children = []
    for child in token.lefts:
        if child.dep_ in ['det', 'amod', 'compound', 'nummod', 'nmod', 'poss']:
            left_children.append(child)
            # Also get compounds of compounds
            for grandchild in child.lefts:
                if grandchild.dep_ in ['compound', 'amod']:
                    left_children.append(grandchild)
    
    # Sort by position
    left_children.sort(key=lambda x: x.i)
    phrase_tokens.extend(left_children)
    
    # Add the main token
    phrase_tokens.append(token)
    
    # Add right children (prepositional phrases, noun modifiers)
    for child in token.rights:
        if child.dep_ in ['prep', 'pobj', 'npadvmod']:
            phrase_tokens.append(child)
            # Add children of preposition
            for grandchild in child.children:
                if grandchild.dep_ in ['pobj', 'compound']:
                    phrase_tokens.append(grandchild)
    
    # Sort by position and join
    phrase_tokens.sort(key=lambda x: x.i)
    phrase = ' '.join([t.text for t in phrase_tokens])
    
    # Clean up
    phrase = phrase.strip()
    
    # Remove leading articles
    phrase = re.sub(r'^(the|a|an)\s+', '', phrase, flags=re.IGNORECASE)
    
    return phrase


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
    
    # Pattern 1: "GDP increases by 20%" or "GDP increases by 20 percent"
    pattern1 = r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:continued\s+to\s+)?(?:increases?|decreases?|grows?|grown|falls?|fallen|rises?|risen|declines?|declined)\s+(?:by\s+)?(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)'
    matches = re.finditer(pattern1, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        # Clean parameter name
        parameter = re.sub(r'\s+has$', '', parameter)
        direction_word = 'increase' if any(word in match.group(0).lower() for word in ['increase', 'grow', 'rise', 'risen', 'grown']) else 'decrease'
        value = float(match.group(2))
        unit = '%'
        
        items.append({
            'parameter': parameter,
            'direction': direction_word,
            'value': value,
            'unit': unit,
            'value_type': 'percent',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 2: "GDP reaches 3.2 trillion" or "GDP reaches 3.2 trillion dollars"
    pattern2 = r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:reaches?|reached|hits?|attains?|to|of)\s+(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gw|twh|kt|mt|gt)'
    matches = re.finditer(pattern2, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        parameter = re.sub(r'\s+has$', '', parameter)
        value = float(match.group(2))
        unit = match.group(3).lower()
        
        items.append({
            'parameter': parameter,
            'direction': 'target',
            'value': value,
            'unit': unit,
            'value_type': 'absolute',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 3: "20% increase in GDP" or "20 percent increase in GDP"
    pattern3 = r'(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)\s+(?:increase|decrease|reduction|growth|decline|rise|fall)\s+(?:in|of)\s+([A-Z][A-Za-z0-9\s]+?)(?:\s|$|,|\.)'
    matches = re.finditer(pattern3, sentence, re.IGNORECASE)
    
    for match in matches:
        value = float(match.group(1))
        direction_word = 'increase' if any(word in match.group(0).lower() for word in ['increase', 'growth', 'rise']) else 'decrease'
        parameter = match.group(3).strip()
        
        items.append({
            'parameter': parameter,
            'direction': direction_word,
            'value': value,
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 4: "GDP doubles" or "emissions halve"
    pattern4 = r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:doubles?|doubled|halves?|halved|triples?|tripled)'
    matches = re.finditer(pattern4, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        parameter = re.sub(r'\s+has$', '', parameter)
        verb = match.group(0).lower()
        
        if 'double' in verb:
            direction = 'double'
            value = 100.0
            unit = '%'
        elif 'halve' in verb or 'halved' in verb:
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
    
    # Pattern 5: "GDP of 3.2 trillion" or "emissions of 5 MtCO2"
    pattern5 = r'([A-Z][A-Za-z0-9\s]+?)\s+of\s+(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(trillion|billion|million|thousand|mtco2|gw|twh|%|percent)'
    matches = re.finditer(pattern5, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(2))
        unit = match.group(3).lower()
        unit = '%' if unit in ['%', 'percent'] else unit
        
        items.append({
            'parameter': parameter,
            'direction': 'target',
            'value': value,
            'unit': unit,
            'value_type': 'percent' if unit == '%' else 'absolute',
            'confidence': 0.8,
            'source_sentence': sentence
        })
    
    # Pattern 6: "GDP: 20%", "Emissions: 5 MtCO2" (colon format)
    pattern6 = r'([A-Z][A-Za-z0-9\s]+?):\s*(?:around\s+|about\s+)?(\d+\.?\d*)\s*(%|percent|trillion|billion|million|mtco2|gw|twh)?'
    matches = re.finditer(pattern6, sentence, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(2))
        unit = match.group(3).lower() if match.group(3) else ''
        unit = '%' if unit in ['%', 'percent'] else unit
        
        items.append({
            'parameter': parameter,
            'direction': 'target',
            'value': value,
            'unit': unit if unit else 'absolute',
            'value_type': 'percent' if unit == '%' else 'absolute',
            'confidence': 0.75,
            'source_sentence': sentence
        })
    
    # Pattern 7: "falling by approximately 40 percent" or "increasing by around 25 percent"
    pattern7 = r'(?:falling|rising|increasing|decreasing|growing|declining)\s+by\s+(?:around\s+|about\s+|approximately\s+)?(\d+\.?\d*)\s*(%|percent|pct)'
    matches = re.finditer(pattern7, sentence, re.IGNORECASE)
    
    for match in matches:
        direction_word = 'increase' if any(word in match.group(0).lower() for word in ['rising', 'increasing', 'growing']) else 'decrease'
        value = float(match.group(1))
        
        # Try to find subject before this phrase
        # Look backwards in sentence
        before_text = sentence[:match.start()]
        # Find capitalized noun phrase
        subject_match = re.search(r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has|have)\s*$', before_text)
        
        if subject_match:
            parameter = subject_match.group(1).strip()
        else:
            # Default to generic name
            parameter = "Value"
        
        items.append({
            'parameter': parameter,
            'direction': direction_word,
            'value': value,
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.8,
            'source_sentence': sentence
        })
    
    # Pattern 8: "reduction in carbon emissions of only around 10 percent"
    pattern8 = r'(?:increase|decrease|reduction|growth)\s+in\s+([A-Za-z0-9\s]+?)\s+of\s+(?:only\s+)?(?:around\s+|about\s+)?(\d+\.?\d*)\s*(%|percent)'
    matches = re.finditer(pattern8, sentence, re.IGNORECASE)
    
    for match in matches:
        direction_word = 'increase' if 'increase' in match.group(0).lower() or 'growth' in match.group(0).lower() else 'decrease'
        parameter = match.group(1).strip()
        value = float(match.group(2))
        
        items.append({
            'parameter': parameter,
            'direction': direction_word,
            'value': value,
            'unit': '%',
            'value_type': 'percent',
            'confidence': 0.85,
            'source_sentence': sentence
        })
    
    # Pattern 9: "GDP increases" (direction only) - ONLY if no value was already extracted
    if not items:
        pattern9 = r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:increases?|increased|decreases?|decreased|grows?|grown|falls?|fallen|rises?|risen|declines?|declined|remains?\s+stable)'
        matches = re.finditer(pattern9, sentence, re.IGNORECASE)
        
        for match in matches:
            parameter = match.group(1).strip()
            parameter = re.sub(r'\s+has$', '', parameter)
            verb = match.group(0).lower()
            
            if 'stable' in verb or 'remain' in verb:
                direction = 'stable'
            elif any(word in verb for word in ['increase', 'grow', 'rise', 'risen', 'grown']):
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
