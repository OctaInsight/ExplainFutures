"""
Parameter extraction module
Extracts parameters, values, and directions from scenario text

PHASE 1 IMPROVEMENTS:
- Atomic statement splitter for better context handling
- Configurable ML extraction with threshold controls
- Deterministic extraction order
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


def extract_parameters_from_scenarios(
    scenarios: List[Dict],
    enable_ml: bool = True,
    ml_confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract parameters from each scenario
    
    Parameters:
    -----------
    scenarios : List[Dict]
        List of scenario dictionaries with 'id', 'title', 'text'
    enable_ml : bool
        Whether to enable ML (GLiNER) extraction
    ml_confidence_threshold : float
        Minimum confidence threshold for ML extraction (0.0-1.0)
        
    Returns:
    --------
    scenarios_with_params : List[Dict]
        Scenarios with added 'items' and 'horizon' fields
    """
    scenarios_with_params = []
    
    for scenario in scenarios:
        # Extract horizon (year)
        horizon = extract_horizon(scenario['text'])
        
        # Extract items (parameters) with configurable ML
        items = extract_items_from_text(
            scenario['text'],
            enable_ml=enable_ml,
            ml_confidence_threshold=ml_confidence_threshold
        )
        
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


def extract_items_from_text(
    text: str,
    enable_ml: bool = True,
    ml_confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Extract parameter items from text using HYBRID approach
    
    PHASE 1 IMPROVEMENTS:
    - Atomic statement splitting prevents context bleeding
    - Configurable ML extraction
    - Deterministic processing order
    
    PHASE 2 ENHANCEMENTS:
    - Time/horizon extraction as first-class output
    - Value semantics classification (target, delta, range, rate)
    - Enhanced context with change detection
    
    HYBRID SYSTEM (Option B + GLiNER):
    - Pass 1: Split into atomic statements
    - Pass 2: Template matching (with Phase 2 time/value extraction)
    - Pass 3: spaCy extraction  
    - Pass 4: Regex extraction
    - Pass 5: Unify parameter names (CENTRALIZED)
    - Pass 6: Deduplicate
    - Pass 7: GLiNER ML extraction (if enabled)
    - Pass 8: Merge Option B + GLiNER
    - Pass 9: PHASE 2 - Extract time expressions and attach to items
    
    Returns list of items with:
    - parameter name (original and canonical)
    - direction (increase/decrease/target/stable)
    - value (if specified)
    - unit
    - PHASE 2: value_type (absolute_target, delta, range, rate, percent_point)
    - PHASE 2: baseline_year, target_year, time_expression, time_confidence
    - PHASE 2: value_min, value_max (for ranges)
    - PHASE 2: base_value, target_value (for "from A to B")
    - confidence score
    - extraction_method (hybrid_both, optionb_only, gliner_only)
    """
    
    # PHASE 1: Split into atomic statements FIRST
    statements = split_into_atomic_statements(text)
    
    all_items = []
    context = {'last_subject': None}
    
    # PHASE 2: Extract time expressions from full text
    from .value_parse import extract_time_expressions
    time_expressions = extract_time_expressions(text)
    
    # ===== OPTION B EXTRACTION =====
    
    # PASS 2: Template-based extraction (highest priority)
    # PHASE 2: Templates now extract time and value semantics
    from .templates import extract_with_templates
    
    for statement in statements:
        # PHASE 2: Enhanced context with change detection
        has_change_word = any(word in statement.lower() for word in 
                             ['increase', 'decrease', 'grow', 'fall', 'rise', 'decline', 'by'])
        context['has_change_word'] = has_change_word
        
        template_items = extract_with_templates(statement, context)
        if template_items:
            all_items.extend(template_items)
            # Update context with last extracted parameter
            if template_items:
                context['last_subject'] = template_items[-1].get('parameter')
    
    # PASS 3: spaCy extraction (if available)
    if SPACY_AVAILABLE:
        for statement in statements:
            spacy_items = extract_with_spacy(statement)
            if spacy_items:
                all_items.extend(spacy_items)
    
    # PASS 4: Regex extraction (fallback)
    for statement in statements:
        regex_items = extract_with_regex(statement)
        if regex_items:
            all_items.extend(regex_items)
    
    # PASS 5: Unify parameter names (CENTRALIZED - only here)
    from .unification import unify_extracted_items, deduplicate_items, merge_similar_parameters
    
    all_items = unify_extracted_items(all_items)
    
    # PASS 6: Deduplicate and merge
    all_items = deduplicate_items(all_items)
    all_items = merge_similar_parameters(all_items, similarity_threshold=0.85)
    
    # ===== HYBRID WITH GLINER ML =====
    
    # PASS 7 & 8: Use hybrid extraction (Option B + GLiNER) with configuration
    try:
        from .ml_extractor import hybrid_extract, is_gliner_available
        
        if is_gliner_available() and enable_ml:
            # Combine Option B results with GLiNER
            all_items = hybrid_extract(
                text, 
                all_items,
                confidence_threshold=ml_confidence_threshold,
                enable_ml=enable_ml
            )
        else:
            # GLiNER not available or disabled - mark items as Option B only
            for item in all_items:
                if 'extraction_method' not in item:
                    item['extraction_method'] = 'optionb'
    
    except ImportError:
        # ml_extractor not available - just use Option B
        for item in all_items:
            if 'extraction_method' not in item:
                item['extraction_method'] = 'optionb'
    
    # PASS 9: PHASE 2 - Attach time expressions to items that don't have them
    for item in all_items:
        # If item doesn't have time info, try to find closest time expression
        if not item.get('target_year') and not item.get('time_expression'):
            # Find closest time expression
            item_sentence = item.get('source_sentence', '')
            for time_expr in time_expressions:
                # Check if time expression is in same sentence or nearby
                if time_expr['expression'] in item_sentence or time_expr['expression'] in text:
                    item['target_year'] = time_expr.get('target_year')
                    item['baseline_year'] = time_expr.get('baseline_year')
                    item['time_expression'] = time_expr.get('expression', '')
                    item['time_confidence'] = time_expr.get('confidence', 0.0)
                    break
        
        # PHASE 2: Ensure all new fields exist (backward compatibility)
        if 'baseline_year' not in item:
            item['baseline_year'] = None
        if 'target_year' not in item:
            item['target_year'] = None
        if 'time_expression' not in item:
            item['time_expression'] = ''
        if 'time_confidence' not in item:
            item['time_confidence'] = 0.0
        if 'value_min' not in item:
            item['value_min'] = None
        if 'value_max' not in item:
            item['value_max'] = None
        if 'base_value' not in item:
            item['base_value'] = None
        if 'target_value' not in item:
            item['target_value'] = None
        if 'is_range' not in item:
            item['is_range'] = False
        if 'is_rate' not in item:
            item['is_rate'] = False
        if 'rate_period' not in item:
            item['rate_period'] = ''
        
        # Sync target_year with horizon for backward compatibility
        if item.get('target_year') and not item.get('horizon'):
            item['horizon'] = item['target_year']
        elif item.get('horizon') and not item.get('target_year'):
            item['target_year'] = item['horizon']
    
    # Sort by confidence (deterministic - highest first, then by parameter name)
    all_items.sort(key=lambda x: (-x.get('confidence', 0), x.get('parameter', '')))
    
    return all_items


def split_into_atomic_statements(text: str) -> List[str]:
    """
    PHASE 1: Split text into atomic statements
    
    Atomic statements are:
    - Individual bullet points (lines starting with -, *, •, or numbers)
    - "Parameter: value" lines
    - Sentences (as fallback)
    
    This prevents sentence splitters from merging distinct parameter statements.
    """
    statements = []
    
    # First, split by newlines to preserve bullet structure
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a bullet point or numbered list
        is_bullet = re.match(r'^[\-\*•●○▪▫]\s+', line)
        is_numbered = re.match(r'^\d+[\.\)]\s+', line)
        is_parameter_value = re.match(r'^[A-Z][A-Za-z0-9\s]+:\s*\d', line)
        
        if is_bullet or is_numbered or is_parameter_value:
            # This is an atomic statement - don't split further
            # Clean the bullet/number prefix
            if is_bullet:
                line = re.sub(r'^[\-\*•●○▪▫]\s+', '', line)
            elif is_numbered:
                line = re.sub(r'^\d+[\.\)]\s+', '', line)
            
            statements.append(line)
        else:
            # Not a bullet - split into sentences
            subsents = split_sentences_smart(line)
            statements.extend(subsents)
    
    # Clean and filter
    cleaned = [s.strip() for s in statements if s.strip() and len(s.strip()) > 10]
    
    return cleaned


def split_sentences_smart(text: str) -> List[str]:
    """
    Smart sentence splitting that preserves parameter boundaries
    """
    # Handle abbreviations to avoid false splits
    text = text.replace('e.g.', 'eg').replace('i.e.', 'ie').replace('etc.', 'etc')
    
    # Split on period, exclamation, or question mark followed by space and capital letter
    # But NOT on decimal numbers (e.g., "3.2 trillion")
    sentences = re.split(r'(?<![0-9])[.!?]+(?=\s+[A-Z])', text)
    
    # Also split on semicolons (common in parameter lists)
    all_sentences = []
    for sent in sentences:
        subsents = sent.split(';')
        all_sentences.extend(subsents)
    
    return [s.strip() for s in all_sentences if s.strip()]


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
                'unit': unit if unit else '',
                'value_type': 'percent' if unit == '%' else ('absolute' if value else 'direction_only'),
                'confidence': confidence,
                'source_sentence': sentence
            })
    
    return items


def get_noun_phrase(token) -> str:
    """Extract full noun phrase from token"""
    # Get the subtree
    phrase_tokens = list(token.subtree)
    phrase = ' '.join([t.text for t in phrase_tokens])
    return phrase.strip()


def extract_value_from_token(verb_token, doc) -> Tuple[Optional[float], str]:
    """Extract numeric value and unit near a verb"""
    value = None
    unit = ''
    
    # Look for numbers in the sentence
    for token in doc:
        if token.like_num:
            try:
                # Check if this number is related to our verb (within 5 tokens)
                if abs(token.i - verb_token.i) <= 5:
                    value = float(token.text.replace(',', ''))
                    
                    # Look for unit nearby
                    for neighbor in [doc[i] for i in range(max(0, token.i-2), min(len(doc), token.i+3))]:
                        if neighbor.text in ['%', 'percent', 'pct']:
                            unit = '%'
                            break
                        elif neighbor.text.lower() in ['trillion', 'billion', 'million', 'thousand']:
                            unit = neighbor.text.lower()
                            break
                        elif neighbor.text.lower() in ['mtco2', 'gtco2', 'gw', 'mw', 'twh']:
                            unit = neighbor.text.lower()
                            break
                    
                    break
            except:
                continue
    
    return value, unit


def extract_with_regex(sentence: str) -> List[Dict]:
    """
    Robust regex extraction for common patterns
    """
    items = []
    
    # Pattern 1: "GDP has increased/decreased by 20%"
    pattern1 = r'([A-Z][A-Za-z0-9\s]+?)\s+(?:has\s+)?(?:increased|decreased|grown|fallen|risen|declined)\s+by\s+(?:around\s+|about\s+)?(\d+\.?\d*)\s*%'
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
        before_text = sentence[:match.start()]
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
