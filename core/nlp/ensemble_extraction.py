"""
Ensemble Parameter Extraction System
Multi-method extraction with intelligent reconciliation

PHASE 1 IMPROVEMENTS:
- Removed duplicated normalization routines
- Centralized all normalization to unification.py
- Consistent ensemble merging logic
- No competing/conflicting normalization code
- Deterministic processing
"""

import re
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple
import numpy as np


# ============================================================================
# ENSEMBLE EXTRACTION - DELEGATES TO parameter_extract.py
# ============================================================================

def ensemble_extract_parameters(text: str, enable_ml: bool = True, ml_confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Main ensemble extraction entry point
    
    PHASE 1: This now delegates to parameter_extract.py which handles
    the complete pipeline including normalization via unification.py
    
    Parameters:
    -----------
    text : str
        Text to extract from
    enable_ml : bool
        Whether to enable ML (GLiNER) extraction
    ml_confidence_threshold : float
        Minimum confidence for ML extraction
        
    Returns:
    --------
    items : List[Dict]
        Extracted and normalized items
    """
    # Import here to avoid circular dependencies
    from .parameter_extract import extract_items_from_text
    
    # Delegate to parameter_extract which handles all extraction + normalization
    items = extract_items_from_text(
        text,
        enable_ml=enable_ml,
        ml_confidence_threshold=ml_confidence_threshold
    )
    
    return items


# ============================================================================
# UTILITY FUNCTION FOR BACKWARD COMPATIBILITY
# ============================================================================

def clean_parameter_name(parameter: str) -> str:
    """
    PHASE 1: DEPRECATED - Use unification.unify_parameter instead
    
    This function is kept for backward compatibility only.
    All normalization should go through unification.py
    """
    # Just basic cleaning - full normalization is in unification.py
    param = parameter.strip()
    
    # Remove common direction words that shouldn't be part of parameter name
    direction_words = [
        'increases', 'increase', 'increasing', 'increased',
        'decreases', 'decrease', 'decreasing', 'decreased',
        'grows', 'grow', 'growing', 'grown',
        'falls', 'fall', 'falling', 'fallen',
        'rises', 'rise', 'rising', 'risen',
        'declines', 'decline', 'declining', 'declined',
        'doubles', 'double', 'doubling', 'doubled',
        'halves', 'halve', 'halving', 'halved',
        'remains', 'remain', 'remaining', 'remained',
        'reaches', 'reach', 'reaching', 'reached',
        'targets', 'target', 'targeting', 'targeted',
        'by', 'to', 'of', 'in', 'the', 'a', 'an'
    ]
    
    words = param.split()
    cleaned_words = []
    
    for word in words:
        if word.lower() not in direction_words:
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words).strip()


# ============================================================================
# TITLE & YEAR EXTRACTION
# ============================================================================

def extract_title_method_1_header(text: str) -> str:
    """Extract title from first line/header"""
    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # Remove year mentions
        title = re.sub(r'\s*\(?\s*(?:by\s+)?(?:in\s+)?\d{4}\s*\)?', '', first_line)
        title = title.rstrip(':').strip()
        return title
    return None


def extract_title_method_2_scenario_pattern(text: str) -> str:
    """Extract title from 'Scenario X: Title' pattern"""
    patterns = [
        r'Scenario\s*\d*\s*[—–\-:]\s*([^\n(]+)',  # Support em-dash (—), en-dash (–), hyphen (-), colon (:)
        r'Scenario\s*[A-Z]?\s*[—–\-:]\s*([^\n(]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Remove year in parentheses or standalone
            title = re.sub(r'\s*\(?\s*(?:by\s+)?(?:in\s+)?\d{4}\s*\)?', '', title)
            return title.strip()
    
    return None


def extract_title_method_3_spacy_ner(text: str) -> str:
    """Extract title using spaCy NER"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        first_line = text.strip().split('\n')[0]
        doc = nlp(first_line)
        
        # Get largest noun chunk
        noun_chunks = list(doc.noun_chunks)
        if noun_chunks:
            title_chunk = max(noun_chunks, key=lambda x: len(x.text))
            title = title_chunk.text.strip()
            
            # Clean
            title = re.sub(r'^Scenario\s*\d*\s*[—–\-:]\s*', '', title, flags=re.IGNORECASE)
            title = re.sub(r'\s*\(?\s*(?:by\s+)?(?:in\s+)?\d{4}\s*\)?', '', title)
            
            return title.strip()
    except:
        pass
    
    return None


def extract_title_method_4_capital_words(text: str) -> str:
    """Extract title from capitalized words"""
    first_line = text.strip().split('\n')[0]
    
    # Remove year
    clean = re.sub(r'\s*\(?\s*(?:by\s+)?(?:in\s+)?\d{4}\s*\)?', '', first_line)
    
    # Remove scenario prefix
    clean = re.sub(r'^Scenario\s*\d*\s*[—–\-:]\s*', '', clean, flags=re.IGNORECASE)
    
    # Extract capitalized words
    words = clean.split()
    capital_words = [w for w in words if w and w[0].isupper()]
    
    if capital_words:
        return ' '.join(capital_words).strip()
    
    return None


def extract_year_method_1_parentheses(text: str) -> int:
    """Extract year from (by YYYY) or (in YYYY) format"""
    match = re.search(r'\((?:by\s+|in\s+)?(\d{4})\)', text[:200])
    if match:
        year = int(match.group(1))
        if 2020 <= year <= 2100:
            return year
    return None


def extract_year_method_2_temporal_keywords(text: str) -> int:
    """Extract year near temporal keywords"""
    temporal_keywords = ['by', 'in', 'until', 'to', 'year', 'horizon', 'target', 'projected']
    
    for keyword in temporal_keywords:
        pattern = rf'\b{keyword}\b\s+(\d{{4}})'
        match = re.search(pattern, text[:500], re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 2020 <= year <= 2100:
                return year
    
    return None


def extract_year_method_3_spacy_date(text: str) -> int:
    """Extract year using spaCy DATE entity"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text[:500])
        
        for ent in doc.ents:
            if ent.label_ == "DATE":
                year_match = re.search(r'\b(20\d{2})\b', ent.text)
                if year_match:
                    year = int(year_match.group(1))
                    if 2020 <= year <= 2100:
                        return year
    except:
        pass
    
    return None


def extract_year_method_4_any_year(text: str) -> int:
    """Extract any 4-digit year"""
    years = re.findall(r'\b(20\d{2})\b', text[:500])
    
    for year_str in years:
        year = int(year_str)
        if 2020 <= year <= 2100:
            return year
    
    return None


def reconcile_titles(titles: List[str]) -> str:
    """Choose best title from multiple extractions"""
    # Remove None values
    valid_titles = [t for t in titles if t and len(t) > 3]
    
    if not valid_titles:
        return None
    
    # Use longest title (usually most complete)
    return max(valid_titles, key=len)


def reconcile_years(years: List[int]) -> int:
    """Choose best year from multiple extractions"""
    # Remove None values
    valid_years = [y for y in years if y is not None]
    
    if not valid_years:
        return None
    
    # Use most common year
    year_counts = Counter(valid_years)
    return year_counts.most_common(1)[0][0]


def extract_title_and_year_ensemble(text: str) -> Tuple[str, int]:
    """
    Extract scenario title and year using multiple methods
    """
    # Extract title with all methods
    titles = [
        extract_title_method_1_header(text),
        extract_title_method_2_scenario_pattern(text),
        extract_title_method_3_spacy_ner(text),
        extract_title_method_4_capital_words(text)
    ]
    
    # Extract year with all methods
    years = [
        extract_year_method_1_parentheses(text),
        extract_year_method_2_temporal_keywords(text),
        extract_year_method_3_spacy_date(text),
        extract_year_method_4_any_year(text)
    ]
    
    # Reconcile
    best_title = reconcile_titles(titles)
    best_year = reconcile_years(years)
    
    return best_title, best_year
