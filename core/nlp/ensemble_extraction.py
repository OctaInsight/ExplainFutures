"""
Ensemble Parameter Extraction System
Multi-method extraction with intelligent reconciliation
"""

import re
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple
import numpy as np


# ============================================================================
# METHOD 1: TEMPLATE-BASED EXTRACTION
# ============================================================================

def method_1_template_extraction(text: str) -> pd.DataFrame:
    """
    Fast template-based extraction using known patterns
    """
    results = []
    
    # Pattern 1: "Parameter increases/decreases by X%"
    pattern1 = r'([A-Za-z\s]+?)\s+(increases|decreases|remains|stable)\s+(?:by\s+)?(\d+\.?\d*)\s*%'
    matches = re.finditer(pattern1, text, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        direction = match.group(2).lower()
        value = float(match.group(3))
        
        # Map direction
        direction_map = {
            'increases': 'increase',
            'decreases': 'decrease',
            'remains': 'stable',
            'stable': 'stable'
        }
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': '%',
            'direction': direction_map.get(direction, 'target'),
            'reference_point': 'baseline',
            'confidence': 0.95,
            'source_sentence': match.group(0),
            'method': 'template'
        })
    
    # Pattern 2: "Parameter reaches X%"
    pattern2 = r'([A-Za-z\s]+?)\s+(?:reaches|targets?|achieves?)\s+(\d+\.?\d*)\s*%'
    matches = re.finditer(pattern2, text, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(2))
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': '%',
            'direction': 'target',
            'reference_point': 'target',
            'confidence': 0.90,
            'source_sentence': match.group(0),
            'method': 'template'
        })
    
    # Pattern 3: "Parameter to X unit"
    pattern3 = r'([A-Za-z\s]+?)\s+(?:to|at)\s+(\d+\.?\d*)\s+(MtCO2|GtCO2|GW|MW|TWh|billion|million)'
    matches = re.finditer(pattern3, text, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(2))
        unit = match.group(3)
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': unit,
            'direction': 'target',
            'reference_point': 'absolute',
            'confidence': 0.90,
            'source_sentence': match.group(0),
            'method': 'template'
        })
    
    return pd.DataFrame(results)


# ============================================================================
# METHOD 2: ADVANCED REGEX EXTRACTION
# ============================================================================

def method_2_regex_extraction(text: str) -> pd.DataFrame:
    """
    Flexible regex patterns for various formats
    """
    results = []
    
    # Pattern 1: "X% increase/decrease in Parameter"
    pattern1 = r'(\d+\.?\d*)\s*%\s+(increase|decrease|rise|fall|drop|growth|decline)\s+in\s+([A-Za-z\s]+?)(?:\.|,|;|\n|$)'
    matches = re.finditer(pattern1, text, re.IGNORECASE)
    
    for match in matches:
        value = float(match.group(1))
        direction_word = match.group(2).lower()
        parameter = match.group(3).strip()
        
        # Map direction
        if direction_word in ['increase', 'rise', 'growth']:
            direction = 'increase'
        elif direction_word in ['decrease', 'fall', 'drop', 'decline']:
            direction = 'decrease'
        else:
            direction = 'target'
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': '%',
            'direction': direction,
            'reference_point': 'baseline',
            'confidence': 0.85,
            'source_sentence': match.group(0),
            'method': 'regex'
        })
    
    # Pattern 2: "Parameter: X%"
    pattern2 = r'([A-Za-z\s]+?):\s*(\d+\.?\d*)\s*%'
    matches = re.finditer(pattern2, text, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        value = float(match.group(2))
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': '%',
            'direction': 'target',
            'reference_point': 'target',
            'confidence': 0.80,
            'source_sentence': match.group(0),
            'method': 'regex'
        })
    
    # Pattern 3: "Parameter grows/falls to X%"
    pattern3 = r'([A-Za-z\s]+?)\s+(grows?|falls?|rises?|drops?)\s+to\s+(\d+\.?\d*)\s*%'
    matches = re.finditer(pattern3, text, re.IGNORECASE)
    
    for match in matches:
        parameter = match.group(1).strip()
        direction_word = match.group(2).lower()
        value = float(match.group(3))
        
        direction = 'increase' if direction_word in ['grows', 'grow', 'rises', 'rise'] else 'decrease'
        
        results.append({
            'parameter': parameter,
            'value': value,
            'unit': '%',
            'direction': direction,
            'reference_point': 'target',
            'confidence': 0.85,
            'source_sentence': match.group(0),
            'method': 'regex'
        })
    
    return pd.DataFrame(results)


# ============================================================================
# METHOD 3: SPACY SEMANTIC EXTRACTION
# ============================================================================

def method_3_semantic_extraction(text: str) -> pd.DataFrame:
    """
    Deep semantic understanding with spaCy
    """
    results = []
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            return pd.DataFrame(results)  # Return empty if model not available
    except:
        return pd.DataFrame(results)
    
    doc = nlp(text)
    
    # Extract using dependency parsing
    for sent in doc.sents:
        # Find numbers
        numbers = []
        for token in sent:
            if token.like_num or token.ent_type_ in ['PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL']:
                numbers.append({
                    'value': token.text,
                    'position': token.i,
                    'entity_type': token.ent_type_
                })
        
        if not numbers:
            continue
        
        # Find parameter names (noun chunks)
        for chunk in sent.noun_chunks:
            chunk_text = chunk.text
            
            # Check if this looks like a parameter
            parameter_indicators = [
                'emission', 'energy', 'gdp', 'income', 'productivity',
                'efficiency', 'investment', 'pollution', 'growth',
                'rate', 'level', 'index', 'ratio', 'share', 'cohesion',
                'employment', 'health', 'education', 'inequality'
            ]
            
            if not any(ind in chunk_text.lower() for ind in parameter_indicators):
                continue
            
            # Find closest number
            for num_info in numbers:
                # Check proximity (within 10 tokens)
                if abs(num_info['position'] - chunk.start) <= 10:
                    try:
                        # Parse value
                        value_str = num_info['value']
                        value = float(re.sub(r'[^\d.]', '', value_str))
                        
                        # Determine unit
                        if '%' in sent.text or 'percent' in sent.text.lower():
                            unit = '%'
                        elif 'billion' in sent.text.lower():
                            unit = 'billion'
                        elif 'million' in sent.text.lower():
                            unit = 'million'
                        else:
                            unit = 'absolute'
                        
                        # Determine direction
                        sent_lower = sent.text.lower()
                        if any(word in sent_lower for word in ['increase', 'grow', 'rise', 'up', 'higher']):
                            direction = 'increase'
                        elif any(word in sent_lower for word in ['decrease', 'decline', 'fall', 'reduce', 'lower', 'down']):
                            direction = 'decrease'
                        elif any(word in sent_lower for word in ['reach', 'target', 'achieve', 'at']):
                            direction = 'target'
                        else:
                            direction = 'target'
                        
                        results.append({
                            'parameter': chunk_text,
                            'value': value,
                            'unit': unit,
                            'direction': direction,
                            'reference_point': 'target',
                            'confidence': 0.75,
                            'source_sentence': sent.text,
                            'method': 'semantic'
                        })
                        
                        break  # Only use first number match
                    
                    except:
                        pass
    
    return pd.DataFrame(results)


# ============================================================================
# METHOD 4: STATISTICAL PATTERN EXTRACTION
# ============================================================================

def method_4_statistical_extraction(text: str) -> pd.DataFrame:
    """
    Statistical analysis of text structure
    """
    results = []
    
    lines = text.strip().split('\n')
    
    # Find lines with numbers
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Look for percentage patterns
        if '%' in line or 'percent' in line.lower():
            # Extract all numbers
            numbers = re.findall(r'\d+\.?\d*', line)
            
            if numbers:
                # Get parameter name (text before number)
                for num in numbers:
                    parts = line.split(num)
                    if len(parts) >= 2:
                        param_candidate = parts[0].strip()
                        
                        # Clean parameter name
                        param_candidate = re.sub(r'^\W+|\W+$', '', param_candidate)
                        
                        # Skip if too short or too long
                        if len(param_candidate) < 3 or len(param_candidate) > 100:
                            continue
                        
                        # Determine direction
                        line_lower = line.lower()
                        if 'increase' in line_lower or 'grow' in line_lower:
                            direction = 'increase'
                        elif 'decrease' in line_lower or 'decline' in line_lower:
                            direction = 'decrease'
                        elif 'reach' in line_lower or 'target' in line_lower:
                            direction = 'target'
                        else:
                            direction = 'target'
                        
                        try:
                            value = float(num)
                            
                            results.append({
                                'parameter': param_candidate,
                                'value': value,
                                'unit': '%',
                                'direction': direction,
                                'reference_point': 'baseline',
                                'confidence': 0.65,
                                'source_sentence': line,
                                'method': 'statistical'
                            })
                        except:
                            pass
    
    return pd.DataFrame(results)


# ============================================================================
# METHOD 5: GLINER ML EXTRACTION (IF AVAILABLE)
# ============================================================================

def method_5_gliner_extraction(text: str) -> pd.DataFrame:
    """
    GLiNER machine learning extraction
    """
    results = []
    
    try:
        from core.nlp.ml_extractor import extract_with_gliner
        
        # Use GLiNER extraction
        gliner_results = extract_with_gliner(text)
        
        if gliner_results:
            for item in gliner_results:
                results.append({
                    'parameter': item.get('parameter', ''),
                    'value': item.get('value', 0.0),
                    'unit': item.get('unit', '%'),
                    'direction': item.get('direction', 'target'),
                    'reference_point': 'baseline',
                    'confidence': item.get('confidence', 0.70),
                    'source_sentence': item.get('source_sentence', ''),
                    'method': 'gliner'
                })
    
    except:
        pass  # GLiNER not available
    
    return pd.DataFrame(results)


# ============================================================================
# PARAMETER NAME CLEANING & DEDUPLICATION
# ============================================================================

def clean_parameter_name(param: str) -> str:
    """
    Clean parameter name by removing direction words and artifacts
    
    Examples:
    "GDP increases by" → "GDP"
    "Economic parameters decreases" → "Economic parameters"
    "CO2 emissions reaches" → "CO2 emissions"
    """
    if not param:
        return param
    
    # Convert to string and strip
    param = str(param).strip()
    
    # Remove direction words and artifacts
    direction_patterns = [
        r'\s+increases?\s+by\s*$',
        r'\s+decreases?\s+by\s*$',
        r'\s+increases?\s*$',
        r'\s+decreases?\s*$',
        r'\s+reaches?\s*$',
        r'\s+targets?\s*$',
        r'\s+achieves?\s*$',
        r'\s+grows?\s+to\s*$',
        r'\s+falls?\s+to\s*$',
        r'\s+rises?\s+to\s*$',
        r'\s+drops?\s+to\s*$',
        r'\s+remains?\s*$',
        r'\s+stable\s*$',
        r'\s+by\s*$',
        r'\s+to\s*$',
        r'\s+at\s*$',
    ]
    
    for pattern in direction_patterns:
        param = re.sub(pattern, '', param, flags=re.IGNORECASE)
    
    # Remove trailing/leading whitespace and punctuation
    param = param.strip().rstrip(':').rstrip(',').rstrip(';').strip()
    
    return param


def calculate_parameter_similarity(param1: str, param2: str) -> float:
    """
    Calculate similarity between two parameter names
    Uses multiple methods for robust matching
    """
    # Normalize both
    p1 = param1.lower().strip()
    p2 = param2.lower().strip()
    
    # Exact match
    if p1 == p2:
        return 1.0
    
    # Character-level similarity (Levenshtein-like)
    char_similarity = SequenceMatcher(None, p1, p2).ratio()
    
    # Word-level similarity
    words1 = set(p1.split())
    words2 = set(p2.split())
    
    if words1 and words2:
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
    else:
        word_overlap = 0.0
    
    # Containment check (one is substring of other)
    if p1 in p2 or p2 in p1:
        containment = 0.9
    else:
        containment = 0.0
    
    # Combined score (weighted average)
    similarity = (char_similarity * 0.4 + word_overlap * 0.4 + containment * 0.2)
    
    return similarity


def find_duplicate_parameters(param_list: List[str], threshold: float = 0.85) -> Dict[str, List[str]]:
    """
    Find groups of duplicate/similar parameters
    
    Returns:
    --------
    dict: {canonical_name: [similar_names]}
    
    Example:
    {
        "GDP": ["GDP", "GDP increases by", "gdp", "GDP growth"],
        "CO2 emissions": ["CO2 emissions", "co2", "carbon emissions"]
    }
    """
    if not param_list:
        return {}
    
    # Clean all parameters first
    cleaned_params = [(p, clean_parameter_name(p)) for p in param_list]
    
    # Group similar parameters
    groups = {}
    used = set()
    
    for i, (original_i, cleaned_i) in enumerate(cleaned_params):
        if original_i in used:
            continue
        
        # Start a new group
        group = [original_i]
        
        # Find all similar parameters
        for j, (original_j, cleaned_j) in enumerate(cleaned_params):
            if i != j and original_j not in used:
                similarity = calculate_parameter_similarity(cleaned_i, cleaned_j)
                
                if similarity >= threshold:
                    group.append(original_j)
                    used.add(original_j)
        
        # Choose canonical name (shortest cleaned version)
        canonical = min(group, key=lambda x: len(clean_parameter_name(x)))
        canonical_cleaned = clean_parameter_name(canonical)
        
        groups[canonical_cleaned] = group
        used.add(original_i)
    
    return groups


def deduplicate_extraction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate parameters in an extraction table
    Merges similar parameter names
    """
    if df.empty:
        return df
    
    # Get all unique parameters
    all_params = df['parameter'].unique().tolist()
    
    # Find duplicates
    duplicate_groups = find_duplicate_parameters(all_params, threshold=0.85)
    
    # Create mapping from any variant to canonical name
    param_mapping = {}
    for canonical, variants in duplicate_groups.items():
        for variant in variants:
            param_mapping[variant] = canonical
    
    # Apply mapping
    df['parameter'] = df['parameter'].map(lambda x: param_mapping.get(x, clean_parameter_name(x)))
    
    # Group by parameter and aggregate (keep first value for each parameter)
    if not df.empty:
        # For duplicates within same table, keep the one with highest confidence
        df_dedup = df.sort_values('confidence', ascending=False).groupby('parameter', as_index=False).first()
        return df_dedup
    
    return df


def advanced_parameter_deduplication(all_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Advanced deduplication across all extraction tables
    
    Process:
    1. Collect all unique parameter names from all methods
    2. Find duplicate groups
    3. Create canonical mapping
    4. Apply to all tables
    """
    # Collect all parameters from all tables
    all_params = []
    for method, table in all_tables.items():
        if not table.empty and 'parameter' in table.columns:
            all_params.extend(table['parameter'].unique().tolist())
    
    if not all_params:
        return all_tables
    
    # Find duplicate groups across ALL methods
    duplicate_groups = find_duplicate_parameters(all_params, threshold=0.80)
    
    # Create global mapping
    global_mapping = {}
    for canonical, variants in duplicate_groups.items():
        for variant in variants:
            global_mapping[variant] = canonical
    
    # Apply mapping to all tables
    cleaned_tables = {}
    for method, table in all_tables.items():
        if not table.empty and 'parameter' in table.columns:
            # Apply global mapping
            table_copy = table.copy()
            table_copy['parameter'] = table_copy['parameter'].map(
                lambda x: global_mapping.get(x, clean_parameter_name(x))
            )
            
            # Deduplicate within table
            table_copy = deduplicate_extraction_table(table_copy)
            cleaned_tables[method] = table_copy
        else:
            cleaned_tables[method] = table
    
    return cleaned_tables
    """
    Standardize parameter names
    """
    # Convert to lowercase
    param = param.lower().strip()
    
    # Remove special characters except spaces
    param = re.sub(r'[^\w\s]', ' ', param)
    
    # Remove extra spaces
    param = re.sub(r'\s+', ' ', param).strip()
    
    # Common substitutions
    substitutions = {
        'gdp growth': 'gdp',
        'gross domestic product': 'gdp',
        'co2 emissions': 'co2 emissions',
        'co2 emission': 'co2 emissions',
        'carbon dioxide emissions': 'co2 emissions',
        'carbon dioxide emission': 'co2 emissions',
        'carbon emissions': 'co2 emissions',
        'renewable energy share': 'renewable energy',
        'renewable share': 'renewable energy',
        'renewables': 'renewable energy',
        'income inequality': 'inequality',
        'gini index': 'inequality',
        'gini coefficient': 'inequality',
        'air pollution levels': 'air pollution',
        'pollution levels': 'air pollution',
        'employment rate': 'employment',
        'unemployment': 'employment',
        'public health outcomes': 'health outcomes',
        'health outcome': 'health outcomes',
        'disease rates': 'disease rate',
        'energy efficiency': 'energy efficiency',
        'productivity growth': 'productivity',
        'public investment': 'investment',
        'social cohesion': 'social cohesion',
        'access to education': 'education access',
        'educational access': 'education access'
    }
    
    # Apply substitutions
    for old, new in substitutions.items():
        if param == old:
            return new
    
    return param


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


# ============================================================================
# RECONCILIATION LOGIC
# ============================================================================

def reconcile_extractions(param_name: str, extractions: List[Dict]) -> Dict:
    """
    Reconcile multiple extractions for same parameter
    """
    if not extractions:
        return None
    
    # Extract values and confidences - filter out None values
    raw_values = [e.get('value') for e in extractions]
    raw_confidences = [e.get('confidence', 0.5) for e in extractions]
    methods = [e.get('method', 'unknown') for e in extractions]
    
    # Filter to only valid numeric values
    valid_entries = []
    for i, val in enumerate(raw_values):
        if val is not None:
            try:
                numeric_val = float(val)
                if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                    valid_entries.append({
                        'value': numeric_val,
                        'confidence': float(raw_confidences[i]),
                        'method': methods[i]
                    })
            except (ValueError, TypeError):
                pass  # Skip non-numeric values
    
    if not valid_entries:
        # No valid values, return None
        return None
    
    values = [e['value'] for e in valid_entries]
    confidences = [e['confidence'] for e in valid_entries]
    methods_valid = [e['method'] for e in valid_entries]
    
    # Strategy 1: Voting (most common value)
    value_counts = Counter(values)
    most_common_value, vote_count = value_counts.most_common(1)[0]
    
    # Strategy 2: Weighted average
    try:
        weighted_sum = sum(v * c for v, c in zip(values, confidences))
        total_confidence = sum(confidences)
        weighted_avg = weighted_sum / total_confidence if total_confidence > 0 else most_common_value
    except:
        weighted_avg = most_common_value
    
    # Strategy 3: Highest confidence value
    max_conf_idx = confidences.index(max(confidences))
    highest_conf_value = values[max_conf_idx]
    
    # Decision logic
    if vote_count >= len(values) / 2:  # Majority agrees
        final_value = most_common_value
        confidence_level = 'HIGH'
        decision_method = 'majority_vote'
    elif len(set(values)) == 1:  # All agree
        final_value = values[0]
        confidence_level = 'VERY_HIGH'
        decision_method = 'unanimous'
    elif abs(weighted_avg - most_common_value) < 5:  # Close enough
        final_value = weighted_avg
        confidence_level = 'MEDIUM'
        decision_method = 'weighted_average'
    else:  # Use highest confidence
        final_value = highest_conf_value
        confidence_level = 'MEDIUM'
        decision_method = 'highest_confidence'
    
    # Get most common unit and direction from ALL extractions (not just valid ones)
    units = [e.get('unit', '') for e in extractions if e.get('unit')]
    directions = [e.get('direction', 'target') for e in extractions if e.get('direction')]
    
    unit_counts = Counter(units) if units else Counter([''])
    direction_counts = Counter(directions) if directions else Counter(['target'])
    
    final_unit = unit_counts.most_common(1)[0][0] if unit_counts else ''
    final_direction = direction_counts.most_common(1)[0][0] if direction_counts else 'target'
    
    # Aggregate source sentences
    sources = list(set([e.get('source_sentence', '') for e in extractions if e.get('source_sentence')]))
    
    return {
        'parameter': extractions[0].get('parameter', param_name),  # Use first original name
        'parameter_normalized': param_name,
        'value': round(final_value, 2),
        'unit': final_unit,
        'direction': final_direction,
        'confidence_level': confidence_level,
        'confidence_score': round(total_confidence / len(valid_entries), 2) if valid_entries else 0.5,
        'extraction_count': len(valid_entries),
        'methods_used': ', '.join(sorted(set(methods_valid))),
        'decision_method': decision_method,
        'all_values': values,
        'source_sentences': sources[:3]  # Limit to 3 sources
    }


# ============================================================================
# MAIN ENSEMBLE EXTRACTION
# ============================================================================

def extract_parameters_ensemble(text: str) -> pd.DataFrame:
    """
    Extract parameters using all methods and reconcile
    """
    # Run all extraction methods
    tables = {}
    
    tables['template'] = method_1_template_extraction(text)
    tables['regex'] = method_2_regex_extraction(text)
    tables['semantic'] = method_3_semantic_extraction(text)
    tables['statistical'] = method_4_statistical_extraction(text)
    tables['gliner'] = method_5_gliner_extraction(text)
    
    # === CRITICAL: DEDUPLICATE PARAMETER NAMES ===
    # This removes duplicates like "GDP", "GDP increases by", "GDP decreases"
    tables = advanced_parameter_deduplication(tables)
    
    # Normalize parameter names in all tables
    for method, table in tables.items():
        if not table.empty:
            table['parameter_normalized'] = table['parameter'].apply(normalize_parameter_name)
    
    # Find all unique normalized parameters
    all_params = set()
    for table in tables.values():
        if not table.empty:
            all_params.update(table['parameter_normalized'].unique())
    
    # For each parameter, gather all extractions
    unified_results = []
    
    for param_normalized in all_params:
        extractions = []
        
        for method, table in tables.items():
            if table.empty:
                continue
            
            matches = table[table['parameter_normalized'] == param_normalized]
            
            for _, row in matches.iterrows():
                extractions.append({
                    'method': method,
                    'parameter': row.get('parameter', param_normalized),
                    'value': row.get('value'),
                    'unit': row.get('unit', ''),
                    'direction': row.get('direction', 'target'),
                    'confidence': row.get('confidence', 0.5),
                    'source_sentence': row.get('source_sentence', '')
                })
        
        # Reconcile
        if extractions:
            try:
                reconciled = reconcile_extractions(param_normalized, extractions)
                if reconciled:
                    unified_results.append(reconciled)
            except Exception as e:
                # Skip this parameter if reconciliation fails
                print(f"Warning: Failed to reconcile parameter '{param_normalized}': {e}")
                continue
    
    return pd.DataFrame(unified_results)


# ============================================================================
# CROSS-SCENARIO NORMALIZATION
# ============================================================================

def normalize_across_scenarios(scenarios: List[Dict]) -> List[Dict]:
    """
    Normalize parameter names across all scenarios
    """
    # Collect all parameters from all scenarios
    all_params = {}  # {normalized_name: [original_names]}
    
    for scenario in scenarios:
        for item in scenario.get('items', []):
            param_original = item.get('parameter', '')
            param_normalized = normalize_parameter_name(param_original)
            
            if param_normalized not in all_params:
                all_params[param_normalized] = []
            
            all_params[param_normalized].append(param_original)
    
    # For each normalized parameter, choose best canonical name
    canonical_names = {}
    
    for normalized, originals in all_params.items():
        # Count occurrences
        counter = Counter(originals)
        most_common = counter.most_common(1)[0][0]
        
        # Use most common as canonical
        canonical_names[normalized] = most_common
    
    # Apply canonical names to all scenarios
    for scenario in scenarios:
        for item in scenario.get('items', []):
            param_original = item.get('parameter', '')
            param_normalized = normalize_parameter_name(param_original)
            
            # Set canonical name
            item['parameter_canonical'] = canonical_names.get(param_normalized, param_original)
            item['parameter_normalized'] = param_normalized
    
    return scenarios


# ============================================================================
# MULTI-METHOD TITLE & YEAR EXTRACTION
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
        r'Scenario\s*\d*\s*[-:]\s*([^\n(]+)',
        r'Scenario\s*[A-Z]?\s*[-:]\s*([^\n(]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            title = match.group(1).strip()
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
            title = re.sub(r'^Scenario\s*\d*\s*[-:]\s*', '', title, flags=re.IGNORECASE)
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
    clean = re.sub(r'^Scenario\s*\d*\s*[-:]\s*', '', clean, flags=re.IGNORECASE)
    
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


def normalize_parameter_name(param: str) -> str:
    """
    Standardize parameter names after cleaning
    """
    # First, clean the parameter name (remove direction words)
    param = clean_parameter_name(param)
    
    # Convert to lowercase
    param = param.lower().strip()
    
    # Remove special characters except spaces
    param = re.sub(r'[^\w\s]', ' ', param)
    
    # Remove extra spaces
    param = re.sub(r'\s+', ' ', param).strip()
    
    # Common substitutions
    substitutions = {
        'gdp growth': 'gdp',
        'gross domestic product': 'gdp',
        'co2 emissions': 'co2 emissions',
        'co2 emission': 'co2 emissions',
        'carbon dioxide emissions': 'co2 emissions',
        'carbon dioxide emission': 'co2 emissions',
        'carbon emissions': 'co2 emissions',
        'renewable energy share': 'renewable energy',
        'renewable share': 'renewable energy',
        'renewables': 'renewable energy',
        'income inequality': 'inequality',
        'gini index': 'inequality',
        'gini coefficient': 'inequality',
        'air pollution levels': 'air pollution',
        'pollution levels': 'air pollution',
        'employment rate': 'employment',
        'unemployment': 'employment',
        'public health outcomes': 'health outcomes',
        'health outcome': 'health outcomes',
        'disease rates': 'disease rate',
        'energy efficiency': 'energy efficiency',
        'productivity growth': 'productivity',
        'public investment': 'investment',
        'social cohesion': 'social cohesion',
        'access to education': 'education access',
        'educational access': 'education access',
        'economic parameters': 'economic indicator',
        'environmental parameters': 'environmental indicator'
    }
    
    # Apply substitutions
    for old, new in substitutions.items():
        if param == old:
            return new
    
    return param
