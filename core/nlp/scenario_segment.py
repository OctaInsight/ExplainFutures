"""
Scenario segmentation module
Detects and splits multiple scenarios in text
"""

import re
from typing import List, Dict


def segment_scenarios(text: str) -> List[Dict[str, str]]:
    """
    Segment text into individual scenarios
    
    Uses hierarchical detection:
    1. Explicit scenario headers (e.g., "Scenario 1:", "Scenario A:")
    2. Named scenarios (e.g., "Baseline", "Optimistic", "Pessimistic")
    3. Paragraph breaks (double newlines)
    4. Single scenario (no segmentation)
    
    Parameters:
    -----------
    text : str
        Input text potentially containing multiple scenarios
        
    Returns:
    --------
    scenarios : List[Dict]
        List of dictionaries with:
        {
            'id': str (S1, S2, ...),
            'title': str,
            'text': str (scenario content)
        }
    """
    # Clean text
    text = text.strip()
    
    if not text:
        return []
    
    # Level 1: Explicit numbered/lettered scenario headers
    scenarios = detect_explicit_headers(text)
    if scenarios:
        return scenarios
    
    # Level 2: Named scenario patterns
    scenarios = detect_named_scenarios(text)
    if scenarios:
        return scenarios
    
    # Level 3: Paragraph breaks
    scenarios = split_by_paragraphs(text)
    if len(scenarios) > 1:
        return scenarios
    
    # Level 4: Single scenario
    return [{
        'id': 'S1',
        'title': 'Main Scenario',
        'text': text
    }]


def detect_explicit_headers(text: str) -> List[Dict[str, str]]:
    """
    Detect explicit scenario headers like:
    - "Scenario 1:", "Scenario A:", "Scenario I:"
    - "### Scenario 1"
    - "## 1. Optimistic Growth"
    """
    # Pattern for scenario headers
    patterns = [
        r'(?:^|\n)(?:#{1,3}\s*)?Scenario\s+([0-9]+|[A-Z]|[IVX]+)\s*:?\s*([^\n]*)',
        r'(?:^|\n)(?:#{1,3}\s*)?([0-9]+|[A-Z])\.\s+([^\n]+)',
    ]
    
    scenarios = []
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if matches:
            for i, match in enumerate(matches):
                # Extract ID and title
                scenario_id = f"S{i+1}"
                
                if len(match.groups()) >= 2:
                    title = match.group(2).strip() or f"Scenario {match.group(1)}"
                else:
                    title = f"Scenario {i+1}"
                
                # Extract text (from current match to next match or end)
                start_pos = match.end()
                end_pos = matches[i+1].start() if i+1 < len(matches) else len(text)
                scenario_text = text[start_pos:end_pos].strip()
                
                scenarios.append({
                    'id': scenario_id,
                    'title': title,
                    'text': scenario_text
                })
            
            return scenarios
    
    return []


def detect_named_scenarios(text: str) -> List[Dict[str, str]]:
    """
    Detect scenarios by common names:
    - Baseline, Business as Usual, BAU
    - Optimistic, Best Case
    - Pessimistic, Worst Case
    - Conservative, Moderate, Aggressive
    """
    scenario_names = [
        'Baseline',
        'Business as Usual',
        'BAU',
        'Optimistic',
        'Best Case',
        'Pessimistic',
        'Worst Case',
        'Conservative',
        'Moderate',
        'Aggressive',
        'Low Growth',
        'High Growth',
        'Accelerated',
        'Delayed'
    ]
    
    # Create pattern
    names_pattern = '|'.join([re.escape(name) for name in scenario_names])
    pattern = rf'(?:^|\n)(?:#{1,3}\s*)?({names_pattern})(?:\s+Scenario)?\s*:?\s*([^\n]*)'
    
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
    
    if not matches:
        return []
    
    scenarios = []
    
    for i, match in enumerate(matches):
        scenario_id = f"S{i+1}"
        title = match.group(1).strip()
        
        # Extract text
        start_pos = match.end()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(text)
        scenario_text = text[start_pos:end_pos].strip()
        
        scenarios.append({
            'id': scenario_id,
            'title': title,
            'text': scenario_text
        })
    
    return scenarios


def split_by_paragraphs(text: str) -> List[Dict[str, str]]:
    """
    Split text by paragraph breaks (double newlines)
    """
    # Split on double newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= 1:
        return []
    
    # Create scenarios
    scenarios = []
    for i, para in enumerate(paragraphs):
        scenarios.append({
            'id': f"S{i+1}",
            'title': f"Scenario {i+1}",
            'text': para
        })
    
    return scenarios


def merge_scenarios(scenarios: List[Dict[str, str]], indices: List[int]) -> Dict[str, str]:
    """
    Merge multiple scenarios into one
    
    Parameters:
    -----------
    scenarios : List[Dict]
        List of scenarios
    indices : List[int]
        Indices of scenarios to merge
        
    Returns:
    --------
    merged : Dict
        Merged scenario
    """
    if not indices:
        return {}
    
    # Collect texts
    texts = [scenarios[i]['text'] for i in indices if i < len(scenarios)]
    merged_text = '\n\n'.join(texts)
    
    # Create title
    titles = [scenarios[i]['title'] for i in indices if i < len(scenarios)]
    merged_title = ' + '.join(titles)
    
    return {
        'id': f"S{indices[0]+1}_merged",
        'title': merged_title,
        'text': merged_text
    }
