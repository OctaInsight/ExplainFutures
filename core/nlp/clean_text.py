"""
Clean text generation module
Generate cleaned, structured narrative from scenario data

PHASE 2 ENHANCEMENTS:
- Display time expressions and ranges
- Show value semantics (target, delta, range, rate)
- Improved human-readable formatting
"""

from typing import Dict, Optional


def generate_cleaned_scenario_text(scenario: Dict, mappings: Optional[Dict[str, str]] = None) -> str:
    """
    Generate cleaned, structured narrative for a scenario
    
    PHASE 2: Enhanced with time and value semantics display
    
    Parameters:
    -----------
    scenario : dict
        Scenario data with items
    mappings : dict (optional)
        Parameter to variable mappings
        
    Returns:
    --------
    cleaned_text : str
        Structured narrative in markdown format
    """
    output = []
    
    # Title
    output.append(f"## {scenario['title']}")
    output.append("")
    
    # PHASE 2: Enhanced metadata with time range
    baseline = scenario.get('baseline_year')
    horizon = scenario.get('horizon')
    
    if baseline and horizon:
        output.append(f"**Time Period:** {baseline}–{horizon}")
    elif horizon:
        output.append(f"**Target Horizon:** {horizon}")
    else:
        output.append("**Target Horizon:** Not specified")
    
    # PHASE 2: Time confidence
    time_conf = scenario.get('time_confidence', 0)
    if time_conf > 0:
        output.append(f"**Time Confidence:** {time_conf:.0%}")
    
    output.append("")
    
    # Items
    if scenario['items']:
        output.append("### Parameters:")
        output.append("")
        
        for item in scenario['items']:
            line = format_item(item, mappings)
            output.append(line)
        
    else:
        output.append("*No parameters specified*")
    
    return "\n".join(output)


def format_item(item: Dict, mappings: Optional[Dict[str, str]] = None) -> str:
    """
    Format a single item for display
    
    PHASE 2: Enhanced with time and value semantics
    
    Parameters:
    -----------
    item : dict
        Item data
    mappings : dict (optional)
        Parameter to variable mappings
        
    Returns:
    --------
    formatted : str
        Formatted line
    """
    parts = []
    
    # Parameter name
    param_name = item['parameter']
    parts.append(f"**{param_name}**")
    
    # Add mapping if available
    if mappings and param_name in mappings:
        parts.append(f"*(maps to: {mappings[param_name]})*")
    
    # Direction and value with PHASE 2 semantics
    direction = item['direction']
    value = item.get('value')
    unit = item.get('unit', '')
    value_type = item.get('value_type', 'absolute')
    
    # PHASE 2: Handle different value types
    if value_type == 'range':
        value_min = item.get('value_min')
        value_max = item.get('value_max')
        if value_min is not None and value_max is not None:
            parts.append(f"ranges **{format_value(value_min, unit)} to {format_value(value_max, unit)}**")
    
    elif value_type == 'rate':
        if value is not None:
            rate_period = item.get('rate_period', 'per year')
            parts.append(f"grows at **{format_value(value, unit)}** {rate_period}")
    
    elif value_type == 'percent_point':
        if value is not None:
            verb = "increases" if direction == 'increase' else "decreases"
            parts.append(f"{verb} by **{value:.1f} percentage points**")
    
    elif value_type == 'delta':
        # Change from baseline
        base_val = item.get('base_value')
        target_val = item.get('target_value')
        
        if base_val is not None and target_val is not None:
            parts.append(f"changes from **{format_value(base_val, unit)}** to **{format_value(target_val, unit)}**")
        elif value is not None:
            verb = "increases" if direction == 'increase' else "decreases"
            parts.append(f"{verb} by **{format_value(value, unit)}**")
        else:
            verb = "increases" if direction == 'increase' else "decreases"
            parts.append(f"**{verb}** (magnitude to be determined)")
    
    elif value_type == 'absolute_target':
        if value is not None:
            parts.append(f"reaches **{format_value(value, unit)}**")
        else:
            parts.append("reaches target value")
    
    elif direction == 'stable':
        parts.append("remains **stable**")
    
    elif direction == 'double':
        parts.append("**doubles** (+100%)")
    
    elif direction == 'halve':
        parts.append("**halves** (-50%)")
    
    elif direction == 'triple':
        parts.append("**triples** (+200%)")
    
    elif direction in ['increase', 'decrease']:
        if value is not None:
            verb = "increases" if direction == 'increase' else "decreases"
            parts.append(f"{verb} by **{format_value(value, unit)}**")
        else:
            verb = "increases" if direction == 'increase' else "decreases"
            parts.append(f"**{verb}** (magnitude to be determined)")
    
    else:
        parts.append(f"direction: {direction}")
    
    # PHASE 2: Add time information
    baseline_year = item.get('baseline_year')
    target_year = item.get('target_year')
    time_expr = item.get('time_expression', '')
    
    if baseline_year and target_year:
        parts.append(f"**({baseline_year}–{target_year})**")
    elif target_year:
        parts.append(f"**by {target_year}**")
    elif time_expr:
        parts.append(f"**({time_expr})**")
    elif item.get('horizon'):
        parts.append(f"by **{item['horizon']}**")
    
    return "- " + " ".join(parts)


def format_value(value, unit: str) -> str:
    """
    Format value with unit
    
    PHASE 2: Enhanced formatting
    
    Parameters:
    -----------
    value : float or str
        Numeric value (can be string from table)
    unit : str
        Unit string
        
    Returns:
    --------
    formatted : str
    """
    # Convert to float if string
    try:
        if isinstance(value, str):
            value = float(value) if value.strip() else 0.0
        elif value is None:
            value = 0.0
    except (ValueError, AttributeError):
        value = 0.0
    
    if unit == '%':
        return f"{value:.1f}%"
    elif unit == 'pp':
        return f"{value:.1f} pp"
    elif unit in ['billion', 'million', 'thousand', 'trillion']:
        return f"{value:.2f} {unit}"
    elif unit in ['MtCO2', 'GtCO2', 'ktCO2']:
        return f"{value:.2f} {unit}"
    elif unit in ['GW', 'MW', 'TWh', 'GWh']:
        return f"{value:.2f} {unit}"
    elif unit == 'absolute':
        return f"{value:.2f}"
    else:
        return f"{value:.2f} {unit}"


def export_to_json(scenario: Dict) -> str:
    """
    Export scenario to JSON string
    
    Parameters:
    -----------
    scenario : dict
        Scenario data
        
    Returns:
    --------
    json_str : str
        JSON formatted string
    """
    import json
    return json.dumps(scenario, indent=2)


def export_all_to_json(scenarios: list) -> str:
    """
    Export all scenarios to JSON string
    
    Parameters:
    -----------
    scenarios : list
        List of scenario dictionaries
        
    Returns:
    --------
    json_str : str
        JSON formatted string
    """
    import json
    return json.dumps(scenarios, indent=2)


def generate_comparison_table(scenarios: list) -> str:
    """
    Generate markdown table comparing all scenarios
    
    PHASE 2: Enhanced with time and value type display
    
    Parameters:
    -----------
    scenarios : list
        List of scenario dictionaries
        
    Returns:
    --------
    table : str
        Markdown table
    """
    if not scenarios:
        return "*No scenarios to compare*"
    
    # Collect all unique parameters
    all_params = set()
    for scenario in scenarios:
        for item in scenario['items']:
            all_params.add(item['parameter'])
    
    all_params = sorted(list(all_params))
    
    if not all_params:
        return "*No parameters to compare*"
    
    # Create table header
    lines = []
    header = "| Parameter | " + " | ".join([s['title'] for s in scenarios]) + " |"
    separator = "|" + "|".join(["---"] * (len(scenarios) + 1)) + "|"
    
    lines.append(header)
    lines.append(separator)
    
    # Create rows
    for param in all_params:
        row_parts = [param]
        
        for scenario in scenarios:
            # Find item for this parameter
            item = None
            for i in scenario['items']:
                if i['parameter'] == param:
                    item = i
                    break
            
            if item:
                # PHASE 2: Format value with type awareness
                value_type = item.get('value_type', 'absolute')
                
                if value_type == 'range':
                    v_min = item.get('value_min')
                    v_max = item.get('value_max')
                    if v_min is not None and v_max is not None:
                        cell = f"{v_min}-{v_max}{item.get('unit', '')}"
                    else:
                        cell = format_value(item['value'], item.get('unit', ''))
                
                elif item['value'] is not None:
                    cell = format_value(item['value'], item.get('unit', ''))
                    
                    # Add type indicator
                    if value_type == 'delta':
                        cell = f"Δ{cell}"
                    elif value_type == 'rate':
                        cell = f"{cell}/yr"
                else:
                    cell = item['direction']
            else:
                cell = "-"
            
            row_parts.append(cell)
        
        row = "| " + " | ".join(row_parts) + " |"
        lines.append(row)
    
    return "\n".join(lines)
