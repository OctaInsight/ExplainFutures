"""
NLP module for scenario analysis
"""

from .lang_detect import detect_language, validate_english
from .scenario_segment import segment_scenarios
from .parameter_extract import extract_parameters_from_scenarios
from .value_parse import parse_value_string, normalize_unit, format_value
from .mapping import suggest_variable_mapping, create_mapping, apply_mapping
from .clean_text import generate_cleaned_scenario_text, export_to_json, generate_comparison_table
from .schema import Scenario, ScenarioItem

__all__ = [
    'detect_language',
    'validate_english',
    'segment_scenarios',
    'extract_parameters_from_scenarios',
    'parse_value_string',
    'normalize_unit',
    'format_value',
    'suggest_variable_mapping',
    'create_mapping',
    'apply_mapping',
    'generate_cleaned_scenario_text',
    'export_to_json',
    'generate_comparison_table',
    'Scenario',
    'ScenarioItem'
]
