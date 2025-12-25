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
from .unification import (
    unify_parameter, 
    unify_extracted_items, 
    deduplicate_items,
    merge_similar_parameters,
    add_user_synonym,
    get_all_canonical_parameters
)
from .templates import extract_with_templates, get_template_count

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
    'ScenarioItem',
    'unify_parameter',
    'unify_extracted_items',
    'deduplicate_items',
    'merge_similar_parameters',
    'add_user_synonym',
    'get_all_canonical_parameters',
    'extract_with_templates',
    'get_template_count'
]
