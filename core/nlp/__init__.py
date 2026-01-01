"""
NLP module for scenario analysis

PHASE 1 IMPROVEMENTS:
- Centralized parameter unification
- ML extraction with configurable thresholds
- Atomic statement splitting
- Reduced false positives

PHASE 2 ENHANCEMENTS:
- First-class time/horizon extraction
- Richer value semantics (target, delta, range, rate)
- Enhanced mapping with token similarity
- Category-aware parameter classification
"""

from .lang_detect import detect_language, validate_english
from .scenario_segment import segment_scenarios
from .parameter_extract import extract_parameters_from_scenarios
from .value_parse import (
    parse_value_string, 
    normalize_unit, 
    format_value,
    parse_range,
    parse_rate,
    parse_from_to,
    parse_percentage_point,
    extract_time_expressions
)
from .mapping import (
    suggest_variable_mapping, 
    create_mapping, 
    apply_mapping,
    get_category_for_parameter
)
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
from .ml_extractor import (
    hybrid_extract,
    extract_with_gliner,
    is_gliner_available,
    get_gliner_status,
    load_gliner_model
)

__all__ = [
    'detect_language',
    'validate_english',
    'segment_scenarios',
    'extract_parameters_from_scenarios',
    'parse_value_string',
    'normalize_unit',
    'format_value',
    'parse_range',
    'parse_rate',
    'parse_from_to',
    'parse_percentage_point',
    'extract_time_expressions',
    'suggest_variable_mapping',
    'create_mapping',
    'apply_mapping',
    'get_category_for_parameter',
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
    'get_template_count',
    'hybrid_extract',
    'extract_with_gliner',
    'is_gliner_available',
    'get_gliner_status',
    'load_gliner_model'
]
