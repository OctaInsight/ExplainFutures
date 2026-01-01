"""
Evaluation Harness for NLP Extraction Pipeline

PHASE 3: Regression protection and quality metrics
- Loads gold test cases
- Runs extraction pipeline
- Computes precision/recall metrics
- Reports per-case results and overall statistics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp.parameter_extract import extract_parameters_from_scenarios
from core.nlp.scenario_segment import segment_scenarios


class EvaluationMetrics:
    """Tracks evaluation metrics"""
    
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        
        # Parameter detection
        self.param_true_positives = 0
        self.param_false_positives = 0
        self.param_false_negatives = 0
        
        # Direction accuracy
        self.direction_correct = 0
        self.direction_total = 0
        
        # Value type accuracy
        self.value_type_correct = 0
        self.value_type_total = 0
        
        # Horizon accuracy
        self.horizon_correct = 0
        self.horizon_total = 0
        
        # Value accuracy (within tolerance)
        self.value_correct = 0
        self.value_total = 0
        
        # Per-case results
        self.case_results = []
    
    def precision(self) -> float:
        """Parameter detection precision"""
        if self.param_true_positives + self.param_false_positives == 0:
            return 0.0
        return self.param_true_positives / (self.param_true_positives + self.param_false_positives)
    
    def recall(self) -> float:
        """Parameter detection recall"""
        if self.param_true_positives + self.param_false_negatives == 0:
            return 0.0
        return self.param_true_positives / (self.param_true_positives + self.param_false_negatives)
    
    def f1_score(self) -> float:
        """F1 score for parameter detection"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def direction_accuracy(self) -> float:
        """Direction classification accuracy"""
        if self.direction_total == 0:
            return 0.0
        return self.direction_correct / self.direction_total
    
    def value_type_accuracy(self) -> float:
        """Value type classification accuracy"""
        if self.value_type_total == 0:
            return 0.0
        return self.value_type_correct / self.value_type_total
    
    def horizon_accuracy(self) -> float:
        """Horizon/target year accuracy"""
        if self.horizon_total == 0:
            return 0.0
        return self.horizon_correct / self.horizon_total
    
    def value_accuracy(self) -> float:
        """Value accuracy (within tolerance)"""
        if self.value_total == 0:
            return 0.0
        return self.value_correct / self.value_total
    
    def overall_pass_rate(self) -> float:
        """Overall test case pass rate"""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases


def load_gold_cases(filepath: str = None) -> Dict:
    """Load gold test cases from JSON file"""
    if filepath is None:
        filepath = Path(__file__).parent / "gold_cases.json"
    
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_parameter_name(param: str) -> str:
    """Normalize parameter name for comparison"""
    # Convert to lowercase, remove extra spaces
    param = param.lower().strip()
    
    # Common normalizations
    normalizations = {
        'co2 emissions': 'emissions',
        'carbon dioxide emissions': 'emissions',
        'carbon emissions': 'emissions',
        'renewable energy': 'renewable',
        'renewable energy share': 'renewable',
        'gross domestic product': 'gdp',
        'public investment': 'investment'
    }
    
    return normalizations.get(param, param)


def fuzzy_match_parameter(actual: str, expected: str, threshold: float = 0.8) -> bool:
    """Check if two parameter names match (fuzzy)"""
    actual_norm = normalize_parameter_name(actual)
    expected_norm = normalize_parameter_name(expected)
    
    # Exact match after normalization
    if actual_norm == expected_norm:
        return True
    
    # Fuzzy match
    similarity = SequenceMatcher(None, actual_norm, expected_norm).ratio()
    return similarity >= threshold


def find_matching_item(actual_item: Dict, expected_items: List[Dict]) -> Optional[Dict]:
    """Find matching expected item for an actual extracted item"""
    for expected in expected_items:
        if fuzzy_match_parameter(actual_item.get('parameter', ''), expected.get('parameter', '')):
            return expected
    return None


def compare_values(actual: Optional[float], expected: Optional[float], tolerance: float = 0.1) -> bool:
    """Compare two values with tolerance"""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    
    # Allow small tolerance for floating point
    return abs(actual - expected) <= tolerance


def evaluate_case(case: Dict, enable_ml: bool = True, ml_threshold: float = 0.5) -> Dict:
    """
    Evaluate a single test case
    
    Returns:
    --------
    result : dict
        {
            'case_id': str,
            'passed': bool,
            'errors': List[str],
            'metrics': dict
        }
    """
    case_id = case['id']
    text = case['text']
    expected = case['expected']
    
    result = {
        'case_id': case_id,
        'description': case.get('description', ''),
        'passed': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Run extraction
        scenarios = segment_scenarios(text)
        extracted_scenarios = extract_parameters_from_scenarios(
            scenarios,
            enable_ml=enable_ml,
            ml_confidence_threshold=ml_threshold
        )
        
        # Check number of scenarios
        if len(extracted_scenarios) != expected['num_scenarios']:
            result['errors'].append(
                f"Scenario count mismatch: expected {expected['num_scenarios']}, got {len(extracted_scenarios)}"
            )
            result['passed'] = False
        
        # Evaluate each scenario
        for i, expected_scenario in enumerate(expected['scenarios']):
            if i >= len(extracted_scenarios):
                result['errors'].append(f"Missing scenario {i+1}")
                result['passed'] = False
                continue
            
            actual_scenario = extracted_scenarios[i]
            
            # Check title (if specified)
            expected_title = expected_scenario.get('title')
            if expected_title:
                actual_title = actual_scenario.get('title', '')
                if expected_title.lower() not in actual_title.lower():
                    result['warnings'].append(
                        f"Title mismatch: expected '{expected_title}', got '{actual_title}'"
                    )
            
            # Check horizon (if specified)
            expected_horizon = expected_scenario.get('horizon')
            if expected_horizon is not None:
                actual_horizon = actual_scenario.get('horizon')
                if actual_horizon != expected_horizon:
                    result['errors'].append(
                        f"Horizon mismatch: expected {expected_horizon}, got {actual_horizon}"
                    )
                    result['passed'] = False
            
            # Evaluate items
            expected_items = expected_scenario.get('items', [])
            actual_items = actual_scenario.get('items', [])
            
            matched_expected = set()
            matched_actual = set()
            
            # Match actual items to expected items
            for j, actual_item in enumerate(actual_items):
                matching_expected = find_matching_item(actual_item, expected_items)
                
                if matching_expected is None:
                    # False positive
                    result['warnings'].append(
                        f"Extra parameter detected: {actual_item.get('parameter', 'unknown')}"
                    )
                    continue
                
                expected_idx = expected_items.index(matching_expected)
                matched_expected.add(expected_idx)
                matched_actual.add(j)
                
                # Compare fields
                errors = compare_item_fields(actual_item, matching_expected)
                if errors:
                    result['errors'].extend([f"Parameter '{actual_item.get('parameter', 'unknown')}': {e}" for e in errors])
                    result['passed'] = False
            
            # Check for false negatives
            for idx, expected_item in enumerate(expected_items):
                if idx not in matched_expected:
                    result['errors'].append(
                        f"Missing parameter: {expected_item.get('parameter', 'unknown')}"
                    )
                    result['passed'] = False
        
    except Exception as e:
        result['errors'].append(f"Exception during extraction: {str(e)}")
        result['passed'] = False
    
    return result


def compare_item_fields(actual: Dict, expected: Dict) -> List[str]:
    """Compare fields of an extracted item with expected item"""
    errors = []
    
    # Direction
    expected_direction = expected.get('direction')
    if expected_direction:
        actual_direction = actual.get('direction')
        if actual_direction != expected_direction:
            errors.append(f"direction: expected '{expected_direction}', got '{actual_direction}'")
    
    # Value type
    expected_value_type = expected.get('value_type')
    if expected_value_type:
        actual_value_type = actual.get('value_type')
        if actual_value_type != expected_value_type:
            errors.append(f"value_type: expected '{expected_value_type}', got '{actual_value_type}'")
    
    # Value (with tolerance)
    expected_value = expected.get('value')
    if expected_value is not None:
        actual_value = actual.get('value')
        if not compare_values(actual_value, expected_value, tolerance=0.5):
            errors.append(f"value: expected {expected_value}, got {actual_value}")
    
    # Unit
    expected_unit = expected.get('unit')
    if expected_unit:
        actual_unit = actual.get('unit', '')
        if actual_unit != expected_unit:
            errors.append(f"unit: expected '{expected_unit}', got '{actual_unit}'")
    
    # Target year
    expected_target_year = expected.get('target_year')
    if expected_target_year is not None:
        actual_target_year = actual.get('target_year')
        if actual_target_year != expected_target_year:
            errors.append(f"target_year: expected {expected_target_year}, got {actual_target_year}")
    
    # Baseline year
    expected_baseline_year = expected.get('baseline_year')
    if expected_baseline_year is not None:
        actual_baseline_year = actual.get('baseline_year')
        if actual_baseline_year != expected_baseline_year:
            errors.append(f"baseline_year: expected {expected_baseline_year}, got {actual_baseline_year}")
    
    # Range fields
    expected_is_range = expected.get('is_range', False)
    if expected_is_range:
        actual_is_range = actual.get('is_range', False)
        if not actual_is_range:
            errors.append("is_range: expected True, got False")
        
        expected_min = expected.get('value_min')
        expected_max = expected.get('value_max')
        if expected_min is not None:
            actual_min = actual.get('value_min')
            if not compare_values(actual_min, expected_min):
                errors.append(f"value_min: expected {expected_min}, got {actual_min}")
        if expected_max is not None:
            actual_max = actual.get('value_max')
            if not compare_values(actual_max, expected_max):
                errors.append(f"value_max: expected {expected_max}, got {actual_max}")
    
    # Rate fields
    expected_is_rate = expected.get('is_rate', False)
    if expected_is_rate:
        actual_is_rate = actual.get('is_rate', False)
        if not actual_is_rate:
            errors.append("is_rate: expected True, got False")
        
        expected_rate_period = expected.get('rate_period')
        if expected_rate_period:
            actual_rate_period = actual.get('rate_period', '')
            if expected_rate_period.lower() not in actual_rate_period.lower():
                errors.append(f"rate_period: expected '{expected_rate_period}', got '{actual_rate_period}'")
    
    # From-to fields
    expected_base = expected.get('base_value')
    expected_target = expected.get('target_value')
    if expected_base is not None:
        actual_base = actual.get('base_value')
        if not compare_values(actual_base, expected_base):
            errors.append(f"base_value: expected {expected_base}, got {actual_base}")
    if expected_target is not None:
        actual_target = actual.get('target_value')
        if not compare_values(actual_target, expected_target):
            errors.append(f"target_value: expected {expected_target}, got {actual_target}")
    
    return errors


def compute_aggregate_metrics(case_results: List[Dict]) -> EvaluationMetrics:
    """Compute aggregate metrics from case results"""
    metrics = EvaluationMetrics()
    metrics.total_cases = len(case_results)
    metrics.case_results = case_results
    
    for result in case_results:
        if result['passed']:
            metrics.passed_cases += 1
    
    return metrics


def run_evaluation(
    gold_cases_path: str = None,
    enable_ml: bool = True,
    ml_threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[EvaluationMetrics, List[Dict]]:
    """
    Run full evaluation on all gold cases
    
    Parameters:
    -----------
    gold_cases_path : str, optional
        Path to gold cases JSON file
    enable_ml : bool
        Whether to enable ML extraction
    ml_threshold : float
        ML confidence threshold
    verbose : bool
        Print detailed output
        
    Returns:
    --------
    (metrics, case_results) : Tuple
        Aggregate metrics and per-case results
    """
    # Load gold cases
    gold_data = load_gold_cases(gold_cases_path)
    test_cases = gold_data['test_cases']
    
    if verbose:
        print(f"=" * 80)
        print(f"EVALUATION HARNESS - ExplainFutures NLP Pipeline")
        print(f"=" * 80)
        print(f"Test Cases: {len(test_cases)}")
        print(f"ML Enabled: {enable_ml}")
        print(f"ML Threshold: {ml_threshold}")
        print(f"-" * 80)
    
    # Run evaluation on each case
    case_results = []
    for case in test_cases:
        result = evaluate_case(case, enable_ml=enable_ml, ml_threshold=ml_threshold)
        case_results.append(result)
        
        if verbose:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} | {case['id']}: {case.get('description', '')}")
            if result['errors']:
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"      ERROR: {error}")
            if result['warnings']:
                for warning in result['warnings'][:2]:  # Show first 2 warnings
                    print(f"      WARN:  {warning}")
    
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(case_results)
    
    if verbose:
        print(f"-" * 80)
        print(f"SUMMARY")
        print(f"-" * 80)
        print(f"Total Cases:     {metrics.total_cases}")
        print(f"Passed:          {metrics.passed_cases} ({metrics.overall_pass_rate():.1%})")
        print(f"Failed:          {metrics.total_cases - metrics.passed_cases}")
        print(f"=" * 80)
    
    return metrics, case_results


def generate_report(metrics: EvaluationMetrics, case_results: List[Dict]) -> str:
    """Generate detailed text report"""
    lines = []
    
    lines.append("=" * 80)
    lines.append("EVALUATION REPORT - ExplainFutures NLP Pipeline")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Test Cases:    {metrics.total_cases}")
    lines.append(f"Passed:              {metrics.passed_cases} ({metrics.overall_pass_rate():.1%})")
    lines.append(f"Failed:              {metrics.total_cases - metrics.passed_cases}")
    lines.append("")
    
    # Failed cases
    failed_cases = [r for r in case_results if not r['passed']]
    if failed_cases:
        lines.append("FAILED CASES")
        lines.append("-" * 80)
        for result in failed_cases:
            lines.append(f"❌ {result['case_id']}: {result['description']}")
            for error in result['errors']:
                lines.append(f"   ERROR: {error}")
            lines.append("")
    
    # Warnings
    cases_with_warnings = [r for r in case_results if r['warnings']]
    if cases_with_warnings:
        lines.append("WARNINGS")
        lines.append("-" * 80)
        for result in cases_with_warnings:
            if result['warnings']:
                lines.append(f"⚠️  {result['case_id']}: {result['description']}")
                for warning in result['warnings']:
                    lines.append(f"   WARN: {warning}")
                lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate NLP extraction pipeline")
    parser.add_argument("--gold-cases", type=str, help="Path to gold cases JSON file")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML extraction")
    parser.add_argument("--ml-threshold", type=float, default=0.5, help="ML confidence threshold")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--report", type=str, help="Save report to file")
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics, results = run_evaluation(
        gold_cases_path=args.gold_cases,
        enable_ml=not args.no_ml,
        ml_threshold=args.ml_threshold,
        verbose=not args.quiet
    )
    
    # Generate and save report if requested
    if args.report:
        report = generate_report(metrics, results)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.report}")
    
    # Exit with appropriate code
    sys.exit(0 if metrics.overall_pass_rate() == 1.0 else 1)
