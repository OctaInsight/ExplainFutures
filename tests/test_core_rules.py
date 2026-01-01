"""
Unit Tests for Core NLP Extraction Rules

PHASE 3: Regression protection for critical functionality
Tests individual extraction functions to prevent regressions
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nlp.value_parse import (
    parse_value_string,
    parse_range,
    parse_rate,
    parse_from_to,
    parse_percentage_point,
    extract_time_expressions
)
from core.nlp.templates import extract_with_templates
from core.nlp.unification import unify_parameter


def test_time_extraction_by_year():
    """Test: Time expression 'by YEAR'"""
    text = "GDP increases by 20% by 2040."
    time_exprs = extract_time_expressions(text)
    
    assert len(time_exprs) > 0, "Should extract time expression"
    assert any(expr['target_year'] == 2040 for expr in time_exprs), "Should extract year 2040"
    print("✅ PASS: Time extraction 'by 2040'")


def test_time_extraction_range():
    """Test: Time range 'from YEAR to YEAR'"""
    text = "GDP grows from 2030 to 2050."
    time_exprs = extract_time_expressions(text)
    
    assert len(time_exprs) > 0, "Should extract time expression"
    range_expr = [e for e in time_exprs if e.get('baseline_year')]
    assert len(range_expr) > 0, "Should extract time range"
    assert range_expr[0]['baseline_year'] == 2030, "Should extract baseline year 2030"
    assert range_expr[0]['target_year'] == 2050, "Should extract target year 2050"
    print("✅ PASS: Time extraction '2030 to 2050'")


def test_value_range_detection():
    """Test: Value range 'between X and Y'"""
    result = parse_range("between 10 and 15%")
    
    assert result['is_range'], "Should detect as range"
    assert result['value_min'] == 10.0, "Should extract min value"
    assert result['value_max'] == 15.0, "Should extract max value"
    assert result['unit'] == '%', "Should extract unit"
    print("✅ PASS: Value range 'between 10 and 15%'")


def test_rate_detection():
    """Test: Rate 'X% per year'"""
    result = parse_rate("2% per year")
    
    assert result['is_rate'], "Should detect as rate"
    assert result['value'] == 2.0, "Should extract value"
    assert result['rate_period'] == 'per year', "Should extract rate period"
    print("✅ PASS: Rate '2% per year'")


def test_percentage_point_detection():
    """Test: Percentage points vs percent"""
    result_pp = parse_percentage_point("5 percentage points")
    
    assert result_pp['is_percent_point'], "Should detect percentage points"
    assert result_pp['value'] == 5.0, "Should extract value"
    assert result_pp['unit'] == 'pp', "Should use 'pp' unit"
    print("✅ PASS: Percentage points '5 pp'")


def test_from_to_detection():
    """Test: 'from X to Y' expressions"""
    result = parse_from_to("from 10% to 25%")
    
    assert result['is_from_to'], "Should detect from-to"
    assert result['base_value'] == 10.0, "Should extract base value"
    assert result['target_value'] == 25.0, "Should extract target value"
    assert result['value'] == 15.0, "Should calculate delta"
    print("✅ PASS: From-to 'from 10% to 25%'")


def test_delta_vs_target_by():
    """Test: Delta detection with 'by' keyword"""
    context = {'has_change_word': True}
    result = parse_value_string("20%", context)
    
    assert result['value_type'] == 'delta', "Should classify as delta with change word"
    print("✅ PASS: Delta detection 'increases by 20%'")


def test_delta_vs_target_to():
    """Test: Target detection with 'to/reaches' keyword"""
    context = {'has_change_word': False}
    result = parse_value_string("20%", context)
    
    assert result['value_type'] == 'absolute_target', "Should classify as target without change word"
    print("✅ PASS: Target detection 'reaches 20%'")


def test_template_extraction_basic():
    """Test: Template extraction of simple pattern"""
    text = "GDP increases by 20%."
    items = extract_with_templates(text)
    
    assert len(items) > 0, "Should extract at least one item"
    assert any(i['parameter'] == 'GDP' for i in items), "Should extract GDP parameter"
    assert any(i['value'] == 20.0 for i in items), "Should extract value 20"
    print("✅ PASS: Template extraction 'GDP increases by 20%'")


def test_template_extraction_with_year():
    """Test: Template extraction with year"""
    text = "GDP increases by 20% by 2040."
    items = extract_with_templates(text)
    
    assert len(items) > 0, "Should extract at least one item"
    item = items[0]
    assert item.get('target_year') == 2040, f"Should extract target year 2040, got {item.get('target_year')}"
    print("✅ PASS: Template extraction with year")


def test_parameter_unification_gdp():
    """Test: Parameter unification for GDP"""
    result = unify_parameter("gross domestic product")
    
    assert result['canonical'] == 'GDP', "Should unify to 'GDP'"
    assert result['confidence'] >= 0.8, "Should have high confidence"
    print("✅ PASS: Parameter unification 'GDP'")


def test_parameter_unification_emissions():
    """Test: Parameter unification for emissions"""
    result = unify_parameter("carbon dioxide emissions")
    
    assert result['canonical'] == 'CO2 emissions', "Should unify to 'CO2 emissions'"
    assert result['confidence'] >= 0.8, "Should have high confidence"
    print("✅ PASS: Parameter unification 'CO2 emissions'")


def test_range_hyphen_format():
    """Test: Range with hyphen '10-20%'"""
    result = parse_range("10-20%")
    
    assert result['is_range'], "Should detect hyphen range"
    assert result['value_min'] == 10.0, "Should extract min"
    assert result['value_max'] == 20.0, "Should extract max"
    print("✅ PASS: Range hyphen format '10-20%'")


def test_cagr_detection():
    """Test: CAGR rate detection"""
    result = parse_rate("CAGR 3%")
    
    assert result['is_rate'], "Should detect CAGR as rate"
    assert result['value'] == 3.0, "Should extract value"
    assert result['rate_period'] == 'CAGR', "Should identify as CAGR"
    print("✅ PASS: CAGR detection")


def run_all_tests():
    """Run all unit tests"""
    tests = [
        test_time_extraction_by_year,
        test_time_extraction_range,
        test_value_range_detection,
        test_rate_detection,
        test_percentage_point_detection,
        test_from_to_detection,
        test_delta_vs_target_by,
        test_delta_vs_target_to,
        test_template_extraction_basic,
        test_template_extraction_with_year,
        test_parameter_unification_gdp,
        test_parameter_unification_emissions,
        test_range_hyphen_format,
        test_cagr_detection
    ]
    
    print("=" * 80)
    print("UNIT TESTS - Core NLP Extraction Rules")
    print("=" * 80)
    print(f"Running {len(tests)} tests...\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test.__name__}")
            print(f"   {str(e)}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test.__name__}")
            print(f"   {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} passed")
    if failed > 0:
        print(f"⚠️  {failed} tests failed")
    else:
        print("✅ All tests passed!")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
