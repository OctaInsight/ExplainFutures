# Tests Directory - ExplainFutures NLP Pipeline

## Overview
This directory contains the evaluation harness and regression tests for the NLP extraction pipeline.

## Files

### gold_cases.json
Gold standard test cases with expected outputs.

**Structure:**
- 15 test cases covering Phase 1 & 2 features
- JSON format for easy extension
- Each case has input text + expected extraction results

**Adding New Cases:**
1. Copy an existing test case structure
2. Modify `id`, `description`, `text`, and `expected` fields
3. Run evaluation to verify

### evaluate_extraction.py
Main evaluation runner that tests extraction pipeline against gold cases.

**Usage:**
```bash
# Run all tests
python tests/evaluate_extraction.py

# Run without ML
python tests/evaluate_extraction.py --no-ml

# Adjust ML threshold
python tests/evaluate_extraction.py --ml-threshold 0.6

# Save report
python tests/evaluate_extraction.py --report results.txt

# Quiet mode
python tests/evaluate_extraction.py --quiet
```

**Features:**
- Automated test execution
- Fuzzy parameter matching
- Value tolerance checking
- Detailed error reporting
- Command-line interface

### test_core_rules.py
Unit tests for individual extraction functions.

**Usage:**
```bash
python tests/test_core_rules.py
```

**Coverage:**
- Time extraction (by, from-to)
- Value ranges
- Rate detection
- Percentage points
- Delta vs target classification
- Parameter unification
- Template matching

**Adding New Tests:**
```python
def test_your_feature():
    """Test: Description"""
    # Arrange
    input_data = ...
    
    # Act
    result = function(input_data)
    
    # Assert
    assert condition, "Error message"
    print("✅ PASS: Test name")
```

Then add to `run_all_tests()` list.

## Quick Start

### Run Full Evaluation
```bash
cd /path/to/project
python tests/evaluate_extraction.py
```

Expected output:
```
================================================================================
EVALUATION HARNESS - ExplainFutures NLP Pipeline
================================================================================
Test Cases: 15
ML Enabled: True
ML Threshold: 0.5
--------------------------------------------------------------------------------
✅ PASS | basic_percent_increase: Simple percentage increase with year
✅ PASS | multiple_params_same_year: Multiple parameters with same target year
...
--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
Total Cases:     15
Passed:          14 (93.3%)
Failed:          1
================================================================================
```

### Run Unit Tests
```bash
python tests/test_core_rules.py
```

### Run from Streamlit
1. Open Scenario Analysis (NLP) page
2. Extract some scenarios
3. Open "🧪 Phase 3 Evaluation Harness" expander
4. Click "▶️ Run Tests"
5. View results and download report

## Metrics

### Current:
- Overall pass rate
- Per-case pass/fail
- Detailed error messages

### Coming Soon:
- Parameter precision/recall
- Direction accuracy
- Value type accuracy
- Horizon accuracy
- Value accuracy

## Best Practices

### Before Code Changes:
1. Run evaluation: `python tests/evaluate_extraction.py`
2. Record baseline pass rate
3. Note any existing failures

### After Code Changes:
1. Run evaluation again
2. Compare pass rates
3. Investigate new failures
4. Fix regressions before committing

### Adding Features:
1. Write test case first (TDD)
2. Implement feature
3. Run tests
4. Iterate until passing

### Regression Protection:
- Add test case for every bug found
- Run tests before every commit
- Include in CI/CD pipeline

## CI/CD Integration

### GitHub Actions Example:
```yaml
name: NLP Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: python tests/test_core_rules.py
      - name: Run evaluation
        run: python tests/evaluate_extraction.py
```

## Troubleshooting

### ImportError: No module named 'core.nlp'
Make sure you run from project root:
```bash
cd /path/to/project
python tests/evaluate_extraction.py
```

### Tests failing unexpectedly
1. Check if gold cases match current expected behavior
2. Verify ML is enabled/disabled correctly
3. Check ML threshold settings
4. Review error messages for specifics

### Adding dependencies
If tests need new dependencies, add to project requirements.txt

## Contributing

When adding test cases:
1. Make them minimal and focused
2. Test one thing per case
3. Use clear, descriptive IDs
4. Add helpful descriptions
5. Document expected behavior

When modifying tests:
1. Update gold cases if behavior changed intentionally
2. Don't modify tests to make them pass - fix the code
3. Add new tests for new features
4. Keep tests independent

## Support

For issues or questions about testing:
1. Check CHANGELOG_PHASE3.md
2. Review test case structure in gold_cases.json
3. Look at existing test examples
4. Check error messages carefully
