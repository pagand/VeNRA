import pytest
import re

def is_exact_match_logic(predicted: str, golden: str) -> bool:
    """
    Stand-alone implementation of the ε-Match logic for testing.
    Mirroring extract_metrics.py implementation.
    """
    def normalize_numeric(text: str):
        # Basic numeric extractor
        clean = text.replace(",", "").replace("$", "").replace("%", "")
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", clean)]

    pred_nums = normalize_numeric(predicted)
    gold_nums = normalize_numeric(golden)

    if not gold_nums:
        return False

    for g in gold_nums:
        for p in pred_nums:
            # Absolute tolerance
            if abs(g - p) < 0.015:
                return True
            # Relative error tolerance (0.5%)
            if g != 0:
                relative_error = abs(g - p) / abs(g)
                if relative_error <= 0.005:
                    return True
    return False

def test_relative_error_tolerance():
    """
    Scientific Test: Verifies the 0.5% Relative Error Tolerance (ε-Match).
    Rescues predictions that are mathematically correct but differ in precision.
    """
    # Netflix Case: Prediction is more precise than gold due to unit scale
    gold = "$5466.00"
    pred = "5,466.312" 
    # error = 0.312 / 5466 = 0.000057 (0.005%) -> Should PASS
    assert is_exact_match_logic(pred, gold) == True

    # Case: Slightly outside tolerance (0.6%)
    gold_fail = "1000.00"
    pred_fail = "1006.50" 
    # error = 0.65% -> Should FAIL
    assert is_exact_match_logic(pred_fail, gold_fail) == False

    # Case: Exact match (Absolute tolerance)
    assert is_exact_match_logic("12.0001", "12.0") == True

    # Case: Percentage conversion (Mocking the cross-scale check in actual scorer)
    # The actual scorer has explicit percentage handling, here we test the float core.
    assert is_exact_match_logic("0.71", "0.71") == True
