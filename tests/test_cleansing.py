"""Unit tests for validation rules and cleansing (TC-U1..U5)."""
import numpy as np

from fraudxai import cleansing


def _rule(results, rid):
    return next(r for r in results if r.rule_id == rid)


def test_clean_table_passes_all_blocking_rules(tiny):
    res = cleansing.validate(tiny)
    for rid in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        assert _rule(res, rid).violations == 0


def test_duplicates_and_zero_amount_are_counted(tiny):
    res = cleansing.validate(tiny)
    assert _rule(res, "R8").violations == 1
    assert _rule(res, "R9").violations == 2


def test_missing_negative_and_bad_label_are_caught(tiny):
    bad = tiny.copy()
    bad.loc[0, "V1"] = np.nan
    bad.loc[4, "Amount"] = -1.0
    bad.loc[5, "Class"] = 2
    res = cleansing.validate(bad)
    assert _rule(res, "R3").violations == 1
    assert _rule(res, "R6").violations == 1
    assert _rule(res, "R4").violations == 1


def test_time_order_rule_detects_unsorted_stream(tiny):
    bad = tiny.iloc[::-1].reset_index(drop=True)
    assert _rule(cleansing.validate(bad), "R5").violations > 0


def test_clean_sorts_drops_duplicates_and_logs(tiny):
    out, log = cleansing.clean(tiny.iloc[::-1])
    assert out["Time"].is_monotonic_increasing
    assert log["duplicates_dropped"] == 1 and len(out) == 5
    assert log["fraud_after"] == 1
