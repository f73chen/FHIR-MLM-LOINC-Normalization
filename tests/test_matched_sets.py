"""Tests for the matched-observation-set helpers in analysis/lib/dumps.py.

An arm that removes an input field fits more of each record into the window. It can therefore score
more observations than the arm that keeps the field. Their accuracies then have different
denominators, and comparing them needs a paired analysis on the observations both arms scored. These
tests check the structural facts that this analysis relies on, so that a new arm cannot break them.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from lib.dumps import (KEY, coverage, matched, assert_coverage_nested,  # noqa: E402
                       native_vs_matched)


def _frame(keys, correct):
    return pd.DataFrame({"patient_uuid": [k[0] for k in keys],
                         "obs_ordinal": [k[1] for k in keys],
                         "correct": correct})


def test_coverage_returns_the_key_set():
    d = _frame([("a", 0), ("a", 1), ("b", 0)], [True, False, True])
    assert coverage(d) == {("a", 0), ("a", 1), ("b", 0)}


def test_nesting_gate_passes_when_a_field_removal_only_adds():
    less = _frame([("a", 0), ("a", 1)], [True, True])
    more = _frame([("a", 0), ("a", 1), ("a", 2), ("b", 0)], [True] * 4)
    assert assert_coverage_nested(less, more) == 2          # gained 2


def test_nesting_gate_raises_when_observations_are_lost():
    """Catches an ablated arm that drops observations."""
    less = _frame([("a", 0), ("a", 1)], [True, True])
    more = _frame([("a", 1), ("a", 2)], [True, True])       # lost ('a', 0)
    with pytest.raises(AssertionError, match="coverage NOT nested"):
        assert_coverage_nested(less, more, names=("para", "para_norr"))


def test_matched_is_the_intersection_and_row_aligned():
    a = _frame([("a", 0), ("a", 1), ("a", 2)], [True, True, False])
    b = _frame([("a", 1), ("a", 2), ("b", 0)], [False, True, True])
    ma, mb = matched(a, b)
    assert len(ma) == len(mb) == 2
    assert list(zip(ma.patient_uuid, ma.obs_ordinal)) == [("a", 1), ("a", 2)]
    assert (ma[KEY].values == mb[KEY].values).all()


def test_matched_raises_on_a_duplicate_key():
    """A duplicated key would multiply rows and break the row-for-row pairing."""
    a = _frame([("a", 0), ("a", 0)], [True, False])
    b = _frame([("a", 0)], [True])
    with pytest.raises(AssertionError):
        matched(a, b)


def test_native_vs_matched_reports_both_denominators():
    a = _frame([("a", 0), ("a", 1), ("a", 2)], [True, True, True])       # native 1.0
    b = _frame([("a", 0), ("a", 1)], [True, False])                      # native 0.5
    out = native_vs_matched({"a": a, "b": b}).set_index("arm")
    assert out.loc["a", "n_native"] == 3 and out.loc["a", "n_matched"] == 2
    assert out.loc["a", "top1_native"] == 1.0 and out.loc["a", "top1_matched"] == 1.0
    assert out.loc["b", "top1_matched"] == 0.5
    assert out.loc["a", "delta_pp"] == pytest.approx(0.0)


def test_narrower_claim_keeps_more_data():
    """A claim about 2 arms must not be restricted by a 3rd, irrelevant arm."""
    mlm_para = _frame([("a", i) for i in range(10)], [True] * 10)
    mlm_norr = _frame([("a", i) for i in range(12)], [True] * 12)
    bcm_para = _frame([("a", i) for i in range(4)], [True] * 4)
    two = matched(mlm_para, mlm_norr)
    three = matched(mlm_para, mlm_norr, bcm_para)
    assert len(two[0]) == 10 and len(three[0]) == 4
    assert len(two[0]) > len(three[0]), "scoping to the compared arms must retain more observations"


@pytest.mark.slow
def test_real_dumps_are_nested_and_seed_invariant():
    """The empirical facts the matched-set rules rest on, checked on the real v3 dumps at the headline
    cell. In v3 only FHIR-MLM and TF-IDF have a NoDesc arm."""
    from lib import registry as R
    from lib.dumps import load_cell
    D = dict(train_arm="pert", test_arm="pert")
    for m in R.CONDITIONS["nodesc_xsite"]:
        para = load_cell(R.cell(m, 0, condition="para_xsite", **D))
        nodesc = load_cell(R.cell(m, 0, condition="nodesc_xsite", **D))
        gained = assert_coverage_nested(para, nodesc, names=(f"{m}/para", f"{m}/nodesc"))
        assert gained >= 0
    # coverage must not depend on the seed: the joint (code x seed) bootstrap assumes it
    sets = {frozenset(coverage(load_cell(c))) for c in R.cells("mlm", **R.HEADLINE)}
    assert len(sets) == 1, "FHIR-MLM/para coverage differs across seeds"
