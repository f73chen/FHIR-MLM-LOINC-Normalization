"""Tests for analysis/a17_refrange_calibration.py.

Each statistic is checked against a case whose answer is known by construction. One of the checks
is that interval_stats tells a range that is centred but narrow apart from one that is the right
width but shifted.

  python -m pytest tests/test_a17_refrange_calibration.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
_s = importlib.util.spec_from_file_location("a17", ROOT / "analysis" / "a17_refrange_calibration.py")
a17 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(a17)

NAMES = {"X": ("Analyte X", "Ser/Plas", "MCnc"), "Y": ("Analyte Y", "Bld", "MCnc")}


# --------------------------- interval_stats ---------------------------
def test_perfectly_calibrated_range():
    """Uniform(0,100) with a stated range equal to its central 95%."""
    x = np.linspace(0, 100, 200_001)
    r = a17.interval_stats(x, 2.5, 97.5)
    assert r["containment_pct"] == pytest.approx(95, abs=0.02)
    assert r["att_pct_lo"] == pytest.approx(2.5, abs=0.02)
    assert r["att_pct_hi"] == pytest.approx(97.5, abs=0.02)
    assert r["width_ratio"] == pytest.approx(1.0, abs=0.002)
    assert r["centre_offset"] == pytest.approx(0.0, abs=0.002)


def test_range_too_narrow_is_distinguishable_from_range_shifted():
    """The per-code statistics depend on this distinction. Both cases contain ~50% of the data, and
    only the attained percentiles tell them apart."""
    x = np.linspace(0, 100, 200_001)
    narrow = a17.interval_stats(x, 25, 75)          # centred, half the width it should be
    shifted = a17.interval_stats(x, 0, 50)          # right width, slid to the bottom
    assert narrow["containment_pct"] == pytest.approx(50, abs=0.1)
    assert shifted["containment_pct"] == pytest.approx(50, abs=0.1)
    # centred-but-narrow: bounds land symmetrically about the median
    assert narrow["att_pct_lo"] == pytest.approx(25, abs=.1)
    assert narrow["att_pct_hi"] == pytest.approx(75, abs=.1)
    assert narrow["centre_offset"] == pytest.approx(0, abs=.01)
    assert narrow["width_ratio"] > 1.8
    # shifted: both bounds sit in the lower half, and the centre offset is large and positive
    assert shifted["att_pct_lo"] == pytest.approx(0, abs=.1)
    assert shifted["att_pct_hi"] == pytest.approx(50, abs=.1)
    assert shifted["centre_offset"] == pytest.approx(0.5, abs=.01)


def test_attained_percentile_is_the_empirical_cdf():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    r = a17.interval_stats(x, 3.0, 8.0)
    assert r["att_pct_lo"] == pytest.approx(20.0)     # P(x < 3) = 2/10
    assert r["att_pct_hi"] == pytest.approx(80.0)     # P(x <= 8) = 8/10
    assert r["containment_pct"] == pytest.approx(60.0)


def test_interval_stats_degenerate_inputs():
    assert a17.interval_stats(np.array([]), 1, 2) is None
    assert a17.interval_stats(np.array([np.nan, np.nan]), 1, 2) is None
    r = a17.interval_stats(np.array([1.0, 2, 3]), 2.0, 2.0)     # zero-width stated range
    assert np.isnan(r["width_ratio"]) and np.isnan(r["centre_offset"])
    r2 = a17.interval_stats(np.array([1.0, np.nan, 3.0]), 1.0, 3.0)
    assert r2["n"] == 2                                         # NaNs dropped from the count


# --------------------------- frame helpers ---------------------------
def _frame(recs):
    """recs: (pid, y, ordinal, val, lo, hi, gender, age)"""
    return pd.DataFrame(recs, columns=["pid", "y", "obs_ordinal", "val", "rr_low", "rr_high",
                                       "gender", "age_years"])


def test_first_per_patient_takes_lowest_ordinal():
    d = _frame([(1, "X", 5, 50.0, 0, 10, "male", 60), (1, "X", 2, 11.0, 0, 10, "male", 60),
                (1, "Y", 9, 7.0, 0, 10, "male", 60), (2, "X", 1, 3.0, 0, 10, "female", 70)])
    f = a17.first_per_patient(d)
    assert len(f) == 3
    assert f[(f.pid == 1) & (f.y == "X")].val.iloc[0] == 11.0


def test_containment_own_uses_each_observations_own_range():
    """A code with two ranges, each fitting its own subset perfectly. Against a median range the
    containment looks bad. Against the range that each observation carried, it is 100%."""
    recs = [(i, "X", 0, 1.0, 0.0, 2.0, "male", 50) for i in range(2000)]
    recs += [(2000 + i, "X", 0, 101.0, 100.0, 102.0, "female", 50) for i in range(2000)]
    d = _frame(recs)
    r = a17.test_C(d, NAMES, "t", []).iloc[0]
    assert r.containment_own_pct == pytest.approx(100.0)
    assert r.containment_pct < 1.0          # the median range [50, 52] contains nothing
    assert not r.constant_range and r.n_ranges == 2
