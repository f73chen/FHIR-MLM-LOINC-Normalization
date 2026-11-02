"""Tests for analysis/lib/codes.py, the join that attaches local codes to the dumps, on synthetic frames
only.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from lib import codes as K  # noqa: E402


# -- codes.attach ------------------------------------------------------------
@pytest.fixture
def fake_lookup(monkeypatch):
    lk = pd.DataFrame({"patient_uuid": ["p", "p", "q"], "obs_ordinal": [0, 1, 0],
                       "true_class": [10, 11, 12], "true_loinc": ["L1", "L2", "L3"]})
    monkeypatch.setattr(K, "_LOOKUP", lk)
    return lk


def test_attach_joins_codes_in_the_dump_s_own_order(fake_lookup):
    d = pd.DataFrame({"patient_uuid": ["q", "p"], "obs_ordinal": [0, 1], "true_loinc": ["L3", "L2"]})
    out = K.attach(d)
    assert out.true_class.tolist() == [12, 11]
    assert out.patient_uuid.tolist() == ["q", "p"]


def test_attach_refuses_an_observation_with_no_code(fake_lookup):
    d = pd.DataFrame({"patient_uuid": ["r"], "obs_ordinal": [0], "true_loinc": ["L9"]})
    with pytest.raises(AssertionError, match="no local code"):
        K.attach(d)


def test_attach_refuses_a_gold_disagreement(fake_lookup):
    """The ordinal names a different observation in the two dumps, and the join must refuse."""
    d = pd.DataFrame({"patient_uuid": ["p"], "obs_ordinal": [0], "true_loinc": ["L2"]})
    with pytest.raises(AssertionError, match="disagree on the gold"):
        K.attach(d)


def test_attach_needs_the_dump_s_gold(fake_lookup):
    with pytest.raises(ValueError):
        K.attach(pd.DataFrame({"patient_uuid": ["p"], "obs_ordinal": [0]}))
