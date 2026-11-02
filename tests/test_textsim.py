"""Tests for the text-similarity helpers in analysis/lib/textsim.py.

A query equal to the name of its target must retrieve that target. If it did not, the retrieval
would be broken, and so would every misassignment rate computed from it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_S = importlib.util.spec_from_file_location(
    "textsim", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "textsim.py")
T = importlib.util.module_from_spec(_S)
_S.loader.exec_module(T)

TARGETS = ["Glucose [Mass/volume] in Serum or Plasma",
           "Creatinine [Mass/volume] in Serum or Plasma",
           "Hemoglobin [Mass/volume] in Blood",
           "Potassium [Moles/volume] in Serum or Plasma"]


def _vec(extra=()):
    return T.fit_char_tfidf(list(TARGETS) + list(extra))


def test_verbatim_query_retrieves_its_own_target():
    """A verbatim query retrieves its target, with similarity 1."""
    v = _vec()
    bi, bs, ss = T.nearest_and_self(TARGETS, TARGETS, np.arange(len(TARGETS)), v)
    assert list(bi) == list(range(len(TARGETS)))
    assert np.allclose(bs, 1.0, atol=1e-6)
    assert np.allclose(ss, 1.0, atol=1e-6)


def test_self_score_reports_the_assigned_target_not_the_best():
    """A query assigned to the wrong target must report that target's (low) score as self_score."""
    v = _vec()
    bi, bs, ss = T.nearest_and_self([TARGETS[0]], TARGETS, np.array([2]), v)   # assigned Hemoglobin
    assert bi[0] == 0                      # best match is still Glucose
    assert ss[0] < bs[0]                   # but self_score refers to Hemoglobin
    assert bs[0] == pytest.approx(1.0, abs=1e-6)


def test_similarity_is_case_and_punctuation_tolerant():
    v = _vec(["glucose mass/volume in serum or plasma"])
    _, bs, _ = T.nearest_and_self(["GLUCOSE  [mass/volume]  in SERUM or plasma"], TARGETS,
                                  np.array([0]), v)
    assert bs[0] > 0.7


def test_disjoint_text_scores_low():
    v = _vec(["completely unrelated zzzz qqqq"])
    _, bs, _ = T.nearest_and_self(["completely unrelated zzzz qqqq"], TARGETS, np.array([0]), v)
    assert bs[0] < 0.3


def test_abbreviation_beats_unrelated_but_is_not_perfect():
    """Character n-grams should rank the right analyte first even for a terse local name."""
    v = _vec(["Gluc Ser"])
    bi, bs, _ = T.nearest_and_self(["Gluc Ser"], TARGETS, np.array([0]), v)
    assert bi[0] == 0 and bs[0] < 0.95


def test_chunking_does_not_change_results():
    """A chunk-boundary bug would corrupt a subset of rows."""
    v = _vec()
    q = TARGETS * 7
    si = np.tile(np.arange(len(TARGETS)), 7)
    a = T.nearest_and_self(q, TARGETS, si, v, chunk=100)
    b = T.nearest_and_self(q, TARGETS, si, v, chunk=3)
    assert list(a[0]) == list(b[0])
    assert np.allclose(a[1], b[1], atol=1e-6) and np.allclose(a[2], b[2], atol=1e-6)


def test_negative_self_index_yields_nan_not_a_crash():
    v = _vec()
    _, _, ss = T.nearest_and_self([TARGETS[0]], TARGETS, np.array([-1]), v)
    assert np.isnan(ss[0])
