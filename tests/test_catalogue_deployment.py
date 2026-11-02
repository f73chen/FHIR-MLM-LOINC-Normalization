"""Tests for the catalogue deployment metrics in analysis/a9_catalogue_deployment.py.

The calibration and risk-coverage code comes from `lib/calib.py`, which has its own tests. These tests
cover only what a9 adds: per-code scoring, the deployed-accuracy metric and the blast-radius summary.

Deployed accuracy is volume-weighted and per-code accuracy is not, because the two answer different
questions. A bug that collapsed one into the other would still give numbers near 1, so the tests use
small catalogues where the two differ by a known amount.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_S = importlib.util.spec_from_file_location("a9", ROOT / "analysis" / "a9_catalogue_deployment.py")


def _mod():
    m = importlib.util.module_from_spec(_S)
    _S.loader.exec_module(m)
    return m


SPACE = np.array(["1-1", "2-2", "3-3"], dtype=object)


# -- per-code scoring --------------------------------------------------------

def test_confidence_is_the_winning_share_of_mass():
    m = _mod()
    mass = np.array([[6.0, 3.0, 1.0]])          # winner holds 6 of 10
    s = m.code_scores(mass, np.array(["1-1"], dtype=object), SPACE)
    assert s["pred"][0] == "1-1"
    assert s["correct"][0]
    assert s["conf"][0] == pytest.approx(0.6)


def test_confidence_is_scale_invariant():
    """A code with 1,000 observations and one with 10 must not differ in confidence for that reason."""
    m = _mod()
    a = m.code_scores(np.array([[6.0, 3.0, 1.0]]), np.array(["1-1"], dtype=object), SPACE)
    b = m.code_scores(np.array([[600.0, 300.0, 100.0]]), np.array(["1-1"], dtype=object), SPACE)
    assert a["conf"][0] == pytest.approx(b["conf"][0])


def test_zero_mass_code_is_counted_wrong_not_dropped():
    """A code whose mass all fell outside the label space must not improve the metric by being dropped."""
    m = _mod()
    mass = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    s = m.code_scores(mass, np.array(["1-1", "1-1"], dtype=object), SPACE)
    assert s["conf"][0] == 0.0
    assert not s["correct"][0]
    assert s["correct"][1]
    assert s["correct"].mean() == 0.5, "the zero-mass code must stay in the denominator"


# -- deployed accuracy -------------------------------------------------------

def test_deployed_accuracy_is_volume_weighted():
    """A wrong code with almost no data barely lowers deployed accuracy, but halves per-code accuracy.

    This is the case the metric is for. There are two codes, one right with 999 observations and one
    wrong with 1. Per-code accuracy is 0.5 and deployed accuracy is 0.999.
    """
    m = _mod()
    correct = np.array([True, False])
    n_obs = np.array([999, 1])
    assert correct.mean() == 0.5
    assert m.deployed_accuracy(correct, n_obs) == pytest.approx(0.999)


def test_deployed_accuracy_inverts_when_the_frequent_code_is_wrong():
    """Same code counts, opposite assignment: deployed accuracy must swing to 0.001."""
    m = _mod()
    assert m.deployed_accuracy(np.array([False, True]), np.array([999, 1])) == pytest.approx(0.001)


def test_deployed_equals_per_code_when_volumes_are_equal():
    """With uniform volume the two metrics must coincide. This is a sanity check on the weighting."""
    m = _mod()
    correct = np.array([True, False, True, True])
    n_obs = np.array([10, 10, 10, 10])
    assert m.deployed_accuracy(correct, n_obs) == pytest.approx(correct.mean())


# -- blast radius ------------------------------------------------------------

def test_blast_radius_complements_deployed_accuracy_exactly():
    """1 − deployed accuracy must equal the share of volume sitting on wrong codes, by construction.

    A gate in the script asserts this on real data. Here it is tested as an algebraic identity on
    random catalogues.
    """
    m = _mod()
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(3, 40)
        correct = rng.random(n) > 0.4
        n_obs = rng.integers(1, 5000, n)
        dep = m.deployed_accuracy(correct, n_obs)
        br = m.blast_radius(correct, n_obs)
        assert (1.0 - dep) == pytest.approx(br["pct_obs_affected"])


def test_blast_radius_reports_the_concentration_of_errors():
    m = _mod()
    correct = np.array([True, True, False, False])
    n_obs = np.array([500, 300, 2, 4])
    br = m.blast_radius(correct, n_obs)
    assert br["n_codes_wrong"] == 2
    assert br["pct_codes_wrong"] == pytest.approx(0.5)
    assert br["obs_affected"] == 6
    assert br["pct_obs_affected"] == pytest.approx(6 / 806)
    assert br["median_obs_wrong_code"] == pytest.approx(3.0)
    assert br["median_obs_right_code"] == pytest.approx(400.0)
