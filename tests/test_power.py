"""Tests for the paired-McNemar power and minimum-detectable-effect (MDE) helpers in
analysis/lib/power.py.

The analytic power formula is checked against a Monte Carlo simulation. If the two disagreed, the
MDE would be wrong and could mislabel tail comparisons as adequately powered.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_S = importlib.util.spec_from_file_location(
    "power", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "power.py")
P = importlib.util.module_from_spec(_S)
_S.loader.exec_module(P)


def test_zero_effect_has_power_at_about_alpha():
    assert P.power_mcnemar(100_000, 0.02, 0.0, alpha=0.05) < 0.10


def test_power_increases_with_n():
    ps = [P.power_mcnemar(n, 0.02, 0.002) for n in (1_000, 10_000, 100_000)]
    assert ps[0] < ps[1] < ps[2]


def test_power_increases_with_effect_size():
    ps = [P.power_mcnemar(50_000, 0.02, d) for d in (0.0005, 0.001, 0.004)]
    assert ps[0] < ps[1] < ps[2]


def test_mde_shrinks_as_n_grows():
    m = [P.mde_mcnemar(n, 0.02) for n in (1_000, 10_000, 100_000, 1_000_000)]
    assert all(a > b for a, b in zip(m[:-1], m[1:]))


def test_mde_scales_roughly_as_one_over_sqrt_n():
    """MDE ~ 2.8*sqrt(psi/n), so a 100x larger n should shrink it by about 10x."""
    a = P.mde_mcnemar(10_000, 0.02)
    b = P.mde_mcnemar(1_000_000, 0.02)
    assert 8.0 < a / b < 12.0


def test_mde_grows_with_discordance():
    """Two models that disagree more often need a larger gap before it is detectable."""
    assert P.mde_mcnemar(100_000, 0.10) > P.mde_mcnemar(100_000, 0.01)


def test_mde_delivers_the_requested_power():
    for n, psi in ((5_000, 0.05), (50_000, 0.02), (460_625, 0.004)):
        d = P.mde_mcnemar(n, psi, power=0.80)
        assert P.power_mcnemar(n, psi, d) == pytest.approx(0.80, abs=0.01)


@pytest.mark.parametrize("n,psi,delta", [(2_000, 0.10, 0.02), (10_000, 0.05, 0.006),
                                         (50_000, 0.02, 0.0015)])
def test_analytic_power_matches_monte_carlo(n, psi, delta):
    """The main check: the analytic formula must agree with a simulation of the exact binomial test
    the analysis runs."""
    a = P.power_mcnemar(n, psi, delta)
    s = P.simulate_power(n, psi, delta, n_sim=3000, seed=1)
    assert abs(a - s) < 0.06, f"analytic {a:.3f} vs simulated {s:.3f}"


def test_mde_is_nan_when_effect_cannot_reach_power():
    """With a handful of discordant pairs, no effect reaches 80% power, so the result must be NaN."""
    assert np.isnan(P.mde_mcnemar(20, 0.05))


# -- patients as clusters ----------------------------------------------------
def test_design_effect_is_one_with_one_observation_per_patient():
    rng = np.random.default_rng(0)
    a, b = rng.random(5_000) < 0.9, rng.random(5_000) < 0.8
    assert P.design_effect(np.arange(5_000), a, b) == pytest.approx(1.0)


def test_design_effect_is_the_cluster_size_when_a_patients_observations_are_copies():
    """m copies of one draw carry one observation's information, so the design effect is m."""
    rng = np.random.default_rng(1)
    a, b = rng.random(2_000) < 0.9, rng.random(2_000) < 0.8
    m = 7
    assert P.design_effect(np.repeat(np.arange(2_000), m), np.repeat(a, m), np.repeat(b, m)) == pytest.approx(m)


def test_patient_mde_is_the_independent_mde_at_the_effective_sample_size():
    assert P.mde_mcnemar_patients(400_000, 0.01, 1.0) == pytest.approx(P.mde_mcnemar(400_000, 0.01))
    assert P.mde_mcnemar_patients(400_000, 0.01, 4.0) == pytest.approx(P.mde_mcnemar(100_000, 0.01))
    assert np.isnan(P.mde_mcnemar_patients(400_000, 0.01, float("nan")))


def _clustered_differences(rng, k, m, psi, delta, rho):
    """k patients with m observations each. Returns (patient, d) with d = a_correct - b_correct. An
    observation is discordant with probability psi. A discordant one takes the shared direction of its
    patient with probability rho and its own direction otherwise. Each direction favours a with
    probability 0.5 + delta / (2 psi), so E[d] = delta."""
    pi = 0.5 + delta / (2 * psi)
    shared = np.where(rng.random(k) < pi, 1, -1)
    own = np.where(rng.random((k, m)) < pi, 1, -1)
    sign = np.where(rng.random((k, m)) < rho, shared[:, None], own)
    return np.repeat(np.arange(k), m), np.where(rng.random((k, m)) < psi, sign, 0).ravel()


def test_clustered_mde_delivers_the_requested_power():
    """The main check for patients. Measure the design effect on clustered data, take the MDE at n / deff,
    and simulate the clustered McNemar test (the statistic in dumps.mcnemar_patients) at that difference.
    It must reject about 80% of the time. At the independent MDE it falls far short."""
    rng = np.random.default_rng(2)
    k, m, psi, rho = 1_000, 20, 0.2, 0.9
    pat, d = _clustered_differences(rng, 20 * k, m, psi, 0.0, rho)
    deff = P.design_effect(pat, d == 1, d == -1)
    assert 3.5 < deff < 4.7                         # 1 + (m - 1) * psi * rho^2 = 4.08 under the null
    z = 1.959964

    def power_at(delta, n_sim=2_000):
        hits = 0
        for _ in range(n_sim):
            s = _clustered_differences(rng, k, m, psi, delta, rho)[1].reshape(k, m).sum(1)
            hits += abs(s.sum() / np.sqrt((s ** 2).sum())) > z
        return hits / n_sim

    assert power_at(P.mde_mcnemar_patients(k * m, psi, deff)) == pytest.approx(0.80, abs=0.04)
    assert power_at(P.mde_mcnemar(k * m, psi)) < 0.45
