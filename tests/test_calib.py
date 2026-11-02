"""Tests for the calibration and selective-prediction primitives in analysis/lib/calib.py.

Among other things, they check that:
  - temperature scaling does not change the prediction in the class space (accuracy-preserving)
  - with no sibling classes, marginalisation leaves the argmax unchanged
  - equal-mass bins have equal counts (a binning bug would distort every ECE)
  - a perfect confidence ranking gives E-AURC == 0
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_S = importlib.util.spec_from_file_location(
    "calib", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "calib.py")
C = importlib.util.module_from_spec(_S)
_S.loader.exec_module(C)


def centred(z):
    return z - z.max(axis=1, keepdims=True)


# -- marginalisation ----------------------------------------------------------

def test_marginal_logprob_rows_sum_to_one():
    rng = np.random.default_rng(0)
    z = centred(rng.normal(size=(50, 12)).astype(np.float32))
    group = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    p = np.exp(C.marginal_logprob(z, group, 6))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_marginal_logprob_sums_siblings_not_maxes_them():
    """Two classes with equal probability mapping to one group must yield their SUM."""
    z = np.array([[0.0, 0.0, -np.inf]], dtype=np.float32)   # two equal classes, one impossible
    group = np.array([0, 0, 1])
    p = np.exp(C.marginal_logprob(z, group, 2))
    assert p[0, 0] == pytest.approx(1.0, abs=1e-6)          # 0.5 + 0.5
    assert p[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_marginal_logprob_drops_negative_group_and_renormalises():
    z = np.zeros((1, 4), dtype=np.float32)                  # uniform over 4 -> 0.25 each
    group = np.array([0, 1, -1, -1])                        # drop half the mass
    p = np.exp(C.marginal_logprob(z, group, 2))
    assert np.allclose(p, [[0.5, 0.5]], atol=1e-6)          # renormalised over the kept half


def test_dropped_mass_matches_the_dropped_half():
    z = np.zeros((1, 4), dtype=np.float32)
    group = np.array([0, 1, -1, -1])
    assert C.dropped_mass(z, group)[0] == pytest.approx(0.5, abs=1e-6)


def test_marginalisation_preserves_argmax_when_no_siblings():
    rng = np.random.default_rng(1)
    z = centred(rng.normal(size=(200, 8)).astype(np.float32))
    group = np.arange(8)                                     # one-to-one
    logp = C.marginal_logprob(z, group, 8)
    assert (logp.argmax(1) == z.argmax(1)).all()


# -- row_stats ----------------------------------------------------------------

def test_row_stats_fields_are_consistent():
    rng = np.random.default_rng(2)
    z = centred(rng.normal(size=(300, 10)).astype(np.float32))
    logp = C.marginal_logprob(z, np.arange(10), 10)
    true_idx = rng.integers(0, 10, size=300)
    st = C.row_stats(logp, true_idx)
    p = np.exp(logp)
    assert (st["p_top1"] >= st["p_true"] - 1e-6).all()       # top-1 is the max by definition
    assert np.allclose(st["p_top1"], p.max(axis=1), atol=1e-5)
    assert (st["margin"] >= -1e-6).all()
    assert (st["correct"] == (p.argmax(1) == true_idx)).all()
    ent = -(p * np.log(p)).sum(axis=1)
    assert np.allclose(st["entropy"], ent, atol=1e-4)


def test_row_stats_uniform_distribution_has_max_entropy():
    logp = np.log(np.full((1, 8), 1 / 8, dtype=np.float32))
    st = C.row_stats(logp, np.array([0]))
    assert st["entropy"][0] == pytest.approx(np.log(8), abs=1e-4)
    assert st["margin"][0] == pytest.approx(0.0, abs=1e-6)


# -- scalar metrics -----------------------------------------------------------

def test_nll_of_certain_correct_prediction_is_zero():
    assert C.nll(np.array([1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)


def test_nll_matches_hand_computation():
    assert C.nll(np.array([0.5, 0.25])) == pytest.approx(-(np.log(0.5) + np.log(0.25)) / 2)


def test_brier_top_extremes():
    assert C.brier_top(np.array([1.0, 1.0]), np.array([True, True])) == pytest.approx(0.0)
    assert C.brier_top(np.array([1.0, 1.0]), np.array([False, False])) == pytest.approx(1.0)
    assert C.brier_top(np.array([0.5]), np.array([True])) == pytest.approx(0.25)


# -- binning / ECE ------------------------------------------------------------

def test_equal_mass_bins_have_equal_counts():
    """A binning bug here would distort every reported ECE."""
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1.0, size=10_000)
    b = C.calibration_bins(conf, rng.random(10_000) < 0.9, n_bins=10, scheme="mass")
    counts = b["count"][b["nonempty"]]
    assert counts.max() - counts.min() <= 2               # equal up to ties at the edges


def test_equal_width_bins_are_skewed_at_high_confidence():
    """This is why equal-mass bins are preferred. At 0.99 accuracy, equal-width bins put nearly
    everything in one bin."""
    conf = np.random.default_rng(4).uniform(0.98, 1.0, size=5000)
    b = C.calibration_bins(conf, np.ones(5000, bool), n_bins=10, scheme="width")
    assert (b["count"] > 0).sum() <= 2                     # only the top bins populated


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    rng = np.random.default_rng(5)
    conf = rng.uniform(0.0, 1.0, size=200_000)
    correct = rng.random(200_000) < conf                   # accuracy == confidence by construction
    assert C.ece(conf, correct, n_bins=20, scheme="mass") < 0.01


def test_ece_detects_maximal_overconfidence():
    conf = np.ones(1000)                                   # claims certainty
    correct = np.zeros(1000, dtype=bool)                   # always wrong
    assert C.ece(conf, correct, n_bins=10, scheme="mass") == pytest.approx(1.0, abs=1e-6)
    assert C.mce(conf, correct, n_bins=10, scheme="mass") == pytest.approx(1.0, abs=1e-6)


def test_mce_is_at_least_ece():
    rng = np.random.default_rng(6)
    conf = rng.uniform(0, 1, 20_000)
    correct = rng.random(20_000) < conf ** 2
    assert C.mce(conf, correct) >= C.ece(conf, correct) - 1e-12


# -- risk-coverage ------------------------------------------------------------

def test_perfect_ranking_gives_zero_excess_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, True, False, False])   # confidence ranks errors perfectly
    rc = C.risk_coverage(conf, correct)
    assert rc["e_aurc"] == pytest.approx(0.0, abs=1e-12)


def test_full_coverage_risk_equals_base_error_rate():
    rng = np.random.default_rng(7)
    correct = rng.random(5000) < 0.8
    rc = C.risk_coverage(rng.random(5000), correct)
    assert rc["risk"][-1] == pytest.approx(1 - correct.mean(), abs=1e-12)


def test_inverted_ranking_is_worse_than_random():
    rng = np.random.default_rng(8)
    correct = rng.random(4000) < 0.7
    good = C.risk_coverage(correct.astype(float), correct)["aurc"]
    bad = C.risk_coverage(1 - correct.astype(float), correct)["aurc"]
    assert bad > good


def test_risk_at_and_coverage_at_round_trip():
    conf = np.linspace(1, 0, 1000)
    correct = np.arange(1000) < 900                        # top 900 correct
    rc = C.risk_coverage(conf, correct)
    assert C.risk_at_coverage(rc, (0.9,))[0.9] == pytest.approx(0.0, abs=1e-9)
    assert C.coverage_at_risk(rc, 0.0) == pytest.approx(0.9, abs=1e-3)
    assert C.coverage_at_risk(rc, -1.0) == 0.0             # unachievable


# -- temperature scaling ------------------------------------------------------

def test_temperature_scaling_never_changes_the_prediction():
    """T-scaling must preserve accuracy. This catches almost any indexing error."""
    rng = np.random.default_rng(9)
    z = centred(rng.normal(size=(500, 20)).astype(np.float32))
    base = C.marginal_logprob(z, np.arange(20), 20, T=1.0).argmax(1)
    for T in (0.25, 0.5, 2.0, 5.0, 10.0):
        assert (C.marginal_logprob(z, np.arange(20), 20, T=T).argmax(1) == base).all()


def test_grouping_BREAKS_argmax_invariance_in_general():
    """Documents a real caveat.

    Temperature scaling preserves accuracy in the native class space, because softmax is monotone and
    the argmax cannot move. Once sibling classes are summed into a LOINC, that guarantee is gone. As T
    rises, a group of several mediocre classes gains relative to one sharp class, and the argmax LOINC
    can flip. So "temperature scaling does not change accuracy" must not be asserted for the
    marginalised numbers. It has to be measured. Here, with 2 siblings per group and diffuse logits,
    15% of rows flip at T=3."""
    rng = np.random.default_rng(10)
    z = centred(rng.normal(size=(400, 12)).astype(np.float32))
    group = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    base = C.marginal_logprob(z, group, 6, T=1.0).argmax(1)
    changed = (C.marginal_logprob(z, group, 6, T=3.0).argmax(1) != base).mean()
    assert changed > 0.0, "expected the invariance to break; if it holds the caveat can be dropped"


def test_grouping_argmax_is_stable_for_PEAKED_distributions():
    """On real data the caveat has little effect. The model predictions are sharply peaked, and the
    summed siblings cannot overtake a dominant class. The calibration analysis measures the flip rate
    on real data. This test does not stand in for that measurement."""
    rng = np.random.default_rng(10)
    group = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def agreement(scale):
        z = centred(rng.normal(size=(400, 12)).astype(np.float32) * scale)
        base = C.marginal_logprob(z, group, 6, T=1.0).argmax(1)
        return (C.marginal_logprob(z, group, 6, T=3.0).argmax(1) == base).mean()

    diffuse, peaked = agreement(1), agreement(24)
    assert peaked > diffuse, "peakedness should stabilise the argmax"
    assert peaked > 0.99


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(11)
    n, k, T_true = 40_000, 15, 2.5
    z0 = rng.normal(size=(n, k)).astype(np.float32)          # "true" calibrated logits
    p = np.exp(z0 - np.log(np.exp(z0).sum(1, keepdims=True)))
    labels = np.array([rng.choice(k, p=row) for row in p])
    z_obs = centred(z0 * T_true)                             # observed logits are T_true too sharp
    fit = C.fit_temperature(z_obs, labels)
    assert fit["T"] == pytest.approx(T_true, rel=0.08)


def test_fit_temperature_improves_or_matches_nll():
    rng = np.random.default_rng(12)
    z = centred(rng.normal(size=(5000, 10)).astype(np.float32) * 3)
    labels = rng.integers(0, 10, 5000)
    fit = C.fit_temperature(z, labels)
    assert fit["nll_at_T"] <= fit["nll_at_1"] + 1e-9


def test_fit_temperature_returns_one_for_already_calibrated_logits():
    rng = np.random.default_rng(13)
    n, k = 40_000, 12
    z0 = rng.normal(size=(n, k)).astype(np.float32)
    p = np.exp(z0 - np.log(np.exp(z0).sum(1, keepdims=True)))
    labels = np.array([rng.choice(k, p=row) for row in p])
    assert C.fit_temperature(centred(z0), labels)["T"] == pytest.approx(1.0, rel=0.10)


# -- fast temperature objective -----------------------------------------------

def _true_cols_for(group, n_groups, true_idx):
    """columns of z_kept belonging to each row's true group, -1 padded."""
    kept = np.flatnonzero(group >= 0)
    by_g = {}
    for j, c in enumerate(kept):
        by_g.setdefault(group[c], []).append(j)
    m = max(len(v) for v in by_g.values())
    out = np.full((len(true_idx), m), -1, dtype=np.int64)
    for i, g in enumerate(true_idx):
        cols = by_g.get(g, [])
        out[i, :len(cols)] = cols
    return out, kept


def test_fast_objective_matches_the_reference_marginal():
    """The fast path exists only for speed, so it must agree with marginal_logprob."""
    rng = np.random.default_rng(20)
    z = centred(rng.normal(size=(300, 12)).astype(np.float32))
    group = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, -1, 5, 5])   # includes a dropped class
    true_idx = rng.integers(0, 6, size=300)
    tc, kept = _true_cols_for(group, 6, true_idx)
    f = C.fast_temperature_objective(z[:, kept], tc)
    for T in (0.4, 1.0, 2.7, 8.0):
        ref = C.temperature_nll(z, true_idx, T, group, 6)
        assert f(T) == pytest.approx(ref, abs=1e-5), f"mismatch at T={T}"


def test_fast_objective_recovers_a_known_temperature():
    rng = np.random.default_rng(21)
    n, k, T_true = 30_000, 12, 0.6                     # T<1: an underconfident model, as observed
    z0 = rng.normal(size=(n, k)).astype(np.float32)
    p = np.exp(z0 - np.log(np.exp(z0).sum(1, keepdims=True)))
    labels = np.array([rng.choice(k, p=row) for row in p])
    z_obs = centred(z0 * T_true)
    group = np.arange(k)
    tc, kept = _true_cols_for(group, k, labels)
    fit = C.minimise_scalar(C.fast_temperature_objective(z_obs[:, kept], tc))
    assert fit["T"] == pytest.approx(T_true, rel=0.10)


def test_minimise_scalar_finds_a_known_minimum():
    fit = C.minimise_scalar(lambda t: (t - 3.0) ** 2 + 1.0)
    assert fit["T"] == pytest.approx(3.0, abs=1e-3)
    assert fit["nll_at_T"] == pytest.approx(1.0, abs=1e-5)


# -- clustered bootstrap ------------------------------------------------------

def test_bootstrap_clustered_point_matches_direct_computation():
    rng = np.random.default_rng(30)
    n = 4000
    pat = rng.integers(0, 200, size=n)
    conf, corr = rng.uniform(0.5, 1, n), rng.random(n) < 0.9
    r = C.bootstrap_clustered(pat, lambda conf, correct: C.ece(conf, correct),
                              n_boot=20, conf=conf, correct=corr)
    assert r["point"] == pytest.approx(C.ece(conf, corr), abs=1e-12)
    assert r["lo"] <= r["point"] <= r["hi"] or r["n_boot"] < 5


def test_bootstrap_clustered_resamples_whole_patients():
    """The ragged gather must return the full row block of each chosen patient."""
    pat = np.array([0, 0, 0, 1, 1, 2])
    val = np.array([10., 10., 10., 20., 20., 30.])
    seen = []
    C.bootstrap_clustered(pat, lambda v: seen.append(np.sort(v)) or 0.0, n_boot=25, v=val)
    for s in seen:
        # every value present must come in whole patient blocks (3, 2 and 1 rows)
        for value, mult in ((10., 3), (20., 2), (30., 1)):
            cnt = int((s == value).sum())
            assert cnt % mult == 0, f"patient block split: {value} appeared {cnt} times"


def test_bootstrap_clustered_interval_widens_with_clustering():
    """Clustered CIs must be wider than a naive observation-level bootstrap when rows within a
    patient are correlated. This is the reason for clustering."""
    rng = np.random.default_rng(31)
    n_pat, per = 60, 50
    pat = np.repeat(np.arange(n_pat), per)
    # strong within-patient correlation: each patient is entirely right or entirely wrong
    corr = np.repeat(rng.random(n_pat) < 0.7, per)
    conf = np.full(n_pat * per, 0.8)
    clustered = C.bootstrap_clustered(pat, lambda conf, correct: correct.mean(),
                                      n_boot=300, conf=conf, correct=corr)
    naive = C.bootstrap_clustered(np.arange(len(corr)), lambda conf, correct: correct.mean(),
                                  n_boot=300, conf=conf, correct=corr)
    assert (clustered["hi"] - clustered["lo"]) > (naive["hi"] - naive["lo"]) * 3


def test_identity_grouping_fast_path_is_bit_identical_to_the_loop():
    """TF-IDF's one-column-per-LOINC grouping takes a direct log-softmax. It must equal the logaddexp
    accumulation exactly, or the calibration numbers would stop reproducing."""
    import numpy as np
    from scipy.special import logsumexp
    rng = np.random.default_rng(1)
    z = (rng.normal(size=(800, 50)) * 5).astype(np.float32)
    z -= z.max(axis=1, keepdims=True)
    g = np.arange(50)
    fast = C.marginal_logprob(z, g, 50)
    out = np.full_like(z, -np.inf)
    for c in range(50):
        out[:, c] = np.logaddexp(out[:, c], z[:, c])
    out -= logsumexp(out, axis=1, keepdims=True)
    assert fast.dtype == out.dtype and np.array_equal(fast, out)
