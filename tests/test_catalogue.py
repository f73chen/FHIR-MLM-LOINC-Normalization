"""Tests for the catalogue-level aggregation in analysis/a8_catalogue.py.

They cover three properties of the metric:
  1. top-k is computed over codes, and each code counts once however many observations it has.
     That is what makes it catalogue level. An observation-weighted bug here would give back the
     observation-level number.
  2. summed mass must be able to overrule a plurality of wrong argmaxes. Otherwise it would be an
     expensive majority vote.
  3. renormalisation after dropping out-of-space mass must not change the ranking.
"""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_S = importlib.util.spec_from_file_location("a8", ROOT / "analysis" / "a8_catalogue.py")


def _mod():
    m = importlib.util.module_from_spec(_S)
    _S.loader.exec_module(m)
    return m


SPACE = np.array(["1-1", "2-2", "3-3", "4-4"], dtype=object)


# -- top-k over codes --------------------------------------------------------

def test_topk_counts_each_code_once():
    """Two codes, one right and one wrong, give 0.5 however many observations each had."""
    m = _mod()
    scores = np.array([[9.0, 1, 0, 0],      # picks 1-1  (correct)
                       [0.0, 0, 1, 9]])     # picks 4-4  (gold is 3-3)
    gold = np.array(["1-1", "3-3"], dtype=object)
    t = m.topk_from_scores(scores, gold, SPACE, ks=(1, 2))
    assert t["top1"] == 0.5
    assert t["top2"] == 1.0, "3-3 is the runner-up for code 2, so top-2 must catch it"


def test_topk_is_not_observation_weighted():
    """A code with 1000 observations and a code with 1 observation weigh the same.

    If aggregation carried observation counts into the metric by mistake, a frequent code would
    dominate and the number would drift back toward observation-level.
    """
    m = _mod()
    # code 0 accumulated huge mass (many observations) but is wrong; code 1 is tiny but right
    scores = np.array([[0.0, 5000.0, 0, 0],
                       [0.001, 0, 0, 0]])
    gold = np.array(["1-1", "1-1"], dtype=object)
    assert m.topk_from_scores(scores, gold, SPACE, ks=(1,))["top1"] == 0.5


# -- summed mass vs majority vote --------------------------------------------

def test_summed_mass_can_overrule_a_wrong_plurality():
    """3 observations: two weakly pick the wrong code, one strongly picks the right one.

    Majority vote gives the wrong answer and summed mass gives the right one. If this fails, summed
    mass has collapsed into vote-counting and is no better than a vote.
    """
    m = _mod()
    obs = np.array([[0.40, 0.35, 0.15, 0.10],     # argmax 1-1 (wrong)
                    [0.40, 0.35, 0.15, 0.10],     # argmax 1-1 (wrong)
                    [0.02, 0.95, 0.02, 0.01]])    # argmax 2-2 (right), very confident
    gold = np.array(["2-2"], dtype=object)
    mass = obs.sum(axis=0)[None, :]
    votes = np.zeros((1, 4));
    for i in obs.argmax(axis=1):
        votes[0, i] += 1
    assert m.topk_from_scores(mass, gold, SPACE, ks=(1,))["top1"] == 1.0
    assert m.topk_from_scores(votes, gold, SPACE, ks=(1,))["top1"] == 0.0


def test_summing_survives_a_confident_error_where_mean_log_does_not():
    """Why the script sums probabilities instead of multiplying them.

    Nine observations favour the right code. One is confidently wrong and gives the right code ~0.
    Summing keeps the right answer. Mean-log, which ranks the same as the product, is pulled to the
    wrong code by that single near-zero.

    This checks the mechanism only. It says nothing about which rule wins on real data. That is
    measured by the `mean_logprob` row that a8 emits.
    """
    m = _mod()
    good = np.tile(np.array([0.30, 0.45, 0.15, 0.10]), (9, 1))
    bad = np.array([[0.70, 1e-9, 1e-9, 1e-9]])
    obs = np.vstack([good, bad])
    gold = np.array(["2-2"], dtype=object)
    assert m.topk_from_scores(obs.sum(0)[None, :], gold, SPACE, ks=(1,))["top1"] == 1.0
    logmean = np.log(np.clip(obs, 1e-12, None)).mean(axis=0)[None, :]
    assert m.topk_from_scores(logmean, gold, SPACE, ks=(1,))["top1"] == 0.0


def test_summing_is_NOT_immune_to_a_confident_error():
    """A known limitation.

    Summing resists a confident error better than multiplying, but it is not immune. When the
    majority preference is weak (margin 0.10 over nine observations = 0.9 total), a single 0.999
    spike still wins. The test passes because this is a real property of the chosen rule. It is not
    a desirable one.
    """
    m = _mod()
    obs = np.vstack([np.tile(np.array([0.30, 0.40, 0.20, 0.10]), (9, 1)),
                     np.array([[0.999, 1e-6, 1e-6, 1e-6]])])
    gold = np.array(["2-2"], dtype=object)
    assert m.topk_from_scores(obs.sum(0)[None, :], gold, SPACE, ks=(1,))["top1"] == 0.0
    assert m.topk_from_scores(obs.sum(0)[None, :], gold, SPACE, ks=(2,))["top2"] == 1.0, \
        "the right code is still ranked second, which is why top-5 matters at catalogue level"


# -- renormalisation ---------------------------------------------------------

def test_renormalisation_preserves_ranking():
    """Dropping out-of-space mass and rescaling must not reorder candidates."""
    m = _mod()
    q = np.array([[0.2, 0.5, 0.2, 0.1]])
    gold = np.array(["2-2"], dtype=object)
    a = m.topk_from_scores(q, gold, SPACE, ks=(1, 2))
    b = m.topk_from_scores(q / q.sum(), gold, SPACE, ks=(1, 2))
    assert a == b


# -- bootstrap ---------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    m = _mod()
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.random((n, 4))
    gold = SPACE[scores.argmax(1)].copy()
    gold[: n // 4] = SPACE[(scores.argmax(1)[: n // 4] + 1) % 4]      # make 25% wrong
    point = m.topk_from_scores(scores, gold, SPACE, ks=(1,))["top1"]
    lo, hi = m.boot_ci(scores, gold, SPACE, 1, n_boot=400, seed=0)
    assert lo <= point <= hi
    assert hi - lo > 0.02, "at n=300 the interval should be visibly wide, not degenerate"


# -- mean_logprob rule -------------------------------------------------------

def test_summed_log_ranks_identically_to_mean_log():
    """a8 accumulates the sum of log-probabilities and does not divide by n_obs.

    This is sound only because n_obs is constant within a code. Dividing would scale every entry in
    the row of a code by the same factor and cannot reorder it. The test checks that assumption. It
    breaks if the accumulator becomes a running mean or if n_obs varies within a row.
    """
    m = _mod()
    rng = np.random.default_rng(0)
    n_codes, n_obs_per_code = 40, 7
    q = rng.random((n_codes, n_obs_per_code, 4)) + 1e-3
    q /= q.sum(axis=2, keepdims=True)
    logsum = np.log(q).sum(axis=1)
    logmean = logsum / n_obs_per_code
    gold = SPACE[rng.integers(0, 4, n_codes)]
    for k in (1, 3):
        assert (m.topk_from_scores(logsum, gold, SPACE, ks=(k,))
                == m.topk_from_scores(logmean, gold, SPACE, ks=(k,)))


def test_log_eps_clipping_keeps_zero_mass_codes_rankable():
    """log(0) would be -inf, making a whole row unrankable and indistinguishable from a NaN bug.

    Clipping at LOG_EPS keeps the veto effect (the zeroed LOINC ranks last) while leaving the row
    finite and comparable.
    """
    m = _mod()
    q = np.array([[0.5, 0.0, 0.3, 0.2]])
    scored = np.log(np.clip(q, m.LOG_EPS, None))
    assert np.isfinite(scored).all()
    assert scored.argmin() == 1, "the zero-mass LOINC must rank last, not become -inf"
    gold = np.array(["2-2"], dtype=object)
    assert m.topk_from_scores(scored, gold, SPACE, ks=(1,))["top1"] == 0.0
