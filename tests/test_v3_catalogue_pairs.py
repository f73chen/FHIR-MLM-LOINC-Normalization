"""
Tests for analysis/v3_catalogue.py's `pairwise_rows` with `same_runs`: when two names are the
same trained models (RQ3's scenarios A and B), one seed pick per bootstrap replicate serves both, so the
interval of their difference no longer carries the spread of the accuracy between seeds. Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_catalogue_t", HERE / "v3_catalogue.py")
C = importlib.util.module_from_spec(_S); _S.loader.exec_module(C)

N_CODES = 300


def frame(seed_acc=(0.60, 0.70, 0.80, 0.90, 0.95), gap_codes=6, seed=0):
    """A: each seed gets its own share of codes right (a large spread between seeds). B: the same runs, with
    `gap_codes` of each run's right codes turned wrong (a small, steady gap). C: independent runs."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, acc in enumerate(seed_acc):
        a = rng.random(N_CODES) < acc
        b = a.copy()
        b[np.flatnonzero(a)[:gap_codes]] = False
        c = rng.random(N_CODES) < 0.75
        for name, v in (("A", a), ("B", b), ("C", c)):
            rows += [{"model": name, "seed": s, "code": k, "correct": float(v[k])} for k in range(N_CODES)]
    return pd.DataFrame(rows)


def row(rows, a, b):
    return next(r for r in rows if {r["model_a"], r["model_b"]} == {a, b})


def test_same_runs_narrow_the_interval_and_keep_the_estimate():
    pc = frame()
    plain = C.pairwise_rows(pc, "x", names=["A", "B", "C"])
    paired = C.pairwise_rows(pc, "x", names=["A", "B", "C"], same_runs=(("A", "B"),))
    u, p = row(plain, "A", "B"), row(paired, "A", "B")
    assert p["diff_mean"] == u["diff_mean"] == pytest.approx(6 / N_CODES)
    assert u["lo"] < 0 < u["hi"]                                   # the seed spread (0.60 to 0.95) swamps a 2-point gap
    assert p["lo"] > 0 and p["excludes_zero"] and not u["excludes_zero"]
    assert (p["hi"] - p["lo"]) < 0.25 * (u["hi"] - u["lo"])
    assert p["seeds_paired"] and not u["seeds_paired"]


def test_other_pairs_are_untouched():
    pc = frame()
    plain = C.pairwise_rows(pc, "x", names=["A", "B", "C"])
    paired = C.pairwise_rows(pc, "x", names=["A", "B", "C"], same_runs=(("A", "B"),))
    for a, b in (("A", "C"), ("B", "C")):
        u, p = row(plain, a, b), row(paired, a, b)
        assert (u["lo"], u["hi"], u["diff_mean"]) == (p["lo"], p["hi"], p["diff_mean"]) and not p["seeds_paired"]


def test_the_order_of_a_pair_does_not_matter_and_unequal_seeds_fail():
    pc = frame()
    a = row(C.pairwise_rows(pc, "x", names=["A", "B", "C"], same_runs=(("A", "B"),)), "A", "B")
    b = row(C.pairwise_rows(pc, "x", names=["A", "B", "C"], same_runs=(("B", "A"),)), "A", "B")
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])
    short = pc[~((pc.model == "B") & (pc.seed == 4))]                       # one seed fewer
    with pytest.raises(AssertionError):
        C.pairwise_rows(short, "x", names=["A", "B", "C"], same_runs=(("A", "B"),))
    other = pc.assign(seed=pc.seed.where(~((pc.model == "B") & (pc.seed == 4)), 7))   # five seeds, another label
    with pytest.raises(AssertionError):
        C.pairwise_rows(other, "x", names=["A", "B", "C"], same_runs=(("A", "B"),))
    assert row(C.pairwise_rows(other, "x", names=["A", "B", "C"]), "A", "B")["seeds_paired"] is False   # unpaired: allowed


def test_rq3_pairs_are_the_scenarios_that_share_their_trained_models():
    same = {frozenset(p) for p in C.RQ3_SAME_RUNS}
    for x, (train_x, _) in C.LETTERS.items():
        for y, (train_y, _) in C.LETTERS.items():
            if x < y:
                assert (frozenset((x, y)) in same) == (train_x == train_y), (x, y)
