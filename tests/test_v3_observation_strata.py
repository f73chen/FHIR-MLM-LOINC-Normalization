"""
Tests for analysis/v3_observation.py's `strata_joint_rows`: the joint patient × seed
bootstrap interval of each training-frequency bin's seed-mean accuracy, which F-02 draws as its bars.
Synthetic frames only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_observation_t", HERE / "v3_observation.py")
O = importlib.util.module_from_spec(_S); _S.loader.exec_module(O)

TC = {"rare": 20, "mid": 500, "freq": 200_000}          # bins 0, 3 and 8 of the nine log-spaced bins


def frames(n_seeds=3, n_pat=40, seed=0):
    """Every patient has 3 `freq` and 2 `mid` observations; only patient 0 has the one `rare` observation.
    `freq` is always right; `mid` is right with a probability that differs by seed."""
    rng = np.random.default_rng(seed)
    rows = [(p, i, "freq" if i < 3 else "mid") for p in range(n_pat) for i in range(5)] + [(0, 5, "rare")]
    base = pd.DataFrame(rows, columns=["patient_uuid", "obs_ordinal", "true_loinc"])
    out = []
    for s in range(n_seeds):
        f = base.copy()
        f["correct"] = np.where(f.true_loinc == "freq", True,
                                np.where(f.true_loinc == "rare", s == 0, rng.random(len(f)) < 0.3 + 0.2 * s))
        out.append(f)
    return out


def test_counts_means_and_bounds():
    fs = frames()
    r = pd.DataFrame(O.strata_joint_rows(fs, TC, n_boot=400, seed=1, model="X")).set_index("stratum")
    assert list(r.index) == [0, 3, 8] and (r.model == "X").all() and (r.n_seeds == 3).all()
    assert r.n_obs.to_dict() == {0: 1, 3: 80, 8: 120} and r.n_patients.to_dict() == {0: 1, 3: 40, 8: 40}
    mid = np.mean([f.correct[f.true_loinc == "mid"].mean() for f in fs])
    assert r.acc_mean[3] == pytest.approx(mid) and r.acc_mean[8] == 1.0 and r.acc_mean[0] == pytest.approx(1 / 3)
    assert r.boot_lo[8] == r.boot_hi[8] == 1.0                      # always right: no spread
    assert r.boot_lo[3] < mid < r.boot_hi[3] and 0 <= r.boot_lo[3] and r.boot_hi[3] <= 1
    assert r.n_empty[3] == r.n_empty[8] == 0


def test_a_bin_missing_from_a_replicate_is_left_out():
    r = pd.DataFrame(O.strata_joint_rows(frames(), TC, n_boot=400, seed=1)).set_index("stratum")
    # the rare bin's only patient is absent from about (1 - 1/40)^40 = 36% of the patient draws
    assert 80 < r.n_empty[0] < 220
    # its seeds score 1, 0, 0, so a replicate's seed mean is 0, 1/3, 2/3 or (three picks of seed 0: 1 in 27) 1
    assert r.boot_lo[0] == 0.0 and r.boot_hi[0] == pytest.approx(2 / 3)


def test_seeds_widen_the_interval_and_one_seed_runs():
    fs = frames(n_seeds=3)
    joint = pd.DataFrame(O.strata_joint_rows(fs, TC, n_boot=400, seed=1)).set_index("stratum")
    one = pd.DataFrame(O.strata_joint_rows(fs[:1], TC, n_boot=400, seed=1)).set_index("stratum")
    assert (one.n_seeds == 1).all() and one.boot_lo[3] < one.boot_hi[3]
    mean_only = []                                                  # patients resampled, the three seeds kept
    f = fs[0].assign(correct=np.mean([g.correct.to_numpy() for g in fs], axis=0))
    by = f[f.true_loinc == "mid"].groupby("patient_uuid").correct.agg(["sum", "count"])
    rng = np.random.default_rng(5)
    for _ in range(400):
        i = rng.integers(0, len(by), len(by))
        mean_only.append(by["sum"].to_numpy()[i].sum() / by["count"].to_numpy()[i].sum())
    lo, hi = np.percentile(mean_only, [2.5, 97.5])
    assert joint.boot_hi[3] - joint.boot_lo[3] > hi - lo            # the seeds differ (0.3, 0.5, 0.7), so resampling them adds width


def test_same_seed_same_rows_and_misaligned_frames_fail():
    fs = frames()
    a = O.strata_joint_rows(fs, TC, n_boot=200, seed=3)
    assert a == O.strata_joint_rows(fs, TC, n_boot=200, seed=3)
    with pytest.raises(AssertionError):
        O.strata_joint_rows([fs[0], fs[1].iloc[::-1].reset_index(drop=True)], TC, n_boot=10)
