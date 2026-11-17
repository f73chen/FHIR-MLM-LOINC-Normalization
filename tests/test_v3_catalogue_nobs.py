"""
Tests for analysis/v3_catalogue.py's `nobs_rows` and `ece_floor_rows`: per-code confidence
against per-code accuracy by a local code's number of test observations, and the per-code ECE that
sampling alone gives. Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_catalogue_n", HERE / "v3_catalogue.py")
C = importlib.util.module_from_spec(_S); _S.loader.exec_module(C)


def frame():
    """Six local codes: two with one test observation (one right, one wrong, both confident), four with more."""
    n_obs = [1, 1, 5, 50, 500, 5000]
    correct = [1, 0, 1, 1, 1, 1]
    conf = [0.9, 0.8, 0.6, 0.7, 0.9, 0.99]
    n_right = [1, 0, 3, 40, 480, 4990]
    rows = [{"model": "M", "seed": s, "code": k, "n_obs": n_obs[k], "correct": float(correct[k]), "conf": conf[k],
             "n_correct_obs": n_right[k]} for s in (0, 1) for k in range(6)]
    return pd.DataFrame(rows)


def by(rows, seed=0):
    return {r["test_obs"]: r for r in rows if r["seed"] == seed}


def test_groups_and_their_numbers():
    r = by(C.nobs_rows(frame(), "x"))
    assert set(r) == {"all", "1", "2+", "2-9", "10-99", "100-999", "1000+"}
    assert (r["1"]["n_codes"], r["2+"]["n_codes"], r["all"]["n_codes"]) == (2, 4, 6)
    assert r["1"]["acc"] == pytest.approx(0.5) and r["1"]["conf"] == pytest.approx(0.85)
    assert r["1"]["conf_minus_acc"] == pytest.approx(0.35)                    # overconfident where per-code is per-observation
    assert r["2+"]["acc"] == pytest.approx(1.0) and r["2+"]["conf_minus_acc"] == pytest.approx((0.6 + 0.7 + 0.9 + 0.99) / 4 - 1)
    assert r["1"]["obs_right"] == pytest.approx(0.5)                          # one observation: the code's share IS its accuracy
    assert r["2+"]["obs_right"] == pytest.approx((0.6 + 0.8 + 0.96 + 0.998) / 4)
    assert r["2+"]["n_obs"] == 5555 and r["1000+"]["n_codes"] == 1


def test_the_two_parts_add_up_to_the_whole():
    r = by(C.nobs_rows(frame(), "x"))
    assert r["1"]["gap_part"] + r["2+"]["gap_part"] == pytest.approx(r["all"]["conf_minus_acc"])
    assert r["all"]["gap_part"] == pytest.approx(r["all"]["conf_minus_acc"])
    fine = sum(r[g]["gap_part"] for g in ("2-9", "10-99", "100-999", "1000+"))
    assert fine == pytest.approx(r["2+"]["gap_part"])


def test_empty_groups_are_left_out_and_seeds_are_separate():
    f = frame()
    f = f[f.n_obs < 100]
    rows = C.nobs_rows(f, "x")
    assert {r["test_obs"] for r in rows} == {"all", "1", "2+", "2-9", "10-99"}
    assert {r["seed"] for r in rows} == {0, 1}
    assert "obs_right" in rows[0] and np.isnan(C.nobs_rows(f.drop(columns="n_correct_obs"), "x")[0]["obs_right"])


def test_the_floor_is_above_zero_shrinks_with_more_codes_and_is_reproducible():
    rng = np.random.default_rng(1)
    def cal(n):                                    # a frame that is calibrated by construction
        conf = rng.uniform(0.5, 1.0, n)
        return pd.DataFrame({"model": "M", "seed": 0, "code": np.arange(n), "n_obs": 3, "conf": conf,
                             "correct": (rng.random(n) < conf).astype(float)})
    small, large = cal(300), cal(20000)
    a, b = C.ece_floor_rows(small, "x", n_sim=200)[0], C.ece_floor_rows(large, "x", n_sim=50)[0]
    assert a["floor_mean"] > 0.02 and b["floor_mean"] < 0.01 and a["floor_mean"] > 3 * b["floor_mean"]
    assert a["floor_p95"] >= a["floor_mean"] and a["n_codes"] == 300 and a["n_sim"] == 200
    assert a["ece"] < a["floor_p95"] * 1.5                                    # a calibrated frame sits near its own floor
    assert C.ece_floor_rows(small, "x", n_sim=200)[0] == a                    # fixed generator: the same numbers again
    assert a["ece"] == pytest.approx(C.A10.C.ece(small.conf.to_numpy(), small.correct.to_numpy().astype(bool), n_bins=C.A10.N_BINS))
