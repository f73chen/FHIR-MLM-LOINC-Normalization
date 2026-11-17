"""
Tests for analysis/v3_calibration.py's `frequency_rows`: per-observation calibration by the
training frequency of the true LOINC code. Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_calibration_f", HERE / "v3_calibration.py")
K = importlib.util.module_from_spec(_S); _S.loader.exec_module(K)


def test_rows_per_group_with_their_own_gap():
    group_of_label = np.array(["rare", "common", "frequent", "frequent"], dtype=object)
    df = pd.DataFrame({"true_idx": [0, 0, 1, 1, 2, 3, 3, 3],
                       "p_top1": [0.9, 0.9, 0.6, 0.8, 1.0, 1.0, 0.98, 0.99],
                       "correct": [True, False, True, True, True, True, True, True]})
    rows = {r["group"]: r for r in K.frequency_rows(df, group_of_label, model="M", seed=0)}
    assert list(rows) == ["rare", "common", "frequent"] and rows["rare"]["model"] == "M"
    assert (rows["rare"]["n"], rows["common"]["n"], rows["frequent"]["n"]) == (2, 2, 4)
    assert rows["rare"]["acc"] == pytest.approx(0.5) and rows["rare"]["gap_conf_minus_acc"] == pytest.approx(0.4)   # overconfident
    assert rows["common"]["gap_conf_minus_acc"] == pytest.approx(-0.3)                                              # underconfident
    assert rows["frequent"]["mean_conf"] == pytest.approx((1.0 + 1.0 + 0.98 + 0.99) / 4)
    assert rows["rare"]["ece_mass15"] == pytest.approx(K.C.ece(np.array([0.9, 0.9]), np.array([True, False]), 15, "mass"))


def test_a_group_without_observations_is_left_out():
    df = pd.DataFrame({"true_idx": [1, 1], "p_top1": [0.7, 0.9], "correct": [True, True]})
    rows = K.frequency_rows(df, np.array(["rare", "common"], dtype=object))
    assert [r["group"] for r in rows] == ["common"] and rows[0]["n"] == 2


def test_the_groups_follow_the_training_counts():
    g = K.frequency_groups()
    assert len(g) == len(K.A.canonical_space()) and set(g) <= {"rare", "common", "frequent"}
    tc = K.OBS.train_count_map()
    for label, name in list(zip(K.A.canonical_space(), g))[:50]:
        n = tc.get(label, 0)
        assert name == ("rare" if n < 1_000 else "common" if n < 100_000 else "frequent")
