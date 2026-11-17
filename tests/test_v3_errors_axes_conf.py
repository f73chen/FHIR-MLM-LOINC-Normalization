"""
Tests for the confidence-by-axes columns behind T-20: `a5_error_taxonomy.rows_for_run` stores the mean
top-1 confidence of the errors that differ from the truth on 1 axis, on 2 axes and on 3 or more, so that every error is
in one group, and `v3_tables.t20` shows the three groups beside the correct predictions. Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


A5 = _load("a5_axes_t", HERE / "a5_error_taxonomy.py")
T = _load("v3_tables_axes_t", HERE / "v3_tables.py")
L = A5.L


def axis_table():
    """Five codes. Against code 0, codes 1 to 4 differ on 1, 2, 3 and 5 axes (a Method that only one code has counts)."""
    rows = [("A", "P", "PT", "SER", "QN", "M"),        # 0: the truth
            ("B", "P", "PT", "SER", "QN", "M"),        # 1: Component
            ("B", "P", "PT", "UR", "QN", "M"),         # 2: Component, System
            ("B", "Q", "PT", "UR", "QN", "M"),         # 3: Component, Property, System
            ("B", "Q", "PT", "UR", "ORD", L.MISSING.upper())]   # 4: Component, Property, System, Scale, Method
    return pd.DataFrame(rows, columns=L.AXES)


def reduction(preds, confs):
    """A reduction frame: every observation's truth is code 0."""
    preds = np.asarray(preds)
    return pd.DataFrame({"true_idx": np.zeros(len(preds), int), "pred_idx": preds, "correct": preds == 0,
                         "p_top1": np.asarray(confs, float)})


def run(d):
    table = axis_table()
    marg, sev, conf, ex = [], [], [], []
    A5.rows_for_run(d, "M", "para_xsite", 0, table, list(range(len(table))), np.zeros(len(table), bool), marg, sev, conf, ex)
    rows = [r for r in marg if r["subset"] == "all"]
    assert len(rows) == 1
    return rows[0]


def test_the_three_groups_hold_the_right_errors():
    #            correct      1 axis      2 axes   3 axes  5 axes
    d = reduction([0, 0, 0,   1, 1,       2,       3,      4], [0.99, 0.97, 0.98,   0.8, 0.6,   0.5,   0.4,    0.2])
    r = run(d)
    assert r["n_errors"] == 5
    assert r["mean_conf_correct"] == pytest.approx(0.98)
    assert r["mean_conf_1axis"] == pytest.approx(0.7)
    assert r["mean_conf_2axes"] == pytest.approx(0.5)
    assert r["mean_conf_3plus_axes"] == pytest.approx(0.3)
    assert (r["pct_axes_1"], r["pct_axes_2"], r["pct_axes_3"], r["pct_axes_5"]) == pytest.approx((40.0, 20.0, 20.0, 20.0))


def test_every_error_is_in_one_of_the_three_groups():
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 5, 400)
    d = reduction(preds, rng.uniform(0.05, 1.0, 400))
    r = run(d)
    share3 = 100.0 - r["pct_axes_1"] - r["pct_axes_2"]                     # no error differs on 0 axes
    assert r["pct_axes_0"] == 0.0
    pooled = (r["pct_axes_1"] * r["mean_conf_1axis"] + r["pct_axes_2"] * r["mean_conf_2axes"]
              + share3 * r["mean_conf_3plus_axes"]) / 100.0
    assert pooled == pytest.approx(d.p_top1[~d.correct].mean())            # the groups partition the errors


def test_an_empty_group_is_missing_not_zero():
    r = run(reduction([0, 1, 3], [0.9, 0.6, 0.3]))                          # no error differs on exactly 2 axes
    assert np.isnan(r["mean_conf_2axes"])
    assert r["mean_conf_1axis"] == pytest.approx(0.6) and r["mean_conf_3plus_axes"] == pytest.approx(0.3)


def marginals(with_two=True):
    rows = []
    for m, c in (("FHIR-MLM", 0.9), ("BCM", 0.8), ("Llama-3.1-8B", 0.85), ("TF-IDF", 0.7)):
        for seed in (0, 1):
            row = {"model": m, "condition": "para_xsite", "seed": seed, "subset": "all", "mean_conf_correct": 0.99,
                   "mean_conf_1axis": c, "mean_conf_3plus_axes": c - 0.2}
            if with_two:
                row["mean_conf_2axes"] = c - 0.1 + 0.01 * seed
            rows.append(row)
    other = {"model": "FHIR-MLM", "condition": "nodesc_xsite", "seed": 0, "subset": "all", "mean_conf_correct": 0.5,
             "mean_conf_1axis": 0.1, "mean_conf_3plus_axes": 0.1}                # another condition: not in T-20
    if with_two:
        other["mean_conf_2axes"] = 0.1
    return pd.DataFrame(rows + [other])


def test_t20_shows_the_two_axis_column_between_one_and_three():
    out = T.t20({"errors_axis_marginals.csv": marginals()})
    assert out["header"] == ["Model", "Correct", "1 axis differs", "2 axes differ", "≥ 3 axes differ"]
    assert out["right"] == {1, 2, 3, 4} and out["incomplete"] == []
    body = {r[0]: r for r in out["body"]}
    assert list(body) == T.ALL4
    assert body["FHIR-MLM"] == ["FHIR-MLM", "0.990", "0.900", "0.805", "0.700"]     # seed means, para only
    assert all(len(r) == 5 for r in out["body"])


def test_t20_flags_outputs_from_before_the_two_axis_column():
    out = T.t20({"errors_axis_marginals.csv": marginals(with_two=False)})
    assert out["incomplete"] and all(r[3] == "—" for r in out["body"])
