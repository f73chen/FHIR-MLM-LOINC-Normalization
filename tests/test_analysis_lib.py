"""Unit tests for analysis/lib/dumps.py, on synthetic fixtures only.

Several tests cover `matched()`. Runs differ in which observations they score. RELMA drops the
observations of codes without a RELMA name, S2 truncates more than S1, and the model families truncate
differently. Every cross-run comparison must therefore be an explicit intersection, and a paired test on
unequal arms must raise an error.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_S = importlib.util.spec_from_file_location(
    "dumps", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_S)
_S.loader.exec_module(D)


def mk(keys, correct, true=None, topk=None):
    """keys = [(uuid, ordinal), ...].
    topk must agree with `correct`: the true LOINC is at rank 1 only when the row is correct. If it
    were at rank 1 in every row, micro_topk would score every row as a hit, and the metric tests
    would pass whatever the metric did."""
    n = len(keys)
    true = true or ["L1"] * n
    return pd.DataFrame({
        "patient_uuid": [k[0] for k in keys],
        "obs_ordinal": [k[1] for k in keys],
        "true_loinc": true,
        "correct": np.array(correct, dtype=bool),
        "pred_loinc": [t if c else "OTHER" for t, c in zip(true, correct)],
        "topk_loinc": topk or [[t if c else "OTHER"] + ["X"] * 9
                               for t, c in zip(true, correct)],
    })


# -- matched(): the central guard -------------------------------------------
def test_matched_reduces_to_intersection():
    a = mk([("u1", 0), ("u1", 1), ("u2", 0)], [1, 1, 1])
    b = mk([("u1", 1), ("u2", 0), ("u3", 0)], [1, 0, 1])
    ma, mb = D.matched(a, b)
    assert len(ma) == len(mb) == 2
    assert list(zip(ma.patient_uuid, ma.obs_ordinal)) == [("u1", 1), ("u2", 0)]


def test_matched_arms_are_row_aligned():
    """Row i must be the same observation in both arms, or a paired test compares unrelated rows."""
    a = mk([("u2", 5), ("u1", 0), ("u1", 3)], [1, 1, 0])
    b = mk([("u1", 3), ("u2", 5), ("u1", 0)], [0, 1, 1])
    ma, mb = D.matched(a, b)
    assert (ma[D.KEY].values == mb[D.KEY].values).all()


def test_matched_three_way():
    a = mk([("u", 0), ("u", 1), ("u", 2)], [1, 1, 1])
    b = mk([("u", 1), ("u", 2)], [1, 1])
    c = mk([("u", 2), ("u", 3)], [1, 1])
    out = D.matched(a, b, c)
    assert all(len(x) == 1 for x in out)


def test_matched_empty_intersection_is_equal_not_silent():
    a = mk([("u1", 0)], [1]); b = mk([("u2", 0)], [1])
    out = D.matched(a, b)
    assert all(len(x) == 0 for x in out), "disjoint runs must yield empty arms"


def test_matched_is_idempotent():
    a = mk([("u", 0), ("u", 1)], [1, 0]); b = mk([("u", 0), ("u", 1)], [0, 1])
    once = D.matched(a, b); twice = D.matched(*once)
    assert len(twice[0]) == len(once[0]) == 2


# -- metrics ----------------------------------------------------------------
def test_micro_topk_known_answer():
    df = mk([("u", 0), ("u", 1)], [1, 0], true=["A", "B"],
            topk=[["A"] + ["X"] * 9, ["Z", "B"] + ["X"] * 8])
    m = D.micro_topk(df)
    assert m["top1"] == pytest.approx(0.5)
    assert m["top3"] == pytest.approx(1.0)


def test_macro_weights_classes_equally_not_observations():
    """Under macro averaging, a 100-observation class and a 1-observation class count the same."""
    df = mk([("u", i) for i in range(101)], [1] * 100 + [0], true=["BIG"] * 100 + ["RARE"])
    assert D.micro_topk(df)["top1"] == pytest.approx(100 / 101)
    assert D.macro_top1(df) == pytest.approx(0.5)


def test_macro_restricted_to_shared_classes():
    """Comparing arms with different class coverage must average over the same class set."""
    df = mk([("u", 0), ("u", 1)], [1, 0], true=["A", "B"])
    assert D.macro_top1(df) == pytest.approx(0.5)
    assert D.macro_top1(df, classes={"A"}) == pytest.approx(1.0)


# -- McNemar ----------------------------------------------------------------
def test_mcnemar_matches_scipy_on_known_table():
    from scipy.stats import binomtest
    a = mk([("u", i) for i in range(10)], [1] * 7 + [0] * 3)
    b = mk([("u", i) for i in range(10)], [1] * 2 + [0] * 5 + [1] * 3)
    r = D.mcnemar(a, b)
    assert r["a_only"] == 5 and r["b_only"] == 3
    assert r["p"] == pytest.approx(binomtest(5, 8, 0.5).pvalue)


def test_mcnemar_identical_arms_is_p1():
    a = mk([("u", i) for i in range(5)], [1, 0, 1, 0, 1])
    r = D.mcnemar(a, a.copy())
    assert r["a_only"] == r["b_only"] == 0 and r["p"] == 1.0
    assert r["delta"] == pytest.approx(0.0)


def test_mcnemar_ignores_concordant_pairs():
    """Adding agreements must not change p, because only discordant pairs carry information."""
    a = mk([("u", i) for i in range(4)], [1, 0, 1, 1])
    b = mk([("u", i) for i in range(4)], [0, 1, 1, 1])
    r1 = D.mcnemar(a, b)
    a2 = mk([("u", i) for i in range(6)], [1, 0, 1, 1, 1, 1])
    b2 = mk([("u", i) for i in range(6)], [0, 1, 1, 1, 1, 1])
    assert D.mcnemar(a2, b2)["p"] == pytest.approx(r1["p"])


def test_mcnemar_patients_with_one_observation_per_patient_is_mcnemars_chi_square():
    from scipy.stats import chi2
    a = mk([(f"u{i}", 0) for i in range(10)], [1] * 7 + [0] * 3)
    b = mk([(f"u{i}", 0) for i in range(10)], [1] * 2 + [0] * 5 + [1] * 3)
    r = D.mcnemar_patients(a, b)
    assert r["z"] == pytest.approx((5 - 3) / np.sqrt(5 + 3))
    assert r["p"] == pytest.approx(chi2.sf((5 - 3) ** 2 / (5 + 3), 1))


def test_mcnemar_patients_counts_a_patients_disagreements_together():
    """The four disagreements of patient u0 all favour a and move together. The clustered z is
    therefore smaller than if the same counts came from separate patients."""
    a = mk([("u0", i) for i in range(4)] + [("u1", 0), ("u2", 0)], [1, 1, 1, 1, 0, 1])
    b = mk([("u0", i) for i in range(4)] + [("u1", 0), ("u2", 0)], [0, 0, 0, 0, 1, 0])
    r = D.mcnemar_patients(a, b)
    assert r["z"] == pytest.approx((4 - 1 + 1) / np.sqrt(16 + 1 + 1))
    assert r["z"] < (5 - 1) / np.sqrt(5 + 1)
    assert r["n_patients"] == 3


def test_mcnemar_patients_refuses_unaligned_arms():
    a = mk([("u1", 0), ("u2", 0)], [1, 0])
    b = mk([("u2", 0), ("u1", 0)], [0, 1])
    with pytest.raises(AssertionError):
        D.mcnemar_patients(a, b)


# -- clustered bootstrap ----------------------------------------------------
def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    df = mk([(f"u{i//5}", i % 5) for i in range(200)], [1] * 150 + [0] * 50)
    lo, hi = D.bootstrap_ci(df, n_boot=300, seed=0)
    assert lo <= df["correct"].mean() <= hi
    assert (lo, hi) == D.bootstrap_ci(df, n_boot=300, seed=0)


def test_bootstrap_clusters_by_patient_not_observation():
    """With all observations inside one patient, a patient-level bootstrap can only resample that single
    cluster, so the interval has zero width. An observation-level bootstrap would report a spuriously
    tight non-zero CI."""
    df = mk([("only_patient", i) for i in range(100)], [1] * 60 + [0] * 40)
    lo, hi = D.bootstrap_ci(df, n_boot=200, seed=0)
    assert lo == hi == pytest.approx(0.6), "CI must collapse when there is only one cluster"


def test_bootstrap_widens_with_fewer_clusters():
    many = mk([(f"u{i}", 0) for i in range(200)], [1] * 120 + [0] * 80)
    few = mk([(f"u{i//20}", i % 20) for i in range(200)], [1] * 120 + [0] * 80)
    w_many = np.diff(D.bootstrap_ci(many, n_boot=400, seed=1))[0]
    w_few = np.diff(D.bootstrap_ci(few, n_boot=400, seed=1))[0]
    assert w_few > w_many, "fewer independent clusters must give a wider interval"


# -- sidecar join -----------------------------------------------------------
def test_attach_relma_flags_and_marks_uncovered():
    df = mk([("u1", 0), ("u1", 1)], [1, 1], true=["A", "NOCOVER"])
    side = pd.DataFrame({"patient_uuid": ["u1"], "loinc": ["A"],
                         "relma_name": ["HCT"], "clean_same_loinc": [True], "clean_global": [True]})
    out = D.attach_relma(df, side)
    assert out["has_relma"].tolist() == [True, False]
    assert out.loc[out.true_loinc == "A", "clean_same_loinc"].iloc[0] == True


def test_attach_relma_does_not_change_row_count():
    """A left join must not duplicate or drop rows, because the assignment gives one name per
    (patient, LOINC)."""
    df = mk([("u1", i) for i in range(5)], [1] * 5, true=["A"] * 5)
    side = pd.DataFrame({"patient_uuid": ["u1"], "loinc": ["A"], "relma_name": ["HCT"],
                         "clean_same_loinc": [True], "clean_global": [True]})
    assert len(D.attach_relma(df, side)) == 5


# -----------------------------------------------------------------------------
# label_arms guards against sign errors. `delta` is always a-arm minus b-arm, but which arm is `a`
# is visible only at the call site. That makes it easy to state a conclusion backwards, for example
# by printing -delta under a "drop" heading while writing +delta to CSV, or by naming a table
# s1_vs_s2 when its `a` arm is S2 (q5). Column names that state the direction prevent this.
# -----------------------------------------------------------------------------

def test_label_arms_delta_column_names_its_own_direction():
    a = mk([("p", 1), ("p", 2), ("p", 3), ("p", 4)], [1, 1, 1, 0])   # 0.75
    b = mk([("p", 1), ("p", 2), ("p", 3), ("p", 4)], [1, 0, 0, 0])   # 0.25
    out = D.label_arms(D.mcnemar(a, b), "relma", "mimic")
    assert out["acc_relma"] == pytest.approx(0.75)
    assert out["acc_mimic"] == pytest.approx(0.25)
    # the column name states the direction, and the value must match it
    assert out["delta_relma_minus_mimic"] == pytest.approx(0.75 - 0.25)
    assert "delta" not in out and "a_acc" not in out and "b_acc" not in out


def test_label_arms_discordant_counts_follow_the_names():
    a = mk([("p", 1), ("p", 2)], [1, 0])
    b = mk([("p", 1), ("p", 2)], [0, 1])
    out = D.label_arms(D.mcnemar(a, b), "relma", "mimic")
    assert out["n_relma_only"] == 1 and out["n_mimic_only"] == 1


def test_q2_arm_order_makes_negative_delta_mean_relma_is_harder():
    """The stated convention is 'negative drop = RELMA is harder'. Build RELMA strictly worse and
    confirm the emitted column is negative."""
    keys = [("p", i) for i in range(10)]
    mimic = mk(keys, [1] * 9 + [0])   # 0.9
    relma = mk(keys, [1] * 6 + [0] * 4)  # 0.6  -> RELMA harder
    mr, mb = D.matched(relma, mimic)
    out = D.label_arms(D.mcnemar(mr, mb), "relma", "mimic")   # a-arm = RELMA, as q2 calls it
    assert out["delta_relma_minus_mimic"] < 0


def test_q5_arm_order_labels_s2_as_s2():
    """q5 passes (S2, S1), so delta reads S2 - S1. The S2 column must hold the S2 accuracy even
    though the table is named s1_vs_s2."""
    keys = [("p", i) for i in range(10)]
    s1 = mk(keys, [1] * 9 + [0])       # 0.9
    s2 = mk(keys, [1] * 5 + [0] * 5)   # 0.5, battery hurts
    out = D.label_arms(D.mcnemar(s2, s1), "s2", "s1")
    assert out["acc_s2"] == pytest.approx(0.5) and out["acc_s1"] == pytest.approx(0.9)
    assert out["delta_s2_minus_s1"] == pytest.approx(-0.4)
