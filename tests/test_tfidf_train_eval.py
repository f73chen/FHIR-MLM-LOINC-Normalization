"""Tests for the downsampling and metric helpers of pipeline/07_tfidf_train_eval.py (synthetic data)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "tte", Path(__file__).resolve().parents[1] / "pipeline" / "07_tfidf_train_eval.py"
)
tte = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tte)


def test_threshold_targets():
    t = tte.threshold_targets({"A": 100, "B": 1000, "C": 3000}, subset_size=2000, downsample_min=500)
    assert t["A"] == 100                      # <=500 -> take all
    assert t["B"] == 500                      # scaled below floor -> floored to 500
    assert t["C"] == 1425                     # int(3000/4000 * 1900)
    assert all(t[c] <= n for c, n in {"A": 100, "B": 1000, "C": 3000}.items())  # never exceed available


def test_downsample_deterministic_and_capped():
    labels = ["A"] * 50 + ["B"] * 1000        # A small (take all), B large
    i1 = tte.downsample_indices(labels, subset_size=600, downsample_min=100, seed=3)
    i2 = tte.downsample_indices(labels, subset_size=600, downsample_min=100, seed=3)
    assert i1 == i2                           # deterministic under fixed seed
    kept = [labels[i] for i in i1]
    assert kept.count("A") == 50              # take-all class fully kept
    assert kept.count("B") <= 1000 and kept.count("B") >= 100  # downsampled, floored
    assert tte.downsample_indices(labels, 600, 100, seed=4) != i1  # different seed -> different


def test_topk_hits_rank():
    # obs0 true=1 (max -> rank0), obs1 true=0 (0.2, one greater -> rank1), obs2 true=2 (0.1, two greater -> rank2)
    proba = np.array([[0.1, 0.7, 0.2], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
    y = np.array([1, 0, 2])
    h = tte.topk_hits(proba, y, [1, 2, 3])
    assert h[1] == 1                          # only obs0 is top-1
    assert h[2] == 2                          # obs0 + obs1 (rank < 2)
    assert h[3] == 3                          # all within top-3


def test_make_logreg_wraps_liblinear_in_ovr():
    import pytest
    pytest.importorskip("sklearn")
    ovr = tte.make_logreg(6.59, "liblinear", 1500, fit_jobs=8)
    assert type(ovr).__name__ == "OneVsRestClassifier" and ovr.n_jobs == 8
    base = ovr.estimator
    assert base.C == 6.59 and base.solver == "liblinear" and base.max_iter == 1500
    assert type(tte.make_logreg(1.0, "lbfgs", 200)).__name__ == "LogisticRegression"  # multinomial as-is


def test_eval_on_test_parallel_matches_sequential():
    """The parallel eval (eval_jobs>1) must produce results identical to sequential (eval_jobs=1)."""
    import pytest
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    rng = np.random.RandomState(0)
    words = [f"w{c}" for c in range(6)]
    Xtr = [f"{words[c]} {words[c]} common" for c in range(6) for _ in range(20)]
    ytr = np.array([c for c in range(6) for _ in range(20)])
    # lbfgs (not liblinear) so this test runs under base sklearn too; eval logic is solver-agnostic.
    pipe = Pipeline([("tfidf", TfidfVectorizer()),
                     ("clf", tte.make_logreg(1.0, "lbfgs", 300))]).fit(Xtr, ytr)
    Xte = [f"{words[rng.randint(6)]} common e{rng.randint(3)}" for _ in range(257)]
    yte = np.array([int(t.split()[0][1:]) for t in Xte])
    seq = tte.eval_on_test(pipe, Xte, yte, 6, [1, 3], eval_batch=37, eval_jobs=1)
    par = tte.eval_on_test(pipe, Xte, yte, 6, [1, 3], eval_batch=37, eval_jobs=3)
    assert seq[0] == par[0]                       # micro hit counts identical
    assert np.array_equal(seq[1], par[1])         # argmax preds identical
    assert np.allclose(seq[2], par[2])            # max-prob confidence identical
    assert abs(seq[3] - par[3]) < 1e-6            # Brier sum identical (up to FP order)


def test_brier_sum():
    # perfect prediction -> brier 0; uniform-ish -> positive
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert abs(tte.brier_sum(proba, y)) < 1e-12
    p2 = np.array([[0.5, 0.5]])
    assert abs(tte.brier_sum(p2, np.array([0])) - 0.5) < 1e-12   # (0.25+0.25)-2*0.5+1 = 0.5


def test_ece_perfectly_calibrated():
    # confidence == accuracy in every populated bin -> ECE 0
    conf = np.array([0.95, 0.95, 0.05])
    correct = np.array([1.0, 1.0, 0.0])       # 0.95 bin: acc 1.0 vs conf 0.95 -> small but nonzero
    e = tte.ece_from_bins(conf, correct, n_bins=10)
    assert 0.0 <= e <= 0.1
    # exact-calibration construction: bin centered, acc matches conf
    conf2 = np.array([0.5, 0.5])
    correct2 = np.array([1.0, 0.0])           # acc 0.5 == conf 0.5
    assert tte.ece_from_bins(conf2, correct2, n_bins=10) < 1e-9
