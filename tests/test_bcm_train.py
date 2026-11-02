"""Tests for the helpers of pipeline/13_bcm_train.py, on synthetic fixtures and the real class->LOINC
mapping. They run in the deep-models environment because they import torch and transformers. No GPU
or patient data is needed."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "bcm", Path(__file__).resolve().parents[1] / "pipeline" / "13_bcm_train.py")
bcm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcm)


def test_build_class_to_loinc_array_real():
    """(1623,) class->LOINC array from the real code files. About 828 classes map to LOINC, and the
    rest get unique '?{i}' sentinels."""
    arr = bcm.build_class_to_loinc_array()
    assert arr.shape == (bcm.N_LAB_CLASSES,) == (1623,)
    mapped = sum(1 for x in arr if not str(x).startswith("?"))
    assert 700 < mapped < 900                      # ~828 LOINC-mappable classes
    # unmapped sentinels are unique per class, so an unmapped true label can't spuriously match another class
    unmapped = [str(x) for x in arr if str(x).startswith("?")]
    assert len(unmapped) == len(set(unmapped))


def test_preprocess_logits_topk_over_class_dim():
    """Top-K reduction returns the K largest class indices directly (no vocab slice)."""
    logits = torch.full((1, 1, 10), -5.0)
    logits[0, 0, 5] = 9.0     # largest class
    logits[0, 0, 3] = 4.0     # 2nd
    idx = bcm.preprocess_logits_for_metrics(logits, torch.zeros(1, 1, dtype=torch.long))
    assert idx.shape == (1, 1, bcm.TOP_K_MAX)
    assert idx[0, 0, 0].item() == 5
    assert idx[0, 0, 1].item() == 3


def test_compute_metrics_topk_accuracy():
    """Top-K LOINC accuracy: one top-1 hit, one top-3-only hit, one ignored (-100)."""
    arr = np.array(["A", "B", "C", "?3", "?4"], dtype=object)
    preds = np.zeros((1, 3, bcm.TOP_K_MAX), dtype=np.int64)   # default class 0 ("A")
    preds[0, 1, 1] = 1                                          # pos1 has class 1 ("B") at rank 2
    labels = np.array([[0, 1, -100]], dtype=np.int64)          # pos0 "A", pos1 "B", pos2 ignored
    m = bcm.compute_metrics((preds, labels), arr)
    assert m["n_active_positions"] == 2
    assert m["loinc_top1"] == 0.5    # pos0 A@1 hit; pos1 A@1 != B miss
    assert m["loinc_top3"] == 1.0    # pos1 B@2 within top3


def test_compute_metrics_unmapped_and_empty():
    arr = np.array(["A", "B", "?2", "?3"], dtype=object)
    # unmapped true label (class 2 -> "?2") matches only a same-class prediction
    p_hit = np.full((1, 1, bcm.TOP_K_MAX), 2, dtype=np.int64)   # predict class 2
    m_hit = bcm.compute_metrics((p_hit, np.array([[2]], dtype=np.int64)), arr)
    assert m_hit["loinc_top1"] == 1.0                           # same unmapped class = correct
    p_miss = np.full((1, 1, bcm.TOP_K_MAX), 3, dtype=np.int64)  # predict class 3 ("?3")
    m_miss = bcm.compute_metrics((p_miss, np.array([[2]], dtype=np.int64)), arr)
    assert m_miss["loinc_top1"] == 0.0                          # "?3" != "?2"
    m0 = bcm.compute_metrics((p_hit, np.array([[-100]], dtype=np.int64)), arr)  # all ignored
    assert m0["n_active_positions"] == 0 and m0["loinc_top1"] == 0.0
