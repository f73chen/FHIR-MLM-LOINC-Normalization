"""Unit tests for the pure logic of pipeline/15_eval_runner.py, on synthetic fixtures and the real
class->LOINC mapping. Needs the deep-models environment (torch, transformers) but no GPU, no patient
data and no checkpoints.

Covers the happy path and checks that each guardrail fires on injected bad input. Whether a
checkpoint loads correctly is checked separately, by `--verify` against the training manifests.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "evalrunner", Path(__file__).resolve().parents[1] / "pipeline" / "15_eval_runner.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


# ------------------------------------------------------------------------------
# class -> LOINC mapping
# ------------------------------------------------------------------------------
def test_build_class_to_loinc_array_real():
    """(1623,) class->LOINC from the real code files. About 828 classes are mapped, and unmapped ones
    get unique sentinels."""
    arr = ev.build_class_to_loinc_array()
    assert arr.shape == (ev.N_LAB_CLASSES,) == (1623,)
    mapped = sum(1 for x in arr if not str(x).startswith("?"))
    assert 700 < mapped < 900
    unmapped = [str(x) for x in arr if str(x).startswith("?")]
    assert len(unmapped) == len(set(unmapped)), "sentinels must be unique per class"


# ------------------------------------------------------------------------------
# topk_reduce: the core contract of the evaluation dump's columns
# ------------------------------------------------------------------------------
def test_topk_reduce_shapes_and_ordering():
    logits = torch.full((3, ev.N_LAB_CLASSES), -8.0)
    logits[0, 5] = 9.0; logits[0, 3] = 4.0          # top1=5, top2=3
    logits[1, 100] = 7.0
    logits[2, 7] = 2.0
    labels = torch.tensor([5, 100, 999])
    d = ev.topk_reduce(logits, labels)
    assert d["topk_class"].shape == (3, ev.TOP_K_MAX)
    assert d["topk_logit"].shape == (3, ev.TOP_K_MAX)
    assert d["true_class_logit"].shape == (3,)
    assert d["row_logsumexp"].shape == (3,)
    assert d["topk_class"][0, 0] == 5 and d["topk_class"][0, 1] == 3
    # topk_logit must be sorted descending
    assert np.all(np.diff(d["topk_logit"][0]) <= 1e-5)


def test_topk_reduce_true_class_logit_is_gathered_correctly():
    logits = torch.arange(2 * ev.N_LAB_CLASSES, dtype=torch.float32).reshape(2, ev.N_LAB_CLASSES)
    labels = torch.tensor([11, 22])
    d = ev.topk_reduce(logits, labels)
    assert d["true_class_logit"][0] == pytest.approx(float(logits[0, 11]))
    assert d["true_class_logit"][1] == pytest.approx(float(logits[1, 22]))


def test_row_logsumexp_reconstructs_exact_probabilities():
    """The top-K logits, the true-class logit and the row logsumexp in the dump must reconstruct
    exact softmax probabilities. Calibration, risk-coverage and NLL then do not need the full
    1623-wide row."""
    torch.manual_seed(0)
    logits = torch.randn(4, ev.N_LAB_CLASSES) * 3.0
    labels = torch.randint(0, ev.N_LAB_CLASSES, (4,))
    d = ev.topk_reduce(logits, labels)
    full_probs = torch.softmax(logits.float(), dim=-1)

    # top-1 probability from the dump alone
    p_top1 = np.exp(d["topk_logit"][:, 0] - d["row_logsumexp"])
    assert np.allclose(p_top1, full_probs.max(dim=-1).values.numpy(), atol=1e-5)

    # true-class probability from the dump alone (works even if truth is outside top-K)
    p_true = np.exp(d["true_class_logit"] - d["row_logsumexp"])
    expected = full_probs[torch.arange(4), labels].numpy()
    assert np.allclose(p_true, expected, atol=1e-5)


def test_true_class_logit_survives_truth_outside_topk():
    """Truth outside the top-K is the case that would break NLL if only the top-K were stored."""
    logits = torch.full((1, ev.N_LAB_CLASSES), -20.0)
    for i in range(ev.TOP_K_MAX):
        logits[0, i] = 10.0 - i          # classes 0..9 dominate
    labels = torch.tensor([500])          # truth not in top-10
    d = ev.topk_reduce(logits, labels)
    assert 500 not in d["topk_class"][0]
    assert d["true_class_logit"][0] == pytest.approx(-20.0, abs=1e-4)


# ------------------------------------------------------------------------------
# accuracy_from_dump
# ------------------------------------------------------------------------------
def test_accuracy_from_dump_known_answer():
    arr = np.array(["A", "B", "C", "D"], dtype=object)
    topk = np.array([[0, 1, 2, 3], [1, 0, 2, 3]])      # row0 top1 right; row1 top1 wrong, top3 right
    true = np.array([0, 0])
    m = ev.accuracy_from_dump(topk, true, arr)
    assert m["loinc_top1"] == pytest.approx(0.5)
    assert m["loinc_top3"] == pytest.approx(1.0)


def test_unmapped_sentinel_never_matches_another_class():
    """Guardrail: an unmapped truth must not spuriously match a different unmapped prediction."""
    arr = ev.build_class_to_loinc_array()
    unmapped = [i for i, x in enumerate(arr) if str(x).startswith("?")]
    a, b = unmapped[0], unmapped[1]
    topk = np.array([[b] * ev.TOP_K_MAX])
    true = np.array([a])
    m = ev.accuracy_from_dump(topk, true, arr)
    assert m["loinc_top1"] == 0.0 and m["loinc_top10"] == 0.0


def test_two_classes_sharing_a_loinc_count_as_correct():
    """Scoring is at LOINC level, so distinct MIMIC classes that map to the same LOINC match."""
    arr = np.array(["X", "X", "Y"], dtype=object)
    topk = np.array([[1, 2, 2, 2]])
    true = np.array([0])                                # class 0 and 1 are both LOINC "X"
    assert ev.accuracy_from_dump(topk, true, arr)["loinc_top1"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------
# per-family label / logit adapters
# ------------------------------------------------------------------------------
def test_labels_to_class_mlm_shifts_and_preserves_ignore_index():
    labels = torch.tensor([[ev.LAB_TOKEN_MIN, ev.LAB_TOKEN_MIN + 42, -100]])
    out = ev.labels_to_class("mlm", labels)
    assert out[0, 0].item() == 0
    assert out[0, 1].item() == 42
    assert out[0, 2].item() == -100, "-100 (the ignore index) must not be shifted"


def test_labels_to_class_cls_is_identity():
    labels = torch.tensor([[0, 42, -100]])
    assert torch.equal(ev.labels_to_class("bcm", labels), labels)
    assert torch.equal(ev.labels_to_class("llama", labels), labels)


def test_family_logits_mlm_slices_the_lab_band():
    vocab = ev.LAB_TOKEN_MAX + 50
    logits = torch.zeros(1, 2, vocab)
    logits[0, 0, ev.LAB_TOKEN_MIN] = 5.0        # class 0
    logits[0, 0, ev.LAB_TOKEN_MAX] = 7.0        # class 1622
    out = ev.family_logits("mlm", logits)
    assert out.shape[-1] == ev.N_LAB_CLASSES == 1623
    assert out[0, 0, 0].item() == 5.0
    assert out[0, 0, -1].item() == 7.0


def test_family_logits_cls_passthrough():
    logits = torch.zeros(1, 2, ev.N_LAB_CLASSES)
    assert ev.family_logits("cls", logits) is logits


def test_mlm_label_and_logit_adapters_are_mutually_consistent():
    """The class index from labels_to_class must index the same class in the sliced logits."""
    vocab = ev.LAB_TOKEN_MAX + 50
    tok = ev.LAB_TOKEN_MIN + 314
    logits = torch.full((1, 1, vocab), -9.0)
    logits[0, 0, tok] = 12.0
    sliced = ev.family_logits("mlm", logits)
    cls = ev.labels_to_class("mlm", torch.tensor([[tok]]))[0, 0].item()
    assert sliced[0, 0].argmax().item() == cls


# ------------------------------------------------------------------------------
# resolve_best_checkpoint: the guardrails must fire
# ------------------------------------------------------------------------------
def test_resolve_best_checkpoint_ok(tmp_path, monkeypatch):
    ck = tmp_path / "checkpoint-100"; ck.mkdir()
    res = tmp_path / "res"; res.mkdir()
    (res / "manifest_para_seed0.json").write_text(json.dumps(
        {"best_model_checkpoint": str(ck), "best_metric": 0.9, "best_step": 100}))
    monkeypatch.setitem(ev.TRAIN_RESULTS, "bcm", res)
    path, metric, step = ev.resolve_best_checkpoint("bcm", "para", 0)
    assert path == str(ck) and metric == 0.9 and step == 100


def test_resolve_best_checkpoint_raises_on_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setitem(ev.TRAIN_RESULTS, "bcm", tmp_path)
    with pytest.raises(FileNotFoundError):
        ev.resolve_best_checkpoint("bcm", "para", 0)


def test_resolve_best_checkpoint_raises_when_checkpoint_dir_absent(tmp_path, monkeypatch):
    """Guardrail: a manifest that points at a deleted checkpoint must raise an error. It must not
    fall back to an untrained model."""
    res = tmp_path / "res"; res.mkdir()
    (res / "manifest_para_seed0.json").write_text(json.dumps(
        {"best_model_checkpoint": str(tmp_path / "gone"), "best_metric": 0.9}))
    monkeypatch.setitem(ev.TRAIN_RESULTS, "bcm", res)
    with pytest.raises(FileNotFoundError):
        ev.resolve_best_checkpoint("bcm", "para", 0)


def test_constants_match_training_scripts():
    """Drift guard: these must equal the values in pipeline/12_mlm_train.py and 13_bcm_train.py, or
    the evaluation mis-scores."""
    assert ev.N_LAB_CLASSES == 1623
    assert ev.LAB_TOKEN_MIN == 34951 and ev.LAB_TOKEN_MAX == 36573
    assert ev.LAB_TOKEN_MAX - ev.LAB_TOKEN_MIN + 1 == ev.N_LAB_CLASSES
    assert ev.TEXT_VOCAB_SIZE == 30522
    assert ev.TOP_KS == (1, 3, 5, 10) and ev.TOP_K_MAX == 10
