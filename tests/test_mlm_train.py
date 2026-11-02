"""Tests for the helpers of pipeline/12_mlm_train.py, on synthetic fixtures and the real code mapping.
They run in the deep-models environment because they import torch and transformers. No GPU or patient
data is needed."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_SPEC = importlib.util.spec_from_file_location(
    "mlm", Path(__file__).resolve().parents[1] / "pipeline" / "12_mlm_train.py")
mlm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mlm)

LO = mlm.LAB_TOKEN_MIN
HI = mlm.LAB_TOKEN_MAX


def test_build_loinc_mappings_real():
    """Token->LOINC mapping from the real code files (aggregate counts only)."""
    tokens, t2l = mlm.build_loinc_mappings(mlm.CODES_TO_TOKENS_PATH, mlm.SEEN_CODES_PATH,
                                           mlm.LAB_CODE_SYSTEM, mlm.TEXT_VOCAB_SIZE)
    assert tokens == set(t2l.keys())
    assert 700 < len(t2l) < 900                  # ~828 LOINC-mappable lab codes
    assert all(LO <= t <= HI for t in tokens)    # every lab token id within the [4429,6051]+offset range


def test_preprocess_logits_topk():
    """Top-K reduction returns the absolute ids of the K largest lab logits, in order. Non-lab logits
    are ignored."""
    vocab = HI + 100
    logits = torch.full((1, 1, vocab), -5.0)
    logits[0, 0, LO + 5] = 10.0     # largest in lab range
    logits[0, 0, LO + 3] = 7.0      # 2nd
    logits[0, 0, 999] = 99.0        # non-lab (outside the lab slice): must be ignored
    ids, _ = mlm.preprocess_logits_for_metrics(logits, torch.zeros(1, 1), LO, HI)
    assert ids.shape[-1] == mlm.TOP_K_MAX
    assert ids[0, 0, 0].item() == LO + 5
    assert ids[0, 0, 1].item() == LO + 3


def test_compute_metrics_topk_accuracy():
    """Top-K LOINC accuracy: one top-1 hit, one top-3-only hit, one ignored (-100)."""
    t2l = {LO: "A", LO + 1: "B", LO + 2: "C"}
    preds = np.full((1, 3, mlm.TOP_K_MAX), LO, dtype=np.int64)   # everything -> "A" by default
    preds[0, 1, 1] = LO + 1                                       # pos1 has "B" at rank 2
    labels = np.array([[LO, LO + 1, -100]], dtype=np.int64)       # pos0 "A", pos1 "B", pos2 ignored
    m = mlm.compute_metrics((preds, labels), t2l)
    assert m["eval_samples"] == 2
    assert m["loinc_top1"] == 0.5    # pos0 A@1 hit; pos1 A@1 != B miss
    assert m["loinc_top3"] == 1.0    # pos1 B@2 within top3
    assert m["loinc_accuracy"] == m["loinc_top1"]


def test_compute_metrics_unmappable_and_empty():
    t2l = {LO: "A"}
    preds = np.full((1, 1, mlm.TOP_K_MAX), LO, dtype=np.int64)
    m = mlm.compute_metrics((preds, np.array([[LO + 999]], dtype=np.int64)), t2l)  # unmappable true -> "?"
    assert m["eval_samples"] == 1 and m["loinc_top1"] == 0.0
    m0 = mlm.compute_metrics((preds, np.array([[-100]], dtype=np.int64)), t2l)     # all ignored
    assert m0["eval_samples"] == 0 and m0["loinc_top1"] == 0.0


def test_apply_lab_logit_mask():
    """lm_head hook biases non-lab logits to -inf, leaves lab logits finite."""
    vocab = HI + 50

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(4, vocab, bias=False)

        def forward(self, x):
            return self.lm_head(x)

    m = M()
    mlm.apply_lab_logit_mask(m, LO, HI, vocab)
    out = m(torch.randn(2, 4))
    assert torch.isinf(out[:, :LO]).all() and (out[:, :LO] < 0).all()   # below lab range -> -inf
    assert torch.isfinite(out[:, LO:HI + 1]).all()                       # lab range -> finite
    assert torch.isinf(out[:, HI + 1:]).all()                            # above lab range -> -inf
