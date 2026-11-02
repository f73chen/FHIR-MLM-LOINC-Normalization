"""Tests for the full-logit capture helpers in pipeline/15_eval_runner.py.

The capture takes days of GPU time and is run once. An error not caught at write time is found only
after the run. Two properties are tested directly:

    1. Centring is lossless. The capture stores (logit - row_max) as float16 instead of raw float32.
       This is safe only because softmax is shift-invariant at every temperature:
           softmax((z - c)/T) == softmax(z/T)
       If that failed for T != 1, every temperature-scaled number would be wrong, and temperature
       scaling is the reason for the capture.

    2. The alignment assert fires. Row i of logits.npy must be row i of the parquet. A misalignment
       changes no accuracy figure, so without this assert it would show up only during analysis,
       after the run. It must abort the capture at write time.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_S = importlib.util.spec_from_file_location(
    "ev", Path(__file__).resolve().parents[1] / "pipeline" / "15_eval_runner.py")
ev = importlib.util.module_from_spec(_S)
_S.loader.exec_module(ev)


def _logits(n=32, c=1623, seed=0, scale=8.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g) * scale


# -- 1. centring is lossless for softmax, at every temperature --------------
@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
def test_centring_is_softmax_lossless_at_every_temperature(T):
    """The stored format rests on this claim. If it fails for any T, centred logits cannot be stored."""
    lg = _logits()
    cen, rmax = ev.center_logits(lg)
    raw = lg.numpy().astype(np.float32)
    rec = cen.astype(np.float32) + rmax[:, None]          # reconstructed raw logits

    def sm(z, T):
        z = z / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    assert np.abs(sm(rec, T) - sm(raw, T)).max() < 1e-3


def test_centred_max_is_zero():
    cen, _ = ev.center_logits(_logits())
    assert np.allclose(cen.max(axis=1), 0.0, atol=1e-2)


def test_row_max_reconstructs_raw_logits():
    lg = _logits()
    cen, rmax = ev.center_logits(lg)
    assert np.abs((cen.astype(np.float32) + rmax[:, None]) - lg.numpy()).max() < 0.05


def test_dtypes_are_as_specified():
    cen, rmax = ev.center_logits(_logits())
    assert cen.dtype == np.float16, "centred logits must be float16 (halves a 390 GB footprint)"
    assert rmax.dtype == np.float32


def test_float16_does_not_destroy_top1_probability():
    """float16 is precise near 0, so the stored form must still give accurate probabilities."""
    lg = _logits(scale=12.0)
    cen, rmax = ev.center_logits(lg)
    e16 = np.exp(cen.astype(np.float32)); p16 = e16 / e16.sum(axis=1, keepdims=True)
    z = lg.numpy(); z = z - z.max(axis=1, keepdims=True)
    e32 = np.exp(z); p32 = e32 / e32.sum(axis=1, keepdims=True)
    assert np.abs(p16.max(axis=1) - p32.max(axis=1)).max() < 1e-4


# -- 2. true_rank -----------------------------------------------------------
def test_true_rank_matches_bruteforce():
    lg = _logits(n=64)
    tc = torch.randint(0, lg.shape[1], (64,))
    got = ev.true_rank_from_logits(lg, tc).numpy()
    exp = np.array([1 + int((lg[i].numpy() > lg[i, tc[i]].item()).sum()) for i in range(64)])
    assert (got == exp).all()


def test_true_rank_is_one_when_true_class_is_argmax():
    lg = torch.full((3, 10), -5.0)
    lg[torch.arange(3), torch.tensor([2, 7, 0])] = 9.0
    r = ev.true_rank_from_logits(lg, torch.tensor([2, 7, 0]))
    assert (r == 1).all()


def test_true_rank_is_nclasses_when_true_class_is_worst():
    lg = torch.arange(10, dtype=torch.float32).repeat(2, 1)   # class 0 is the lowest
    r = ev.true_rank_from_logits(lg, torch.tensor([0, 0]))
    assert (r == 10).all()


def test_true_rank_tie_convention_favours_the_truth():
    """All-equal logits give rank 1, which matches the tie handling in topk_reduce (strict >)."""
    lg = torch.zeros(2, 5)
    assert (ev.true_rank_from_logits(lg, torch.tensor([0, 3])) == 1).all()


# -- 3. the alignment assert fires ------------------------------------------
def _ok(n=10, c=1623):
    L = np.zeros((n, c), dtype=np.float16); L[:, 0] = 0.0
    L[:, 1:] = -1.0
    return L, np.zeros(n, dtype=np.float32)


def test_alignment_assert_passes_on_valid_input():
    L, RM = _ok()
    ev.assert_logit_dump_valid(L, RM, 10, 10, 1623)          # must not raise


@pytest.mark.parametrize("bad", ["short_logits", "short_rowmax", "parquet_mismatch", "npositions_mismatch"])
def test_alignment_assert_fires_on_every_length_mismatch(bad):
    L, RM = _ok()
    args = [L, RM, 10, 10, 1623]
    if bad == "short_logits":       args[0] = L[:9]
    if bad == "short_rowmax":       args[1] = RM[:9]
    if bad == "parquet_mismatch":   args[2] = 9
    if bad == "npositions_mismatch":args[3] = 9
    with pytest.raises(AssertionError):
        ev.assert_logit_dump_valid(*args)


def test_alignment_assert_fires_on_wrong_class_count():
    L, RM = _ok(c=679)
    with pytest.raises(AssertionError, match="expected 1623"):
        ev.assert_logit_dump_valid(L, RM, 10, 10, 1623)


def test_alignment_assert_fires_when_centring_is_broken():
    """Uncentred logits are rejected, so that a later change cannot store the wrong thing."""
    L = np.full((10, 1623), 3.0, dtype=np.float16)           # max != 0
    with pytest.raises(AssertionError, match="centring broken"):
        ev.assert_logit_dump_valid(L, np.zeros(10, dtype=np.float32), 10, 10, 1623)


# -- 4. TF-IDF log-prob surrogate -------------------------------------------
def test_softmax_of_log_probs_recovers_the_probabilities():
    """The TF-IDF classifier (one-vs-rest logistic regression) has no logit vector, so the capture
    stores log(predict_proba). This is a valid surrogate because softmax(log p) == p when p sums to 1.
    T=1 is therefore the identity."""
    rng = np.random.default_rng(0)
    p = rng.random((16, 679)); p /= p.sum(axis=1, keepdims=True)
    lp = np.log(p)
    lp = lp - lp.max(axis=1, keepdims=True)                  # centred, as stored
    e = np.exp(lp)
    assert np.abs(e / e.sum(axis=1, keepdims=True) - p).max() < 1e-6


def test_logprob_floor_keeps_float16_finite():
    """log(0) = -inf would become -inf in float16 and poison the row, so the writer floors at -60."""
    p = np.array([[0.5, 0.5, 0.0]])
    with np.errstate(divide="ignore"):
        lp = np.log(p)
    lp = np.maximum(lp, -60.0)
    assert np.isfinite(lp.astype(np.float16)).all()
    e = np.exp((lp - lp.max()).astype(np.float32))
    assert abs((e / e.sum())[0, 2]) < 1e-20, "floored class must still carry ~zero probability"
