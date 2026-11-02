#!/usr/bin/env python
"""Evaluation-time sliding windows. Does accuracy hold on the observations that truncation leaves
unscored? This module has the integrity checks, the by-window table and the window-0-vs-later gap,
for analysis/v3_sliding.py (thesis Chapter 6).

Every headline number is computed on a single 8,192-token window per patient, so long stays are scored
only up to the truncation point. Two different things could be true of the unscored tail:

    (a) it is more of the same, and the reported accuracy generalises to it; or
    (b) accuracy decays with position (later observations are rarer, sicker, more ambiguous), and
        the reported number is optimistic because truncation hid the hard part.

The headline evaluation cannot tell these apart, because the tail is never scored. The sliding
evaluation runs the same trained checkpoints over non-overlapping windows that cover the whole record,
so the tail is measured.

The windowing is at evaluation time only, and the checkpoints were trained on single truncated
windows. So windows 1+ are a mild distribution shift. The model sees a mid-record slice, with only the
Patient and most recent Encounter as preamble, while in training every sequence began at the record
start. Any window-0-vs-later gap therefore confounds "later observations are harder" with "starting
mid-record is out of distribution". That is a limitation of the design, and it is why the gap is
reported instead of a single pooled accuracy.

Two confounds must be controlled before reading any gap:
  * Patient mix. Many patients need only a single window, so the window-0 pool contains every patient
    while the later pool contains only long-stay patients. A raw window-0-vs-later comparison is
    partly a short-stay-vs-long-stay comparison. This is controlled by restricting to multi-window
    patients.
  * Code mix. Later windows carry a different LOINC distribution (more repeats of common panels, and
    a different rare-code tail). This is controlled by reweighting later accuracy to the code mix of
    window 0.
  Removing the patient mix alone can reverse the sign of a raw gap, so the raw gap could show an
  effect that is not there.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_D = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_D); _D.loader.exec_module(D)


# Roughly geometric bins. Window counts are heavily right-skewed (median 1, max 128), so equal-width
# bins would put ~99% of positions in the first bin and leave the tail unreadable.
WBINS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 10_000]
WLABS = ["0", "1", "2", "3-4", "5-7", "8-12", "13-20", "21-33", "34-54", "55+"]

N_BOOT, BOOT_SEED = 2000, 0


def integrity(model: str, sl: pd.DataFrame, can: pd.DataFrame) -> dict:
    """The join key must survive windowing, or every cross-model comparison downstream is corrupt.

    obs_ordinal is emitted as range(obs_offset, obs_offset+n) per window. If obs_offset counted raw
    lab tokens, which include unlabelled ones, window 2 would start past where window 1 ended and
    leave a gap. So contiguity is checked directly. The ordinals of every patient must run from 0 to
    n-1 with no gaps.
    """
    dup = int(sl.duplicated(["patient_uuid", "obs_ordinal"]).sum())
    g = sl.groupby("patient_uuid")["obs_ordinal"]
    noncontig = int(((g.min() != 0) | (g.max() != g.count() - 1)).sum())
    wmax = sl.groupby("patient_uuid")["window_idx"].max()

    # Window 0 must be a prefix subset of the canonical run, and need not be an equal set. The
    # canonical build truncates at a token boundary (mid-resource), while window 0 ends at a resource
    # boundary. So window 0 holds <= the canonical positions and must agree with them where they overlap.
    w0 = sl[~sl.late]
    j = can.merge(w0[["patient_uuid", "obs_ordinal", "correct"]],
                  on=["patient_uuid", "obs_ordinal"], suffixes=("_can", "_w0"))
    return {
        "model": model,
        "n_positions_sliding": len(sl), "n_positions_canonical": len(can),
        "coverage_gain": len(sl) / len(can),
        "n_patients": int(sl.patient_uuid.nunique()),
        "duplicate_join_keys": dup,
        "patients_with_noncontiguous_ordinals": noncontig,
        "max_window_idx": int(wmax.max()),
        "patients_at_window_cap": int((wmax >= 127).sum()),
        "n_window0": len(w0),
        "n_window0_joined_to_canonical": len(j),
        "window0_join_rate_vs_canonical": len(j) / len(can),
        "window0_agreement_with_canonical": float((j.correct_can == j.correct_w0).mean()),
        "acc_canonical": float(can.correct.mean()),
        "acc_sliding_all_windows": float(sl.correct.mean()),
    }


def by_window(sl: pd.DataFrame) -> pd.DataFrame:
    b = sl.assign(wb=pd.cut(sl.window_idx, bins=WBINS, right=False, labels=WLABS))
    t = (b.groupby("wb", observed=True)
          .agg(n=("correct", "size"), n_patients=("patient_uuid", "nunique"),
               acc=("correct", "mean"))
          .reset_index())
    t["err_per_10k"] = (1 - t.acc) * 10_000
    return t


def gap(sl: pd.DataFrame) -> dict:
    """window-0 minus later accuracy, raw and under both controls, with a patient-clustered CI.

    The bootstrap resamples patients. Observations within a patient share a record, a paraphrase
    assignment and a clinical trajectory, so a position-level interval would be far too narrow. The
    calibration bootstrap clusters for the same reason.
    """
    nw = sl.groupby("patient_uuid")["window_idx"].max()
    multi = sl[sl.patient_uuid.isin(nw[nw >= 1].index)]

    raw_gap = float(sl[~sl.late].correct.mean() - sl[sl.late].correct.mean())
    ctl_gap = float(multi[~multi.late].correct.mean() - multi[multi.late].correct.mean())

    # code-matched: reweight later accuracy to window 0's LOINC distribution (direct standardisation)
    a = (multi.groupby(["true_loinc", "late"])["correct"]
              .agg(["size", "mean"]).unstack("late").dropna())
    w = a[("size", False)]
    std_w0 = float((w * a[("mean", False)]).sum() / w.sum())
    std_late = float((w * a[("mean", True)]).sum() / w.sum())

    g = multi.groupby(["patient_uuid", "late"])["correct"].agg(["sum", "size"]).unstack("late").fillna(0)
    e0, n0 = g[("sum", False)].values, g[("size", False)].values
    e1, n1 = g[("sum", True)].values, g[("size", True)].values
    rng = np.random.default_rng(BOOT_SEED)
    P = len(e0)
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, P, P)
        bs[b] = e0[i].sum() / n0[i].sum() - e1[i].sum() / n1[i].sum()
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    return {
        "gap_raw_per10k": raw_gap * 10_000,
        "gap_patient_matched_per10k": ctl_gap * 10_000,
        "gap_code_matched_per10k": (std_w0 - std_late) * 10_000,
        "gap_ci_lo_per10k": lo * 10_000, "gap_ci_hi_per10k": hi * 10_000,
        "n_patients_multi_window": P,
        "significant": bool(lo * hi > 0),
        "n_shared_codes": len(a),
    }
