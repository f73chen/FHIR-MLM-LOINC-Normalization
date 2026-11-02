#!/usr/bin/env python
"""The fixed-cohort curve and trend test behind F-10a and F-10b. They show accuracy by window index in
the sliding-window evaluation (thesis Chapter 6).

The curve supports the claim that accuracy does not decay at late positions in the record. If that
holds, the truncated denominators that every headline number uses are not optimistic. The
patient-matched window-0-vs-later test (`b3_sliding.gap`) is the rigorous version of that claim. The
curve lets a reader see the flatness directly instead of taking it from a confidence interval.

The raw accuracy by window index (`b3_sliding.by_window`) is not plotted, because the patient cohort
shrinks across bins. It goes from every patient at window 0 to a few dozen in the deepest bins. A line
through those points changes population at every step. This is the same patient-mix confound that can
flip the sign of a raw window-0 gap once controlled. A deep bin of two patients would also show as a
spike.

A fixed cohort is plotted instead. For each model, only patients whose record reaches window index
DEPTH are kept, i.e. DEPTH+1 windows or more. Windows 0..DEPTH-1 are plotted, so every x-position
describes the same patients.

The cohort needs one more window than is plotted. The extra window is intended. With a window
beyond the plotted range, every plotted window is non-final. Only non-final windows are full,
because a last window is only partly filled. Under a rule of "DEPTH windows", the rightmost position
would mix full and partial windows while every other position is all full. That position also has the
most leverage on the fitted slope. Depth differs by model because record lengths in windows differ.

Within a fixed cohort the error rate still wanders from window to window. That is noise, because
observations cluster within patients and the effective sample size is far below the raw count. Without
intervals, the wiggle would look like a real positional pattern, and the figure would argue against
the conclusion it is meant to support. The intervals come from a patient-clustered bootstrap, as in
the calibration analysis.

Used by analysis/v3_sliding.py --stage run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_D = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_D); _D.loader.exec_module(D)
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)



# Chosen per model from the data. Each is the deepest index at which the fixed cohort still holds
# enough patients to say anything. FHIR-MLM's records span fewer windows, so very few of its patients
# reach index 16 and it stops at 8. v3_sliding.py halves a depth while its cohort is below MIN_COHORT.
DEPTH = {"mlm": 8, "bcm": 16, "llama": 16}
N_BOOT, BOOT_SEED = 500, 0


def curve(g: pd.DataFrame, depth: int):
    """Error per 10k by window index, with a patient-clustered bootstrap interval, plus a trend test.

    The bootstrap resamples patients. Two observations from one patient share a record and a
    paraphrase assignment, so an observation-level interval would be far too narrow.

    The figure claims "no decay with position", so the test asks whether the error rate has a slope
    in window index. Comparing each window's interval with the window-0 point estimate would ignore
    the uncertainty of window 0 itself. It would also make `depth` comparisons with no multiplicity
    control and flag windows as different in a scatter with no trend. The slope is refit inside every
    bootstrap resample, so its interval is patient-clustered too.
    """
    pats = g.patient_uuid.unique()
    idx = {p: i for i, p in enumerate(pats)}
    N = np.zeros((len(pats), depth)); K = np.zeros((len(pats), depth))
    for r in g.itertuples():
        N[idx[r.patient_uuid], r.window_idx] = r.n
        K[idx[r.patient_uuid], r.window_idx] = r.ok
    x = np.arange(depth)
    point = (1 - K.sum(0) / N.sum(0)) * 10_000
    rng = np.random.default_rng(BOOT_SEED)
    bs = np.empty((N_BOOT, depth)); slopes = np.empty(N_BOOT)
    lines = np.empty((N_BOOT, depth))
    for b in range(N_BOOT):
        s = rng.integers(0, len(pats), len(pats))
        bs[b] = (1 - K[s].sum(0) / N[s].sum(0)) * 10_000
        cf = np.polyfit(x, bs[b], 1)
        slopes[b] = cf[0]
        lines[b] = np.polyval(cf, x)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    slope = float(np.polyfit(x, point, 1)[0])
    s_lo, s_hi = (float(v) for v in np.percentile(slopes, [2.5, 97.5]))
    # Envelope of the bootstrap regression lines (lo and hi above are the per-window intervals). It
    # makes "no trend" visible. If a horizontal line fits inside the envelope, the data cannot tell
    # the fitted slope from flat. Drawing the single fitted line alone fails the other way, because
    # the eye reads a slope that the interval says is not there.
    t_lo, t_hi = np.percentile(lines, [2.5, 97.5], axis=0)
    return point, lo, hi, N.sum(0), (slope, s_lo, s_hi, t_lo, t_hi)
