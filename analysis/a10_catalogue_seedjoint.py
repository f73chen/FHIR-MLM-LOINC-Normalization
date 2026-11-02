#!/usr/bin/env python
"""Seed-averaged catalogue numbers with joint (code x seed) bootstrap intervals (thesis Chapter 6).

This module holds `joint_ci` and the bootstrap constants for the tables stage of
analysis/v3_catalogue.py. Appendix D quotes the constants.

The catalogue unit is more sensitive to the seed than the observation unit. Many rare codes carry a
tiny share of the observations, so a few flips on rare codes can change per-code accuracy while
per-observation accuracy barely moves. A catalogue table from a single seed only reports one draw.

The point estimate is therefore the mean over seeds, and the interval has to cover both sources of
variation. Adding the two variances is wrong because it assumes they are independent. The codes that
are hard are hard for every seed, so the two sources overlap. On this data, adding the variances is
conservative. Its interval is wider than the joint bootstrap interval.

There are two estimands, and the tables use A:
    A  pipeline mean: how precisely is this pipeline's mean per-code accuracy known? Resample codes,
       and resample which seeds were trained. This is what a table reporting "the model scores X"
       claims.
    B  one new model: where would a single newly trained model land? Resample codes and draw one
       seed. This is the right interval for a deployment forecast. It is wider, and using it for A
       overstates the uncertainty about the pipeline.

The code draw is shared across seeds within a replicate. This keeps the correlation between the two
sources in the interval.

Llama is trained once, so its seed resample is degenerate. Its interval has no seed component. It is
not comparable in kind to the other intervals, although the width is similar.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_C = importlib.util.spec_from_file_location("calib", HERE/"lib"/"calib.py")
C = importlib.util.module_from_spec(_C); _C.loader.exec_module(C)
_9 = importlib.util.spec_from_file_location("a9", HERE/"a9_catalogue_deployment.py")
A9 = importlib.util.module_from_spec(_9); _9.loader.exec_module(A9)

N_BOOT, BOOT_SEED, N_BINS = 4000, 0, 15
# per-code error budgets for the catalogue's coverage columns
BUDGETS = (0.01, 0.05, 0.10)


def joint_ci(mat, weights=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CIs for both estimands from one shared set of code draws.

    mat is codes x seeds. weights (per code) turn the mean into the observation-weighted deployed
    figure. The weights are resampled with the codes, because a code draw also changes the volume mix.
    """
    rng = np.random.default_rng(seed)
    nc, ns = mat.shape
    A = np.empty(n_boot); B = np.empty(n_boot)
    for b in range(n_boot):
        ci = rng.integers(0, nc, nc)
        sub = mat[ci]                                   # one code draw, shared across every seed
        if weights is None:
            per_seed = sub.mean(axis=0)
        else:
            w = weights[ci]; tot = w.sum()
            per_seed = (sub * w[:, None]).sum(axis=0) / tot if tot else np.full(ns, np.nan)
        A[b] = per_seed[rng.integers(0, ns, ns)].mean()  # resample which seeds were trained
        B[b] = per_seed[rng.integers(0, ns)]             # a single new model
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(A), q(B)
