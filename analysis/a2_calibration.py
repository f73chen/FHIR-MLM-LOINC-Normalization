#!/usr/bin/env python
"""Calibration helpers (RQ4): the canonical 679-code label space, the map from each model's logit
columns onto it, and the reduction of one evaluation dump to per-row scalars.

The reduction is one streaming pass over the full logits of a dump. This is the expensive step, at
gigabytes per test run. It gives a small table with one row per observation: top-1 probability,
margin, probability of the true LOINC, entropy, correctness, predicted index and the probability mass
outside the label space. Every calibration and selective-prediction metric is then computed from
these tables.

All probabilities are marginalised into the 679 primary LOINC codes before any metric (see
lib/calib.py). The deep models emit 1,623 classes and TF-IDF emits 679. Correctness is at the LOINC
level for both, so confidence must be too.

Reductions carry patient_uuid because the patient-clustered bootstrap needs it. They therefore stay
in WORK_DIR/analysis/a2_reductions. Stdout is aggregate only.

Used by analysis/v3_calibration.py (thesis Chapter 6):
  python analysis/v3_calibration.py --stage reduce
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
_S = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_S); _S.loader.exec_module(D)
_C = importlib.util.spec_from_file_location("calib", HERE / "lib" / "calib.py")
C = importlib.util.module_from_spec(_C); _C.loader.exec_module(C)

EVAL = D.EVAL_BASE
RED = locations.WORK / "analysis/a2_reductions"
CHUNK = 20_000



# -----------------------------------------------------------------------------
# label space
# -----------------------------------------------------------------------------

def canonical_space() -> np.ndarray:
    """The 679 primary LOINC codes, in the order of the TF-IDF logit columns. Every model is mapped
    onto this order so that column i means the same LOINC everywhere.

    The list is read from the manifest of one TF-IDF dump. `v3_calibration.py --stage gates` checks
    that every v3 TF-IDF dump carries the same list."""
    man = json.load(open(EVAL / "tfidf__para_xsite__pert__seed0__mimictest" / "manifest.json"))
    return np.array(man["label_space_679"], dtype=object)


def group_for(model: str, space: np.ndarray) -> np.ndarray:
    """group[c] = column in `space` that logit column c contributes to, or -1 to drop it."""
    pos = {l: i for i, l in enumerate(space)}
    if model == "tfidf":
        return np.arange(len(space))                       # already in canonical order
    arr = D.class_to_loinc_array()                         # (1623,) class -> LOINC or '?i'
    return np.array([pos.get(a, -1) for a in arr], dtype=np.int64)


# -----------------------------------------------------------------------------
# reduce
# -----------------------------------------------------------------------------


def reduce_cell(cell, T=1.0, limit_rows=None, force=False, tag_suffix="") -> Path | None:
    """Reduce the dump of one `lib.cells.Cell` at temperature T. The reduction file is named after the
    cell."""
    return _reduce_dir(cell.path, cell.model, cell.name, T, limit_rows, force, tag_suffix)


def _reduce_dir(d: Path, model: str, base_tag: str, T=1.0, limit_rows=None, force=False,
                tag_suffix="") -> Path | None:
    space = canonical_space()
    g = group_for(model, space)
    if not (d / "logits_centered_f16.npy").exists():
        return None
    # A truncated run must not take the canonical filename. Partial runs overwriting real outputs, or a
    # validation run colliding with a test run, are easy mistakes. So the suffix rule is applied here,
    # when the file is created, and is not left to the caller.
    tag = base_tag + tag_suffix
    if limit_rows is not None:
        tag += "__smoke"
    out = RED / f"{tag}.parquet"
    if out.exists() and not force:
        print(f"  SKIP {tag} (exists)"); return out
    RED.mkdir(parents=True, exist_ok=True)

    pq = pd.read_parquet(d / "part-0000.parquet", columns=["patient_uuid", "obs_ordinal", "true_loinc"])
    pos = {l: i for i, l in enumerate(space)}
    true_idx = np.array([pos.get(t, -1) for t in pq.true_loinc.values], dtype=np.int64)
    keep = true_idx >= 0
    n_drop = int((~keep).sum())

    L = np.load(d / "logits_centered_f16.npy", mmap_mode="r")
    n = L.shape[0] if limit_rows is None else min(limit_rows, L.shape[0])
    cols = {k: [] for k in ("p_top1", "margin", "p_true", "entropy", "correct", "pred_idx", "dropped")}
    t0 = time.time()
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        z = np.asarray(L[a:b], dtype=np.float32)
        logp = C.marginal_logprob(z, g, len(space), T=T)
        ti = np.clip(true_idx[a:b], 0, None)
        st = C.row_stats(logp, ti)
        for k in ("p_top1", "margin", "p_true", "entropy", "pred_idx"):
            cols[k].append(st[k])
        cols["correct"].append(st["correct"])
        cols["dropped"].append(C.dropped_mass(z, g, T=T).astype(np.float32))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["patient_uuid"] = pq.patient_uuid.values[:n]
    df["obs_ordinal"] = pq.obs_ordinal.values[:n]
    df["true_idx"] = true_idx[:n]
    df = df[keep[:n]]
    # Atomic: write beside, then rename. Two reduction workers can meet on one cell. With a rename,
    # the visible file is always one complete write and cannot be an interleaving of two.
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"  {tag}: {len(df):,} rows in {time.time()-t0:.1f}s "
          f"(dropped {n_drop} out-of-space, mean outside-mass {df.dropped.mean():.6f})")
    return out
