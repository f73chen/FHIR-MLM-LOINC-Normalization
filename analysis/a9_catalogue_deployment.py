#!/usr/bin/env python
"""Catalogue-level selective prediction, and what a deployed catalogue mapping costs: the decision for
each local code, its confidence and correctness, deployed accuracy, and how concentrated the errors are.
Used by analysis/v3_catalogue.py --stage percode (thesis Chapter 6).

Per-code accuracy leaves two questions open:

(1) How much of a catalogue can be auto-mapped? Per-code accuracy without a confidence cannot say which
    codes a terminologist still has to review. A migration needs to know which codes it can trust to
    map automatically. That is selective prediction at the code level. Whole codes are routed to a
    reviewer in the same way as whole observations in a live feed.

(2) What does a wrong code cost? By design, per-code accuracy treats a code seen twice and a code seen
    a hundred thousand times alike. A deployment cares about both how many codes need review and how
    much of the incoming data a mis-mapped code corrupts. Per-code and per-observation accuracy can
    differ widely for the same model. They answer different questions, and the volume-weighted view
    reconciles them.

Each of the three numbers answers a different question:
  per-observation accuracy   the model predicts each incoming result independently    (live feed)
  per-code accuracy          one decision per code, scored per code                   (migration effort)
  deployed accuracy          one decision per code, applied to every observation      (migration outcome)

Deployed accuracy is what a site experiences after migrating. The per-code decision is written into
the catalogue once, and every future result for that code inherits it.

Confidence is not a posterior. It is the share of aggregated probability mass on the winning LOINC.
The analysis measures whether it behaves like a probability of being correct.

Tested by tests/test_catalogue_deployment.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_S = importlib.util.spec_from_file_location("a8", HERE / "a8_catalogue.py")
A8 = importlib.util.module_from_spec(_S); _S.loader.exec_module(A8)
_C = importlib.util.spec_from_file_location("calib", HERE / "lib" / "calib.py")
C = importlib.util.module_from_spec(_C); _C.loader.exec_module(C)


# -- pure logic (unit-tested in tests/test_catalogue_deployment.py) ----------
def code_scores(mass: np.ndarray, gold: np.ndarray, space: np.ndarray) -> dict:
    """Per-code prediction, confidence and correctness from the a8_catalogue summed-mass matrix.

    `conf` is the share of the code's total aggregated mass that sits on the winning LOINC. Rows with
    zero mass would give 0/0. These are codes where every observation put all its probability
    outside the 679 space. They get confidence 0 and are counted as wrong, because dropping them would
    flatter the metric.
    """
    tot = mass.sum(axis=1)
    safe = np.where(tot > 0, tot, 1.0)
    p = mass / safe[:, None]
    idx = p.argmax(axis=1)
    conf = np.where(tot > 0, p[np.arange(len(p)), idx], 0.0)
    pred = space[idx]
    correct = (pred == gold) & (tot > 0)
    return {"pred": pred, "conf": conf, "correct": correct}


def deployed_accuracy(correct: np.ndarray, n_obs: np.ndarray) -> float:
    """Observation-level accuracy obtained by applying each code's single decision to all its rows.

    This is the migration outcome. There is one decision per code, and every result of that code
    inherits it. It differs from per-observation prediction accuracy, which lets the model answer
    differently on each observation. A migrated catalogue cannot do that.
    """
    n_obs = np.asarray(n_obs, dtype=np.float64)
    total = n_obs.sum()
    return float((n_obs * np.asarray(correct, dtype=np.float64)).sum() / total) if total else float("nan")


def blast_radius(correct: np.ndarray, n_obs: np.ndarray) -> dict:
    """How concentrated are catalogue errors, in codes and in the data volume they affect?"""
    correct = np.asarray(correct, dtype=bool)
    n_obs = np.asarray(n_obs, dtype=np.float64)
    wrong = ~correct
    return {"n_codes": int(len(correct)),
            "n_codes_wrong": int(wrong.sum()),
            "pct_codes_wrong": float(wrong.mean()),
            "obs_affected": int(n_obs[wrong].sum()),
            "pct_obs_affected": float(n_obs[wrong].sum() / n_obs.sum()) if n_obs.sum() else float("nan"),
            "median_obs_wrong_code": float(np.median(n_obs[wrong])) if wrong.any() else float("nan"),
            "median_obs_right_code": float(np.median(n_obs[correct])) if correct.any() else float("nan")}
