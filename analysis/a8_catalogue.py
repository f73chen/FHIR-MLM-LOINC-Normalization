#!/usr/bin/env python
"""Catalogue-level evaluation: one mapping decision per local lab code, aggregated from a dump's
per-observation probabilities (thesis Chapters 5 and 6).

Observation-level accuracy treats each lab observation as an independent prediction, weighted by
observation. LOINC mapping in practice, such as migrating an existing catalogue, makes one decision
per local code. The two differ. Observation-level accuracy is dominated by a handful of very frequent
codes, while at the catalogue level a code that appears twice weighs as much as hematocrit.

The unit of evaluation is the local MIMIC code (`true_class`), which is what a site has at migration
time. Grouping by the gold LOINC would form the group from the answer.

Aggregation rules, all computed so the choice stays visible:
  summed_mass   sum P(LOINC | observation) across every observation of the code, then rank.
                The primary rule. It gives a full ranking, so top-k comes for free.
  majority_vote per-observation argmax, most frequent wins. The ranking is coarse (ties are common),
                so top-k from it is not meaningful beyond k=1.
  single_obs    one randomly chosen observation per code. This is the no-aggregation reference.
                The gap between it and summed_mass is what aggregation buys.

  mean_logprob   sum log P(LOINC | observation) across observations, then rank. Equivalent to
                 multiplying the probabilities. Reported but not primary, because a single
                 confidently wrong observation drives the product for the correct code toward
                 zero and vetoes it.

Probabilities are marginalised to the 679 LOINC codes (sibling classes summed) before aggregation,
because P(LOINC | observation) is the right probabilistic object, as in lib/calib.py.

Reads the part-0000.parquet and logits_centered_f16.npy of a dump. Used by analysis/v3_catalogue.py
(--stage percode), a9_catalogue_deployment.py and a9_percode_dump.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_A = importlib.util.spec_from_file_location("a2c", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)
_D = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_D); _D.loader.exec_module(D)


TOP_KS = (1, 3, 5, 10)
CHUNK = 100_000
N_BOOT, BOOT_SEED = 2000, 0
LOG_EPS = 1e-12   # floor for log(0) under the mean_logprob rule, as in tests/test_catalogue.py


# -- aggregation -------------------------------------------------------------
def marginaliser(model: str, space: np.ndarray):
    """Sparse (n_logit_cols -> 679) 0/1 matrix, so marginalisation is one matmul per chunk.

    lib/calib.marginal_logprob loops over 1,623 columns doing logaddexp. That is correct but far too
    slow to run over every observation of every arm. Summing probabilities (not log-probs) is the
    same operation here and vectorises.
    """
    from scipy.sparse import csr_matrix
    g = A.group_for(model, space)
    keep = np.flatnonzero(g >= 0)
    return csr_matrix((np.ones(len(keep), dtype=np.float32), (keep, g[keep])),
                      shape=(len(g), len(space))), g


def obs_table_cell(cell) -> pd.DataFrame:
    """The observations of a `lib.cells.Cell`, in dump order, with the local code. The TF-IDF dumps
    do not carry it, so it comes through lib/codes.attach, which refuses unmatched rows and gold
    disagreements."""
    cols = ["patient_uuid", "obs_ordinal", "true_loinc"] + ([] if cell.model == "tfidf" else ["true_class"])
    df = pd.read_parquet(cell.path / "part-0000.parquet", columns=cols)
    if cell.model == "tfidf":
        from lib import codes as K
        df = K.attach(df)
    df["row"] = np.arange(len(df))
    return df


def aggregate_cell(cell, space: np.ndarray, keep_keys: set | None = None, limit: int | None = None,
                   topk_only: int | None = None):
    """aggregate_rows() over a Cell's dump and its centred logits."""
    L = np.load(cell.path / "logits_centered_f16.npy", mmap_mode="r")
    return aggregate_rows(obs_table_cell(cell), L, cell.model, space, limit=limit, topk_only=topk_only,
                          keep_keys=keep_keys)


def aggregate_rows(df: pd.DataFrame, L, model: str, space: np.ndarray, limit: int | None = None,
                   topk_only: int | None = None, keep_keys: set | None = None):
    """The aggregation itself, over `df` (dump-ordered rows with `row`, `true_class`, `true_loinc`)
    and the centred logits `L` of the dump. `model` is the model family of the dump, which decides
    the marginaliser."""
    from scipy.sparse import csr_matrix
    if keep_keys is not None:
        keep = [k in keep_keys for k in zip(df.patient_uuid, df.obs_ordinal)]
        df = df[np.asarray(keep)]
    if limit:
        df = df.iloc[:limit]
    M, _ = marginaliser(model, space)

    codes = np.sort(df.true_class.unique())
    cidx = {c: i for i, c in enumerate(codes)}
    code_of_row = df.true_class.map(cidx).to_numpy()

    mass = np.zeros((len(codes), len(space)), dtype=np.float64)
    # Summed log probability. This is the multiply-the-probabilities rule, in log space so that it
    # does not underflow over the thousands of observations a common code carries. Ranking by the
    # sum is the same as ranking by the mean within a code, because n_obs is a per-code constant. So
    # no division is needed. Zeros are clipped to LOG_EPS before the log. Left as -inf, every code whose
    # gold LOINC was ever given exactly zero mass would be unrankable. That is the veto effect
    # itself, and it would look the same as a NaN bug in the output.
    logmass = np.zeros((len(codes), len(space)), dtype=np.float64)
    votes = np.zeros((len(codes), len(space)), dtype=np.float64)
    n_obs = np.zeros(len(codes), dtype=np.int64)
    # Per-code correct counts, so that observation-level accuracy can be reported on the same
    # observations and codes that the catalogue metric uses. Reading it from the full dump would mix
    # denominators. A native rate beside a matched-code catalogue rate is not a like-for-like drop.
    n_correct = np.zeros(len(codes), dtype=np.int64)
    truth_all = df.true_loinc.to_numpy()

    for s in range(0, len(df), CHUNK):
        e = min(s + CHUNK, len(df))
        z = np.asarray(L[df.row.values[s:e]], dtype=np.float32)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        # sparse.T @ dense.T keeps scipy on the left, which returns a plain ndarray. `dense @ sparse`
        # can return an np.matrix, which changes how broadcasting behaves downstream.
        q = np.asarray(M.T.dot(p.T).T)                     # -> (chunk, 679), sibling classes summed
        if topk_only:                                      # emulate a top-k-truncated dump
            cut = np.partition(q, -topk_only, axis=1)[:, -topk_only][:, None]
            q = np.where(q >= cut, q, 0.0)
        tot = q.sum(axis=1, keepdims=True)
        q = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)   # renormalise over kept groups
        g = code_of_row[s:e]
        # Group-sum via a sparse indicator. np.add.at is an unbuffered scatter-add and is orders of
        # magnitude slower over a (chunk x 679) block, which matters over millions of observations.
        S = csr_matrix((np.ones(e - s, dtype=np.float32), (g, np.arange(e - s))),
                       shape=(len(codes), e - s))
        mass += S.dot(q)
        logmass += S.dot(np.log(np.clip(q, LOG_EPS, None)))
        np.add.at(votes, (g, q.argmax(axis=1)), 1.0)
        np.add.at(n_obs, g, 1)
        np.add.at(n_correct, g, (space[q.argmax(axis=1)] == truth_all[s:e]).astype(np.int64))

    gold = df.groupby("true_class").true_loinc.agg(lambda s: s.iloc[0]).reindex(codes).to_numpy()
    return {"codes": codes, "mass": mass, "logmass": logmass, "votes": votes, "n_obs": n_obs,
            "n_correct": n_correct, "gold": gold, "df": df, "space": space}


# -- metrics -----------------------------------------------------------------
def topk_from_scores(scores: np.ndarray, gold: np.ndarray, space: np.ndarray, ks=TOP_KS) -> dict:
    """Fraction of codes whose gold LOINC is in the top-k of the aggregated ranking."""
    order = np.argsort(-scores, axis=1)[:, :max(ks)]
    ranked = space[order]
    return {f"top{k}": float((ranked[:, :k] == gold[:, None]).any(axis=1).mean()) for k in ks}


def boot_ci(scores: np.ndarray, gold: np.ndarray, space: np.ndarray, k: int,
            n_boot=N_BOOT, seed=BOOT_SEED) -> tuple[float, float]:
    """Bootstrap over codes, so n is the number of codes (hundreds), far below the number of
    observations (millions). The intervals are therefore wide and must be reported."""
    order = np.argsort(-scores, axis=1)[:, :k]
    hit = (space[order] == gold[:, None]).any(axis=1)
    rng = np.random.default_rng(seed)
    n = len(hit)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc = hit[idx].mean(axis=1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def single_obs_scores_from(agg: dict, L, model: str, pick_seed: int = 0):
    """Scores for the single_obs rule: one randomly chosen observation per code (`pick_seed`),
    marginalised and renormalised, from an already-open logits array `L`."""
    df = agg["df"].sample(frac=1.0, random_state=pick_seed).drop_duplicates("true_class")
    df = df.set_index("true_class").reindex(agg["codes"]).reset_index()
    M, _ = marginaliser(model, agg["space"])
    z = np.asarray(L[df.row.values.astype(int)], dtype=np.float32)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q = np.asarray(p @ M)
    tot = q.sum(axis=1, keepdims=True)
    return np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)
