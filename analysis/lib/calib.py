"""Calibration and selective-prediction functions (RQ4, thesis Chapters 5 and 6): marginal
probabilities, calibration metrics, risk-coverage, temperature scaling and a patient-clustered
bootstrap. They are pure functions without I/O, tested by tests/test_calib.py.

Probabilities are marginalised into LOINC space before any metric is computed. The deep models emit a
1,623-way softmax over MIMIC lab classes. TF-IDF emits a 679-way softmax directly over LOINC codes.
Correctness is judged in LOINC space for both. Some LOINC codes receive more than one deep class (up
to four). A prediction can then be scored correct while its confidence is split across the sibling
classes, and calibration would compare a LOINC-level correctness label with a class-level
probability. Summing the sibling probabilities gives the marginal P(LOINC) = sum_c P(class c), which
puts confidence and correctness in the same space. Over the cell-D test sets it changes the argmax
LOINC on very few rows (at most about 0.04%), so accuracy is practically unaffected there.
analysis/v3_checks.py (--stage casting) measures this. Low-confidence cells move more (gate G7 in
analysis/v3_calibration.py).

Used by analysis/a2_calibration.py, v3_calibration.py, a9_catalogue_deployment.py,
a10_catalogue_seedjoint.py and b4_oneshot_baselines.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


# -----------------------------------------------------------------------------
# probability construction
# -----------------------------------------------------------------------------

def marginal_logprob(z: np.ndarray, group: np.ndarray, n_groups: int, T: float = 1.0) -> np.ndarray:
    """Softmax at temperature T, then sum probabilities within each group, in log space.

    z       : (n, n_classes) centred logits (row max == 0). float16 is promoted internally.
    group   : (n_classes,) int, group[c] = target column for class c, or -1 to drop the class.
    n_groups: width of the output.
    Returns (n, n_groups) log-probabilities renormalised over the kept groups, so each row sums to 1.

    Dropped mass is renormalised away, because the evaluated label space is the 679 primary LOINC
    codes and the mass outside it is small (0.04% on average). `dropped_mass` reports it.
    """
    z = np.asarray(z, dtype=np.float32)
    if T != 1.0:
        z = z / np.float32(T)
    keep = group >= 0
    if n_groups == len(group) and np.array_equal(group, np.arange(len(group))):
        # Identity grouping (TF-IDF has one column per LOINC, already in canonical order). The loop below
        # would logaddexp each column onto -inf, which returns the column unchanged. This shortcut gives
        # the same result bit for bit and skips 679 Python-level passes over every chunk.
        return z - logsumexp(z, axis=1, keepdims=True)
    out = np.full((z.shape[0], n_groups), -np.inf, dtype=np.float32)
    # logaddexp-accumulate each class into its group column
    for c in np.flatnonzero(keep):
        g = group[c]
        out[:, g] = np.logaddexp(out[:, g], z[:, c])
    out -= logsumexp(out, axis=1, keepdims=True)          # renormalise over kept groups
    return out


def dropped_mass(z: np.ndarray, group: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Fraction of probability mass falling outside the kept groups, per row. Diagnostic only."""
    z = np.asarray(z, dtype=np.float32)
    if T != 1.0:
        z = z / np.float32(T)
    total = logsumexp(z, axis=1)
    kept = logsumexp(z[:, group >= 0], axis=1)
    return 1.0 - np.exp(kept - total)


def row_stats(logp: np.ndarray, true_idx: np.ndarray) -> dict:
    """Per-row scalars from a (n, k) log-probability matrix. Only these are kept after the expensive
    pass over the logits, so they carry everything the calibration and risk-coverage metrics need."""
    order = np.argpartition(-logp, 1, axis=1)[:, :2]
    l0 = logp[np.arange(len(logp)), order[:, 0]]
    l1 = logp[np.arange(len(logp)), order[:, 1]]
    hi = np.where(l0 >= l1, order[:, 0], order[:, 1])
    lo = np.where(l0 >= l1, order[:, 1], order[:, 0])
    p_top1 = np.exp(logp[np.arange(len(logp)), hi])
    p_top2 = np.exp(logp[np.arange(len(logp)), lo])
    p = np.exp(logp)
    return {
        "pred_idx": hi.astype(np.int32),
        "p_top1": p_top1.astype(np.float32),
        "margin": (p_top1 - p_top2).astype(np.float32),
        "p_true": p[np.arange(len(logp)), true_idx].astype(np.float32),
        "entropy": (-(p * np.where(p > 0, logp, 0.0)).sum(axis=1)).astype(np.float32),
        "correct": (hi == true_idx),
    }


# -----------------------------------------------------------------------------
# calibration metrics
# -----------------------------------------------------------------------------

def nll(p_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true label."""
    return float(-np.log(np.maximum(p_true, eps)).mean())


def brier_top(conf: np.ndarray, correct: np.ndarray) -> float:
    """Top-label Brier score: mean (confidence - correctness)^2."""
    return float(((conf - correct.astype(np.float64)) ** 2).mean())


def _bin_edges(conf: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "mass":
        # equal-mass bins: quantile edges. At 0.99+ accuracy nearly every row sits in the top
        # equal-width bin, which is the standard criticism of ECE. Equal-mass bins avoid that.
        q = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        q[0], q[-1] = -np.inf, np.inf
        return np.unique(q)
    raise ValueError(f"unknown binning scheme {scheme!r}")


def calibration_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15,
                     scheme: str = "mass") -> dict:
    """Per-bin count / mean confidence / mean accuracy. Basis for ECE, MCE and reliability diagrams."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct).astype(np.float64)
    edges = _bin_edges(conf, n_bins, scheme)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    nb = len(edges) - 1
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    sconf = np.bincount(idx, weights=conf, minlength=nb)
    sacc = np.bincount(idx, weights=correct, minlength=nb)
    nz = cnt > 0
    mconf, macc = np.zeros(nb), np.zeros(nb)
    mconf[nz], macc[nz] = sconf[nz] / cnt[nz], sacc[nz] / cnt[nz]
    return {"edges": edges, "count": cnt, "conf": mconf, "acc": macc, "nonempty": nz}


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, scheme: str = "mass") -> float:
    """Expected Calibration Error: count-weighted mean |confidence - accuracy| across bins."""
    b = calibration_bins(conf, correct, n_bins, scheme)
    w = b["count"] / b["count"].sum()
    return float((w * np.abs(b["conf"] - b["acc"]))[b["nonempty"]].sum())


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15, scheme: str = "mass") -> float:
    """Maximum Calibration Error: worst per-bin gap. Empty bins excluded."""
    b = calibration_bins(conf, correct, n_bins, scheme)
    gaps = np.abs(b["conf"] - b["acc"])[b["nonempty"]]
    return float(gaps.max()) if gaps.size else 0.0


# -----------------------------------------------------------------------------
# selective prediction / risk-coverage
# -----------------------------------------------------------------------------

def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Risk (error rate) as a function of coverage, accepting most-confident-first.

    Returns coverage/risk curves plus AURC and E-AURC. E-AURC subtracts the AURC of the oracle
    ranking (all correct predictions first). This is the irreducible part, driven by the base error
    rate. Without it, AURC mostly measures accuracy and says little about ranking quality.
    """
    correct = np.asarray(correct).astype(np.float64)
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    err = 1.0 - correct[order]
    n = len(err)
    k = np.arange(1, n + 1)
    risk = np.cumsum(err) / k
    aurc = float(risk.mean())
    opt = np.cumsum(np.sort(err, kind="stable")) / k     # oracle: every correct one first
    return {"coverage": k / n, "risk": risk, "aurc": aurc,
            "aurc_optimal": float(opt.mean()), "e_aurc": aurc - float(opt.mean())}


def risk_at_coverage(rc: dict, coverages=(1.0, 0.99, 0.95, 0.9, 0.8, 0.5)) -> dict:
    """Error rate when only the most-confident `c` fraction is auto-accepted."""
    n = len(rc["risk"])
    return {c: float(rc["risk"][min(max(int(round(c * n)), 1), n) - 1]) for c in coverages}


def coverage_at_risk(rc: dict, target_risk: float) -> float:
    """Largest coverage whose risk stays at or below `target_risk`, or 0.0 if no coverage achieves it.
    It answers the deployment question of what fraction can be auto-mapped at an acceptable error rate."""
    ok = np.flatnonzero(rc["risk"] <= target_risk)
    return float(rc["coverage"][ok[-1]]) if ok.size else 0.0


# -----------------------------------------------------------------------------
# temperature scaling
# -----------------------------------------------------------------------------

def temperature_nll(z: np.ndarray, true_idx: np.ndarray, T: float,
                    group: np.ndarray | None = None, n_groups: int | None = None) -> float:
    """NLL of the true label at temperature T, in the same (marginalised) space as the metrics."""
    if group is None:
        zz = np.asarray(z, dtype=np.float32) / np.float32(T)
        logp = zz - logsumexp(zz, axis=1, keepdims=True)
    else:
        logp = marginal_logprob(z, group, n_groups, T=T)
    return float(-logp[np.arange(len(logp)), true_idx].mean())


def minimise_scalar(f, lo: float = 0.05, hi: float = 20.0, coarse: int = 40,
                    refine_iters: int = 40) -> dict:
    """Coarse log-grid then golden-section refinement of a 1-D objective.

    NLL(T) is well-behaved in practice but not provably convex here, because marginalisation makes it a
    sum of softmaxes. The coarse sweep brackets the minimum, so the search does not assume that a local
    minimum is the global one. The sweep is also returned for inspection.
    """
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), coarse))
    vals = np.array([f(float(t)) for t in grid])
    j = int(vals.argmin())
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(float(c)), f(float(d))
    for _ in range(refine_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a); fc = f(float(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a); fd = f(float(d))
        if abs(b - a) < 1e-6:
            break
    T = float((a + b) / 2.0)
    return {"T": T, "nll_at_T": f(T), "nll_at_1": f(1.0), "grid": grid, "grid_nll": vals}


def fast_temperature_objective(z_kept: np.ndarray, true_cols: np.ndarray):
    """NLL(T) for the marginalised distribution, without materialising the full 679-way matrix.

    log p_true(T) = logsumexp_{c in true LOINC}(z_c/T) - logsumexp_{c kept}(z_c/T)

    Only the classes of the true LOINC (at most 4) are needed for the numerator, so both terms are
    single vectorised logsumexps. Rebuilding the whole marginal for each candidate T would cost a Python
    loop over ~747 classes per evaluation, ~80 times per run.

    z_kept   : (n, n_kept) centred logits, kept classes only
    true_cols: (n, m) column indices into z_kept for the true LOINC's classes, -1 padded
    """
    z_kept = np.asarray(z_kept, dtype=np.float32)
    pad = true_cols < 0
    safe = np.where(pad, 0, true_cols)
    gathered = np.take_along_axis(z_kept, safe, axis=1)
    gathered = np.where(pad, -np.inf, gathered)

    def f(T: float) -> float:
        num = logsumexp(gathered / np.float32(T), axis=1)
        den = logsumexp(z_kept / np.float32(T), axis=1)
        return float(-(num - den).mean())

    return f


def fit_temperature(z: np.ndarray, true_idx: np.ndarray, group: np.ndarray | None = None,
                    n_groups: int | None = None, lo: float = 0.05, hi: float = 20.0,
                    coarse: int = 40, refine_iters: int = 40) -> dict:
    """Fit a single scalar T by minimising NLL on a held-out split.

    Uses a coarse log-grid then golden-section refinement. A bare optimiser would assume convexity in T.
    NLL(T) is well-behaved in practice, but nothing guarantees that a local minimum is the global one.
    The coarse sweep is also returned so the curve can be inspected.
    """
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), coarse))
    vals = np.array([temperature_nll(z, true_idx, float(t), group, n_groups) for t in grid])
    j = int(vals.argmin())
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc = temperature_nll(z, true_idx, float(c), group, n_groups)
    fd = temperature_nll(z, true_idx, float(d), group, n_groups)
    for _ in range(refine_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = temperature_nll(z, true_idx, float(c), group, n_groups)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = temperature_nll(z, true_idx, float(d), group, n_groups)
        if abs(b - a) < 1e-6:
            break
    T = float((a + b) / 2.0)
    return {"T": T, "nll_at_T": temperature_nll(z, true_idx, T, group, n_groups),
            "nll_at_1": temperature_nll(z, true_idx, 1.0, group, n_groups),
            "grid": grid, "grid_nll": vals}


def bootstrap_clustered(patient: np.ndarray, stat, n_boot: int = 200, seed: int = 0,
                        alpha: float = 0.05, **arrays) -> dict:
    """Patient-clustered bootstrap CI for any statistic (ECE, Brier, AURC, ...).

    `bootstrap_ci` in dumps.py handles accuracy only. It uses the fact that a mean can be recovered from
    per-patient sums. ECE cannot be recovered this way, because it needs the raw values re-binned on every
    resample. So the rows themselves have to be gathered.

    Patients are resampled with replacement, and each patient's observations stay together.
    Observations within a patient share a history and a description assignment, so an observation-level
    bootstrap understates the interval.

    stat: callable(**arrays) -> float, receiving the resampled arrays under their original names.
    """
    codes, _ = pd.factorize(patient)
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    arrays = {k: np.asarray(v)[order] for k, v in arrays.items()}
    counts = np.bincount(codes_sorted)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_pat = len(counts)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_pat, size=n_pat)
        ln, st = counts[pick], starts[pick]
        total = int(ln.sum())
        if total == 0:
            continue
        # vectorised ragged gather: expand each chosen patient's contiguous row block
        idx = (np.arange(total) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(st, ln))
        out.append(stat(**{k: v[idx] for k, v in arrays.items()}))
    out = np.asarray(out, dtype=float)
    return {"point": float(stat(**arrays)),
            "lo": float(np.percentile(out, 100 * alpha / 2)),
            "hi": float(np.percentile(out, 100 * (1 - alpha / 2))),
            "n_boot": len(out)}
