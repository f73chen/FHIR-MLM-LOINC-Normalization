#!/usr/bin/env python
"""Calibration and selective prediction at the observation unit (RQ4, thesis Chapter 6), plus the
per-cell ECE of the RQ3 tables. Built on the reduction in a2_calibration (`a2.reduce_cell`) and the
metrics in lib/calib.py.

  --stage gates        label space, logit integrity, val ∩ test = ∅, marginalisation, reduced vs recorded
  --stage reduce       one streaming pass per dump -> per-row scalars (the expensive step)
  --stage temperature  one T per (model, arm, seed), fitted on that arm's validation split
  --stage apply-t      re-reduce cell D (tag __Tfit) and RELMA D (tag __Tind) at the pert arm's T
  --stage metrics      T-21, T-22, ECE CIs, F-05/F-06 data, RQ3's per-cell ECE
  --stage relma        does the in-domain temperature transfer to RELMA names?

Which rows:
  - The headline cell is D (train pert, test pert, para). Cross-model tables use the observations every
    model scored at D, from the seed-0 reductions (the scored set does not move with the seed).
  - T-21 and T-22 give means over seeds on the matched set. The ECE CIs and the printed headline use the
    first seed.
  - RQ3's per-cell ECE is per model (each model on its own rows) and goes to `rq3/`.
  - Normalised AURC is not reported.

Reads the evaluation dumps in WORK_DIR/eval. The reductions carry patient_uuid and stay in
WORK_DIR/analysis/a2_reductions. The outputs (results/analysis/v3/calibration/ and
results/analysis/v3/rq3/) are aggregate only. CPU only.

  python analysis/v3_calibration.py --stage reduce
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import calib as C  # noqa: E402
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402

_A = importlib.util.spec_from_file_location("a2c", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)

ANALYSIS = "calibration"
KEY = ["patient_uuid", "obs_ordinal"]
LETTERS = {"A": ("clean", "clean"), "B": ("clean", "pert"), "C": ("pert", "clean"), "D": ("pert", "pert")}
N_CURVE = 400                      # points kept per risk-coverage curve for F-06
N_BOOT_ECE = 200                   # replicates of the patient-clustered ECE bootstrap


# -- which cells -------------------------------------------------------------
def _cells(model, condition, letter, evalset="mimictest", scope="history", split="test"):
    tr, te = LETTERS[letter]
    return R.cells(model, condition=condition, train_arm=tr, test_arm=te, evalset=evalset,
                   scope=scope, split=split)


def reduce_list() -> list:
    """Every cell this analysis reduces, most-needed first, so a partial run is still useful."""
    out = []
    for m in R.CONDITIONS["para_xsite"]:                          # 1. val: the temperature fits
        for arm in ("pert", "clean"):
            out += R.cells(m, condition="para_xsite", train_arm=arm, split="val")
    for m in R.CONDITIONS["para_xsite"]:                          # 2. the headline cell D
        out += _cells(m, "para_xsite", "D")
    for m in R.CONDITIONS["nodesc_xsite"]:                        # 3. nodesc D (F-05, F-06)
        out += _cells(m, "nodesc_xsite", "D")
    for m in R.CONDITIONS["para_xsite"]:                          # 4. RQ3: the other para cells
        for L in "ABC":
            out += _cells(m, "para_xsite", L)
    for m in R.CONDITIONS["nodesc_xsite"]:
        for L in "ABC":
            out += _cells(m, "nodesc_xsite", L)
    for cond in R.CONDITIONS:                                     # 5. RQ3: single-obs
        for L in "ABCD":
            out += _cells("mlm", cond, L, evalset="singleobs_trim", scope="singleobs")
    for m in R.CONDITIONS["para_xsite"]:                          # 6. RELMA (temperature transfer; RQ3 B/D)
        for L in "DB":
            out += _cells(m, "para_xsite", L, evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY))
    return out


def reduction_path(cell, tag="") -> Path:
    return A.RED / f"{cell.name}{tag}.parquet"


# -- reductions to rows ------------------------------------------------------
def load_red(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)


def _mi(df) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df.patient_uuid.to_numpy(), df.obs_ordinal.to_numpy()])


def matched_keys(paths) -> pd.MultiIndex:
    keys = None
    for p in paths:
        k = _mi(pd.read_parquet(p, columns=KEY))
        keys = k if keys is None else keys.intersection(k)
    return keys


def restrict(df, keys) -> pd.DataFrame:
    return df[_mi(df).isin(keys)]


def calib_row(df) -> dict:
    """The calibration columns for one set of rows (no AURC)."""
    corr, conf = df.correct.to_numpy(), df.p_top1.to_numpy()
    return {"n": len(df), "acc": corr.mean(), "nll": C.nll(df.p_true.to_numpy()),
            "ece_mass15": C.ece(conf, corr, 15, "mass"), "ece_mass30": C.ece(conf, corr, 30, "mass"),
            "ece_width15": C.ece(conf, corr, 15, "width"), "mce_mass15": C.mce(conf, corr, 15, "mass"),
            "brier_top": C.brier_top(conf, corr), "mean_conf": conf.mean(),
            "gap_conf_minus_acc": conf.mean() - corr.mean(),
            "mean_outside_mass": df.dropped.to_numpy().mean() if "dropped" in df else np.nan}


def selective_row(df, budgets=R.BUDGETS_OBS) -> dict:
    rc = C.risk_coverage(df.p_top1.to_numpy(), df.correct.to_numpy())
    r = C.risk_at_coverage(rc, (1.0, 0.95, 0.90))
    return {**{f"risk@{int(k * 100)}": v for k, v in r.items()},
            **{f"cov@{b:g}": C.coverage_at_risk(rc, b) for b in budgets}}


def bins_rows(df, **ids) -> list[dict]:
    b = C.calibration_bins(df.p_top1.to_numpy(), df.correct.to_numpy(), 15, "mass")
    return [{**ids, "bin": i, "count": int(b["count"][i]), "conf": b["conf"][i], "acc": b["acc"][i],
             "gap_conf_minus_acc": b["conf"][i] - b["acc"][i]}
            for i in range(len(b["count"])) if b["nonempty"][i]]


def curve_rows(df, n_points=N_CURVE, **ids) -> list[dict]:
    """A risk-coverage curve thinned to ~2 × n_points, log-spaced from both ends of the coverage axis. Those
    ends are the low-coverage end and the last few percent, where the errors of the deep models rise and meet
    the budgets of T-22. The output is aggregate, and no row identities survive."""
    rc = C.risk_coverage(df.p_top1.to_numpy(), df.correct.to_numpy())
    n = len(rc["risk"])
    k = np.geomspace(1, n, n_points)
    idx = np.unique(np.clip(np.round(np.concatenate([k, n + 1 - k])).astype(int) - 1, 0, n - 1))
    return [{**ids, "coverage": rc["coverage"][i], "risk": rc["risk"][i]} for i in idx]


# -- stage: gates ------------------------------------------------------------
def stage_gates() -> bool:
    ok = True
    space = A.canonical_space()
    print(f"[G0] canonical label space: {len(space)} LOINCs"); ok &= len(space) == 679
    print("[G1] every v3 TF-IDF dump carries the same label_space_679")
    bad = []
    for c in [c for c in R.scan()[0] if c.v3 and c.model == "tfidf" and c.exists()]:
        if json.loads((c.path / "manifest.json").read_text()).get("label_space_679") != list(space):
            bad.append(c.name)
    print(f"  {'✓' if not bad else '✗'} {len(bad)} mismatched" + (f": {bad[:3]}" if bad else "")); ok &= not bad
    print("[G2] parquet / logits / row_max rows agree, every listed cell with logits")
    n_bad = 0
    for c in [c for c in reduce_list() if R.has(c, logits=True)]:
        d = c.path
        n = pd.read_parquet(d / "part-0000.parquet", columns=["obs_ordinal"]).shape[0]
        Lg = np.load(d / "logits_centered_f16.npy", mmap_mode="r"); Rm = np.load(d / "row_max_f32.npy", mmap_mode="r")
        if not (n == Lg.shape[0] == Rm.shape[0]):
            print(f"  ✗ {c.name}: {n} / {Lg.shape[0]} / {Rm.shape[0]}"); n_bad += 1
    print(f"  {'✓' if not n_bad else '✗'} {n_bad} inconsistent"); ok &= not n_bad
    print("[G3] logits centred (row max == 0), first 2,000 rows of each model's D seed 0")
    for m in R.present(logits=True, **R.HEADLINE):
        Lg = np.load(R.cell(m, 0, **R.HEADLINE).path / "logits_centered_f16.npy", mmap_mode="r")
        mx = np.asarray(Lg[:2000], dtype=np.float32).max(axis=1)
        good = bool(np.all(mx == 0)); ok &= good
        print(f"  {'✓' if good else '✗'} {R.LABEL[m]}")
    print("[G4] val ∩ test patients == ∅ (fitting T on val is valid)")
    for m in R.present(logits=True, **R.HEADLINE):
        v = R.cell(m, 0, condition="para_xsite", train_arm="pert", split="val")
        if not R.has(v):
            print(f"  – {R.LABEL[m]}: no val dump"); continue
        t = set(pd.read_parquet(R.cell(m, 0, **R.HEADLINE).path / "part-0000.parquet", columns=["patient_uuid"]).patient_uuid)
        vv = set(pd.read_parquet(v.path / "part-0000.parquet", columns=["patient_uuid"]).patient_uuid)
        print(f"  {'✓' if not t & vv else '✗'} {R.LABEL[m]}: test {len(t):,}, val {len(vv):,}, overlap {len(t & vv)}")
        ok &= not (t & vv)
    print("[G5] marginalisation preserves the argmax (20k sampled rows, D seed 0)")
    for m in R.present(logits=True, **R.HEADLINE):
        c = R.cell(m, 0, **R.HEADLINE)
        g = A.group_for(c.model, space)
        Lg = np.load(c.path / "logits_centered_f16.npy", mmap_mode="r")
        idx = np.sort(np.random.default_rng(0).choice(Lg.shape[0], min(20_000, Lg.shape[0]), replace=False))
        z = np.asarray(Lg[idx], dtype=np.float32)
        pm = space[C.marginal_logprob(z, g, len(space)).argmax(1)]
        pn = space[z[:, g >= 0].argmax(1)] if c.model == "tfidf" else D.class_to_loinc_array()[z.argmax(1)]
        agree = float((pm == pn).mean()); ok &= agree >= 0.999
        print(f"  {'✓' if agree >= 0.999 else '✗'} {R.LABEL[m]}: agreement {agree:.5f}")
    print("[G7] each reduction matches its dump (in-space rows in dump order, argmax on a 10,000-row block, "
          "top-1 within 5e-3 of the manifest)")
    # The last test only catches a mix-up of cells. The two casting rules (thesis Chapter 5) agree to
    # ~1e-5 in-distribution. They differ by up to ~2.4e-3 in low-confidence cells (single-obs NoDesc
    # under shift), where summing sibling codes moves more argmaxes. This is why the tolerance is 5e-3.
    pos = {l: i for i, l in enumerate(space)}
    worst, n_chk, n_ties, n_rows = (0.0, ""), 0, 0, 0
    for c in reduce_list():
        p = reduction_path(c)
        if not p.exists():
            continue
        d = c.path
        red = pd.read_parquet(p, columns=KEY + ["pred_idx", "correct"])
        dump = pd.read_parquet(d / "part-0000.parquet", columns=KEY + ["true_loinc"])
        keep = dump.true_loinc.map(pos).notna().to_numpy()
        rows = np.flatnonzero(keep)
        keys_ok = len(rows) == len(red) and bool((dump[KEY].to_numpy()[rows] == red[KEY].to_numpy()).all())
        Lg = np.load(d / "logits_centered_f16.npy", mmap_mode="r")
        a0 = int(np.random.default_rng(0).integers(0, max(1, Lg.shape[0] - 10_000)))
        blk = np.arange(a0, min(a0 + 10_000, Lg.shape[0]))
        lp = C.marginal_logprob(np.asarray(Lg[blk[0]:blk[-1] + 1], dtype=np.float32), A.group_for(c.model, space),
                                len(space))[keep[blk]]
        pm = lp.argmax(1)
        rp = red.pred_idx.to_numpy()[np.searchsorted(rows, blk[keep[blk]])]
        # an exact tie at the top may be broken either way. row_stats keeps argpartition's first pick,
        # and argmax keeps the lowest index. Any other disagreement means the file does not match these logits
        ii = np.arange(len(lp))
        tie = lp[ii, pm] == lp[ii, rp]
        argmax_ok = keys_ok and bool(((pm == rp) | tie).all())
        n_ties += int(((pm != rp) & tie).sum()); n_rows += len(lp)
        met = json.loads((d / "manifest.json").read_text())["metrics"]
        dev = abs(red.correct.mean() - met.get("loinc_top1", met.get("top1")))
        worst = max(worst, (dev, c.name)); n_chk += 1
        if not (keys_ok and argmax_ok and dev <= 5e-3):
            print(f"  ✗ {c.name}: keys {keys_ok}, argmax {argmax_ok}, |marginalised − mapping-first| {dev:.2e}")
            ok = False
    print(f"  {n_chk} reductions checked. {n_ties} of {n_rows:,} block rows are exact top ties broken the other "
          f"way. Largest rule difference {worst[0]:.2e} ({worst[1]})")
    print("\n" + ("ALL GATES PASSED" if ok else "GATE FAILURES (see ✗ above)"))
    return ok


# -- stage: reduce -----------------------------------------------------------
def stage_reduce(only=None, force=False, reverse=False):
    todo = [c for c in reduce_list() if only is None or c.model == only]
    if reverse:                      # a second worker walks the list from the other end (writes are atomic)
        todo = todo[::-1]
    print(f"{len(todo)} cells in the reduction list; reductions -> {A.RED}", flush=True)
    done = skipped = missing = 0
    t0 = time.time()
    for c in todo:
        if not R.has(c, logits=True):
            missing += 1
            continue
        if reduction_path(c).exists() and not force:
            skipped += 1
            continue
        A.reduce_cell(c, force=force)
        done += 1
        sys.stdout.flush()
    print(f"\nreduced {done}, already present {skipped}, no dump yet {missing} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


# -- stage: temperature ------------------------------------------------------
def _fit_T_dir(d: Path, model: str) -> dict:
    """The temperature fit for one validation dump. It minimises the NLL of the true LOINC over the
    marginalised 679-way distribution, by a coarse log-grid then golden section (lib/calib.minimise_scalar)."""
    space = A.canonical_space()
    g = A.group_for(model, space)
    kept = np.flatnonzero(g >= 0)
    by_g = {}
    for j, col in enumerate(kept):
        by_g.setdefault(int(g[col]), []).append(j)
    maxm = max(len(v) for v in by_g.values())
    cols_by_group = np.full((len(space), maxm), -1, dtype=np.int64)
    for gi, v in by_g.items():
        cols_by_group[gi, :len(v)] = v
    pq = pd.read_parquet(d / "part-0000.parquet", columns=["true_loinc"])
    pos = {l: i for i, l in enumerate(space)}
    ti = np.array([pos.get(t, -1) for t in pq.true_loinc.values], dtype=np.int64)
    ok = ti >= 0
    Lg = np.load(d / "logits_centered_f16.npy", mmap_mode="r")
    zk = np.asarray(Lg[:, kept], dtype=np.float32)[ok]
    fit = C.minimise_scalar(C.fast_temperature_objective(zk, cols_by_group[ti[ok]]))
    return {"T": fit["T"], "nll_val_T1": fit["nll_at_1"], "nll_val_Tfit": fit["nll_at_T"], "n_val": int(ok.sum())}


def stage_temperature(out: Path):
    rows = []
    for m in R.CONDITIONS["para_xsite"]:
        for arm in ("pert", "clean"):
            for c in R.cells(m, condition="para_xsite", train_arm=arm, split="val"):
                if not R.has(c, logits=True):
                    continue
                t0 = time.time()
                f = _fit_T_dir(c.path, c.model)
                rows.append({"model": R.LABEL[m], "model_key": m, "arm": arm, "seed": c.seed, **f,
                             "fit_sec": round(time.time() - t0, 1)})
                print(f"  {R.LABEL[m]:13s} {arm:5s} seed{c.seed}  T={f['T']:.4f}  NLL {f['nll_val_T1']:.5f}"
                      f" -> {f['nll_val_Tfit']:.5f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "temperatures.csv", index=False)
    return t


def temperatures(out: Path) -> pd.DataFrame:
    p = out / "temperatures.csv"
    if not p.exists():
        raise SystemExit("no temperatures.csv: run --stage temperature first")
    return pd.read_csv(p)


# -- stage: apply-t ----------------------------------------------------------
def stage_apply_t(out: Path, force=False):
    T = temperatures(out)
    T = T[T.arm == "pert"].set_index(["model_key", "seed"])["T"]
    for m in R.CONDITIONS["para_xsite"]:
        for s in R.SEEDS[m]:
            if (m, s) not in T.index:
                continue
            c = R.cell(m, s, **R.HEADLINE)
            if R.has(c, logits=True):
                A.reduce_cell(c, T=float(T.loc[(m, s)]), force=force, tag_suffix="__Tfit")
            rl = R.cell(m, s, condition="para_xsite", train_arm="pert",
                        evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY))
            if R.has(rl, logits=True):
                A.reduce_cell(rl, T=float(T.loc[(m, s)]), force=force, tag_suffix="__Tind")
            sys.stdout.flush()


# -- stage: metrics ----------------------------------------------------------
def _cell_D(m, s, condition="para_xsite"):
    return R.cell(m, s, condition=condition, train_arm="pert", test_arm="pert")


def stage_metrics(out: Path, rq3_out: Path) -> list[str]:
    present = {cond: [m for m in R.CONDITIONS[cond] if reduction_path(_cell_D(m, 0, cond)).exists()]
               for cond in R.CONDITIONS}
    cal, sel, bins, curves, ece_ci = [], [], [], [], []
    for cond, models in present.items():
        keys = matched_keys([reduction_path(_cell_D(m, 0, cond)) for m in models])
        print(f"  matched set [{cond}, cell D]: {len(keys):,} observations over "
              f"{[R.LABEL[m] for m in models]}", flush=True)
        for m in models:
            for s in R.SEEDS[m]:
                for tag, scaling in (("", "T=1"), ("__Tfit", "T=fitted")):
                    p = reduction_path(_cell_D(m, s, cond), tag)
                    if not p.exists():
                        continue
                    df = restrict(load_red(p), keys)
                    ids = {"model": R.LABEL[m], "condition": cond, "seed": s, "scaling": scaling}
                    cal.append({**ids, **calib_row(df)})
                    if scaling == "T=1":
                        sel.append({**ids, **selective_row(df)})
                        curves += curve_rows(df, **ids)      # every seed: F-06 draws the median
                        bins += bins_rows(df, **ids)         # every seed: F-05 draws the mean
                    if s == R.SEEDS[m][0]:
                        r = C.bootstrap_clustered(df.patient_uuid.to_numpy(),
                                                  lambda conf, corr: C.ece(conf, corr, 15, "mass"),
                                                  n_boot=N_BOOT_ECE, seed=0, conf=df.p_top1.to_numpy(),
                                                  corr=df.correct.to_numpy())
                        ece_ci.append({**ids, "n": len(df), "ece": r["point"], "ece_lo": r["lo"],
                                       "ece_hi": r["hi"], "n_boot": N_BOOT_ECE})
    curves += relma_curve_rows()
    pd.DataFrame(cal).to_csv(out / "calibration_matched.csv", index=False)
    pd.DataFrame(sel).to_csv(out / "selective_matched.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "reliability_bins.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "risk_coverage_curves.csv", index=False)
    pd.DataFrame(ece_ci).to_csv(out / "ece_ci.csv", index=False)
    # RQ3: per-cell ECE, each model on its own rows
    rq = []
    for cond in R.CONDITIONS:
        for scope, ev, models in (("history", "mimictest", R.CONDITIONS[cond]),
                                  ("singleobs", "singleobs_trim", R.SINGLEOBS_MODELS)):
            for m in models:
                for L in "ABCD":
                    for c in _cells(m, cond, L, evalset=ev, scope=scope):
                        p = reduction_path(c)
                        if p.exists():
                            df = load_red(p)
                            rq.append({"model": R.LABEL[m], "condition": cond, "scope": scope, "cell": L,
                                       "seed": c.seed, "n": len(df), "acc": df.correct.mean(),
                                       "ece_mass15": C.ece(df.p_top1.to_numpy(), df.correct.to_numpy(), 15, "mass"),
                                       "gap_conf_minus_acc": df.p_top1.mean() - df.correct.mean()})
    rq = pd.DataFrame(rq)
    rq.to_csv(rq3_out / "rq3_ece_by_seed.csv", index=False)
    _print_headline(pd.DataFrame(cal), pd.DataFrame(sel))
    return sorted({m for ms in present.values() for m in ms})


def relma_curve_rows() -> list[dict]:
    """Data for the RELMA panel of F-06: risk-coverage curves on RELMA names (draw a0, the para-trained
    models at cell D, T = 1) for every seed. They use the observations that every model scored with a RELMA
    name in the dump of its first seed. This is the rule of the paraphrase matched set, on RELMA-named rows
    only."""
    space = A.canonical_space()

    def covered(m, s):
        c = R.cell(m, s, condition="para_xsite", train_arm="pert", evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY))
        p = reduction_path(c)
        if not p.exists():
            return None
        b = load_red(p)
        b["true_loinc"] = space[b.true_idx.to_numpy()]
        return b[D.relma_covered(c, b)]

    first = {m: covered(m, R.SEEDS[m][0]) for m in R.CONDITIONS["para_xsite"]}
    first = {m: b for m, b in first.items() if b is not None}
    if not first:
        return []
    keys = None
    for b in first.values():
        keys = _mi(b) if keys is None else keys.intersection(_mi(b))
    print(f"  matched set [RELMA a0, cell D]: {len(keys):,} observations over {[R.LABEL[m] for m in first]}", flush=True)
    rows = []
    for m, b0 in first.items():
        for s in R.SEEDS[m]:
            b = b0 if s == R.SEEDS[m][0] else covered(m, s)
            if b is not None:
                rows += curve_rows(restrict(b, keys), model=R.LABEL[m], condition="relma_a0", seed=s, scaling="T=1")
    return rows


def _print_headline(cal, sel):
    pd.set_option("display.width", 200)
    s0 = cal[(cal.scaling == "T=1")]
    first = s0.groupby(["model", "condition"]).seed.transform("min") == s0.seed
    print("\n  T-21 (cell D, matched, first seed):")
    print(s0[first][["condition", "model", "n", "acc", "ece_mass15", "mce_mass15", "brier_top", "gap_conf_minus_acc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    q = sel[sel.groupby(["model", "condition"]).seed.transform("min") == sel.seed]
    print("\n  T-22 (coverage at an error budget, observation unit, first seed):")
    print(q[["condition", "model"] + [c for c in q.columns if c.startswith("cov@")]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# -- stage: relma -----------------------------------------------------
def stage_relma(out: Path) -> list[str]:
    T = temperatures(out)
    T = T[T.arm == "pert"].set_index(["model_key", "seed"])["T"]
    space = A.canonical_space()
    rows, done = [], set()
    for m in R.CONDITIONS["para_xsite"]:
        for s in R.SEEDS[m]:
            c = R.cell(m, s, condition="para_xsite", train_arm="pert",
                       evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY))
            p1, pT = reduction_path(c), reduction_path(c, "__Tind")
            if not p1.exists():
                continue
            b = load_red(p1)
            b["true_loinc"] = space[b.true_idx.to_numpy()]
            cov = D.relma_covered(c, b)                    # RELMA-named rows only
            b = b[cov]
            rec = {"model": R.LABEL[m], "seed": s, "n": len(b), "T_in_domain": float(T.get((m, s), np.nan)),
                   "acc_T1": b.correct.mean(), "ece_T1": C.ece(b.p_top1.to_numpy(), b.correct.to_numpy()),
                   "nll_T1": C.nll(b.p_true.to_numpy()), "gap_T1": b.p_top1.mean() - b.correct.mean()}
            if pT.exists():
                sc = load_red(pT)
                sc["true_loinc"] = space[sc.true_idx.to_numpy()]
                sc = sc[D.relma_covered(c, sc)]
                rec.update({"acc_Tind": sc.correct.mean(),
                            "ece_Tind": C.ece(sc.p_top1.to_numpy(), sc.correct.to_numpy()),
                            "nll_Tind": C.nll(sc.p_true.to_numpy()), "gap_Tind": sc.p_top1.mean() - sc.correct.mean()})
            rows.append(rec); done.add(m)
    t = pd.DataFrame(rows)
    t.to_csv(out / "relma_transfer.csv", index=False)
    if not t.empty:
        agg = t.groupby("model").agg(n_seeds=("seed", "nunique"), ece_T1=("ece_T1", "mean"),
                                     ece_Tind=("ece_Tind", "mean"), nll_T1=("nll_T1", "mean"),
                                     nll_Tind=("nll_Tind", "mean")).reset_index()
        print("\n  RELMA a0, in-domain T applied, mean over seeds:")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return sorted(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gates", "reduce", "temperature", "apply-t", "metrics", "relma"])
    ap.add_argument("--only", default=None, choices=list(R.MODELS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="reduce: walk the list from the end")
    a = ap.parse_args()
    if a.stage == "reduce":
        return stage_reduce(a.only, a.force, a.reverse)
    if a.stage == "gates":
        sys.exit(0 if stage_gates() else 1)
    out = R.out_dir(ANALYSIS)
    if a.stage == "temperature":
        t = stage_temperature(out)
        R.stamp(out, ANALYSIS, sorted(t.model_key.unique()) if len(t) else [], stage="temperature")
    elif a.stage == "apply-t":
        stage_apply_t(out, a.force)
    elif a.stage == "metrics":
        rq3 = R.out_dir("rq3", cross_model=False)
        models = stage_metrics(out, rq3)
        R.stamp(out, ANALYSIS, models, stage="metrics", note="cross-model matched set at cell D")
        R.stamp(rq3, "rq3", models, stage="ece", note="per-cell ECE, each model on its own rows")
    elif a.stage == "relma":
        R.stamp(out, ANALYSIS, stage_relma(out), stage="relma")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
