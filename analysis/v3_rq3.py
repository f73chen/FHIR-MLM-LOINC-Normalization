#!/usr/bin/env python
"""
RQ3 (numbers): what perturbing the values and reference ranges does, model by model (thesis
Chapter 6).

These are per-model results, each computed on the positions that model scored. Views that compare
models on one matched set are in the cross-model analyses.

  --stage obs        the 2×2 at the observation unit for para, for nodesc, for single-observation para
                     and nodesc, and for RELMA (cells B and D)
  --stage primary    the main test, D vs B at 1× severity, as observation top-1 on paired observations.
                     Gives the mean over seeds with a joint patient × seed CI (what the thesis quotes). At
                     seed 0 it also gives a patient-resampled CI, McNemar's test and its MDE with patients
                     as clusters
  --stage secondary  values vs reference ranges, severity (0.5× and 2×), the five perturbation draws
  --stage fewshot    the off-the-shelf models scored on clean vs perturbed numbers
  --stage all

The cells, as (training numbers, test numbers): A (clean, clean), B (clean, pert), C (pert, clean)
and D (pert, pert). B − A shows how brittle the shortcut is, D − B what training on perturbed numbers
recovers, C − A what that costs on clean numbers, and (D − B)/(A − B) the share of the gap closed.

Statistics:
  - A model's cells are compared on the positions it scored in every compared cell. For TF-IDF and
    FHIR-MLM that is every position. BCM and Llama render clean and perturbed numbers to slightly
    different token counts, so their cells can differ by a few positions. Both the native and the
    matched figures are written.
  - Seeds: per-seed values first, then mean and sd over seeds. `sd` is the sample sd (ddof=1). `sd_pop`
    (ddof=0) is written beside it because pipeline/07_tfidf_train_eval.py reports the population sd.
  - The obs stage checks that each cell's top-1 over all positions equals the value in its
    manifest, and for TF-IDF also the per-seed value from 07.

Reads the evaluation dumps, 07's results and b4's per-observation file. Writes aggregate CSVs to
results/analysis/v3/rq3/. CPU only:
  python analysis/v3_rq3.py --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import power as P  # noqa: E402
from lib import registry as R  # noqa: E402

ANALYSIS = "rq3"
KEY = ["patient_uuid", "obs_ordinal"]
ARMS = {"A": ("clean", "clean"), "B": ("clean", "pert"), "C": ("pert", "clean"), "D": ("pert", "pert")}
RESULTS07 = HERE.parent / "results" / "07_tfidf_train_eval"
B4 = HERE.parent / "results" / "analysis" / "b4" / "b4_per_observation.csv"
N_BOOT, BOOT_SEED = 2000, 0
TOL_MANIFEST = 1e-9
# TF-IDF's dumps come from pipeline/16_tfidf_eval_dump.py, which refits each seed independently of 07
# because 07 saves no fitted pipeline. So a dump and 07 differ by refit noise, measured at up to 1.31e-4
# (always in cell B) with a median of 3.9e-6. That noise is larger than the smallest gap between two
# cells of one seed (para C vs D, 2.86e-5), so a value tolerance cannot prove which cell a dump is.
# Identity is checked by exact equality with the 07 value that 16 itself was checked against
# (`expected_top1_from_07` in the manifest). The value check only applies the noise bound that 16 uses
# (0.001).
TOL_07 = 1e-3

FAMILIES = (
    [dict(family="para", condition="para_xsite", evalset="mimictest", scope="history",
          letters="ABCD", models=R.CONDITIONS["para_xsite"]),
     dict(family="nodesc", condition="nodesc_xsite", evalset="mimictest", scope="history",
          letters="ABCD", models=R.CONDITIONS["nodesc_xsite"]),
     dict(family="singleobs_para", condition="para_xsite", evalset="singleobs_trim",
          scope="singleobs", letters="ABCD", models=R.SINGLEOBS_MODELS),
     dict(family="singleobs_nodesc", condition="nodesc_xsite", evalset="singleobs_trim",
          scope="singleobs", letters="ABCD", models=R.SINGLEOBS_MODELS)]
    + [dict(family=f"relma_a{a}", condition="para_xsite", evalset=R.EVALSET_RELMA.format(a),
            scope="history", letters="BD", models=R.CONDITIONS["para_xsite"])
       for a in R.RELMA_DRAWS])


# -- helpers -----------------------------------------------------------------
def cell_for(model, seed, fam, letter):
    tr, te = ARMS[letter]
    return R.cell(model, seed, condition=fam["condition"], train_arm=tr, test_arm=te,
                  evalset=fam["evalset"], scope=fam["scope"])


def manifest_top1(c) -> float:
    m = json.loads((c.path / "manifest.json").read_text())["metrics"]
    return float(m["loinc_top1"] if "loinc_top1" in m else m["top1"])


_J07: dict = {}


def tfidf07(condition, arm, seed, evalset_name) -> float:
    """07's per-seed observation top-1: the arm's home cell (`evalset_name=None`) or one of its
    `eval_sets` ('pert' = B for the clean arm, 'clean' = C for the pert arm, or a variant)."""
    k = (condition, arm)
    if k not in _J07:
        _J07[k] = json.loads((RESULTS07 / f"tfidf_{condition}__{arm}.json").read_text())
    ps = [p for p in _J07[k]["per_seed"] if p["seed"] == seed]
    if len(ps) != 1:
        raise KeyError(f"07 has no single per-seed entry for {k} seed {seed}")
    p = ps[0]
    return float((p["micro"] if evalset_name is None else p["eval_sets"][evalset_name]["micro"])["top1"])


def summarise(df, by, cols):
    """mean / sample sd / population sd / min / max over seeds, long format."""
    out = []
    for keys, g in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in cols:
            v = g[c].dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            out.append({**dict(zip(by, keys)), "quantity": c, "n_seeds": len(v),
                        "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                        "sd_pop": v.std(ddof=0), "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def paired_cluster_ci(patient, a_correct, b_correct, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI on mean(a) - mean(b), resampling patients. Rows are paired (the same
    observation under both arms), and observations within a patient are not independent."""
    df = pd.DataFrame({"p": patient, "a": a_correct.astype(np.int64), "b": b_correct.astype(np.int64)})
    g = df.groupby("p", sort=False).agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    sa, sb, n = g.a.to_numpy(float), g.b.to_numpy(float), g.n.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    diff = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def joint_seed_ci(SD, SB, N, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI on the seed-mean D − B, by the joint patient × seed bootstrap of v3_singleobs.joint_ci.
    Each replicate shares one patient draw between both arms and resamples each arm's seeds independently,
    because the runs are independent. SD and SB are per-patient correct counts (patients × seeds), and N holds
    the per-patient observation counts."""
    rng = np.random.default_rng(seed)
    P, kd, kb = len(N), SD.shape[1], SB.shape[1]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, P, P)
        n = N[idx].sum()
        out[i] = ((SD[idx].sum(0) / n)[rng.integers(0, kd, kd)].mean()
                  - (SB[idx].sum(0) / n)[rng.integers(0, kb, kb)].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _keys(f):
    return pd.MultiIndex.from_arrays([f.patient_uuid.to_numpy(), f.obs_ordinal.to_numpy()])


def _matched(frames: dict):
    """(boolean masks per letter selecting the rows every frame scored, n_common). Fast path when
    every frame carries the identical key sequence, which TF-IDF and FHIR-MLM do."""
    fs = list(frames.values())
    if all(len(f) == len(fs[0]) and (f.obs_ordinal.to_numpy() == fs[0].obs_ordinal.to_numpy()).all()
           and (f.patient_uuid.to_numpy() == fs[0].patient_uuid.to_numpy()).all() for f in fs[1:]):
        return {L: np.ones(len(f), bool) for L, f in frames.items()}, len(fs[0])
    common = _keys(fs[0])
    for f in fs[1:]:
        common = common.intersection(_keys(f))
    return {L: _keys(f).isin(common) for L, f in frames.items()}, len(common)


# -- stage: obs --------------------------------------------------------------
def stage_obs(out: Path) -> list[str]:
    rows_cell, rows_seed, models_done = [], [], set()
    for fam in FAMILIES:
        for m in fam["models"]:
            need = [cell_for(m, s, fam, L) for s in R.SEEDS[m] for L in fam["letters"]]
            if not all(R.has(c) for c in need):
                print(f"  – {R.LABEL[m]} {fam['family']}: dumps not all present, skipped", flush=True)
                continue
            models_done.add(m)
            t0 = time.time()
            for s in R.SEEDS[m]:
                frames, n_all, top1_all = {}, {}, {}
                for L in fam["letters"]:
                    c = cell_for(m, s, fam, L)
                    f = D.top1_frame(c)
                    n_all[L], top1_all[L] = len(f), float(f.correct.mean())
                    man = manifest_top1(c)              # on all positions, the scope of the manifest
                    if abs(top1_all[L] - man) > TOL_MANIFEST:
                        raise AssertionError(f"manifest check: {c.name} dump top-1 {top1_all[L]:.9f} != manifest {man:.9f}")
                    if fam["evalset"].startswith("relma_"):
                        f = f[D.relma_covered(c, f)]    # only observations shown a RELMA name count
                    frames[L] = f[KEY + ["correct"]]
                masks, n_common = _matched(frames)
                acc = {}
                for L, f in frames.items():
                    c = cell_for(m, s, fam, L)
                    native = float(f.correct.mean())    # = top1_all except on RELMA (covered only)
                    man = manifest_top1(c)              # already checked against top1_all above
                    j07 = np.nan
                    if m == "tfidf" and fam["evalset"] == "mimictest":
                        tr, te = ARMS[L]
                        j07 = tfidf07(fam["condition"], tr, s, None if tr == te else te)
                        rec = json.loads((c.path / "manifest.json").read_text())["expected_top1_from_07"]
                        if j07 != rec:                    # exact: proves which cell this dump is
                            raise AssertionError(f"07 check: {c.name} was verified against 07 value {rec!r}, "
                                                 f"but this letter maps to {j07!r}")
                        if abs(top1_all[L] - j07) > TOL_07:
                            raise AssertionError(f"07 check: {c.name} dump top-1 {top1_all[L]:.9f} vs 07 {j07:.9f}"
                                                 f" exceeds the refit-noise bound {TOL_07:g}")
                    acc[L] = float(f.correct.to_numpy()[masks[L]].mean())
                    rows_cell.append({"model": R.LABEL[m], "family": fam["family"], "seed": s,
                                      "cell": L, "run": c.name, "n_all": n_all[L], "top1_all": top1_all[L],
                                      "n_native": len(f), "top1_native": native, "top1_manifest": man,
                                      "top1_07": j07, "n_matched": n_common, "top1_matched": acc[L]})
                r = {"model": R.LABEL[m], "family": fam["family"], "seed": s, "n_matched": n_common,
                     **{L: acc[L] for L in acc}}
                if fam["letters"] == "ABCD":
                    r.update({"B_minus_A": acc["B"] - acc["A"], "D_minus_B": acc["D"] - acc["B"],
                              "C_minus_A": acc["C"] - acc["A"],
                              "gap_closed": (acc["D"] - acc["B"]) / (acc["A"] - acc["B"])
                              if acc["A"] != acc["B"] else np.nan})
                else:
                    r["D_minus_B"] = acc["D"] - acc["B"]
                rows_seed.append(r)
            print(f"  ✓ {R.LABEL[m]:13s} {fam['family']:17s} {len(R.SEEDS[m])} seed(s) × "
                  f"{len(fam['letters'])} cells, checks passed ({time.time() - t0:.0f}s)", flush=True)
    cells = pd.DataFrame(rows_cell)
    seeds = pd.DataFrame(rows_seed)
    q = [c for c in ["A", "B", "C", "D", "B_minus_A", "D_minus_B", "C_minus_A", "gap_closed"]
         if c in seeds.columns]
    summ = summarise(seeds, ["model", "family"], q)
    cells.to_csv(out / "rq3_obs_cells.csv", index=False)
    seeds.to_csv(out / "rq3_obs_by_seed.csv", index=False)
    summ.to_csv(out / "rq3_obs_summary.csv", index=False)
    print(f"\n  max |dump − manifest| = {(cells.top1_all - cells.top1_manifest).abs().max():.2e}; "
          f"max |dump − 07| = {(cells.top1_all - cells.top1_07).abs().max():.2e}")
    rel = cells[cells.family.str.startswith("relma")]
    if len(rel):
        cov = rel.groupby("model").apply(lambda g: (g.n_native / g.n_all).mean())
        print("  RELMA rows carrying a real RELMA name (share of positions): "
              + ", ".join(f"{k} {100 * v:.2f}%" for k, v in cov.items()))
    native_vs_matched = (cells.n_native != cells.n_matched).sum()
    print(f"  cells whose native positions differ from the matched set: {native_vs_matched}")
    _print_summary(summ)
    return sorted(models_done)


def _print_summary(summ):
    pd.set_option("display.width", 200)
    s = summ.copy()
    s["value"] = [f"{100 * m:.3f} ± {100 * sd:.3f}" if q != "gap_closed" else f"{100 * m:.2f}% ± {100 * sd:.2f}"
                  for m, sd, q in zip(s["mean"], s["sd"].fillna(0), s["quantity"])]
    print("\n  (pp, mean ± sample sd over seeds)")
    print(s.pivot_table(index=["family", "model"], columns="quantity", values="value",
                        aggfunc="first", sort=False).to_string())


# -- stage: primary ----------------------------------------------------------
def stage_primary(out: Path) -> list[str]:
    fam = FAMILIES[0]
    seeds_csv = out / "rq3_obs_by_seed.csv"
    by_seed = pd.read_csv(seeds_csv) if seeds_csv.exists() else None
    rows, done = [], []
    for m in fam["models"]:
        s0 = R.SEEDS[m][0]
        cb, cd = cell_for(m, s0, fam, "B"), cell_for(m, s0, fam, "D")
        if not (R.has(cb) and R.has(cd)):
            continue
        done.append(m)
        fb = D.top1_frame(cb)[KEY + ["correct"]]
        fd = D.top1_frame(cd)[KEY + ["correct"]]
        b, d = D.matched(fb, fd)
        st = D.mcnemar(d, b)                                  # a = D, b = B
        psi = (st["a_only"] + st["b_only"]) / st["n"]
        lo, hi = paired_cluster_ci(d.patient_uuid.to_numpy(), d.correct.to_numpy(), b.correct.to_numpy())
        deff = P.design_effect(d.patient_uuid.to_numpy(), d.correct.to_numpy(), b.correct.to_numpy())
        seed_sd = np.nan
        if by_seed is not None:
            v = by_seed[(by_seed.model == R.LABEL[m]) & (by_seed.family == "para")].D_minus_B
            seed_sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        # Every seed (what the thesis quotes). The mean D − B is taken on the B ∩ D positions, which are the
        # same at every seed, with the joint patient × seed CI of the other per-observation tables.
        codes = pd.factorize(d.patient_uuid.to_numpy())[0]
        N = np.bincount(codes).astype(float)
        SD, SB = [], []
        for s in R.SEEDS[m]:
            bs, ds = (b, d) if s == s0 else D.matched(D.top1_frame(cell_for(m, s, fam, "B"))[KEY + ["correct"]],
                                                       D.top1_frame(cell_for(m, s, fam, "D"))[KEY + ["correct"]])
            if not (bs[KEY].to_numpy() == d[KEY].to_numpy()).all():
                raise AssertionError(f"{R.LABEL[m]} seed {s}: B ∩ D is not seed 0's positions")
            SD.append(np.bincount(codes, weights=ds.correct.to_numpy(float)))
            SB.append(np.bincount(codes, weights=bs.correct.to_numpy(float)))
        SD, SB = np.stack(SD, 1), np.stack(SB, 1)
        per_seed = (SD.sum(0) - SB.sum(0)) / N.sum()
        jlo, jhi = joint_seed_ci(SD, SB, N)
        # mde_80 is the MDE with patients as clusters. The MDE for independent observations is kept beside it.
        rows.append({"model": R.LABEL[m], "seed": s0, "n_paired": st["n"],
                     "top1_D": st["a_acc"], "top1_B": st["b_acc"], "D_minus_B": st["delta"],
                     "ci_lo": lo, "ci_hi": hi, "n_D_only": st["a_only"], "n_B_only": st["b_only"],
                     "mcnemar_p": st["p"], "mcnemar_p_patients": D.mcnemar_patients(d, b)["p"],
                     "discordance": psi, "design_effect": deff,
                     "mde_80": P.mde_mcnemar_patients(st["n"], psi, deff),
                     "mde_80_independent": P.mde_mcnemar(st["n"], psi), "D_minus_B_seed_sd": seed_sd,
                     "n_seeds": len(per_seed), "top1_D_mean": float((SD.sum(0) / N.sum()).mean()),
                     "top1_B_mean": float((SB.sum(0) / N.sum()).mean()), "D_minus_B_mean": float(per_seed.mean()),
                     "joint_ci_lo": jlo, "joint_ci_hi": jhi, "D_minus_B_seed_min": float(per_seed.min()),
                     "D_minus_B_seed_max": float(per_seed.max())})
        print(f"  {R.LABEL[m]:13s} D−B = {100 * st['delta']:+.3f} pp [{100 * lo:+.3f}, {100 * hi:+.3f}] "
              f"n={st['n']:,}  MDE {100 * rows[-1]['mde_80']:.4f} pp (design effect {deff:.2f}; "
              f"{100 * rows[-1]['mde_80_independent']:.4f} pp independent)  seed sd {100 * seed_sd:.3f} pp",
              flush=True)
        print(f"  {'':13s} mean over {len(per_seed)} seed(s): D−B = {100 * per_seed.mean():+.3f} pp "
              f"[{100 * jlo:+.3f}, {100 * jhi:+.3f}] (joint patient × seed)", flush=True)
    pd.DataFrame(rows).to_csv(out / "rq3_primary.csv", index=False)
    return done


# -- stage: secondary --------------------------------------------------------
VARIANTS = ("pert_value", "pert_range", "pert_sev0.5", "pert_sev2.0")
REPLICATES = tuple(f"pert_r{i}" for i in range(1, 6))


def _variant_top1(m, arm, seed, variant):
    """Observation top-1 of `arm`'s model on a perturbed variant of the test data. FHIR-MLM's value comes
    from its dump (cross-checked against the manifest). TF-IDF's comes from 07, which scored every variant
    itself."""
    if m == "tfidf":
        return tfidf07("para_xsite", arm, seed, variant)
    c = R.cell(m, seed, condition="para_xsite", train_arm=arm, test_arm=variant)
    v = float(D.top1_frame(c).correct.mean())
    man = manifest_top1(c)
    if abs(v - man) > TOL_MANIFEST:
        raise AssertionError(f"manifest check: {c.name} dump {v:.9f} != manifest {man:.9f}")
    return v


def _cell_top1(m, arm, seed, letter_test):
    if m == "tfidf":
        return tfidf07("para_xsite", arm, seed, None if arm == letter_test else letter_test)
    c = R.cell(m, seed, condition="para_xsite", train_arm=arm, test_arm=letter_test)
    return float(D.top1_frame(c).correct.mean())


EXTRA = ("bcm", "llama")      # their secondary blocks are used where their evaluations exist


def _block_ready(m, variants, seeds) -> bool:
    """Every (arm, seed, variant) eval of one block exists. TF-IDF scored every variant itself (07)."""
    return m == "tfidf" or all(R.has(R.cell(m, s, condition="para_xsite", train_arm=a, test_arm=v))
                               for a in ("clean", "pert") for s in seeds for v in variants)


def _row_top1(m, arm, seed, tests: dict) -> dict:
    """{label: observation top-1} for one row's cells. TF-IDF values come from 07. FHIR-MLM values come from
    the dump of each cell, because all its test sets score the same positions. BCM and Llama are scored on the
    positions that all the row's cells share. Their subword tokenizers can fit different observations into a
    window when only values or only ranges move."""
    if m in ("tfidf", "mlm"):
        return {k: (_variant_top1(m, arm, seed, t) if t.startswith("pert_") else _cell_top1(m, arm, seed, t))
                for k, t in tests.items()}
    cells = {k: R.cell(m, seed, condition="para_xsite", train_arm=arm, test_arm=t) for k, t in tests.items()}
    frames = D.matched(*[D.top1_frame(c) for c in cells.values()])
    return {k: float(f.correct.mean()) for k, f in zip(cells, frames)}


def stage_secondary(out: Path) -> list[str]:
    """Values vs ranges, severity, and the five draws. TF-IDF's and FHIR-MLM's blocks must be complete.
    BCM's and Llama's are used block by block, where their evaluations exist."""
    from scipy.stats import t as tdist
    vr, sv, rp, used = [], [], [], []
    for m in ("tfidf", "mlm") + EXTRA:
        seeds = R.SEEDS[m]
        s0 = seeds[0]                                          # the draws were run at seed 0 only
        blocks = {"value/range": _block_ready(m, ("pert_value", "pert_range"), seeds),
                  "severity": _block_ready(m, ("pert_sev0.5", "pert_sev2.0"), seeds),
                  "draws": _block_ready(m, REPLICATES, (s0,))}
        if m not in EXTRA and not all(blocks.values()):
            raise AssertionError(f"{R.LABEL[m]}: a planned RQ3 block is incomplete: {blocks}")
        if not any(blocks.values()):
            print(f"  · {R.LABEL[m]}: no block complete yet", flush=True)
            continue
        used.append(m)
        for arm in ("clean", "pert"):
            for s in seeds:
                if blocks["value/range"]:                      # home = A or D; pert = B or D; clean = A or C
                    v = _row_top1(m, arm, s, {"home": arm, "clean": "clean", "pert": "pert",
                                              "pert_value": "pert_value", "pert_range": "pert_range"})
                    vr.append({"model": R.LABEL[m], "arm": arm, "seed": s, "home": v["home"], "clean_test": v["clean"],
                               "pert_both": v["pert"], "pert_value": v["pert_value"], "pert_range": v["pert_range"],
                               "value_minus_home": v["pert_value"] - v["home"],
                               "range_minus_home": v["pert_range"] - v["home"],
                               "both_minus_home": v["pert"] - v["home"]})
                if blocks["severity"]:
                    v = _row_top1(m, arm, s, {"sev0.5": "pert_sev0.5", "sev1.0": "pert", "sev2.0": "pert_sev2.0"})
                    sv.append({"model": R.LABEL[m], "arm": arm, "seed": s, **v,
                               "band_2x_minus_half": v["sev2.0"] - v["sev0.5"]})
            if blocks["draws"]:
                v = _row_top1(m, arm, s0, {"base": "pert", **{r: r for r in REPLICATES}})
                reps = np.array([v[r] for r in REPLICATES])
                half = tdist.ppf(0.975, len(reps) - 1) * reps.std(ddof=1) / np.sqrt(len(reps))
                allsix = np.append(reps, v["base"])
                rp.append({"model": R.LABEL[m], "arm": arm, "seed": s0, "base_draw": v["base"],
                           **{r: x for r, x in zip(REPLICATES, reps)},
                           "r_mean": reps.mean(), "r_ci_lo": reps.mean() - half, "r_ci_hi": reps.mean() + half,
                           "spread_all_six": allsix.max() - allsix.min()})
        print(f"  ✓ {R.LABEL[m]}: " + ", ".join(k for k, ok in blocks.items() if ok)
              + f" ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}; draws at seed 0)", flush=True)
    vr, sv, rp = pd.DataFrame(vr), pd.DataFrame(sv), pd.DataFrame(rp)
    vr.to_csv(out / "rq3_valuerange_by_seed.csv", index=False)
    sv.to_csv(out / "rq3_severity_by_seed.csv", index=False)
    rp.to_csv(out / "rq3_replicates.csv", index=False)
    summarise(vr, ["model", "arm"], ["home", "clean_test", "pert_value", "pert_range", "pert_both", "value_minus_home",
                                     "range_minus_home", "both_minus_home"]).to_csv(
        out / "rq3_valuerange_summary.csv", index=False)
    summarise(sv, ["model", "arm"], ["sev0.5", "sev1.0", "sev2.0", "band_2x_minus_half"]).to_csv(
        out / "rq3_severity_summary.csv", index=False)
    return used


# -- stage: fewshot ----------------------------------------------------------
def stage_fewshot(out: Path) -> list[str]:
    """The off-the-shelf models' share correct with clean vs perturbed numbers, paired on the same
    observations. Reads b4's per-observation file (the v3 few-shot runs). Writes and prints aggregates
    only."""
    d = pd.read_csv(B4, usecols=["model", "patient", "gold", "outcome"])
    rows = []
    for m in sorted({x.split("__")[0] for x in d.model if x.endswith("__fs3")}):
        c = d[d.model == f"{m}__v3clean__fs3"].reset_index(drop=True)
        p = d[d.model == f"{m}__v3pert__fs3"].reset_index(drop=True)
        if not len(c) or len(c) != len(p) or not ((c.patient == p.patient) & (c.gold == p.gold)).all():
            raise AssertionError(f"{m}: the clean and pert few-shot runs are not the same observations")
        cc, pc_ = (c.outcome == "correct").to_numpy(), (p.outcome == "correct").to_numpy()
        lo, hi = paired_cluster_ci(p.patient.to_numpy(), pc_, cc)
        x, y = int((pc_ & ~cc).sum()), int((cc & ~pc_).sum())
        from scipy.stats import binomtest
        # The test is McNemar's (thesis Chapter 5). With a few dozen discordant pairs the percentile
        # bootstrap is unreliable, because its interval can exclude zero where the exact test is not
        # significant. The bootstrap interval is kept as a description only.
        # The interval that goes with the exact test is Clopper-Pearson on the discordant split
        # θ = x / (x + y), conditional on x + y. It is mapped to the difference (x − y)/n = (x + y)/n · (2θ − 1).
        if x + y:
            bt = binomtest(x, x + y, 0.5)
            p_mc, cp = bt.pvalue, bt.proportion_ci(confidence_level=0.95, method="exact")
            k = (x + y) / len(c)
            ex_lo, ex_hi = k * (2 * cp.low - 1), k * (2 * cp.high - 1)
        else:
            p_mc, ex_lo, ex_hi = 1.0, 0.0, 0.0
        rows.append({"model": m, "n": len(c), "correct_clean": cc.mean(), "correct_pert": pc_.mean(),
                     "pert_minus_clean": pc_.mean() - cc.mean(), "exact_ci_lo": ex_lo, "exact_ci_hi": ex_hi,
                     "boot_ci_lo": lo, "boot_ci_hi": hi,
                     "n_pert_only": x, "n_clean_only": y, "mcnemar_p": p_mc,
                     "n_patients": c.patient.nunique()})
    t = pd.DataFrame(rows)
    t.to_csv(out / "rq3_fewshot_control.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["obs", "primary", "secondary", "fewshot", "all"], required=True)
    a = ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    for st, fn in (("obs", stage_obs), ("primary", stage_primary), ("secondary", stage_secondary),
                   ("fewshot", stage_fewshot)):
        if a.stage in (st, "all"):
            print("=" * 78 + f"\nRQ3 · {st}\n" + "=" * 78, flush=True)
            models = set(fn(out))
            R.stamp(out, ANALYSIS, [m for m in R.MODELS if m in models],
                    stage=st, note="per-model results; each model on its own scored positions"
                    + ("; the off-the-shelf models are not in the registry" if st == "fewshot" else ""))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
