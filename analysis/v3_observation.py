#!/usr/bin/env python
"""
The observation unit on v3: T-10's observation columns, T-11 and F-02 (accuracy by training
frequency), T-15's observation columns, and T-17 (thesis Chapter 6).

  --stage headline    T-10: observation top-1 and top-5 at cell D on the cross-model matched set,
                      patient-clustered CI, McNemar pairs (exact, and with patients as clusters) with the
                      MDE that has patients as clusters. Nodesc too
  --stage strata      T-11 (T-04's three groups) and F-02 (nine log-spaced bins), Wilson intervals
  --stage conditions  T-15 (para vs nodesc) and T-17 (RELMA vs para, each model on its own matched rows),
                      with every denominator
  --stage all         all three

Definitions:
  - Correctness is mapping-first, from the dumps (thesis Chapter 5). The argmax of the probabilities
    marginalised onto LOINC, which the calibration reductions use, differs on a small share of rows.
    v3_checks.py measures the difference.
  - Every table gives seed 0 and the mean over the model's seeds, so a caption can say which it uses.
  - Every table states its observation set and n. The sets: `para_D` (every present model at cell D),
    `nodesc_D` (TF-IDF ∩ FHIR-MLM), and per-model sets for T-17.

Writes aggregate CSVs only, to results/analysis/v3/observation/ (T-15's rows also to
results/analysis/v3/observation_t15/).

  python analysis/v3_observation.py --stage all
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import power as P  # noqa: E402
from lib import registry as R  # noqa: E402

_S = importlib.util.spec_from_file_location("a1", HERE / "a1_synthesis.py")
A1 = importlib.util.module_from_spec(_S); _S.loader.exec_module(A1)

ANALYSIS = "observation"
KEY = ["patient_uuid", "obs_ordinal"]
STRATA_EDGES = A1.STRATA_EDGES                                  # nine log-spaced bins (F-02)
GROUPS = (("rare", 0, 1_000), ("common", 1_000, 100_000), ("frequent", 100_000, 10**9))   # T-04's three groups
N_BOOT = 300                                                    # patient-clustered bootstrap resamples
A1_OUT = HERE.parent / "results" / "analysis" / "a1"                # a1_train_counts.py writes the counts here


def train_count_map() -> dict:
    d = pd.read_csv(A1_OUT / "train_loinc_counts.tsv", sep="\t", header=None, names=["n", "loinc"])
    assert int(d.n.sum()) == 38_260_086, "training counts no longer total the v3 training set"
    return dict(zip(d.loinc, d.n))


def cellD(m, s, cond="para_xsite"):
    return R.cell(m, s, condition=cond, train_arm="pert", test_arm="pert")


def relma(m, s, letter="D"):
    tr = "pert" if letter == "D" else "clean"
    return R.cell(m, s, condition="para_xsite", train_arm=tr, test_arm="pert",
                  evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY))


def frame(c, top5=False, covered=False) -> pd.DataFrame:
    """Mapping-first correctness from the dump: keys, gold LOINC, correct (and top5)."""
    f = D.top1_frame(c)[KEY + ["true_loinc", "correct"]]
    if top5:
        f["top5"] = D.topk_hits(c, 5)
    if covered:
        f = f[D.relma_covered(c, f)]
    return f


def mi(f) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([f.patient_uuid.to_numpy(), f.obs_ordinal.to_numpy()])


def on(f, keys) -> pd.DataFrame:
    """`f` restricted to `keys`, sorted by key so rows align across arms."""
    return f[mi(f).isin(keys)].sort_values(KEY).reset_index(drop=True)


def wilson(k, n):
    return A1.wilson(k, n)


# -- stage: headline (T-10) --------------------------------------------------
def stage_headline(out: Path) -> list[str]:
    rows, pw = [], []
    for cond in R.CONDITIONS:
        models = [m for m in R.CONDITIONS[cond] if all(R.has(cellD(m, s, cond)) for s in R.SEEDS[m])]
        f0 = {m: frame(cellD(m, R.SEEDS[m][0], cond), top5=True) for m in models}
        keys = None
        for f in f0.values():
            keys = mi(f) if keys is None else keys.intersection(mi(f))
        print(f"  [{cond} D] matched set: {len(keys):,} observations over {[R.LABEL[m] for m in models]}", flush=True)
        al0 = {m: on(f, keys) for m, f in f0.items()}
        for m in models:
            per_seed = []
            for s in R.SEEDS[m]:
                f = al0[m] if s == R.SEEDS[m][0] else on(frame(cellD(m, s, cond), top5=True), keys)
                per_seed.append((s, f.correct.mean(), f.top5.mean()))
            d0 = al0[m]
            lo, hi = D.bootstrap_ci(d0, n_boot=N_BOOT, seed=0)
            t1 = np.array([x[1] for x in per_seed]); t5 = np.array([x[2] for x in per_seed])
            rows.append({"set": f"{cond}_D", "model": R.LABEL[m], "n": len(keys), "n_seeds": len(t1),
                         "top1_seed0": t1[0], "top1_seed0_ci_lo": lo, "top1_seed0_ci_hi": hi,
                         "top1_mean": t1.mean(), "top1_sd": t1.std(ddof=1) if len(t1) > 1 else np.nan,
                         "top5_seed0": t5[0], "top5_mean": t5.mean()})
        names = list(al0)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                st = D.mcnemar(al0[a], al0[b])
                psi = (st["a_only"] + st["b_only"]) / st["n"]
                deff = P.design_effect(al0[a].patient_uuid.to_numpy(), al0[a].correct.to_numpy(),
                                       al0[b].correct.to_numpy())
                # mde_80 has patients as clusters. The MDE for independent observations is kept beside it.
                pw.append({"set": f"{cond}_D", **D.label_arms(st, "a", "b"), "model_a": R.LABEL[a],
                           "model_b": R.LABEL[b], "p_patients": D.mcnemar_patients(al0[a], al0[b])["p"],
                           "discordance": psi, "design_effect": deff, "mde_80": P.mde_mcnemar_patients(st["n"], psi, deff),
                           "mde_80_independent": P.mde_mcnemar(st["n"], psi)})
    pd.DataFrame(rows).to_csv(out / "obs_headline.csv", index=False)
    pd.DataFrame(pw).to_csv(out / "obs_pairwise.csv", index=False)
    print(pd.DataFrame(rows)[["set", "model", "n", "top1_seed0", "top1_mean", "top5_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return sorted({m for c in R.CONDITIONS for m in R.CONDITIONS[c] if R.has(cellD(m, R.SEEDS[m][0], c))})


# -- stage: strata (T-11, F-02) ----------------------------------------------
def strata_rows(f, tc: dict, **ids) -> list[dict]:
    n_train = f.true_loinc.map(tc).fillna(0).to_numpy()
    out = []
    edges = np.array(STRATA_EDGES)
    b = np.digitize(n_train, edges[1:-1], right=False)
    for k in range(len(edges) - 1):
        msk = b == k
        if msk.any():
            cv = f.correct.to_numpy()[msk]; lo, hi = wilson(int(cv.sum()), int(msk.sum()))
            out.append({**ids, "scheme": "log9", "stratum": k, "lo": int(edges[k]), "hi": int(edges[k + 1]),
                        "n_obs": int(msk.sum()), "n_codes": int(f.true_loinc[msk].nunique()),
                        "acc": float(cv.mean()), "ci_lo": lo, "ci_hi": hi})
    for name, lo_, hi_ in GROUPS:
        msk = (n_train >= lo_) & (n_train < hi_)
        if msk.any():
            cv = f.correct.to_numpy()[msk]; lo, hi = wilson(int(cv.sum()), int(msk.sum()))
            out.append({**ids, "scheme": "t04", "stratum": name, "lo": lo_, "hi": hi_, "n_obs": int(msk.sum()),
                        "n_codes": int(f.true_loinc[msk].nunique()), "acc": float(cv.mean()),
                        "ci_lo": lo, "ci_hi": hi})
    return out


def stage_strata(out: Path) -> list[str]:
    tc = train_count_map()
    models = [m for m in R.CONDITIONS["para_xsite"] if all(R.has(cellD(m, s)) for s in R.SEEDS[m])]
    f0 = {m: frame(cellD(m, R.SEEDS[m][0])) for m in models}
    keys = None
    for f in f0.values():
        keys = mi(f) if keys is None else keys.intersection(mi(f))
    rows = []
    for m in models:
        for s in R.SEEDS[m]:
            f = on(f0[m] if s == R.SEEDS[m][0] else frame(cellD(m, s)), keys)
            rows += strata_rows(f, tc, set="para_D", model=R.LABEL[m], seed=s, n_set=len(keys))
    t = pd.DataFrame(rows)
    t.to_csv(out / "obs_strata.csv", index=False)
    s0 = t[(t.scheme == "t04") & (t.seed == 0)]
    print(s0.pivot_table(index="stratum", columns="model", values="acc").to_string(float_format=lambda v: f"{v:.4f}"))
    print("  smallest log9 stratum:", int(t[(t.scheme == "log9") & (t.seed == 0)].groupby("stratum").n_obs.first().min()),
          "observations")
    return models


# -- stage: conditions (T-15, T-17) ------------------------------------
def stage_conditions(out: Path) -> list[str]:
    rows, t17, dens = [], [], []
    para = [m for m in R.CONDITIONS["para_xsite"] if R.has(cellD(m, 0))]
    nod = [m for m in R.CONDITIONS["nodesc_xsite"] if R.has(cellD(m, 0, "nodesc_xsite"))]
    rel = [m for m in para if R.has(relma(m, 0))]
    fp = {m: frame(cellD(m, 0)) for m in para}
    fn = {m: frame(cellD(m, 0, "nodesc_xsite")) for m in nod}
    fr = {m: frame(relma(m, 0), covered=True) for m in rel}
    # T-15: para vs nodesc for the two nodesc models, on the rows both scored in both conditions
    k15 = None
    for f in [fp[m] for m in nod] + [fn[m] for m in nod]:
        k15 = mi(f) if k15 is None else k15.intersection(mi(f))
    # every seed on the seed-0 set. A model's positions depend only on its test data.
    for m in nod:
        for setup, cond, f0 in (("para", "para_xsite", fp[m]), ("nodesc", "nodesc_xsite", fn[m])):
            for s in R.SEEDS[m]:
                g = on(f0 if s == 0 else frame(cellD(m, s, cond)), k15)
                if len(g) != len(k15):
                    raise AssertionError(f"T-15: {m} {setup} seed {s} lacks some of the seed-0 set's rows")
                rows.append({"table": "T-15", "set": "nodesc_models_x_both_conditions", "n": len(k15),
                             "model": R.LABEL[m], "setup": setup, "seed": s, "acc": g.correct.mean()})
    # T-17: RELMA vs para, each model on its own rows (para D ∩ RELMA D covered), seed 0 and the mean
    for m in rel:
        accs = []
        for s in R.SEEDS[m]:
            a = fp[m] if s == 0 else frame(cellD(m, s))
            b = fr[m] if s == 0 else frame(relma(m, s), covered=True)
            k = mi(a).intersection(mi(b))
            ga, gb = on(a, k), on(b, k)
            accs.append((s, len(k), ga.correct.mean(), gb.correct.mean()))
            if s == 0:
                st = D.mcnemar(gb, ga)                              # a = RELMA, b = para
                t17_0 = {"n_seed0": len(k), "mcnemar_p": st["p"],
                         "mcnemar_p_patients": D.mcnemar_patients(gb, ga)["p"], "n_relma_only": st["a_only"],
                         "n_para_only": st["b_only"]}
        A_ = np.array(accs)
        t17.append({"model": R.LABEL[m], "n_seeds": len(accs), **t17_0,
                    "para_seed0": A_[0, 2], "relma_seed0": A_[0, 3], "change_seed0": A_[0, 3] - A_[0, 2],
                    "para_mean": A_[:, 2].mean(), "relma_mean": A_[:, 3].mean(),
                    "change_mean": (A_[:, 3] - A_[:, 2]).mean(),
                    "change_sd": (A_[:, 3] - A_[:, 2]).std(ddof=1) if len(accs) > 1 else np.nan})
    for name, k in (("para_D (T-10)", None), ("T-15 set", k15)):
        if k is not None:
            dens.append({"set": name, "n": len(k)})
    pd.DataFrame(rows).to_csv(out / "obs_conditions.csv", index=False)
    # T-15 involves only FHIR-MLM and TF-IDF, so its rows also go to a per-model output
    t15 = pd.DataFrame([r for r in rows if r["table"] == "T-15"])
    o15 = R.out_dir("observation_t15", cross_model=False)
    t15.to_csv(o15 / "obs_t15.csv", index=False)
    R.stamp(o15, "observation_t15", nod, stage="conditions",
            note="T-15's rows: FHIR-MLM and TF-IDF on the rows both scored in both conditions, every seed")
    pd.DataFrame(t17).to_csv(out / "obs_relma_vs_para.csv", index=False)
    pd.DataFrame(dens).to_csv(out / "obs_denominators.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["table", "model"], columns="setup", values="acc", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print(pd.DataFrame(t17)[["model", "n_seed0", "para_mean", "relma_mean", "change_mean", "mcnemar_p"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("  denominators:", ", ".join(f"{d['set']} {d['n']:,}" for d in dens))
    return sorted(set(para) | set(nod))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["headline", "strata", "conditions", "all"])
    a = ap.parse_args()
    out = R.out_dir(ANALYSIS)
    for st, fn in (("headline", stage_headline), ("strata", stage_strata), ("conditions", stage_conditions)):
        if a.stage in (st, "all"):
            print("=" * 78 + f"\nobservation · {st}\n" + "=" * 78, flush=True)
            R.stamp(out, ANALYSIS, fn(out), stage=st, note="cross-model matched sets")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
