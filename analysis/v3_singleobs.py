#!/usr/bin/env python
"""What patient history is worth (RQ1) at the observation unit, with the model, weights, tokenizer and
observations held fixed. FHIR-MLM with history is compared with the single-observation-trained FHIR-MLM
(thesis Chapters 6 and 7).

  --stage obs     T-32 (history vs single-obs-trained FHIR-MLM, cells A and D), T-35's observation
                  rows (para vs nodesc), and single-obs FHIR-MLM vs TF-IDF (Chapter 7)
  --stage bands   T-34: history's contribution by training frequency (T-04's three groups), with
                  artefact checks: the per-seed pattern, Wilson intervals, how concentrated the
                  rare-band gain is, and a re-run on observations whose code occurs once in the
                  patient's scored record
  --stage relma   T-35: the history-vs-single-obs gap on RELMA names (draw a0), beside the paraphrase
                  gap
  --stage all     obs, then bands

Design:
  - The single-obs dataset (pipeline/25_singleobs_train_dataset.py) covers the observations of the
    history arm exactly, so every comparison here is paired row for row. The code asserts this.
  - Five seeds per arm. Intervals are joint patient × seed bootstraps. Each replicate shares one patient
    draw between both arms and resamples each arm's seeds independently, because they are independent
    runs. Per-seed paired differences and their SD are reported beside them.
  - The comparison is with the single-observation-trained model only. The history model evaluated on
    single observations is not part of it.
  - FHIR-MLM's positions are a subset of TF-IDF's (which scored every test observation), so these
    comparisons need no other model.

Reads the evaluation dumps in WORK_DIR/eval and results/analysis/a1/train_loinc_counts.tsv. Writes
results/analysis/v3/singleobs/ and results/analysis/v3/singleobs_relma/. Aggregate output only.

  python analysis/v3_singleobs.py --stage all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402

ANALYSIS = "singleobs"
KEY = ["patient_uuid", "obs_ordinal"]
LETTERS = {"A": ("clean", "clean"), "D": ("pert", "pert")}
GROUPS = (("rare", 0, 1_000), ("common", 1_000, 100_000), ("frequent", 100_000, 10**9))
N_BOOT, BOOT_SEED = 2000, 0
COUNTS = HERE.parent / "results" / "analysis" / "a1" / "train_loinc_counts.tsv"


def hist(cond, letter, s):
    tr, te = LETTERS[letter]
    return R.cell("mlm", s, condition=cond, train_arm=tr, test_arm=te)


def so(cond, letter, s):
    tr, te = LETTERS[letter]
    return R.cell("mlm", s, condition=cond, train_arm=tr, test_arm=te, evalset="singleobs_trim",
                  scope="singleobs")


def tfidf(cond, letter, s):
    tr, te = LETTERS[letter]
    return R.cell("tfidf", s, condition=cond, train_arm=tr, test_arm=te)


def load(c, keys=None) -> pd.DataFrame:
    f = D.top1_frame(c)[KEY + ["true_loinc", "correct"]]
    if keys is not None:
        f = f[pd.MultiIndex.from_frame(f[KEY]).isin(keys)]
    return f.sort_values(KEY).reset_index(drop=True)


def per_patient(f, pats) -> tuple[np.ndarray, np.ndarray]:
    g = f.groupby("patient_uuid").correct.agg(["sum", "count"]).reindex(pats)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def joint_ci(arm_a: list, arm_b: list | None = None, n_boot=N_BOOT, seed=BOOT_SEED):
    """Joint patient × seed bootstrap. arm_* = list of per-seed frames on identical rows. Returns the
    CI on arm_a's seed-mean accuracy, or on (arm_a − arm_b) when arm_b is given."""
    pats = np.sort(arm_a[0].patient_uuid.unique())
    Sa = np.stack([per_patient(f, pats)[0] for f in arm_a], axis=1)
    N = per_patient(arm_a[0], pats)[1]
    Sb = np.stack([per_patient(f, pats)[0] for f in arm_b], axis=1) if arm_b else None
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(pats), len(pats))
        n = N[idx].sum()
        va = (Sa[idx].sum(axis=0) / n)[rng.integers(0, Sa.shape[1], Sa.shape[1])].mean()
        if Sb is None:
            out[b] = va
        else:
            out[b] = va - (Sb[idx].sum(axis=0) / n)[rng.integers(0, Sb.shape[1], Sb.shape[1])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def compare(name, a_cells, b_cells, keys=None, **ids) -> dict:
    fa = [load(c, keys) for c in a_cells]
    fb = [load(c, keys) for c in b_cells]
    for f in fa[1:] + fb:
        if not (f[KEY].to_numpy() == fa[0][KEY].to_numpy()).all():
            raise AssertionError(f"{name}: the arms do not cover identical rows")
    acc_a = np.array([f.correct.mean() for f in fa]); acc_b = np.array([f.correct.mean() for f in fb])
    lo_a, hi_a = joint_ci(fa); lo_b, hi_b = joint_ci(fb); lo_d, hi_d = joint_ci(fa, fb)
    k = min(len(fa), len(fb))
    paired = acc_a[:k] - acc_b[:k]                          # seed-matched differences, for the SD
    return {"comparison": name, **ids, "n": len(fa[0]), "seeds_a": len(fa), "seeds_b": len(fb),
            "a_mean": acc_a.mean(), "a_lo": lo_a, "a_hi": hi_a, "b_mean": acc_b.mean(), "b_lo": lo_b,
            "b_hi": hi_b, "a_minus_b": acc_a.mean() - acc_b.mean(), "diff_lo": lo_d, "diff_hi": hi_d,
            "per_seed_diff_sd": paired.std(ddof=1) if k > 1 else np.nan,
            "per_seed_diff_min": paired.min(), "per_seed_diff_max": paired.max()}


# -- stage: obs --------------------------------------------------------------
def stage_obs(out: Path):
    rows = []
    S = R.SEEDS["mlm"]
    for cond in R.CONDITIONS:
        for L in ("A", "D"):
            rows.append(compare("history_vs_singleobs", [hist(cond, L, s) for s in S], [so(cond, L, s) for s in S],
                                condition=cond, cell=L, arm_a="history", arm_b="single obs"))
            print(f"  ✓ history vs single obs  {cond} {L}: {100 * rows[-1]['a_minus_b']:+.3f} pp "
                  f"[{100 * rows[-1]['diff_lo']:+.3f}, {100 * rows[-1]['diff_hi']:+.3f}]", flush=True)
        # single-obs FHIR-MLM vs TF-IDF, on the rows both scored. FHIR-MLM also scores a few positions
        # whose gold LOINC lies outside the 679 space, which TF-IDF never sees.
        keys = pd.MultiIndex.from_frame(load(so(cond, "D", 0))[KEY]).intersection(
            pd.MultiIndex.from_frame(load(tfidf(cond, "D", 0))[KEY]))
        rows.append(compare("singleobs_vs_tfidf", [so(cond, "D", s) for s in S],
                            [tfidf(cond, "D", s) for s in R.SEEDS["tfidf"]], keys=keys,
                            condition=cond, cell="D", arm_a="FHIR-MLM single obs", arm_b="TF-IDF"))
        print(f"  ✓ single-obs FHIR-MLM vs TF-IDF {cond} D: {100 * rows[-1]['a_minus_b']:+.3f} pp", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "singleobs_obs.csv", index=False)
    return t


# -- stage: relma (T-35) -----------------------------------------------------
def hist_relma(s, a=None):
    return R.cell("mlm", s, condition="para_xsite", train_arm="pert", test_arm="pert",
                  evalset=R.EVALSET_RELMA.format(R.RELMA_PRIMARY if a is None else a))


def so_relma(s, a=None):
    ev = R.EVALSET_RELMA.format(R.RELMA_PRIMARY if a is None else a)
    return R.cell("mlm", s, condition="para_xsite", train_arm="pert", test_arm="pert",
                  evalset=f"singleobs_trim_{ev}", scope="singleobs")


def joint_ci_did(hr, sr, hp, sp, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI on (history − single obs) on RELMA names minus the same on paraphrases, by a joint patient ×
    seed bootstrap. The two name conditions have the same patients, and each checkpoint is evaluated
    under both. So one patient draw serves both conditions. The seed draw of the history model is also
    shared across the conditions, and so is the seed draw of the single-obs model."""
    pats = np.sort(hp[0].patient_uuid.unique())
    sums = lambda fr: np.nan_to_num(np.stack([per_patient(f, pats)[0] for f in fr], axis=1))
    Hr, Sr, Hp, Sp = sums(hr), sums(sr), sums(hp), sums(sp)
    Nr, Np = (np.nan_to_num(per_patient(f[0], pats)[1]) for f in (hr, hp))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(pats), len(pats))
        jh, js = rng.integers(0, Hr.shape[1], Hr.shape[1]), rng.integers(0, Sr.shape[1], Sr.shape[1])
        gap = lambda H, Sx, N: (H[i].sum(0) / N[i].sum())[jh].mean() - (Sx[i].sum(0) / N[i].sum())[js].mean()
        out[b] = gap(Hr, Sr, Nr) - gap(Hp, Sp, Np)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def stage_relma() -> None:
    """T-35: what removing patient history costs on real institutional names (RELMA a0, cell D,
    RELMA-named rows only), beside the paraphrase cost, at the observation unit. The per-code columns
    come from the so_* families in v3_catalogue."""
    S = R.SEEDS["mlm"]
    missing = [so_relma(s).name for s in S if not R.has(so_relma(s))]
    if missing:
        raise FileNotFoundError(f"single-obs RELMA dumps missing: {missing}")
    out = R.out_dir("singleobs_relma", cross_model=False)
    h0 = load(hist_relma(S[0]))
    keys = pd.MultiIndex.from_frame(h0[D.relma_covered(hist_relma(S[0]), h0)][KEY])
    hr = [load(hist_relma(s), keys) for s in S]
    hp = [load(hist("para_xsite", "D", s)) for s in S]
    sr = [load(so_relma(s), keys) for s in S]
    sp = [load(so("para_xsite", "D", s)) for s in S]
    for f in sr:
        if not (f[KEY].to_numpy() == hr[0][KEY].to_numpy()).all():
            raise AssertionError("G3's dumps do not cover the history RELMA rows exactly")
    rows = []
    for name, H, Sx in (("relma_s1_a0", hr, sr), ("para", hp, sp)):
        ah, as_ = np.array([f.correct.mean() for f in H]), np.array([f.correct.mean() for f in Sx])
        lo, hi = joint_ci(H, Sx)
        rows.append({"names": name, "n": len(H[0]), "history": ah.mean(), "single_obs": as_.mean(),
                     "gap": ah.mean() - as_.mean(), "gap_lo": lo, "gap_hi": hi,
                     **{f"gap_seed{s}": g for s, g in zip(S, ah - as_)}})
    g = {r["names"]: r for r in rows}
    lo, hi = joint_ci_did(hr, sr, hp, sp)
    per_seed = [g["relma_s1_a0"][f"gap_seed{s}"] - g["para"][f"gap_seed{s}"] for s in S]
    rows.append({"names": "relma − para", "n": np.nan, "gap": g["relma_s1_a0"]["gap"] - g["para"]["gap"],
                 "gap_lo": lo, "gap_hi": hi, "growth": g["relma_s1_a0"]["gap"] / g["para"]["gap"] - 1,
                 "seeds_relma_gap_larger": int(sum(x > 0 for x in per_seed)),
                 **{f"gap_seed{s}": x for s, x in zip(S, per_seed)}})
    t = pd.DataFrame(rows)
    t.to_csv(out / "singleobs_relma.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    R.stamp(out, "singleobs_relma", ["mlm"], stage="relma", note="T-35: single-observation RELMA a0 vs the history model")
    print(f"\nwrote -> {out}")


# -- stage: bands ------------------------------------------------------------
def stage_bands(out: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("a1", HERE / "a1_synthesis.py")
    A1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A1)
    cnt = pd.read_csv(COUNTS, sep="\t", header=None, names=["n", "loinc"]); tc = dict(zip(cnt.loinc, cnt.n))
    rows, conc, single = [], [], []
    cond, L = "para_xsite", "D"
    for s in R.SEEDS["mlm"]:
        h, o = load(hist(cond, L, s)), load(so(cond, L, s))
        assert (h[KEY].to_numpy() == o[KEY].to_numpy()).all()
        n_train = h.true_loinc.map(tc).fillna(0).to_numpy()
        # occurrences of (patient, gold LOINC) in the scored record: the single-occurrence check
        occ = h.groupby(["patient_uuid", "true_loinc"]).obs_ordinal.transform("size").to_numpy()
        for g, lo_, hi_ in GROUPS:
            m = (n_train >= lo_) & (n_train < hi_)
            hc, oc = h.correct.to_numpy()[m], o.correct.to_numpy()[m]
            wl_h, wh_h = A1.wilson(int(hc.sum()), len(hc)); wl_o, wh_o = A1.wilson(int(oc.sum()), len(oc))
            rows.append({"seed": s, "band": g, "n_obs": int(m.sum()), "volume": m.mean(),
                         "n_codes": int(h.true_loinc[m].nunique()), "history": hc.mean(), "single_obs": oc.mean(),
                         "drop": hc.mean() - oc.mean(), "history_lo": wl_h, "history_hi": wh_h,
                         "single_lo": wl_o, "single_hi": wh_o})
            m1 = m & (occ == 1)
            if m1.any():
                single.append({"seed": s, "band": g, "n_obs_once": int(m1.sum()),
                               "history": h.correct.to_numpy()[m1].mean(),
                               "single_obs": o.correct.to_numpy()[m1].mean(),
                               "drop": h.correct.to_numpy()[m1].mean() - o.correct.to_numpy()[m1].mean()})
            if g == "rare":
                # how concentrated is history's rare-band gain? (gain per code, as observations won)
                d = pd.DataFrame({"code": h.true_loinc.to_numpy()[m],
                                  "gain": h.correct.to_numpy()[m].astype(int) - o.correct.to_numpy()[m].astype(int)})
                per = d.groupby("code").gain.sum().sort_values(ascending=False)
                tot = per.sum()
                conc.append({"seed": s, "rare_codes": len(per), "net_obs_gained": int(tot),
                             "codes_with_net_gain": int((per > 0).sum()),
                             "share_from_top5_codes": float(per.head(5).sum() / tot) if tot > 0 else np.nan,
                             "share_from_top10_codes": float(per.head(10).sum() / tot) if tot > 0 else np.nan})
    t, c, s1 = pd.DataFrame(rows), pd.DataFrame(conc), pd.DataFrame(single)
    t.to_csv(out / "singleobs_bands_by_seed.csv", index=False)
    c.to_csv(out / "singleobs_rare_concentration.csv", index=False)
    s1.to_csv(out / "singleobs_bands_single_occurrence.csv", index=False)
    summ = t.groupby("band", sort=False).agg(n_obs=("n_obs", "first"), volume=("volume", "first"),
                                             history=("history", "mean"), single_obs=("single_obs", "mean"),
                                             drop_mean=("drop", "mean"), drop_sd=("drop", "std"),
                                             drop_min=("drop", "min"), drop_max=("drop", "max"))
    summ.to_csv(out / "singleobs_bands_summary.csv")
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  rare-band concentration (per seed):"); print(c.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  single-occurrence re-run (5-seed mean drop):")
    print(s1.groupby("band", sort=False).agg(n=("n_obs_once", "first"), drop=("drop", "mean")).to_string(
        float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["obs", "bands", "relma", "all"])
    a = ap.parse_args()
    if a.stage == "relma":
        stage_relma()
        return
    out = R.out_dir(ANALYSIS, cross_model=False)
    for st, fn in (("obs", stage_obs), ("bands", stage_bands)):
        if a.stage in (st, "all"):
            print("=" * 78 + f"\nsingle-obs · {st}\n" + "=" * 78, flush=True)
            fn(out)
            R.stamp(out, ANALYSIS, ["mlm", "tfidf"], stage=st,
                    note="FHIR-MLM (history and single-obs) and TF-IDF; Llama cannot move these")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
