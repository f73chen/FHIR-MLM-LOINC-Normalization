#!/usr/bin/env python
"""Three small analyses for thesis Chapters 5 and 6.

  --stage casting     the two ways of casting a prediction to LOINC (Chapter 5), mapping-first top-1 (the
                      dump) vs the marginalised argmax (the a2 reduction), at cell D per model and seed.
                      Gives the share of observations whose predicted LOINC differs, and the accuracy
                      difference
  --stage did         the pretraining-overlap difference-in-differences, para, in each 2×2 cell.
                      Gives the overlap-vs-clean top-1 gaps for FHIR-MLM (its own rows) and each control
                      (on FHIR-MLM ∩ control), and the paired DiD on those shared rows with a
                      patient-clustered interval. Also gives the unpaired form of the DiD (`did_v1_form`,
                      FHIR-MLM's own-row gap minus the control's gap) and the overlap facts (test
                      patients, scored patients, scored observations). Cell A (train clean, test clean)
                      is the strictest memorisation test, since stage-1 pretraining saw the real values.
                      D is the headline cell
  --stage ambiguity   top-1 on paraphrase strings claimed by more than one LOINC vs on unique ones, every
                      model and seed in each 2×2 cell, with a patient-clustered interval on the
                      difference. All four models see the same string per observation. `desc_shown`
                      in the TF-IDF dump equals the description sidecar of the deep models, and every v3
                      dataset records the same paraphrase split

Scoring: per-observation accuracy is mapping-first, from the dumps (Chapter 5). Each output is per
model, or per FHIR-MLM–control pair.

Reads the evaluation dumps in WORK_DIR/eval, the a2 reductions, data/pretrain/pretrain_4096_patients.txt
and the paraphrase pools in data/paraphrases/clean/. Writes results/analysis/v3/checks/. Aggregate
output only.

  python analysis/v3_checks.py --stage all
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
from lib import cells as CL  # noqa: E402
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402

_C = importlib.util.spec_from_file_location("v3c", HERE / "v3_calibration.py")
V3C = importlib.util.module_from_spec(_C); _C.loader.exec_module(V3C)
A2 = V3C.A

ANALYSIS = "checks"
KEY = ["patient_uuid", "obs_ordinal"]
PRETRAIN = HERE.parent / "data" / "pretrain" / "pretrain_4096_patients.txt"
POOLS = HERE.parent / "data" / "paraphrases" / "clean"
CONTROLS = ("bcm", "tfidf", "llama")      # models that never saw the stage-1 corpus
N_BOOT = 1000
ARMS = {v: k for k, v in CL.LETTERS.items()}       # "A" -> ("clean", "clean"), ...
NOTES = {"casting": "cell D, both conditions, every seed whose reduction exists",
         "did": "para, cells A-D, mapping-first; controls on FHIR-MLM ∩ control",
         "ambiguity": "para, cells A-D, mapping-first"}


def cellX(m, s, letter, cond="para_xsite"):
    tr, te = ARMS[letter]
    return R.cell(m, s, condition=cond, train_arm=tr, test_arm=te)


def cellD(m, s, cond="para_xsite"):
    return cellX(m, s, "D", cond)


def red(c) -> pd.DataFrame:
    return pd.read_parquet(V3C.reduction_path(c), columns=KEY + ["pred_idx", "correct"])


def pretrain_patients() -> set:
    return set(PRETRAIN.read_text().split())


def clustered_ci(pat, grp, xs: list, signs: list, n_boot=N_BOOT, alpha=0.05) -> tuple:
    """Patient-clustered percentile interval on Σ sign·gap(x), gap(x) = mean(x | grp) − mean(x | not grp).
    The arrays in xs are row-aligned (one per model, same observations). Patients are resampled with
    replacement and every array's rows for a patient move together."""
    up, inv = np.unique(np.asarray(pat), return_inverse=True)
    k = len(up)
    g = np.asarray(grp, bool)
    n1 = np.bincount(inv, g.astype(float), k)
    n0 = np.bincount(inv, (~g).astype(float), k)
    sums = [(np.bincount(inv, np.asarray(x, float) * g, k), np.bincount(inv, np.asarray(x, float) * ~g, k))
            for x in xs]
    rng = np.random.default_rng(0)
    est = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        a1, a0 = w @ n1, w @ n0
        est[b] = sum(sg * ((w @ s1) / a1 - (w @ s0) / a0) for sg, (s1, s0) in zip(signs, sums))
    return float(np.percentile(est, 100 * alpha / 2)), float(np.percentile(est, 100 * (1 - alpha / 2)))


# -- casting -----------------------------------------------------------------
def stage_casting(out: Path) -> list[str]:
    space = A2.canonical_space()
    rows = []
    for cond in R.CONDITIONS:
        for m in R.CONDITIONS[cond]:
            for s in R.SEEDS[m]:
                c = cellD(m, s, cond)
                if not (R.has(c) and V3C.reduction_path(c).exists()):
                    continue
                mf = D.top1_frame(c)
                mg = red(c)
                mg["pred_marg"] = space[mg.pred_idx.to_numpy()]
                j = mf[KEY + ["pred_loinc", "correct"]].merge(
                    mg[KEY + ["pred_marg", "correct"]], on=KEY, how="inner", suffixes=("_mapfirst", "_marg"),
                    validate="one_to_one")
                differs = (j.pred_loinc != j.pred_marg).to_numpy()
                rows.append({"condition": cond, "model": R.LABEL[m], "seed": s, "n_dump": len(mf), "n": len(j),
                             "n_pred_differs": int(differs.sum()), "share_pred_differs": float(differs.mean()),
                             "acc_mapfirst": float(j.correct_mapfirst.mean()),
                             "acc_marginalised": float(j.correct_marg.mean()),
                             "acc_diff": float(j.correct_mapfirst.mean() - j.correct_marg.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(out / "casting_agreement.csv", index=False)
    print(t.groupby(["condition", "model"])[["n_pred_differs", "share_pred_differs", "acc_diff"]]
          .agg(["mean", "min", "max"]).to_string(float_format=lambda v: f"{v:+.2e}"))
    return [m for m in R.MODELS if R.LABEL[m] in set(t.model)]


# -- the difference-in-differences -------------------------------------------
def overlap_facts(f: pd.DataFrame, pretrain: set) -> dict:
    pat = pd.Series(f.patient_uuid.unique())
    return {"n_patients": len(pat), "n_patients_overlap": int(pat.isin(pretrain).sum()),
            "share_patients_overlap": float(pat.isin(pretrain).mean()),
            "n_obs": len(f), "share_obs_overlap": float(f.patient_uuid.isin(pretrain).mean())}


def _gap(f) -> float:
    return float(f.correct[f.ov].mean() - f.correct[~f.ov].mean())


def did_rows(frames: dict, pretrain: set, seed, n_boot=N_BOOT) -> list[dict]:
    """`frames` is {model key: KEY + correct} for one seed, FHIR-MLM's included. The rows give FHIR-MLM's
    overlap-vs-clean gap on its own rows. Then, per control, they give both models' gaps on FHIR-MLM ∩
    control and the paired DiD with a patient-clustered interval. The paired DiD is FHIR-MLM's gap minus
    the control's on the same rows. The unpaired form (`did_v1_form`) is FHIR-MLM's own-row gap minus
    the control's shared-row gap."""
    def mark(f):
        f = f[KEY + ["correct"]].copy()
        f["correct"] = f.correct.astype(bool)
        f["ov"] = f.patient_uuid.isin(pretrain).to_numpy()
        return f
    mlm = mark(frames["mlm"])
    own = _gap(mlm)
    lo, hi = clustered_ci(mlm.patient_uuid.to_numpy(), mlm.ov.to_numpy(), [mlm.correct.to_numpy()], [1], n_boot)
    rows = [{"seed": seed, "model": R.LABEL["mlm"], "rows": "own", "n": len(mlm),
             "gap": own, "gap_lo": lo, "gap_hi": hi}]
    for ctl in CONTROLS:
        if ctl not in frames:
            continue
        a, b = D.matched(mlm, mark(frames[ctl]))
        pat, ov = a.patient_uuid.to_numpy(), a.ov.to_numpy()
        g_t, g_c = _gap(a), _gap(b)
        glo, ghi = clustered_ci(pat, ov, [b.correct.to_numpy()], [1], n_boot)
        dlo, dhi = clustered_ci(pat, ov, [a.correct.to_numpy(), b.correct.to_numpy()], [1, -1], n_boot)
        rows.append({"seed": seed, "model": R.LABEL[ctl], "rows": "FHIR-MLM ∩ control", "n": len(a),
                     "gap": g_c, "gap_lo": glo, "gap_hi": ghi, "mlm_gap_same_rows": g_t,
                     "did": g_t - g_c, "did_lo": dlo, "did_hi": dhi, "did_v1_form": own - g_c})
    return rows


def stage_did(out: Path) -> list[str]:
    pretrain = pretrain_patients()
    rows, facts, used = [], [], set()
    for letter in "ABCD":
        for s in R.SEEDS["mlm"]:
            fr = {}
            for m in ("mlm",) + CONTROLS:
                if s not in R.SEEDS[m] or not R.has(cellX(m, s, letter)):
                    continue
                c = cellX(m, s, letter)
                f = D.top1_frame(c, extra=("in_pretrain_4096",))
                flag = f.in_pretrain_4096
                if flag.isna().any() or (flag.astype(bool).to_numpy() != f.patient_uuid.isin(pretrain).to_numpy()).any():
                    raise AssertionError(f"{c.name}: the dump's in_pretrain_4096 disagrees with {PRETRAIN.name}")
                fr[m] = f
                if letter == "D" and s == R.SEEDS[m][0]:
                    facts.append({"model": R.LABEL[m], **overlap_facts(f, pretrain)})
            if "mlm" in fr:
                rows += [{"cell": letter, **r} for r in did_rows(fr, pretrain, s)]
                used |= set(fr)
    t = pd.DataFrame(rows)
    t.to_csv(out / "pretrain_did.csv", index=False)
    fa = pd.DataFrame(facts)
    fa.to_csv(out / "pretrain_overlap_facts.csv", index=False)
    print(fa.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print("\nmean over seeds:")
    print(t.groupby(["cell", "model"])[["gap", "did", "did_v1_form"]].agg(["mean", "std"]).to_string(
        float_format=lambda v: f"{v:+.5f}"))
    return [m for m in R.MODELS if m in used]


# -- ambiguity ---------------------------------------------------------------
def ambiguous_strings() -> tuple[set, int]:
    """Case-insensitive paraphrase strings claimed by more than one LOINC, over every pool file, and
    the number of distinct strings."""
    owners: dict = {}
    for fp in sorted(POOLS.glob("*.txt")):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                owners.setdefault(ln.strip().lower(), set()).add(fp.stem)
    return {s for s, o in owners.items() if len(o) > 1}, len(owners)


def desc_by_key(amb: set) -> pd.DataFrame:
    """The description each test observation showed, flagged ambiguous or not. It is read from one TF-IDF
    para dump, because TF-IDF scores every test observation and every model and seed of a family shows
    the same per-patient paraphrase."""
    t = pd.read_parquet(cellD("tfidf", 0).path / "part-0000.parquet",
                        columns=KEY + ["true_loinc", "desc_shown"])
    t["ambiguous"] = t.desc_shown.str.strip().str.lower().isin(amb).to_numpy()
    return t


def split(f: pd.DataFrame, desc: pd.DataFrame, n_boot: int = 0) -> dict:
    """Top-1 on ambiguous vs unique strings over f's rows. The join is inner, so the few rows whose gold
    is outside the label space (which TF-IDF does not score) drop out."""
    j = f[KEY + ["correct"]].merge(desc[KEY + ["ambiguous"]], on=KEY, how="inner", validate="one_to_one")
    ok = j.correct.astype(bool).to_numpy()
    amb = j.ambiguous.to_numpy()
    r = {"n": len(j), "n_ambiguous": int(amb.sum()), "share_ambiguous": float(amb.mean()),
         "errors_ambiguous": int((~ok & amb).sum()), "errors_unique": int((~ok & ~amb).sum()),
         "acc_ambiguous": float(ok[amb].mean()), "acc_unique": float(ok[~amb].mean())}
    r["unique_minus_ambiguous"] = r["acc_unique"] - r["acc_ambiguous"]
    if n_boot:
        r["diff_lo"], r["diff_hi"] = clustered_ci(j.patient_uuid.to_numpy(), ~amb, [ok], [1], n_boot)
    return r


def joint_split_ci(frames: list, n_boot=N_BOOT, seed=0) -> dict:
    """Seed-mean accuracy gap (unique − ambiguous) and error-rate ratio (ambiguous ÷ unique) with a joint
    patient × seed bootstrap (patients resampled with replacement, then seeds). frames: one per seed,
    (patient_uuid, correct, ambiguous) on identical rows."""
    f0 = frames[0]
    for f in frames[1:]:
        if not (f.patient_uuid.to_numpy() == f0.patient_uuid.to_numpy()).all():
            raise AssertionError("the seeds do not cover identical rows")
    up, inv = np.unique(f0.patient_uuid.to_numpy(), return_inverse=True)
    k = len(up)
    amb = f0.ambiguous.to_numpy().astype(bool)
    Na, Nu = np.bincount(inv, amb.astype(float), k), np.bincount(inv, (~amb).astype(float), k)
    Sa = np.stack([np.bincount(inv, f.correct.to_numpy().astype(float) * amb, k) for f in frames], axis=1)
    Su = np.stack([np.bincount(inv, f.correct.to_numpy().astype(float) * ~amb, k) for f in frames], axis=1)

    def stats(w, js):
        aa, au = ((w @ Sa[:, js]) / (w @ Na)).mean(), ((w @ Su[:, js]) / (w @ Nu)).mean()
        return au - aa, (1 - aa) / (1 - au), 1 - aa, 1 - au
    g, q, ea, eu = stats(np.ones(k), np.arange(len(frames)))
    rng = np.random.default_rng(seed)
    G, Q = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        G[b], Q[b], _, _ = stats(w, rng.integers(0, len(frames), len(frames)))
    pct = lambda x, v: float(np.percentile(x, v))
    return {"n_seeds": len(frames), "err_ambiguous": ea, "err_unique": eu, "gap": g, "gap_lo": pct(G, 2.5),
            "gap_hi": pct(G, 97.5), "err_ratio": q, "ratio_lo": pct(Q, 2.5), "ratio_hi": pct(Q, 97.5)}


def stage_ambiguity(out: Path) -> list[str]:
    amb, n_str = ambiguous_strings()
    desc = desc_by_key(amb)
    pairs = desc.groupby(["patient_uuid", "true_loinc"]).agg(
        n_desc=("desc_shown", "nunique"), ambiguous=("ambiguous", "any"))
    counts = pd.DataFrame([
        {"unit": "distinct paraphrase strings (case-insensitive)", "n": n_str, "n_ambiguous": len(amb)},
        {"unit": "test (patient, LOINC) pairs", "n": len(pairs), "n_ambiguous": int(pairs.ambiguous.sum()),
         "n_shown_more_than_one_string": int((pairs.n_desc > 1).sum())},
        {"unit": "test observations", "n": len(desc), "n_ambiguous": int(desc.ambiguous.sum())}])
    counts["share"] = counts.n_ambiguous / counts.n
    for letter in "ABC":                    # every cell shows the same string per observation
        o = pd.read_parquet(cellX("tfidf", 0, letter).path / "part-0000.parquet", columns=KEY + ["desc_shown"])
        j = desc[KEY + ["desc_shown"]].merge(o, on=KEY, how="outer", indicator=True)
        if (j._merge != "both").any() or (j.desc_shown_x != j.desc_shown_y).any():
            raise AssertionError(f"cell {letter}'s TF-IDF dump shows different descriptions from cell D's")
    rows, summ = [], []
    for letter in "ABCD":
        for m in R.CONDITIONS["para_xsite"]:
            frames = []
            for s in R.SEEDS[m]:
                c = cellX(m, s, letter)
                if R.has(c):
                    f = D.top1_frame(c)
                    rows.append({"cell": letter, "model": R.LABEL[m], "seed": s, **split(f, desc, N_BOOT)})
                    frames.append(f[KEY + ["correct"]].merge(desc[KEY + ["ambiguous"]], on=KEY, how="inner",
                                                            validate="one_to_one").sort_values(KEY))
            if frames:
                summ.append({"cell": letter, "model": R.LABEL[m], **joint_split_ci(frames)})
    counts.to_csv(out / "ambiguity_counts.csv", index=False)
    a = pd.DataFrame(rows)
    a.to_csv(out / "ambiguity_accuracy.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "ambiguity_summary.csv", index=False)
    print(sm.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print(counts.to_string(index=False))
    print()
    print(a.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nmean over seeds:")
    print(a.groupby(["cell", "model"])[["acc_ambiguous", "acc_unique", "unique_minus_ambiguous"]].agg(["mean", "std"])
          .to_string(float_format=lambda v: f"{v:.5f}"))
    return [m for m in R.MODELS if R.LABEL[m] in set(a.model)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["casting", "did", "ambiguity", "all"])
    a = ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    for st, fn in (("casting", stage_casting), ("did", stage_did), ("ambiguity", stage_ambiguity)):
        if a.stage in (st, "all"):
            print("=" * 78 + f"\nchecks · {st}\n" + "=" * 78, flush=True)
            R.stamp(out, ANALYSIS, fn(out), stage=st, note=NOTES[st])
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
