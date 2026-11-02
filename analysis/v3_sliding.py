#!/usr/bin/env python
"""What truncation at 8,192 tokens leaves unscored, and whether accuracy holds on it. This is the
sliding-window evaluation of the three deep models at cell D, seed 0 (thesis Chapter 6).
It applies b3_sliding's checks and gap (T-26, the by-window table) and b3_figures' fixed-cohort
curve (the data of F-10a and F-10b) to the sliding evaluation dumps.

  --stage run   integrity, T-26 with the denominator decomposition, the by-window table, F-10 data

The methods are b3's, called unchanged. Their docstrings explain the patient-mix and code-mix confounds
and the fixed cohort. T-26 also compares window 0 with windows 1+ on patients with ≥ 2 windows, with a
patient-clustered interval. F-10b is a fixed cohort of patients whose record reaches window index DEPTH,
over windows 0..DEPTH-1. F-10a is the slope of the error rate on window index, refit in every patient
bootstrap. DEPTH starts at b3_figures' values and is halved while a cohort holds fewer than MIN_COHORT
patients. The cohort sizes at every candidate depth are written out so the choice is visible.

T-26's denominators set every model's counts against the full test set, which TF-IDF scores row by row.
They split into observations in the scored windows, observations of patients the model cannot score
(FHIR-MLM needs an Encounter resource), and the shortfall on scored patients.

The windows score most of a model's scorable observations but not all of them. The missing part is
the shortfall. For FHIR-MLM, almost all of it comes from windows that pipeline/23_sliding_dataset.py
skips whole. These windows hold no Encounter, and the tokenizer needs one as its time anchor (KeyError
't0'). They are the windows of records from before a patient's first Encounter, because the packer prepends only an
Encounter at or before the window's start. The rest of the shortfall is truncation. Windows are packed by
estimated token cost up to 8,192, but the tokenizer keeps 8,190 plus [CLS] and [SEP]. So a window can run
over (most by 1–2 tokens), and the next window starts after the cut. BCM and Llama need no Encounter.
Their shortfall is truncation plus one record past the 128-window cap. As a result, past window 0 the
dump's obs_ordinal drifts from the chronological index, because skipped windows and cut tails add no
offset. So a sliding dump must not be joined to another dump on (patient, ordinal) beyond window 0.
Per-model results (T-26, F-10) stay valid, because every scored position is compared with its own gold.

Reads the evaluation dumps in WORK_DIR/eval. Writes results/analysis/v3/sliding/. Aggregate output only.

  python analysis/v3_sliding.py --stage run
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402


def _load(name):
    s = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


B3 = _load("b3_sliding")
B3F = _load("b3_figures")

ANALYSIS = "sliding"
KEY = ["patient_uuid", "obs_ordinal"]
DEEP = ("mlm", "bcm", "llama")
DEPTH = dict(B3F.DEPTH)              # starting depths: FHIR-MLM 8, BCM 16, Llama 16
MIN_COHORT = 100
CANDIDATE_DEPTHS = (4, 8, 12, 16, 20, 24, 32)


def cells_v3(m):
    """(sliding cell, canonical cell): cell D, seed 0."""
    spec = dict(condition="para_xsite", train_arm="pert", test_arm="pert")
    return R.cell(m, 0, evalset="sliding", **spec), R.cell(m, 0, **spec)


def frame(c) -> pd.DataFrame:
    """Top-1 frame (mapping-first) with b3's columns: window_idx and late for a sliding dump."""
    f = D.top1_frame(c, extra=("window_idx",)) if c.evalset == "sliding" else D.top1_frame(c)
    if "window_idx" in f:
        f["late"] = f.window_idx > 0
    return f


def cohort_from(sl: pd.DataFrame, depth: int):
    """b3_figures' fixed cohort, on a frame already in memory: patients whose record reaches window
    index `depth` (depth+1 windows or more), windows 0..depth-1, per-(patient, window) counts."""
    nw = sl.groupby("patient_uuid")["window_idx"].max()
    keep = nw[nw >= depth].index
    sub = sl[sl.patient_uuid.isin(keep) & (sl.window_idx < depth)]
    g = sub.groupby(["patient_uuid", "window_idx"])["correct"].agg(n="size", ok="sum").reset_index()
    return g, len(keep)


def fixed_cohort_rows(m: str, sl: pd.DataFrame, depth: int) -> list[dict]:
    g, n_pat = cohort_from(sl, depth)
    pt, lo, hi, n_obs, trend = B3F.curve(g, depth)
    return [{"model": m, "window_idx": w, "err_per_10k": pt[w], "ci_lo": lo[w], "ci_hi": hi[w],
             "n_obs": int(n_obs[w]), "n_patients": n_pat, "depth": depth, "slope_per_window": trend[0],
             "slope_ci_lo": trend[1], "slope_ci_hi": trend[2]} for w in range(depth)]


def summary_row(m: str, sl: pd.DataFrame, can: pd.DataFrame) -> dict:
    """The per-model record: b3_sliding's integrity checks and gap, the two window accuracies in T-26, and the
    accuracy on the observations only the later windows score."""
    rec = B3.integrity(m, sl, can)
    rec.update(B3.gap(sl))
    nw = sl.groupby("patient_uuid")["window_idx"].max()
    multi = sl[sl.patient_uuid.isin(nw[nw >= 1].index)]          # the two window columns of T-26 (b3's gap() cohort)
    rec["acc_window0_multi"] = float(multi[~multi.late].correct.mean())
    rec["acc_later_multi"] = float(multi[multi.late].correct.mean())
    rec["acc_newly_recovered"] = float(sl[sl.late].correct.mean())
    rec["n_newly_recovered"] = len(sl) - len(can)
    return rec


def denominators(m: str, sl: pd.DataFrame, can: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Where the observations of the full test set go for model m (T-26). They are counted per patient.
    Past window 0 a sliding dump's ordinals drift from the chronological index (see `alignment`), so a
    join on keys would pair different observations."""
    c_sl = sl.groupby("patient_uuid").size()
    c_full = full.groupby("patient_uuid").size()
    scored = c_full.index.isin(c_sl.index)
    diff = c_full[scored] - c_sl.reindex(c_full.index[scored])
    nw = sl.groupby("patient_uuid").window_idx.max()
    return {"model": m, "n_test_obs": len(full), "n_test_patients": len(c_full),
            "n_patients_scored": len(c_sl), "n_first_window": len(can), "n_all_windows": len(sl),
            "coverage_first_window": len(can) / len(sl), "gain": len(sl) / len(can),
            "obs_of_unscored_patients": int(c_full[~scored].sum()),
            "obs_of_scored_patients": int(c_full[scored].sum()),
            "share_of_scored_patients_obs_in_windows": len(sl) / int(c_full[scored].sum()),
            "shortfall_obs": int(diff[diff > 0].sum()), "shortfall_patients": int((diff > 0).sum()),
            "excess_obs": int(-diff[diff < 0].sum()), "excess_patients": int((diff < 0).sum()),
            "patients_at_window_cap": int((nw >= 127).sum()), "max_windows": int(nw.max()) + 1}


def alignment(m: str, sl: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """How far the sliding dump's (patient, ordinal) still names the same observation as the chronological
    index of the test set. It is measured as gold-LOINC agreement with TF-IDF's row at the same key, by
    window. Window 0 agrees. Later windows drift, because a window that loses observations to truncation
    shifts every later ordinal. Per-model sliding results are unaffected, because each position is scored
    against its own gold. Joins to other dumps past window 0 are not valid."""
    j = sl[KEY + ["true_loinc", "window_idx"]].merge(gold, on=KEY, how="left", suffixes=("", "_test"))
    j["agrees"] = j.true_loinc.to_numpy() == j.true_loinc_test.to_numpy()
    wb = pd.cut(j.window_idx, bins=B3.WBINS, right=False, labels=B3.WLABS)
    t = j.groupby(wb, observed=True).agrees.agg(n="size", share_gold_agrees="mean").reset_index()
    return t.rename(columns={"window_idx": "window"}).assign(model=m)


def stage_run(out: Path) -> list[str]:
    full = D.top1_frame(R.cell("tfidf", 0, condition="para_xsite", train_arm="pert", test_arm="pert"))[KEY + ["true_loinc"]]
    summ, dens, coh, sizes, ali, done = [], [], [], [], [], []
    for m in DEEP:
        c_sl, c_can = cells_v3(m)
        if not (R.has(c_sl) and R.has(c_can)):
            print(f"  {R.LABEL[m]}: no v3 sliding dump yet")
            continue
        sl, can = frame(c_sl), frame(c_can)
        summ.append(summary_row(m, sl, can))
        dens.append(denominators(m, sl, can, full))
        ali.append(alignment(m, sl, full))
        B3.by_window(sl).assign(model=m).to_csv(out / f"sliding_by_window__{m}.csv", index=False)
        nw = sl.groupby("patient_uuid").window_idx.max()
        sizes += [{"model": m, "depth": d, "n_patients": int((nw >= d).sum())} for d in CANDIDATE_DEPTHS]
        depth = DEPTH[m]
        while depth > 2 and int((nw >= depth).sum()) < MIN_COHORT:
            depth //= 2
        coh += fixed_cohort_rows(m, sl, depth)
        done.append(m)
    s, d, c = pd.DataFrame(summ), pd.DataFrame(dens), pd.DataFrame(coh)
    s.to_csv(out / "sliding_summary.csv", index=False)
    d.to_csv(out / "sliding_denominators.csv", index=False)
    c.to_csv(out / "sliding_fixed_cohort.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "sliding_cohort_sizes.csv", index=False)
    al = pd.concat(ali, ignore_index=True) if ali else pd.DataFrame()
    al.to_csv(out / "sliding_alignment.csv", index=False)
    t27 = s.assign(acc_w0_minus_later_pp=s.gap_patient_matched_per10k / 100)[
        ["model", "n_patients_multi_window", "gap_patient_matched_per10k", "gap_ci_lo_per10k", "gap_ci_hi_per10k",
         "significant", "patients_at_window_cap"]]
    print("integrity:"); print(s[["model", "duplicate_join_keys", "patients_with_noncontiguous_ordinals",
                                  "window0_join_rate_vs_canonical", "window0_agreement_with_canonical"]].to_string(index=False))
    print("\nT-26 denominators:"); print(d.T.to_string())
    print("\nordinal alignment with the test set, by window:")
    print(al.pivot(index="window", columns="model", values="share_gold_agrees").to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nT-26, window 0 minus later windows (per 10k; patients with ≥ 2 windows):"); print(t27.to_string(index=False))
    print("\nF-10 cohorts:"); print(c.groupby("model")[["depth", "n_patients", "slope_per_window", "slope_ci_lo", "slope_ci_hi"]]
                                    .first().to_string())
    print("\ncohort sizes by depth:"); print(pd.DataFrame(sizes).pivot(index="depth", columns="model", values="n_patients").to_string())
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    R.stamp(out, ANALYSIS, stage_run(out), stage="run", note="cell D seed 0, per model")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
