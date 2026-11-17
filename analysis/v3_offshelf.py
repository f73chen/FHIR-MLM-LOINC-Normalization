#!/usr/bin/env python
"""
The five off-the-shelf models, few-shot with three examples, on the clean and perturbed versions of the
same test observations: T-28's rows and F-11's data (thesis Chapters 5 and 6).

  --stage run   T-28 rows per model × arm. These are outcome shares (summing to 100%) with a 95% interval
                on `correct` from a bootstrap over the sample's 500 patients, each share recounted from
                b4's per-observation file. They are joined to the clean-vs-pert exact McNemar test
                (`v3_rq3.py --stage fewshot`)

Only the few-shot runs are read. The off-the-shelf numbers do not depend on the fine-tuned models.
Reads results/analysis/b4/ and results/analysis/v3/rq3/. Writes aggregate CSVs to
results/analysis/v3/offshelf/.

  python analysis/v3_offshelf.py --stage run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402

ANALYSIS = "offshelf"
N_BOOT, BOOT_SEED = 2000, 0             # as in the other per-observation bootstraps (v3_singleobs, v3_rq3)
B4 = HERE.parent / "results" / "analysis" / "b4"
RQ3 = HERE.parent / "results" / "analysis" / "v3" / "rq3"
RUN = re.compile(r"^(?P<model>.+)__v3(?P<arm>clean|pert)__fs3$")
LABEL = {"openbio-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B", "gemma3-12b": "Gemma-3-12B",
         "llama31-8b": "Llama-3.1-8B", "meditron-7b": "Meditron-7B"}
OUTCOMES = ["correct", "valid_wrong", "not_a_loinc", "unparseable"]


def stage_run(out: Path) -> list[str]:
    s = pd.read_csv(B4 / "b4_summary.csv")
    s = s[s.model.str.match(RUN)].copy()
    p = pd.read_csv(B4 / "b4_per_observation.csv", usecols=["model", "patient", "outcome"])
    counts = pd.crosstab(p.model, p.outcome).reindex(columns=OUTCOMES, fill_value=0)
    by_run = {run: g for run, g in p.groupby("model")}
    ctl = pd.read_csv(RQ3 / "rq3_fewshot_control.csv").set_index("model")
    rows = []
    for r in s.itertuples():
        g = RUN.match(r.model)
        base, arm = g["model"], g["arm"]
        c = counts.loc[r.model]
        if int(c.sum()) != r.n:
            raise AssertionError(f"{r.model}: {int(c.sum())} per-observation rows, summary n {r.n}")
        for o in OUTCOMES:                                       # the summary must equal the recount
            if abs(100 * c[o] / r.n - getattr(r, o)) > 1e-9:
                raise AssertionError(f"{r.model} {o}: recount {100 * c[o] / r.n} vs summary {getattr(r, o)}")
        g = by_run[r.model]                                      # patients as clusters, with 2 to 10 observations each
        lo, hi = D.bootstrap_ci(pd.DataFrame({"patient_uuid": g.patient.to_numpy(),
                                              "correct": (g.outcome == "correct").to_numpy()}),
                                n_boot=N_BOOT, seed=BOOT_SEED)
        rows.append({"model": LABEL[base], "arm": arm, "n": r.n, "n_patients": int(g.patient.nunique()),
                     "n_correct": int(c["correct"]),
                     **{f"{o}_pct": 100 * c[o] / r.n for o in OUTCOMES},
                     "correct_lo_pct": 100 * lo, "correct_hi_pct": 100 * hi,
                     "sum_pct": 100 * c.sum() / r.n,
                     "mcnemar_p_clean_vs_pert": ctl.loc[base, "mcnemar_p"],
                     "pert_minus_clean_pp": 100 * ctl.loc[base, "pert_minus_clean"]})
    t = pd.DataFrame(rows).sort_values(["arm", "correct_pct"], ascending=[False, False])
    t.to_csv(out / "t28_rows.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    stage_run(out)
    R.stamp(out, ANALYSIS, [], stage="run",
            note="off-the-shelf few-shot, both arms; independent of the trained models")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
