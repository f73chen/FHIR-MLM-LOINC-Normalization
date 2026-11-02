#!/usr/bin/env python
"""
Errors by LOINC axis on v3: which axes the errors differ on (T-19), confidence by how far an error is
from the truth (T-20), and the distance and error-rate data behind F-09 (thesis Chapter 6).

  --stage run   a5's per-run taxonomy (`a5.rows_for_run`) on the v3 cell-D reductions, para and nodesc,
                each model on its own rows, every seed

The six axes are weighted equally and counted order-free. So the per-axis shares sum to the mean
number of differing axes instead of 100%. Each model is scored on its own rows, because an error table
describes one model. Writes aggregate CSVs to results/analysis/v3/errors/.

  python analysis/v3_errors.py --stage run
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
from lib import loinc_axes as L  # noqa: E402
from lib import registry as R  # noqa: E402

_S = importlib.util.spec_from_file_location("a5", HERE / "a5_error_taxonomy.py")
A5 = importlib.util.module_from_spec(_S); _S.loader.exec_module(A5)
_C = importlib.util.spec_from_file_location("v3c", HERE / "v3_calibration.py")
V3C = importlib.util.module_from_spec(_C); _C.loader.exec_module(V3C)

ANALYSIS = "errors"


def _table():
    space = A5.A.canonical_space()
    table = L.load_axis_table(space)
    is_chem = (table.get("CLASS") == "CHEM").values if "CLASS" in table else np.zeros(len(table), bool)
    return space, table, is_chem


def stage_run(out: Path) -> list[str]:
    space, table, is_chem = _table()
    marg, sev, conf, ex = [], [], [], []
    done = set()
    for cond in R.CONDITIONS:
        for m in R.CONDITIONS[cond]:
            for s in R.SEEDS[m]:
                p = V3C.reduction_path(R.cell(m, s, condition=cond, train_arm="pert", test_arm="pert"))
                if not p.exists():
                    continue
                A5.rows_for_run(pd.read_parquet(p), R.LABEL[m], cond, s, table, space, is_chem,
                                marg, sev, conf, ex, examples=(s == R.SEEDS[m][0]))
                done.add(m)
    marg = pd.DataFrame(marg)
    marg.to_csv(out / "errors_axis_marginals.csv", index=False)
    pd.DataFrame(sev).to_csv(out / "errors_severity.csv", index=False)
    pd.DataFrame(conf).to_csv(out / "errors_confidence_by_severity.csv", index=False)
    pd.DataFrame(ex).to_csv(out / "errors_examples.csv", index=False)
    a = marg[marg.subset == "all"]
    t19 = a.groupby(["condition", "model"])[["n_errors", "error_rate", "mean_axes_differing"]
                                            + [f"pct_{c}" for c in L.AXES]].mean()
    t20 = a.groupby(["condition", "model"])[["mean_conf_correct", "mean_conf_1axis", "mean_conf_3plus_axes"]].mean()
    print("T-19 (mean over seeds):"); print(t19.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nT-20 (mean over seeds):"); print(t20.to_string(float_format=lambda v: f"{v:.3f}"))
    return sorted(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    R.stamp(out, ANALYSIS, stage_run(out), stage="run",
            note="native rows per model at cell D")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
