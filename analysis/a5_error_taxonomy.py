#!/usr/bin/env python
"""Error taxonomy by LOINC axis: when a model is wrong, how wrong is it? (thesis Chapter 6)

A LOINC code decomposes into six axes, so every error can be classified by the axes it gets wrong.
Serum-vs-urine glucose is a different clinical event from an unrelated analyte, and accuracy cannot
tell them apart.

The models select from a fixed label space, so every prediction is a syntactically valid LOINC by
construction. The hallucination failure mode of Yu et al. (2026) cannot occur here. Every error is
therefore of the "valid-but-wrong" class that they identify as the dangerous one. The taxonomy covers
every error and is not a side analysis.

Yu et al. (2026) found that the errors of commercial LLMs were concentrated in PROPERTY (72-83% of
chemistry discrepancies), with no time, scale or system errors. The rows are reported for CHEM and
for all classes, so the comparison is like-for-like.

The main result is the order-independent axis marginals. The severity ladder is secondary. Its
ordering is an analytic choice, explained in lib/loinc_axes.py.

`rows_for_run` works from the calibration reductions (pred_idx / true_idx at T=1), so no logits are
re-read. analysis/v3_errors.py runs it:

  python analysis/v3_errors.py --stage run
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_A = importlib.util.spec_from_file_location("a2", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)
_L = importlib.util.spec_from_file_location("loinc_axes", HERE / "lib" / "loinc_axes.py")
L = importlib.util.module_from_spec(_L); _L.loader.exec_module(L)


def rows_for_run(d, label, cond, s, table, space, is_chem, marg, sev_rows, conf_rows, ex_rows,
                 examples=False):
    """Append the rows of one run, from its reduction frame `d`, to the four lists: axis marginals,
    severity classes, confidence by severity and examples."""
    err = d[~d.correct]
    if not len(err):
        return
    ti, pi = err.true_idx.values, err.pred_idx.values
    diffs = L.axis_differences(ti, pi, table)
    sev = L.severity_class(diffs)
    nax = L.n_axes_differing(diffs)
    for subset, mask in (("all", np.ones(len(err), bool)), ("chem", is_chem[ti])):
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"model": label, "condition": cond, "seed": s, "subset": subset,
               "n_obs": len(d), "n_errors": n, "error_rate": n / len(d),
               "mean_axes_differing": float(nax[mask].mean())}
        for c in L.AXES:                       # order-independent: the main result
            row[f"pct_{c}"] = 100 * float(diffs[c].values[mask].mean())
        # distance as the count of differing axes (0..6). It is order-free and makes no
        # severity claim. k=0 would mean two distinct codes that share all six axes.
        for k in range(7):
            row[f"pct_axes_{k}"] = 100 * float((nax[mask] == k).mean())
        # Order-free specimen subclass, over every error that differs on SYSTEM. The ladder
        # files only some of them under "2_wrong_system", because that class also requires
        # COMPONENT to match. No literature ranks the LOINC axes, so all six count equally.
        # The subclass must be a share of SYSTEM-touching errors to compare with pct_SYSTEM.
        sysm = mask & diffs["SYSTEM"].values
        if sysm.any():
            sub = L.system_error_subclass(ti[sysm], pi[sysm], table)
            conv = float((sub == "blood_compartment_synonym").mean())
            row["share_of_SYSTEM_errors_that_are_convention"] = 100 * conv
            row["pct_SYSTEM_blood_synonym"] = row["pct_SYSTEM"] * conv
            row["pct_SYSTEM_diff_specimen"] = row["pct_SYSTEM"] * (1 - conv)
        else:
            row["share_of_SYSTEM_errors_that_are_convention"] = np.nan
            row["pct_SYSTEM_blood_synonym"] = 0.0
            row["pct_SYSTEM_diff_specimen"] = 0.0
        # Order-free confidence contrast. The ladder compares "wrong analyte" with
        # "time/scale/method only", which are two exclusive classes. The order-free analogue
        # compares COMPONENT-touching errors with analyte-preserving ones, and near misses
        # (1 axis) with distant ones (>=3). Each is also compared with the correct predictions.
        cp = err.p_top1.values
        cm = mask & diffs["COMPONENT"].values
        nm = mask & ~diffs["COMPONENT"].values
        a1 = mask & (nax == 1)
        a3 = mask & (nax >= 3)
        row["n_COMPONENT"] = int(cm.sum())
        row["n_no_COMPONENT"] = int(nm.sum())
        row["mean_conf_COMPONENT"] = float(cp[cm].mean()) if cm.any() else np.nan
        row["mean_conf_no_COMPONENT"] = float(cp[nm].mean()) if nm.any() else np.nan
        row["mean_conf_1axis"] = float(cp[a1].mean()) if a1.any() else np.nan
        row["mean_conf_3plus_axes"] = float(cp[a3].mean()) if a3.any() else np.nan
        okm = d.correct.values & (np.ones(len(d), bool) if subset == "all"
                                  else is_chem[d.true_idx.values])
        row["mean_conf_correct"] = float(d.p_top1.values[okm].mean()) if okm.any() else np.nan
        marg.append(row)
        srow = {"model": label, "condition": cond, "seed": s, "subset": subset,
                "n_errors": n}
        for lab in L.SEVERITY_LABELS:
            srow[f"pct_{lab}"] = 100 * float((sev[mask] == lab).mean())
        # Split the specimen class. A swap between Ser/Plas and Bld is a labelling convention,
        # but Ser/Plas confused with Urine is a real specimen error. Reported as one class, it
        # makes a model whose specimen errors are mostly convention swaps look far worse than it is.
        sysmask = mask & (sev == "2_wrong_system")
        if sysmask.any():
            sub = L.system_error_subclass(ti[sysmask], pi[sysmask], table)
            srow["pct_2a_blood_synonym"] = 100 * float((sub == "blood_compartment_synonym").mean()) \
                * srow["pct_2_wrong_system"] / 100
            srow["pct_2b_different_specimen"] = srow["pct_2_wrong_system"] - srow["pct_2a_blood_synonym"]
            srow["share_of_specimen_errors_that_are_convention"] = \
                100 * float((sub == "blood_compartment_synonym").mean())
        else:
            srow["pct_2a_blood_synonym"] = 0.0
            srow["pct_2b_different_specimen"] = 0.0
            srow["share_of_specimen_errors_that_are_convention"] = np.nan
        sev_rows.append(srow)
    # confidence by severity: links the taxonomy to the calibration analysis
    conf = err.p_top1.values
    for lab in L.SEVERITY_LABELS:
        k = sev == lab
        if k.sum() >= 50:
            conf_rows.append({"model": label, "condition": cond, "seed": s,
                              "severity": lab, "n": int(k.sum()),
                              "mean_confidence": float(conf[k].mean())})
    if examples:                                # examples from seed 0 only
        e = pd.DataFrame({"true_idx": ti, "pred_idx": pi, "sev": sev})
        for lab, grp in e.groupby("sev"):
            top = grp.groupby(["true_idx", "pred_idx"]).size().nlargest(3)
            for (t_, p_), cnt in top.items():
                ex_rows.append({
                    "model": label, "condition": cond, "severity": lab, "count": int(cnt),
                    "true_loinc": space[t_], "pred_loinc": space[p_],
                    "true_name": table.get("LONG_COMMON_NAME", pd.Series(dtype=str)).iloc[t_]
                    if "LONG_COMMON_NAME" in table else "",
                    "pred_name": table.get("LONG_COMMON_NAME", pd.Series(dtype=str)).iloc[p_]
                    if "LONG_COMMON_NAME" in table else "",
                    "true_component": table.COMPONENT.iloc[t_],
                    "pred_component": table.COMPONENT.iloc[p_],
                    "true_system": table.SYSTEM.iloc[t_], "pred_system": table.SYSTEM.iloc[p_],
                    "true_property": table.PROPERTY.iloc[t_],
                    "pred_property": table.PROPERTY.iloc[p_]})
