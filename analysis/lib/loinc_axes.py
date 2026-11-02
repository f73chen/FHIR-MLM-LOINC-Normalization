"""LOINC six-axis decomposition, for the error analysis by LOINC axis (analysis/v3_errors.py, thesis
Chapter 6). Reads UPSTREAM_DIR/Loinc.csv.

A LOINC code decomposes into six axes, so every wrong prediction can be classified by which axis went
wrong. Predicting serum glucose when the truth is urine glucose is a different clinical event from
predicting an unrelated analyte. Accuracy alone cannot tell them apart.

Missing values are the main pitfall. METHOD_TYP is empty for most of the 679 codes, and a plain
`a != b` comparison treats NaN as unequal to itself. Every pair where both methods are absent would
then be reported as "method differs", which inflates the rarest and most interesting axis. So missing
values are filled with a single sentinel before comparison. A pair with both values missing compares
equal, and a pair with one missing compares different.

The severity ordering is an analytic choice. It was not set by a laboratory clinician and is not
clinical ground truth. So the order-independent axis marginals are the primary result. This hierarchy
is a secondary view. A clinician reader can re-rank it in SEVERITY_RULES, and the other results
remain valid. No published clinical ranking of LOINC axes was found. Plebani
2010 ranks errors by patient outcome, and McDonald 2023 by frequency and correctability. The five eGFR
equations in Ferguson 2012 show that METHOD swaps are not automatically minor. The ladder is therefore
a bookkeeping precedence (the first differing axis wins). It is read after the marginals and the count
of differing axes (`n_axes_differing`).

Used by analysis/a5_error_taxonomy.py and v3_errors.py. a6 and b4 also use it to read Loinc.csv.
"""
from __future__ import annotations


import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
AXES = ["COMPONENT", "PROPERTY", "TIME_ASPCT", "SYSTEM", "SCALE_TYP", "METHOD_TYP"]
MISSING = "<unspecified>"

# Worst first. Each rule is (label, predicate over the per-axis difference flags). The first match wins,
# so the classes are mutually exclusive. `_unclassified` can only appear if two distinct LOINC codes
# share all six axes.
SEVERITY_RULES = [
    ("1_wrong_component", lambda d: d["COMPONENT"]),
    ("2_wrong_system", lambda d: d["SYSTEM"]),
    ("3_wrong_property", lambda d: d["PROPERTY"]),
    ("4_wrong_time_scale_method", lambda d: d["TIME_ASPCT"] | d["SCALE_TYP"] | d["METHOD_TYP"]),
]
SEVERITY_LABELS = [r[0] for r in SEVERITY_RULES] + ["_unclassified"]

SEVERITY_GLOSS = {
    "1_wrong_component": "wrong analyte entirely",
    "2_wrong_system": "right analyte, wrong specimen (e.g. serum vs urine)",
    "3_wrong_property": "right analyte and specimen, wrong quantity kind (e.g. mass vs substance conc.)",
    "4_wrong_time_scale_method": "differs only in timing, scale or method (fewest axes; not necessarily clinically minor)",
    "_unclassified": "identical on all six axes (should be impossible - see gate G3)",
}


def load_axis_table(space) -> pd.DataFrame:
    """Axis table for a label space, row-aligned to it: row i describes space[i].

    Alignment matters because downstream code indexes this by the model's class index. A reindex that
    reordered or dropped rows would mislabel every error.
    """
    space = list(space)
    d = pd.read_csv(LOINC_CSV, low_memory=False)
    d.columns = [c.upper() for c in d.columns]
    keep = ["LOINC_NUM"] + AXES + [c for c in ("CLASS", "STATUS", "LONG_COMMON_NAME", "SHORTNAME")
                                   if c in d.columns]
    d = d[keep].drop_duplicates(subset="LOINC_NUM").set_index("LOINC_NUM")
    out = d.reindex(space)
    for c in AXES:
        out[c] = out[c].fillna(MISSING).astype(str).str.strip().str.upper()
    out.index.name = "loinc"
    return out.reset_index()


def axis_differences(true_idx: np.ndarray, pred_idx: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Boolean per-axis difference flags for each (true, predicted) pair, indexed like the inputs."""
    out = {}
    for c in AXES:
        v = table[c].values
        out[c] = v[true_idx] != v[pred_idx]
    return pd.DataFrame(out)


def severity_class(diffs: pd.DataFrame) -> np.ndarray:
    """Mutually exclusive severity label per row: the first matching rule in SEVERITY_RULES."""
    out = np.full(len(diffs), "_unclassified", dtype=object)
    assigned = np.zeros(len(diffs), dtype=bool)
    for label, pred in SEVERITY_RULES:
        m = pred(diffs).values & ~assigned
        out[m] = label
        assigned |= m
    return out


def n_axes_differing(diffs: pd.DataFrame) -> np.ndarray:
    return diffs[AXES].values.sum(axis=1).astype(int)


# -- sub-classification of specimen (SYSTEM) errors ---------------------------
# Reporting "wrong specimen" as one class is misleading. For some models most specimen errors are swaps
# within the blood compartments, chiefly between Bld and Ser/Plas, because LOINC merges serum and
# plasma into Ser/Plas where they are clinically equivalent (Users' Guide 2.5 p.2). For other models most specimen
# errors are different specimens, such as Ser/Plas versus Urine or Bld versus Body fluid. Collapsing
# the two makes the first kind of model look far worse than it is. Two LOINC documents disagree on
# how much a swap between Bld and Ser/Plas matters. The Mapper's Guide "System Adjusted" column groups
# a subset of Bld and Ser/Plas chemistry codes as near-equivalent. The Users' Guide keeps the axis
# because reference ranges are keyed to the code and a swap can break range checking (2.5 p.7). So
# these swaps are reported as a separate subclass. They are not counted as correct, and they are not
# described as "clinically harmless".
#
# Like SEVERITY_RULES, this set is an analytic judgement and was not made by a clinician. A clinician
# reader can revise it here. It is conservative and holds only blood-compartment synonyms.
BLOOD_COMPARTMENTS = {
    "SER/PLAS", "SER", "PLAS", "BLD", "BLDA", "BLDV", "BLDC",
    "SER/PLAS/BLD", "PLAS/BLD", "PPP", "PLAS^DONOR",
}


def system_error_subclass(true_idx, pred_idx, table) -> np.ndarray:
    """For specimen errors, separate blood-compartment synonyms from different specimens."""
    ts = table["SYSTEM"].values[true_idx]
    ps = table["SYSTEM"].values[pred_idx]
    both_blood = np.array([a in BLOOD_COMPARTMENTS and b in BLOOD_COMPARTMENTS for a, b in zip(ts, ps)])
    return np.where(both_blood, "blood_compartment_synonym", "different_specimen")
