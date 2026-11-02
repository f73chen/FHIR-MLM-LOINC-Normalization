"""The local MIMIC code behind each scored observation, for the catalogue analyses (thesis Chapter 6).

The catalogue unit is the local code. The evaluation dumps of the deep models carry it (`true_class`).
The TF-IDF dumps do not, because TF-IDF predicts directly in the 679-LOINC space and records only the
LOINC. So the local code is joined in on (patient_uuid, obs_ordinal) from the dump that lists every
test observation with its code. That is the dump of the reference-range diagnostic
(analysis/refrange_diagnostic.py). Only its observation-to-code table is used. Its predictions are
ignored.

The diagnostic reads the unperturbed records, yet its table serves every representation and
condition. The join key is the chronological ordinal of the observation within the patient's whole
record, counted independently of model, condition and representation (lib/dumps.py, `coverage`).
Perturbation changes values and ranges but does not change which observation is which. So the
key-to-code mapping cannot move. `attach` checks this on every join: every row must find a code, and
the joined gold LOINC must equal the dump's `true_loinc`. A mismatch means the ordinal no longer
names the same observation. The join then raises an error and attaches no codes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402


LOOKUP_DIR = locations.WORK / "eval/refrange_diagnostic__mimictest"
KEY = ["patient_uuid", "obs_ordinal"]
_LOOKUP: pd.DataFrame | None = None


def lookup() -> pd.DataFrame:
    """(patient_uuid, obs_ordinal) -> true_class (local MIMIC code), true_loinc. Test split only."""
    global _LOOKUP
    if _LOOKUP is None:
        d = pd.read_parquet(LOOKUP_DIR / "part-0000.parquet", columns=KEY + ["true_class", "true_loinc"])
        if d.duplicated(KEY).any():
            raise AssertionError("the code lookup has duplicate (patient, ordinal) keys")
        _LOOKUP = d
    return _LOOKUP


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with `true_class` joined in, in the row order of `df`. Refuses on any unmatched row or any
    gold-LOINC disagreement (the module docstring says why both are checked)."""
    if "true_loinc" not in df.columns:
        raise ValueError("attach needs the dump's own true_loinc, to verify the join")
    lk = lookup().rename(columns={"true_loinc": "_lk_loinc"})
    n0 = len(df)
    out = df.merge(lk, on=KEY, how="left", sort=False, validate="many_to_one")
    if len(out) != n0:
        raise AssertionError("the code join changed the row count")
    miss = int(out.true_class.isna().sum())
    if miss:
        raise AssertionError(f"{miss:,} of {n0:,} observations have no local code in the lookup")
    disagree = int((out._lk_loinc.to_numpy() != out.true_loinc.to_numpy()).sum())
    if disagree:
        raise AssertionError(f"{disagree:,} of {n0:,} joined rows disagree on the gold LOINC: the "
                             "ordinal does not name the same observation in both dumps")
    out["true_class"] = out.true_class.astype(np.int64)
    return out.drop(columns="_lk_loinc")
