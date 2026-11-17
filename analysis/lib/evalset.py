"""The evaluated test set (thesis Chapter 5).

Off by default. Unless N8_EVALSET is set, every analysis reads every scored observation.

With N8_EVALSET=both, every test result of a trained model is restricted to one frozen set of observations.
  main set   The observations scored in every non-RELMA test dataset of every trained model (the four models,
             both number arms, no description, single observation, values only, ranges only, severity, the
             five draws), without the test patients in the first pretraining corpus of FHIR-MLM.
  RELMA set  The observations of the main set that every RELMA test dataset scores and that carry a RELMA
             name. RELMA never filters the main comparisons. The main comparisons filter RELMA.
These are exempt and read every row, as without the switch:
  - validation cells, which chose the checkpoints and fitted the temperatures
  - the sliding-window analysis, which compares the later windows of each model with its own first window
    and keeps all its patients
  - cells of the earlier dataset versions (v1, v2)
  - the off-the-shelf sample, which is read from its own file and never through this module

The set is built once from the dumps (`python analysis/lib/evalset.py build`) and frozen to a key file in
WORK_DIR/analysis/evalset_both, with a manifest of counts. The key file holds patient ids, so it must stay
out of git. Readers call `mask(cell, patients, ordinals)` or `restrict(cell, frame)`. Integrity checks that
compare a dump with its own manifest read unfiltered (`evaluated=False` in lib/dumps.py).

With the switch on, outputs go to their own roots (results/analysis/v3_both and numbers_both.md), beside the
results on every scored observation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

ENV = "N8_EVALSET"
CHOICES = ("both",)
KEYS_DIR = Path(os.environ.get("N8_EVALSET_DIR", locations.WORK / "analysis" / "evalset_both"))
KEYS_FILE = KEYS_DIR / "keys.parquet"
MANIFEST = KEYS_DIR / "manifest.json"
PRETRAIN = Path(__file__).resolve().parents[2] / "data" / "pretrain" / "pretrain_4096_patients.txt"
KEY = ["patient_uuid", "obs_ordinal"]
MULT = 1_000_000                       # obs_ordinal < MULT (checked when the set is built)


def name() -> str:
    """'' when off, otherwise the name of the chosen set. An unknown value is an error."""
    v = os.environ.get(ENV, "")
    if v and v not in CHOICES:
        raise ValueError(f"{ENV}={v!r}: expected one of {CHOICES} or unset")
    return v


def active() -> bool:
    return bool(name())


def suffix() -> str:
    """Appended to output roots and the numbers book when the switch is on."""
    return f"_{name()}" if active() else ""


def is_relma(cell) -> bool:
    return "relma" in cell.evalset


def applies(cell) -> bool:
    """Whether the rows of `cell` are restricted. Only v3 test cells are, and the sliding windows are not."""
    return (active() and getattr(cell, "v3", False) and cell.split == "test" and cell.evalset != "sliding")


_SETS: dict | None = None


def _load() -> dict:
    global _SETS
    if _SETS is None:
        if not KEYS_FILE.exists():
            raise FileNotFoundError(f"{KEYS_FILE} is missing: build the evaluated set first (lib/evalset.py build)")
        k = pd.read_parquet(KEYS_FILE)
        pats = np.array(sorted(k.patient_uuid.unique()), dtype=object)
        pidx = {p: i for i, p in enumerate(pats)}
        code = k.patient_uuid.map(pidx).to_numpy(np.int64) * MULT + k.obs_ordinal.to_numpy(np.int64)
        _SETS = {"pidx": pidx, "main": np.sort(code), "relma": np.sort(code[k.relma.to_numpy(bool)])}
    return _SETS


def mask(cell, patients, ordinals) -> np.ndarray:
    """Boolean mask over rows given as (patient_uuid, obs_ordinal) arrays. It is all True when the cell is
    not restricted. Otherwise it marks the rows of the main set, or of the RELMA set for a RELMA cell."""
    patients = np.asarray(patients, dtype=object)
    if not applies(cell):
        return np.ones(len(patients), bool)
    S = _load()
    idx = pd.Series(patients).map(S["pidx"]).to_numpy()
    known = ~pd.isna(idx)
    out = np.zeros(len(patients), bool)
    if known.any():
        code = idx[known].astype(np.int64) * MULT + np.asarray(ordinals, np.int64)[known]
        out[known] = np.isin(code, S["relma"] if is_relma(cell) else S["main"])
    return out


def restrict(cell, frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` (with patient_uuid and obs_ordinal) restricted to the evaluated set. An exempt cell is returned
    unchanged."""
    if not applies(cell):
        return frame
    return frame[mask(cell, frame.patient_uuid.to_numpy(), frame.obs_ordinal.to_numpy())]


# -- building the set (once) -------------------------------------------------
def build() -> dict:
    """Freeze the 'both' set from the dumps. Every v3 test dataset with a finished dump counts, except the
    sliding windows. One dump per distinct test dataset is read, because the positions a model scores do not
    depend on its seed or training arm."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from lib import registry as R
    from lib import codes as K
    cells, _ = R.scan()
    G = {}
    for c in cells:
        if not (c.v3 and c.split == "test") or c.evalset == "sliding" or not c.exists():   # finished dumps only
            continue
        k = (c.model, c.condition, c.evalset, c.scope, c.test_arm_explicit)
        if k not in G or (c.seed, c.train_arm) < (G[k].seed, G[k].train_arm):
            G[k] = c
    frames = {k: pd.read_parquet(c.path / "part-0000.parquet", columns=KEY) for k, c in sorted(G.items())}
    pats = np.array(sorted(set().union(*[set(f.patient_uuid.unique()) for f in frames.values()])), dtype=object)
    pidx = {p: i for i, p in enumerate(pats)}
    enc = {}
    for k, f in frames.items():
        o = f.obs_ordinal.to_numpy(np.int64)
        if o.min() < 0 or o.max() >= MULT:
            raise ValueError(f"{k}: obs_ordinal outside [0, {MULT})")
        e = f.patient_uuid.map(pidx).to_numpy(np.int64) * MULT + o
        if len(np.unique(e)) != len(e):
            raise ValueError(f"{k}: duplicate keys")
        enc[k] = np.sort(e)

    def inter(arrs):
        out = arrs[0]
        for a in arrs[1:]:
            out = np.intersect1d(out, a, assume_unique=True)
        return out

    relma_k = [k for k in enc if "relma" in k[2]]
    main = inter([v for k, v in enc.items() if k not in relma_k])
    pre = set(PRETRAIN.read_text().split())
    ov = np.array([p in pre for p in pats])
    main = main[~ov[main // MULT]]
    named = enc[("tfidf", "para_xsite", "relma_s1_a0", "history", "pert")]     # the RELMA dumps of TF-IDF hold named rows only
    if not all(np.array_equal(named, enc[("tfidf", "para_xsite", f"relma_s1_a{a}", "history", "pert")]) for a in (1, 2)):
        raise AssertionError("RELMA-named rows differ between draws")
    relma = inter([main, named] + [enc[k] for k in relma_k if k[0] != "tfidf"])
    # local codes, for the counts
    t = frames[("tfidf", "para_xsite", "mimictest", "history", "pert")].copy()
    t["true_loinc"] = pd.read_parquet(G[("tfidf", "para_xsite", "mimictest", "history", "pert")].path / "part-0000.parquet",
                                      columns=["true_loinc"]).true_loinc.to_numpy()
    t = K.attach(t)
    code_of = pd.Series(t.true_class.to_numpy(), index=t.patient_uuid.map(pidx).to_numpy(np.int64) * MULT
                        + t.obs_ordinal.to_numpy(np.int64))
    out = pd.DataFrame({"patient_uuid": pats[main // MULT], "obs_ordinal": (main % MULT).astype(np.int64),
                        "relma": np.isin(main, relma, assume_unique=True)})
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = KEYS_FILE.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False); tmp.replace(KEYS_FILE)
    man = {"set": "both", "built_from_datasets": len(enc), "relma_datasets": len(relma_k),
           "main_observations": int(len(main)), "main_patients": int(len(np.unique(main // MULT))),
           "main_local_codes": int(code_of.loc[main].nunique()),
           "relma_observations": int(len(relma)), "relma_local_codes": int(code_of.loc[relma].nunique()),
           "overlap_patients_excluded": int(ov.sum()),
           "rule": "main = every non-RELMA v3 test dataset of every trained model, minus stage-1 pretraining patients; "
                   "RELMA = main ∩ every RELMA dataset ∩ RELMA-named rows; validation, sliding, v1 exempt"}
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    return man


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        sys.exit("usage: python analysis/lib/evalset.py build")
    print(json.dumps(build(), indent=1))
