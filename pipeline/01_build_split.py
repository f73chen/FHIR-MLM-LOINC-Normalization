"""Build and freeze the patient-level train/val/test split shared by all four model families:
TF-IDF, FHIR-MLM, BCM and Llama (thesis Chapter 4).

The default does not reproduce the split used in the thesis. FHIR-MLM was pretrained in two stages.
The thesis split excluded only the stage-2 (8192) pretraining corpus, so some of its test patients were
in the stage-1 (4096) corpus. Chapter 6 measures the effect. The default excludes both stages, which is
the right choice for a new split. Pass --contamination-set legacy8192 to reproduce the thesis split.
Every trained model depends on the split, so a different split means retraining all of them.

Eligibility and the quantile-bin stratification follow the split of the earlier TF-IDF experiments.
The contamination set comes from fhir_data_utils.enumerate_arrow_patients, run on the `files` column of
the pretraining Arrow datasets. That column is the ground-truth record of who FHIR-MLM was pretrained
on, and it takes precedence over mlm_train_patients.txt.

Rules:
  - Eligible        = LabObservation>=1 & 3 < total_resources <= 8192.
  - Contamination   = all patients in the `files` column of the chosen pretraining Arrow
                      datasets, train and pretraining-val alike. Excluding all of them is
                      conservative. With the default `union` set, no split patient appeared in
                      either pretraining stage.
  - Splits          train/val/test and the reserve are disjoint stratified subsets of the
                    clean pool (eligible - pretrain), binned once so the bins are identical intervals.
                    train=100000, val=2000, test=10000; reserve = the rest.
                    With the default set, no fine-tuning split contains a pretraining patient,
                    and the reserve is contamination-free and safe for a later test extension.
                    Train and test share one distribution, because the excluded pretraining
                    patients are kept out of train too. They have about 3x longer histories,
                    so including them in train would shift train against test.
  - Hard asserts    disjointness, (train|val|test) ∩ pretrain = ∅, eligibility, no duplicates,
                    every UUID resolves to a real <uuid>.ndjson on disk.

The patient lists stay in data/splits/. Stdout shows only aggregates (counts and percentiles).
Determinism: SEED=42 everywhere, so re-running yields an identical split.

  python pipeline/01_build_split.py --contamination-set legacy8192
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

# --- Config -----------------------------------------------------------------
SEED = 42
N_BINS = 10
VAL_SIZE = 2_000
TEST_SIZE = 10_000
TRAIN_SIZE = 100_000

SUMMARY_FILE = locations.UPSTREAM / "loinc_patients_summary.csv"

# Contamination set. FHIR-MLM was pretrained in two stages:
#   stage 1 (from scratch) : ...pretrain-4096-35k
#   stage 2 (warm start)   : ...pretrain-8192-35k
# Stage 2 warm-starts from the stage-1 checkpoint, so the weights carry exposure to both corpora.
# The thesis split excluded only the 8192 set, so some of its test patients are in the stage-1
# corpus.
ARROW_PATH_8192 = str(locations.PRETRAIN_DATASETS / "mimic-hosp-merged-tokenized-pretrain-8192-35k")
ARROW_PATH_4096 = str(locations.PRETRAIN_DATASETS / "mimic-hosp-merged-tokenized-pretrain-4096-35k")
CONTAMINATION_SETS = {
    "union": [ARROW_PATH_4096, ARROW_PATH_8192],   # both stages: use for any new split
    "legacy8192": [ARROW_PATH_8192],               # stage 2 only: reproduces the thesis split
}
NDJSON_DIR = locations.PATIENTS
FHIR_UTILS_DIR = str(locations.COMMON)  # enumerate_arrow_patients
OUTPUT_DIR = locations.REPO / "data/splits"

EXPECTED_COLUMNS = [
    "filename", "total_resources", "Encounter", "Procedure",
    "LabObservation", "ChartObservation", "Specimen",
]
ELIGIBILITY = "LabObservation>=1 & 3 < total_resources <= 8192"


# --- Pure, unit-testable logic (no real-data / heavy deps) ------------------
def strip_ndjson(name: str) -> str:
    """Normalize a patient identifier to a bare UUID (drop a single .ndjson suffix)."""
    return name[:-7] if name.endswith(".ndjson") else name


def apply_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to eligible patients (the bounds in ELIGIBILITY) and add a `uuid`
    column with the bare UUID, for set operations. Uses only the given frame."""
    out = df[
        (df["LabObservation"] >= 1)
        & (df["total_resources"] > 3)
        & (df["total_resources"] <= 8192)
    ].copy()
    out["uuid"] = out["filename"].map(strip_ndjson)
    return out


def add_bins(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Add a quantile bin column on total_resources."""
    out = df.copy()
    out["bin"] = pd.qcut(out["total_resources"], q=n_bins, labels=False, duplicates="drop")
    return out


def stratified_take(df: pd.DataFrame, n: int, strat_col: str, seed: int):
    """Return (taken, remainder) where len(taken)==n, stratified by strat_col.
    Falls back to a non-stratified split if any class has <2 members."""
    if n >= len(df):
        raise ValueError(f"stratified_take: n={n} >= len(df)={len(df)}")
    counts = df[strat_col].value_counts()
    if df[strat_col].nunique() < 2 or counts.min() < 2:
        remainder, taken = train_test_split(df, test_size=n, random_state=seed)
    else:
        remainder, taken = train_test_split(
            df, test_size=n, stratify=df[strat_col], random_state=seed
        )
    return taken, remainder


def build_split(
    df_eligible: pd.DataFrame,
    pretrain_uuids: set[str],
    *,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    train_size: int = TRAIN_SIZE,
    n_bins: int = N_BINS,
    seed: int = SEED,
) -> dict:
    """Carve train, val, test and the reserve from the clean pool (eligible - pretrain), as
    the module docstring describes. No split contains a patient from `pretrain_uuids`. The
    result depends only on the eligible frame and the pretraining set."""
    elig_uuids = set(df_eligible["uuid"])
    clean = df_eligible[~df_eligible["uuid"].isin(pretrain_uuids)].copy()
    need = train_size + val_size + test_size
    if len(clean) < need:
        raise ValueError(f"clean pool {len(clean)} < train+val+test = {need}")

    # Bin once, so all four subsets use the same quantile intervals.
    clean = add_bins(clean, n_bins)
    test, rest = stratified_take(clean, test_size, "bin", seed)
    val, rest = stratified_take(rest, val_size, "bin", seed)
    train, reserve = stratified_take(rest, train_size, "bin", seed)

    return {
        "eligible_uuids": elig_uuids,
        "pretrain_uuids": pretrain_uuids,
        "clean_pool": clean,
        "val": val,
        "test": test,
        "train": train,
        "reserve": reserve,
    }


def run_asserts(
    splits: dict, *, val_size: int = VAL_SIZE, test_size: int = TEST_SIZE,
    train_size: int = TRAIN_SIZE,
) -> None:
    """Assert the split rules and abort on any violation."""
    sets = {k: set(splits[k]["uuid"]) for k in ("train", "val", "test", "reserve")}
    pretrain = splits["pretrain_uuids"]
    elig = splits["eligible_uuids"]
    names = ("train", "val", "test", "reserve")

    # Disjointness across all four carved subsets.
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"{a}∩{b} = {len(sets[a] & sets[b])}"
    # Contamination: no train, val or test patient is in the excluded pretraining set.
    for name in ("train", "val", "test"):
        assert not (sets[name] & pretrain), f"{name}∩pretrain = {len(sets[name] & pretrain)}"
    # Eligibility: every finetune-split member is eligible.
    for name in ("train", "val", "test"):
        assert sets[name] <= elig, f"{name} has {len(sets[name] - elig)} ineligible UUIDs"
    # No dupes within a split.
    for name in ("train", "val", "test"):
        df = splits[name]
        assert len(df) == df["uuid"].nunique(), f"duplicate UUIDs in {name}"
    # Sizes.
    assert len(sets["val"]) == val_size, f"val size {len(sets['val'])} != {val_size}"
    assert len(sets["test"]) == test_size, f"test size {len(sets['test'])} != {test_size}"
    assert len(sets["train"]) == train_size, f"train size {len(sets['train'])} != {train_size}"


def check_files_exist(splits: dict, ndjson_dir: Path) -> None:
    """Assert every train/val/test UUID resolves to a real <uuid>.ndjson on disk."""
    available = set(os.listdir(ndjson_dir))
    for name in ("train", "val", "test"):
        wanted = {u + ".ndjson" for u in splits[name]["uuid"]}
        missing = wanted - available
        assert not missing, f"{len(missing)} {name} patients missing ndjson in {ndjson_dir}"


# --- Reporting / IO ---------------------------------------------------------
def _pctiles(df: pd.DataFrame) -> dict:
    s = df["total_resources"]
    return {
        "n": int(len(df)),
        "min": int(s.min()), "p25": float(s.quantile(0.25)), "median": float(s.median()),
        "p75": float(s.quantile(0.75)), "max": int(s.max()), "mean": round(float(s.mean()), 1),
    }


def _compute_overlaps(splits: dict) -> dict:
    """Pairwise overlaps among the 4 carved subsets, plus each fine-tuning split against
    pretraining. Every value must be 0 (run_asserts checks this)."""
    ft = {k: set(splits[k]["uuid"]) for k in ("train", "val", "test", "reserve")}
    pre = splits["pretrain_uuids"]
    names = ("train", "val", "test", "reserve")
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}_{b}"] = len(ft[a] & ft[b])
    for name in ("train", "val", "test"):
        out[f"{name}_pretrain"] = len(ft[name] & pre)
    return out


def build_manifest(splits: dict, contamination_set: str = "union") -> dict:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_bins": N_BINS,
        "eligibility": ELIGIBILITY,
        "sources": {
            "summary_file": str(SUMMARY_FILE),
            # which contamination set this split was built against. Without it a split is
            # unattributable. The thesis split predates this field and is legacy8192.
            "contamination_set": contamination_set,
            "pretrain_arrow": CONTAMINATION_SETS[contamination_set],
            "ndjson_dir": str(NDJSON_DIR),
        },
        "counts": {
            "eligible": len(splits["eligible_uuids"]),
            "pretrain": len(splits["pretrain_uuids"]),
            "clean_pool": int(len(splits["clean_pool"])),
            "train": int(len(splits["train"])),
            "val": int(len(splits["val"])),
            "test": int(len(splits["test"])),
            "reserve_pool": int(len(splits["reserve"])),
        },
        "overlaps": _compute_overlaps(splits),
        "history_length_pctiles": {
            "train": _pctiles(splits["train"]),
            "val": _pctiles(splits["val"]),
            "test": _pctiles(splits["test"]),
            "clean_pool": _pctiles(splits["clean_pool"]),
        },
    }


def write_outputs(splits: dict, manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_cols = EXPECTED_COLUMNS + ["bin"]
    for name in ("train", "val", "test", "reserve"):
        df = splits[name]
        with open(output_dir / f"{name}_patients.txt", "w") as f:
            for u in df["uuid"]:
                f.write(f"{u}.ndjson\n")
        keep = [c for c in stats_cols if c in df.columns]
        df[keep].to_csv(output_dir / f"{name}_patients_with_stats.csv", index=False)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def print_report(manifest: dict) -> None:
    c = manifest["counts"]
    print("\n" + "=" * 64)
    print("SPLIT SUMMARY")
    print("=" * 64)
    print(f"  eligible           : {c['eligible']:>8,}")
    print(f"  pretrain (exclude) : {c['pretrain']:>8,}  (Arrow files column)")
    print(f"  clean pool         : {c['clean_pool']:>8,}  (eligible - pretrain)")
    print(f"  train              : {c['train']:>8,}  (clean pool; disjoint from pretrain & val/test)")
    print(f"  val                : {c['val']:>8,}  (clean)")
    print(f"  test               : {c['test']:>8,}  (clean)")
    print(f"  reserve pool       : {c['reserve_pool']:>8,}  (clean - train - val - test; for test extension)")
    print("\n  overlaps (all must be 0):")
    for k, v in manifest["overlaps"].items():
        flag = "" if v == 0 else "  <-- VIOLATION"
        print(f"    {k:<20}: {v}{flag}")
    print("\n  history-length (total_resources) percentiles per split:")
    print(f"    {'split':<11}{'n':>8}{'min':>7}{'p25':>8}{'med':>8}{'p75':>8}{'max':>8}{'mean':>9}")
    for name, p in manifest["history_length_pctiles"].items():
        print(f"    {name:<11}{p['n']:>8,}{p['min']:>7}{p['p25']:>8.0f}{p['median']:>8.0f}"
              f"{p['p75']:>8.0f}{p['max']:>8}{p['mean']:>9.1f}")
    print("=" * 64)


# --- Main -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute + assert + print, write nothing")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--skip-file-check", action="store_true", help="skip the on-disk ndjson existence assert")
    ap.add_argument("--contamination-set", choices=list(CONTAMINATION_SETS), default="union",
                    help="'union' (default: both MLM pretrain stages) or 'legacy8192' "
                         "(the 8192 stage only, which reproduces the frozen 2026-06-28 split)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    sys.path.insert(0, FHIR_UTILS_DIR)
    from fhir_data_utils import enumerate_arrow_patients

    print(f"Loading summary: {SUMMARY_FILE}")
    df_raw = pd.read_csv(SUMMARY_FILE)
    assert list(df_raw.columns) == EXPECTED_COLUMNS, (
        f"summary schema drift: {list(df_raw.columns)}"
    )
    df_eligible = apply_eligibility(df_raw)
    print(f"  raw patients     : {len(df_raw):,}")
    print(f"  eligible         : {len(df_eligible):,}  ({ELIGIBILITY})")

    print("\nEnumerating pretrain patients from Arrow `files` column...")
    # Always enumerate both stages so the overlap is visible even when reproducing the legacy split.
    per_corpus = {p: enumerate_arrow_patients(p) for p in CONTAMINATION_SETS["union"]}
    for p, s in per_corpus.items():
        print(f"  {Path(p).name}: {len(s):,} patients")
    union_all = set().union(*per_corpus.values())
    print(f"  union of both stages: {len(union_all):,}")

    pretrain_uuids = set().union(*(per_corpus[p] for p in CONTAMINATION_SETS[args.contamination_set]))
    print(f"  contamination-set='{args.contamination_set}' -> excluding {len(pretrain_uuids):,}")
    if args.contamination_set == "legacy8192":
        missed = union_all - pretrain_uuids
        print(f"  legacy mode: {len(missed):,} stage-1 (4096) patients are not excluded. "
              "This reproduces the frozen split, which may include stage-1 pretraining patients.")

    splits = build_split(df_eligible, pretrain_uuids)
    run_asserts(splits)
    print("\n[asserts] disjointness, contamination, eligibility and sizes: PASSED")

    if not args.skip_file_check:
        check_files_exist(splits, NDJSON_DIR)
        print("[asserts] every train/val/test UUID has an .ndjson file on disk: PASSED")

    manifest = build_manifest(splits, args.contamination_set)
    print_report(manifest)

    if args.dry_run:
        print("\n[dry-run] no files written. Re-run without --dry-run to freeze the split.")
        return
    write_outputs(splits, manifest, Path(args.output_dir))
    print(f"\nWrote split + manifest to {args.output_dir}/")


if __name__ == "__main__":
    main()
