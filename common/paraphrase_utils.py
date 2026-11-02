"""Paraphrase utilities (thesis Chapter 4, Appendix C).

The paraphrase corpus is split into train and validation pools. Each patient then gets one
paraphrase per lab code, which replaces the display text of the observation.

The three main functions:
  1. `load_paraphrases_train_val_split` reads a paraphrase corpus directory
     (UPSTREAM_DIR/paraphrases_ext_verified/) and splits each LOINC's
     paraphrases into train and val sets (seed 42, test_fraction 0.2).
     Duplicates within a LOINC are removed first, so the same paraphrase
     string cannot appear in both splits.
  2. `assign_paraphrases_per_patient` takes a patient's resource list and a
     paraphrase pool that is already split into train or val. It picks one
     paraphrase per (patient, LOINC) pair. Every occurrence of LOINC X in the
     patient's history is patched to the same display string.
  3. `apply_paraphrases_to_resources` patches
     `obs["code"]["coding"][0]["display"]` in place for every lab observation
     whose MIMIC code maps to a LOINC in the assignment dict. Observations
     that do not map keep their existing display.

Design notes:
  - `assign_paraphrases_per_patient` draws from a `random.Random` passed in
     by the caller (the dataset builders pass patient_rng(uuid)).
     `load_paraphrases_train_val_split` builds its own `random.Random(seed)`.
     Both are deterministic for a fixed seed.
  - The FHIR-MLM tokenizer is not subclassed. It writes the description
     through an opt-in branch of its Coding handler
     (third_party/fhir_mlm/fhir_tokenizer.py). The branch runs when
     `relevant_keys` includes a path that ends in `display` (e.g.
     `"code:display"` on Observation). This module only produces and applies
     the display-text patches.
"""

from __future__ import annotations

import glob
import hashlib
import os
import random
from pathlib import Path
from typing import Iterable

from synthea_utils import MIMIC_LAB_SYSTEM, MIMIC_CHARTEVENTS_SYSTEM

# Default systems whose primary coding gets paraphrase-patched. Callers can
# pass other code systems via `lab_systems=...`.
DEFAULT_LAB_SYSTEMS: frozenset[str] = frozenset({MIMIC_LAB_SYSTEM, MIMIC_CHARTEVENTS_SYSTEM})

# Default salt for the per-patient RNG seed. The salt is mixed into the SHA1
# of the patient UUID. A different salt gives a new patient-to-paraphrase
# assignment and leaves the corpus split unchanged.
DEFAULT_PATIENT_RNG_SALT = "fhir-tokenizer/desc-experiment/v1"


def load_paraphrases_train_val_split(
    corpus_dir: str | Path,
    test_fraction: float = 0.2,
    seed: int = 42,
    dedup: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict]:
    """Load all `.txt` files from `corpus_dir` (one file per LOINC, one
    paraphrase per non-empty line) and split each LOINC's paraphrases into
    train and val sets.

    The split:
      - One shared `random.Random(seed)` is advanced across all LOINCs, sorted
        by filename, so the split is deterministic.
      - `n_test = max(1, int(len(lines) * test_fraction))`, so the val pool
        gets at least one paraphrase.
      - `val` is the first `n_test` lines after the shuffle, and `train` is
        the rest.

    Duplicates within a LOINC's file are removed before the split (`dedup`,
    on by default). Without dedup, a paraphrase string that appears twice in
    the same file could put one copy in train and one in val. Val strings
    would then leak into train.

    Returns (train_paraphrases, val_paraphrases, stats).
    """
    rng = random.Random(seed)
    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}

    files = sorted(glob.glob(os.path.join(str(corpus_dir), "*.txt")))
    n_dedup_total = 0
    n_loincs_with_dups = 0

    for fp in files:
        loinc_code = Path(fp).stem
        with open(fp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if dedup:
            seen: set[str] = set()
            uniq: list[str] = []
            for ln in lines:
                if ln not in seen:
                    seen.add(ln)
                    uniq.append(ln)
            if len(uniq) < len(lines):
                n_dedup_total += len(lines) - len(uniq)
                n_loincs_with_dups += 1
            lines = uniq
        if not lines:
            continue
        rng.shuffle(lines)
        n_test = max(1, int(len(lines) * test_fraction))
        val[loinc_code] = lines[:n_test]
        train[loinc_code] = lines[n_test:]

    stats = {
        "n_loincs": len(train),
        "n_train": sum(len(v) for v in train.values()),
        "n_val": sum(len(v) for v in val.values()),
        "n_dedup_removed": n_dedup_total,
        "n_loincs_with_dups": n_loincs_with_dups,
    }
    if verbose:
        print(
            f"Loaded paraphrases for {stats['n_loincs']} LOINCs from {corpus_dir}\n"
            f"  train: {stats['n_train']} paraphrases\n"
            f"  val:   {stats['n_val']} paraphrases\n"
            f"  dedup: removed {stats['n_dedup_removed']} duplicates "
            f"across {stats['n_loincs_with_dups']} LOINCs"
        )
    return train, val, stats


def patient_rng(
    patient_uuid: str,
    salt: str = DEFAULT_PATIENT_RNG_SALT,
) -> random.Random:
    """Build a deterministic `random.Random` instance for one patient.

    The 64-bit seed comes from SHA1 over `salt + ":" + patient_uuid`. SHA1
    gives the same seed in every process, whatever `PYTHONHASHSEED` is set
    to. Python's `hash()` does not. The salt is a cheap way to re-randomize
    every patient's paraphrase assignment together without touching the
    train/val corpus split. An ablation can use it to reshuffle the
    paraphrase-to-patient mapping while everything else stays fixed.
    """
    digest = hashlib.sha1(f"{salt}:{patient_uuid}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return random.Random(seed)


def check_paraphrase_coverage(
    paraphrases: dict[str, list[str]],
    mimic_to_loinc: dict[str, str],
    *,
    label: str = "",
    raise_on_missing: bool = False,
    verbose: bool = True,
) -> dict:
    """Confirm that every LOINC reachable from `mimic_to_loinc.values()` has
    at least one paraphrase in `paraphrases`. Returns a small stats dict and,
    if `raise_on_missing`, raises `ValueError` when any LOINC is missing.

    Run it right after splitting the corpus, so a coverage gap is caught
    before any training starts.
    """
    needed = set(mimic_to_loinc.values())
    have = {lc for lc, lst in paraphrases.items() if lst}
    missing = needed - have
    extra = have - needed
    stats = {
        "n_needed": len(needed),
        "n_have_with_paraphrases": len(have),
        "n_missing": len(missing),
        "n_extra_in_corpus": len(extra),
        "missing_loincs_sample": sorted(missing)[:10],
    }
    if verbose:
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}coverage: {stats['n_needed'] - stats['n_missing']}/{stats['n_needed']} "
            f"LOINCs have paraphrases ({stats['n_extra_in_corpus']} extra LOINCs in corpus "
            "are unused)"
        )
        if missing:
            print(f"{prefix}  missing sample: {stats['missing_loincs_sample']}")
    if raise_on_missing and missing:
        raise ValueError(
            f"paraphrase coverage gap: {len(missing)}/{len(needed)} LOINCs missing."
            f" First few: {stats['missing_loincs_sample']}"
        )
    return stats


def _iter_lab_loincs(
    resources: list[dict],
    mimic_to_loinc: dict[str, str],
    lab_systems: Iterable[str] = DEFAULT_LAB_SYSTEMS,
) -> Iterable[str]:
    """Yield LOINC codes for each lab Observation in `resources`. Yields one
    LOINC per occurrence (so a LOINC seen N times in this patient yields N
    times). Caller dedupes if it wants unique LOINCs."""
    lab_systems = frozenset(lab_systems)
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        codings = r.get("code", {}).get("coding", [])
        if not codings:
            continue
        primary = codings[0]
        if primary.get("system") not in lab_systems:
            continue
        mimic_code = primary.get("code")
        loinc = mimic_to_loinc.get(mimic_code)
        if loinc:
            yield loinc


def assign_paraphrases_per_patient(
    resources: list[dict],
    paraphrases: dict[str, list[str]],
    mimic_to_loinc: dict[str, str],
    rng: random.Random,
    lab_systems: Iterable[str] = DEFAULT_LAB_SYSTEMS,
) -> dict[str, str]:
    """For each *unique* LOINC present in this patient's lab observations,
    sample one paraphrase from `paraphrases[loinc]`. Returns a
    `{loinc: chosen_paraphrase}` dict.

    Every observation of LOINC X in this patient gets the same paraphrase
    string when `apply_paraphrases_to_resources` runs downstream.

    LOINCs not present in `paraphrases` (or with empty paraphrase lists) are
    left out of the returned dict. `apply_paraphrases_to_resources` then
    leaves the displays of those observations untouched.

    `rng` must be a `random.Random` instance. The result is deterministic for
    a seeded rng and a fixed LOINC iteration order. The order here is
    `sorted(set(_iter_lab_loincs(...)))`, so two patients with the same set
    of LOINCs draw paraphrases at the same RNG positions.
    """
    unique_loincs = sorted(set(_iter_lab_loincs(resources, mimic_to_loinc, lab_systems)))
    out: dict[str, str] = {}
    for loinc in unique_loincs:
        candidates = paraphrases.get(loinc)
        if not candidates:
            continue
        out[loinc] = rng.choice(candidates)
    return out


def apply_paraphrases_to_resources(
    resources: list[dict],
    loinc_to_paraphrase: dict[str, str],
    mimic_to_loinc: dict[str, str],
    lab_systems: Iterable[str] = DEFAULT_LAB_SYSTEMS,
) -> int:
    """Patch `obs["code"]["coding"][0]["display"]` in place for every lab
    Observation whose MIMIC code maps to a LOINC present in
    `loinc_to_paraphrase`. Non-lab resources, lab resources whose LOINC is
    unmapped, and resources whose primary coding lacks a "code" field are
    left untouched.

    Returns the number of Observations patched.
    """
    lab_systems = frozenset(lab_systems)
    n_patched = 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        codings = r.get("code", {}).get("coding", [])
        if not codings:
            continue
        primary = codings[0]
        if primary.get("system") not in lab_systems:
            continue
        mimic_code = primary.get("code")
        if not mimic_code:
            continue
        loinc = mimic_to_loinc.get(mimic_code)
        if loinc is None:
            continue
        paraphrase = loinc_to_paraphrase.get(loinc)
        if paraphrase is None:
            continue
        primary["display"] = paraphrase
        n_patched += 1
    return n_patched
