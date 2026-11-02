"""Derive the LOINC label space that all four models report on (thesis Chapter 4).

  - Primary set: codes that are LOINC-mappable and occur at least 10 times in the train split.
  - Top-N core cut: the N most frequent codes. N is not hard-coded. The script prints the floor at
    each candidate rank, so the cut can be chosen from real numbers.
  - Macro core: primary codes with at least 20 test observations, for stable per-class metrics.
  - Aside: the set of codes seen in at least 10 distinct patients, and how it differs from the
    occurrence set. This set is stricter because it rejects a code whose 10 occurrences all come
    from one patient.
  - Coverage of the space on the test split.

TF-IDF is a classifier over this class set. The deep models train on the full LOINC-mappable head and
only report on this space.

How observations are counted:
  - Lab observations are extracted as in paraphrase_utils._iter_lab_loincs. For each Observation,
    take code.coding[0], check its system and read .get("code"). The scan is restricted to
    LAB_CODE_SYSTEM (mimic-d-labitems), the system of the model head. The other default lab system,
    mimic-chartevents-d-items, is out of scope. It holds ICU chart items, which are not in the head
    and do not appear in the files of the hospital patients.
  - Label mapping: build_lab_code_mappings maps mimic_code -> class -> LOINC. This is the same label
    path the deep head uses (fhir_data_utils.tokenize_and_align).
  - A lab observation the model can predict has its code in the head. labitems codes outside the head
    (e.g. 53153) cannot be tokenized or predicted. They are counted and reported as an out-of-head
    gap, and no assert fails on them.
  - Counts are at the LOINC level. Occurrences set the primary threshold, and distinct patients set
    the aside.

Writes data/labelspace/labelspace.csv and manifest.json. They hold LOINC codes and aggregate counts
and no patient identifiers. Prints aggregates only.

  python pipeline/02_derive_labelspace.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
CODES_TO_TOKENS_PATH = str(locations.TOKENIZER / "codes_to_tokens.pkl")
FHIR_UTILS_DIR = str(locations.COMMON)
NDJSON_DIR = locations.PATIENTS
SPLITS_DIR = locations.REPO / "data/splits"
OUTPUT_DIR = locations.REPO / "data/labelspace"

MIN_OCCURRENCES = 10            # primary space: >= this many train occurrences (micro headline)
MACRO_CORE_MIN_TEST_OBS = 20    # macro core: >= this many test obs (stable per-class metrics)
TOPN_CANDIDATES = [500, 550, 600, 650]
FLOOR_BAR_TEST_OBS = 80  # diagnostic bar for the top-N-by-train floor table


# --- Pure, unit-testable logic ----------------------------------------------
def iter_lab_codes(resources: list[dict], lab_systems) -> "list[str]":
    """Return the MIMIC lab code (itemid) of every lab Observation, one list entry per occurrence.
    Applies the same extraction filter as paraphrase_utils._iter_lab_loincs and keeps the raw mimic_code."""
    out = []
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        codings = r.get("code", {}).get("coding", [])
        if not codings:
            continue
        primary = codings[0]
        if primary.get("system") not in lab_systems:
            continue
        code = primary.get("code")
        if code:
            out.append(str(code))
    return out


def tally_patient(resources, lab_systems, head_codes: set, code_to_loinc: dict[str, str]):
    """Return (loinc_occ Counter, n_lab, n_head, n_mapped) for one patient.
    n_lab    = all LAB_CODE_SYSTEM observations;
    n_head   = those whose code is in the model head (tokenizable/predictable);
    n_mapped = those whose code maps to a label LOINC (in the head and LOINC-mappable).
    Invariant: n_mapped <= n_head <= n_lab (code_to_loinc keys subset head_codes)."""
    occ: Counter = Counter()
    n_lab = n_head = n_mapped = 0
    for code in iter_lab_codes(resources, lab_systems):
        n_lab += 1
        if code in head_codes:
            n_head += 1
        loinc = code_to_loinc.get(code)
        if loinc is not None:
            n_mapped += 1
            occ[loinc] += 1
    return occ, n_lab, n_head, n_mapped


def compose_code_to_loinc(mimic_code_to_class: dict[str, int],
                          class_to_loinc: dict[int, str]) -> dict[str, str]:
    """Collapse the model label path to {mimic_code -> LOINC}. Only codes whose class is
    LOINC-mappable are kept. These are the codes the deep head labels (tokenize_and_align)."""
    return {code: class_to_loinc[cls]
            for code, cls in mimic_code_to_class.items() if cls in class_to_loinc}


def build_labelspace(train_occ: Counter, train_pat: Counter,
                     test_occ: Counter, test_pat: Counter, *,
                     min_occ: int = MIN_OCCURRENCES,
                     min_core_test_obs: int = MACRO_CORE_MIN_TEST_OBS,
                     topn_candidates=TOPN_CANDIDATES) -> dict:
    """Pure given the four count maps. Returns the primary micro-headline set (>= min_occ
    train occurrences), the macro core (>= min_core_test_obs test observations, for stable
    per-class metrics), the aside set (>= min_occ distinct patients) and the diagnostic table
    of floors by rank."""
    def rank_key(l):
        return (-train_occ[l], l)

    distinct_loinc = sorted(train_occ, key=rank_key)
    primary = [l for l in distinct_loinc if train_occ[l] >= min_occ]
    by_patients = sorted((l for l in train_pat if train_pat[l] >= min_occ), key=rank_key)

    total_train = sum(train_occ.values())
    total_test = sum(test_occ.values())
    floor_rows = {}
    for N in sorted(set(topn_candidates + [len(primary)])):
        if N == 0 or N > len(primary):
            continue
        topN = primary[:N]
        floor_rows[N] = {
            "floor_train_occ": train_occ[topN[-1]],
            "floor_test_occ": min(test_occ.get(l, 0) for l in topN),
            "floor_test_patients": min(test_pat.get(l, 0) for l in topN),
            "cum_train_coverage": (sum(train_occ[l] for l in topN) / total_train) if total_train else 0.0,
            "cum_test_coverage": (sum(test_occ.get(l, 0) for l in topN) / total_test) if total_test else 0.0,
        }

    macro_core = [l for l in primary if test_occ.get(l, 0) >= min_core_test_obs]  # >= test floor
    set_occ, set_pat = set(primary), set(by_patients)
    return {
        "distinct_loinc_in_train": distinct_loinc,
        "primary": primary,
        "macro_core": macro_core,
        "by_patients": by_patients,
        "only_in_occ": sorted(set_occ - set_pat, key=rank_key),   # >=10 occ but <10 patients
        "only_in_pat": sorted(set_pat - set_occ, key=rank_key),   # impossible (patients<=occ)
        "floor_rows": floor_rows,
    }


# --- Real-data scan (main only; supports multiprocessing) -------------------
def _load_uuids(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip()[:-7] if line.strip().endswith(".ndjson") else line.strip()
                for line in f if line.strip()]


_W: dict = {}


def _winit(lab_systems, head_codes, code_to_loinc, ndjson_dir):
    _W.update(ls=lab_systems, head=head_codes, c2l=code_to_loinc, dir=ndjson_dir)


def _scan_one(uuid):
    import ndjson
    path = _W["dir"] / f"{uuid}.ndjson"
    if not path.exists():
        return None
    with open(path) as f:
        resources = ndjson.load(f)
    occ, n_lab, n_head, n_mapped = tally_patient(resources, _W["ls"], _W["head"], _W["c2l"])
    return dict(occ), n_lab, n_head, n_mapped


def scan_split(uuids, ndjson_dir, lab_systems, head_codes, code_to_loinc, desc, workers=1):
    from tqdm import tqdm
    loinc_occ: Counter = Counter()
    loinc_pat: Counter = Counter()
    totals = {"n_patients": 0, "n_missing": 0, "n_lab_obs": 0, "n_head_obs": 0, "n_mapped_obs": 0}

    def _merge(res):
        if res is None:
            totals["n_missing"] += 1
            return
        occ, n_lab, n_head, n_mapped = res
        totals["n_patients"] += 1
        totals["n_lab_obs"] += n_lab
        totals["n_head_obs"] += n_head
        totals["n_mapped_obs"] += n_mapped
        loinc_occ.update(occ)
        for loinc in occ:
            loinc_pat[loinc] += 1

    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_winit,
                     initargs=(lab_systems, head_codes, code_to_loinc, ndjson_dir)) as pool:
            for res in tqdm(pool.imap_unordered(_scan_one, uuids, chunksize=64),
                            total=len(uuids), desc=desc):
                _merge(res)
    else:
        import ndjson
        for uuid in tqdm(uuids, desc=desc):
            path = ndjson_dir / f"{uuid}.ndjson"
            if not path.exists():
                _merge(None)
                continue
            with open(path) as f:
                resources = ndjson.load(f)
            occ, n_lab, n_head, n_mapped = tally_patient(resources, lab_systems, head_codes, code_to_loinc)
            _merge((dict(occ), n_lab, n_head, n_mapped))
    return loinc_occ, loinc_pat, totals


def print_report(lspace, train_tot, test_tot, test_occ, n_mappable_classes, n_distinct_mappable_loinc):
    primary = lspace["primary"]
    mapped_in_primary = sum(test_occ.get(l, 0) for l in primary)
    cov_all = mapped_in_primary / max(test_tot["n_lab_obs"], 1)
    cov_map = mapped_in_primary / max(test_tot["n_mapped_obs"], 1)

    def gap(tot):
        return tot["n_lab_obs"] - tot["n_head_obs"]
    print("\n" + "=" * 72)
    print("LABEL SPACE")
    print("=" * 72)
    print(f"  LOINC-mappable classes (deep head)        : {n_mappable_classes}")
    print(f"  distinct LOINC, mappable, >=1 occ in train : {n_distinct_mappable_loinc}")
    print(f"  >=10 occurrences   (primary, micro headline): {len(primary)}")
    print(f"  >={MACRO_CORE_MIN_TEST_OBS} test obs     (macro core, per-class) : {len(lspace['macro_core'])}")
    print(f"  >=10 distinct patients (aside)             : {len(lspace['by_patients'])}"
          f"   (>=10 occ but <10 patients: {len(lspace['only_in_occ'])}; impossible-set: {len(lspace['only_in_pat'])})")
    print("\n  lab observations (LAB_CODE_SYSTEM = mimic-d-labitems):")
    print(f"    {'split':<6}{'total':>12}{'in-head':>12}{'LOINC-mapped':>14}{'out-of-head':>13}")
    for nm, t in (("train", train_tot), ("test", test_tot)):
        print(f"    {nm:<6}{t['n_lab_obs']:>12,}{t['n_head_obs']:>12,}{t['n_mapped_obs']:>14,}{gap(t):>13,}")
    print("    (out-of-head = labitems codes not in the 1623 head, e.g. 53153, which the model cannot predict)")
    print("\nCOVERAGE (test, primary space):")
    print(f"  primary LOINC / all labitems test obs     : {cov_all*100:.2f}%")
    print(f"  primary LOINC / LOINC-mappable test obs   : {cov_map*100:.2f}%")
    print(f"\nTOP-N FLOOR (pick the cut; bar = floor_test_occ >= {FLOOR_BAR_TEST_OBS}):")
    print(f"  {'N':>5}  {'floor_train':>11}  {'floor_test':>10}  {'floor_test_pat':>14}  {'cum_train%':>10}  {'cum_test%':>9}  clears")
    for N in sorted(lspace["floor_rows"]):
        r = lspace["floor_rows"][N]
        clears = "yes" if r["floor_test_occ"] >= FLOOR_BAR_TEST_OBS else "no"
        tag = "  (=full primary set)" if N == len(primary) else ""
        print(f"  {N:>5}  {r['floor_train_occ']:>11,}  {r['floor_test_occ']:>10,}  "
              f"{r['floor_test_patients']:>14,}  {r['cum_train_coverage']*100:>9.2f}%  "
              f"{r['cum_test_coverage']*100:>8.2f}%  {clears}{tag}")
    print("=" * 72)


def write_outputs(lspace, train_occ, train_pat, test_occ, test_pat, train_tot, test_tot,
                  n_mappable_classes, n_distinct_mappable_loinc, output_dir: Path):
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = lspace["primary"]
    core = set(lspace["macro_core"])
    with open(output_dir / "labelspace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loinc", "rank_train_occ", "train_occ", "train_patients",
                    "test_occ", "test_patients", "in_primary_ge10occ",
                    "in_ge10patients", "in_macro_core_ge20test"])
        for i, l in enumerate(lspace["distinct_loinc_in_train"], start=1):
            w.writerow([l, i, train_occ[l], train_pat.get(l, 0), test_occ.get(l, 0),
                        test_pat.get(l, 0), int(train_occ[l] >= MIN_OCCURRENCES),
                        int(train_pat.get(l, 0) >= MIN_OCCURRENCES),
                        int(l in core)])
    manifest = {
        "min_occurrences": MIN_OCCURRENCES,
        "n_mappable_classes_head": n_mappable_classes,
        "n_distinct_mappable_loinc_in_train": n_distinct_mappable_loinc,
        "primary_label_space_size": len(primary),
        "macro_core_min_test_obs": MACRO_CORE_MIN_TEST_OBS,
        "macro_core_size": len(lspace["macro_core"]),
        "ge10_distinct_patients_size": len(lspace["by_patients"]),
        "occ_ge10_but_patients_lt10": len(lspace["only_in_occ"]),
        "train_totals": train_tot,
        "test_totals": test_tot,
        "out_of_head_obs": {"train": train_tot["n_lab_obs"] - train_tot["n_head_obs"],
                            "test": test_tot["n_lab_obs"] - test_tot["n_head_obs"]},
        "floor_table": lspace["floor_rows"],
        "sources": {"seen_codes": SEEN_CODES_PATH, "codes_to_tokens": CODES_TO_TOKENS_PATH,
                    "splits_dir": str(SPLITS_DIR), "lab_system": "mimic-d-labitems"},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="scan only the first N train patients, serial, no write")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    sys.path.insert(0, FHIR_UTILS_DIR)
    from fhir_data_utils import build_lab_code_mappings, LAB_CODE_SYSTEM

    mimic_code_to_class, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    code_to_loinc = compose_code_to_loinc(mimic_code_to_class, class_to_loinc)
    head_codes = set(mimic_code_to_class)
    lab_systems = frozenset({LAB_CODE_SYSTEM})
    n_mappable_classes = len(class_to_loinc)
    print(f"mappings: {len(mimic_code_to_class)} head classes, {n_mappable_classes} LOINC-mappable, "
          f"{len(set(class_to_loinc.values()))} distinct LOINC in head")

    train_uuids = _load_uuids(SPLITS_DIR / "train_patients.txt")
    if args.smoke:
        import time
        sub = train_uuids[:args.smoke]
        t0 = time.time()
        occ, pat, tot = scan_split(sub, NDJSON_DIR, lab_systems, head_codes, code_to_loinc, "smoke", workers=1)
        dt = time.time() - t0
        ooh = tot["n_lab_obs"] - tot["n_head_obs"]
        print(f"\n[smoke] {tot['n_patients']} patients in {dt:.1f}s ({dt/max(tot['n_patients'],1)*1000:.1f} ms/pt) "
              f"-> ~{dt/max(tot['n_patients'],1)*len(train_uuids)/60:.1f} min serial for {len(train_uuids):,} train")
        print(f"[smoke] labitems obs: {tot['n_lab_obs']:,}  in-head: {tot['n_head_obs']:,}  "
              f"LOINC-mapped: {tot['n_mapped_obs']:,}  out-of-head: {ooh:,} ({ooh/max(tot['n_lab_obs'],1)*100:.1f}%)")
        print(f"[smoke] distinct LOINC seen: {len(occ)}; >=10 occ here: {sum(1 for l in occ if occ[l]>=10)}")
        return

    test_uuids = _load_uuids(SPLITS_DIR / "test_patients.txt")
    print(f"scanning with {args.workers} workers...")
    train_occ, train_pat, train_tot = scan_split(train_uuids, NDJSON_DIR, lab_systems, head_codes, code_to_loinc, "train", args.workers)
    test_occ, test_pat, test_tot = scan_split(test_uuids, NDJSON_DIR, lab_systems, head_codes, code_to_loinc, "test", args.workers)

    n_distinct_mappable_loinc = len(train_occ)
    lspace = build_labelspace(train_occ, train_pat, test_occ, test_pat)
    print_report(lspace, train_tot, test_tot, test_occ, n_mappable_classes, n_distinct_mappable_loinc)
    write_outputs(lspace, train_occ, train_pat, test_occ, test_pat, train_tot, test_tot,
                  n_mappable_classes, n_distinct_mappable_loinc, Path(args.output_dir))
    print(f"\nWrote labelspace.csv + manifest.json to {args.output_dir}/")


if __name__ == "__main__":
    main()
