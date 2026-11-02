#!/usr/bin/env python
"""Count what the upper bound of the patient filter removes, and whether the removed patients carry codes
that the training split lacks (thesis Chapter 4).

The eligibility rule is LabObservation >= 1 and 3 < total_resources <= 8192 (pipeline/01_build_split.py).
Its upper bound removes patients with very long records. Part 1 counts them and their share of lab
observations. Part 2 counts their LOINC-mappable codes against the label space, using `scan_split`
from pipeline/02_derive_labelspace.py, the same counting that the label space was built with.

Reads UPSTREAM_DIR/loinc_patients_summary.csv, the ndjson files of the removed patients and
data/labelspace/labelspace.csv. Writes aggregates only, to results/analysis/ch4/excluded_patients_codes.txt,
and refuses to overwrite it. A LOINC code is listed only with at least 10 observations.

  python analysis/ch4_excluded_patients_codes.py
"""
import csv
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/ch4/excluded_patients_codes.txt"
SUMMARY = str(locations.UPSTREAM / "loinc_patients_summary.csv")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def main():
    assert not OUT.exists(), f"{OUT} exists (not overwriting)"
    # -- part 1: the filter ----------------------------------------------------------------------------
    df = pd.read_csv(SUMMARY, usecols=["filename", "total_resources", "LabObservation"])
    other = df[(df.LabObservation >= 1) & (df.total_resources > 3)]
    elig, ex = other[other.total_resources <= 8192], other[other.total_resources > 8192]
    say(f"patients in the summary: {len(df):,} | eligible: {len(elig):,} | removed by the upper bound: {len(ex):,} "
        f"= {100 * len(ex) / len(other):.2f}% of the otherwise-eligible")
    say(f"their lab observations: {ex.LabObservation.sum():,} of {other.LabObservation.sum():,} "
        f"= {100 * ex.LabObservation.sum() / other.LabObservation.sum():.1f}%")

    # -- part 2: their codes against the label space --------------------------------------------------
    spec = importlib.util.spec_from_file_location("ls02", ROOT / "pipeline/02_derive_labelspace.py")
    L = importlib.util.module_from_spec(spec); sys.modules["ls02"] = L; spec.loader.exec_module(L)
    sys.path.insert(0, L.FHIR_UTILS_DIR)
    from fhir_data_utils import build_lab_code_mappings, LAB_CODE_SYSTEM
    m2c, c2l = build_lab_code_mappings(L.CODES_TO_TOKENS_PATH, L.SEEN_CODES_PATH)
    code_to_loinc = L.compose_code_to_loinc(m2c, c2l)
    uuids = [f[:-7] if f.endswith(".ndjson") else f for f in ex.filename]
    t0 = time.time()
    occ, pat, tot = L.scan_split(uuids, L.NDJSON_DIR, frozenset({LAB_CODE_SYSTEM}), set(m2c), code_to_loinc,
                                 "excluded", workers=6)
    say(f"scanned in {time.time() - t0:.0f}s | patients {tot['n_patients']:,} (missing {tot['n_missing']}) | "
        f"lab obs {tot['n_lab_obs']:,} | in-head {tot['n_head_obs']:,} | LOINC-mapped {tot['n_mapped_obs']:,}")
    ls = {r["loinc"]: r for r in csv.DictReader(open(ROOT / "data/labelspace/labelspace.csv"))}
    primary = {l for l, r in ls.items() if r["in_primary_ge10occ"] == "1"}
    seen = set(occ)
    a, b, c = seen & primary, (seen & set(ls)) - primary, seen - set(ls)
    say(f"distinct LOINC-mappable codes in the removed patients: {len(seen)} (of {len(set(c2l.values()))} mappable)")
    say(f"  in the 679-code label space (>= 10 train occurrences): {len(a)} -> {sum(occ[l] for l in a):,} obs")
    say(f"  in training but < 10 times (outside the label space):  {len(b)} -> {sum(occ[l] for l in b):,} obs")
    say(f"  never in the training split:                           {len(c)} -> {sum(occ[l] for l in c):,} obs")
    say("  share of their mapped observations outside the label space: "
        f"{100 * sum(occ[l] for l in b | c) / max(sum(occ.values()), 1):.3f}%")
    cross = sorted([l for l in b | c if int(ls.get(l, {}).get("train_occ", 0)) + occ[l] >= 10], key=lambda l: -occ[l])
    say(f"codes outside the label space that reach >= 10 with these patients added (an upper bound): {len(cross)}")
    for l in cross:
        if occ[l] >= 10:
            say(f"    {l}: removed-patient obs {occ[l]:,} in {pat[l]:,} patients | train occ {ls.get(l, {}).get('train_occ', 0)}")
    say(f"  ({sum(1 for l in cross if occ[l] < 10)} more have < 10 such observations and are not listed)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
