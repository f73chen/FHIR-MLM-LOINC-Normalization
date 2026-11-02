"""Write one plain-text concept file per LOINC code, the reference for the LLM review of the
paraphrases (thesis Chapter 4 and Appendix C).

Each file combines:
  - the code's LOINC LONG_COMMON_NAME (LoincTableCore), and
  - the verified descriptions from every coding system (loinc/mimic/cdisc/snomed/ciel/mth in
    UPSTREAM_DIR/loinc_descriptions_enriched.jsonl). The paraphrases were generated from these.
    The name and the descriptions define the accepted names of the concept, so the review does
    not flag a terse alias such as "CA-125" or "CA125AG" as garbage.

Aliases are de-duplicated (case-insensitive) and capped at MAX_ALIASES to keep the reviewer's context
small. Reads LOINC vocabulary only (no patient data). Writes data/paraphrases/concepts/{loinc}.txt.

  python pipeline/03b_build_concepts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ENRICHED = locations.UPSTREAM / "loinc_descriptions_enriched.jsonl"
LOINC_TABLE = locations.UPSTREAM / "Loinc_2.82/LoincTableCore/LoincTableCore.csv"
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
OUTPUT_DIR = locations.REPO / "data/paraphrases/concepts"

BUCKET_ORDER = ["loinc", "mimic", "cdisc", "snomed", "ciel", "mth"]
MAX_ALIASES = 15


def dedup_aliases(descriptions: dict, canonical: str) -> list[str]:
    seen = {canonical.strip().lower()} if canonical else set()
    out: list[str] = []
    for bucket in BUCKET_ORDER + [b for b in descriptions if b not in BUCKET_ORDER]:
        for d in descriptions.get(bucket, []):
            d = d.strip()
            k = d.lower()
            if d and k not in seen:
                seen.add(k)
                out.append(d)
            if len(out) >= MAX_ALIASES:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loinc-list", default=None,
                    help="file of LOINC codes, one per line (default: every code in "
                         "data/labelspace/labelspace.csv)")
    args = ap.parse_args()
    used = ([l.strip() for l in open(args.loinc_list) if l.strip()]
            if args.loinc_list else [r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV))])
    long_name = {r["LOINC_NUM"]: r.get("LONG_COMMON_NAME", "") for r in csv.DictReader(open(LOINC_TABLE))}
    enriched = {}
    with open(ENRICHED) as f:
        for line in f:
            d = json.loads(line)
            enriched[d["loinc_code"]] = d

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_enriched = []
    for loinc in used:
        e = enriched.get(loinc, {})
        meta = e.get("metadata", {})
        descs = e.get("descriptions", {})
        canonical = long_name.get(loinc) or (descs.get("loinc", [""])[0] if descs.get("loinc") else "")
        if not e:
            missing_enriched.append(loinc)
        aliases = dedup_aliases(descs, canonical)
        lines = [
            f"LOINC {loinc}",
            f"Canonical name: {canonical or '(unknown)'}",
            f"Component: {meta.get('loinc_component','?')} | System: {meta.get('loinc_system','?')} | "
            f"Property: {meta.get('loinc_property','?')} | Scale: {meta.get('loinc_scale_type','?')} | "
            f"Method: {meta.get('loinc_method_type','') or '-'}",
            "Verified real descriptions of THIS test (paraphrases were generated from these; "
            "treat as the accepted concept family):",
        ]
        lines += [f"- {a}" for a in aliases] or ["- (none beyond canonical)"]
        with open(OUTPUT_DIR / f"{loinc}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    print(f"wrote {written} concept files to {OUTPUT_DIR}/")
    if missing_enriched:
        print(f"  WARNING: {len(missing_enriched)} LOINC missing from enriched (canonical-only): {missing_enriched[:5]}")


if __name__ == "__main__":
    main()
