"""Structural clean-up of the paraphrase pool, restricted to the LOINC codes in the label space
(thesis Chapter 4 and Appendix C).

This is the cheap rule-based pass, part A in the output names. The thorough per-line check is part B,
an LLM review of the output of this script. pipeline/04_finalize_paraphrases.py applies the part B removals.
Removals here are high-precision only. When a line might be valid, part A keeps it and leaves the
decision to part B.
  1. drop empty and whitespace-only lines
  2. de-duplicate within each code (normalized: lowercased, outer quotes and spaces stripped)
  3. drop lines equal to the code's canonical LONG_COMMON_NAME, so the paraphrased condition never
     shows the canonical name
  4. drop only structural garbage that cannot be a valid line: empty, pure punctuation, non-Latin
     scripts (CJK, Cyrillic), longer than MAX_LEN. Code, markdown, template and HTML cruft and
     corrupted but recognizable terms are left to part B, because part A should not drop a valid
     line, such as the percent terms "%Hct" and "%A1c".

Checks (asserted):
  - leakage: the real train/val paraphrase split, re-run on the cleaned pool, puts no string in both
    train and val for any code;
  - coverage: every code keeps at least COVERAGE_FLOOR paraphrases.

Reads UPSTREAM_DIR/paraphrases_ext_verified/ (the pool), data/labelspace/labelspace.csv and
LoincTableCore.csv. Writes data/paraphrases/partA_pool/ (one txt per code, the input to part B),
partA_removals.csv and manifest.json. LOINC text and counts only, no patient data.

  python pipeline/03_paraphrase_hygiene.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

POOL_DIR = locations.UPSTREAM / "paraphrases_ext_verified"
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
LOINC_TABLE = locations.UPSTREAM / "Loinc_2.82/LoincTableCore/LoincTableCore.csv"
FHIR_UTILS_DIR = str(locations.COMMON)
OUTPUT_DIR = locations.REPO / "data/paraphrases/partA_pool"

MAX_LEN = 150          # MAX_PARAPHRASE_LEN in prep/4.2 (p99 of real long common names is about 134)
COVERAGE_FLOOR = 20    # every used LOINC must retain at least this many paraphrases

# Only structural-garbage patterns with no false positives. Code, markdown and template cruft and
# corrupted but recognizable terms are left to the part B LLM review.
_NONLATIN = re.compile(r"[一-鿿Ѐ-ӿ]")          # CJK / Cyrillic (never a valid English lab name)
_PUNCT_ONLY = re.compile(r"^[\W_]+$")            # all non-word chars (digits are \w, so kept)


# --- Pure, unit-testable logic ----------------------------------------------
def normalize(s: str) -> str:
    """Normalization key for dedup + canonical compare: strip, drop outer quotes, lower, collapse ws."""
    s = s.strip().strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", s).lower()


def is_structural_garbage(line: str) -> str | None:
    """Return a reason string if `line` is obvious structural garbage, else None.
    High precision only: when unsure, return None and let part B decide."""
    s = line.strip()
    if not s:
        return "empty"
    if len(s) > MAX_LEN:
        return "too_long"
    if _PUNCT_ONLY.match(s):
        return "punct_only"
    if _NONLATIN.search(s):
        return "non_latin"
    return None


def clean_loinc(lines: list[str], canonical: str | None):
    """Apply structural hygiene to one LOINC's lines. Returns (kept, removals) where
    removals is a list of (line, reason). Order of `kept` preserves first appearance."""
    removals: list[tuple[str, str]] = []
    kept: list[str] = []
    seen: set[str] = set()
    canon_norm = normalize(canonical) if canonical else None
    for raw in lines:
        line = raw.rstrip("\n")
        reason = is_structural_garbage(line)
        if reason:
            removals.append((line, reason))
            continue
        key = normalize(line)
        if canon_norm is not None and key == canon_norm:
            removals.append((line, "equals_canonical"))
            continue
        if key in seen:
            removals.append((line, "duplicate"))
            continue
        seen.add(key)
        kept.append(line.strip())
    return kept, removals


# --- Real-data IO (main only) -----------------------------------------------
def load_used_loincs(labelspace_csv: Path) -> list[str]:
    with open(labelspace_csv) as f:
        return [r["loinc"] for r in csv.DictReader(f)]


def load_canonical_names(loinc_table: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(loinc_table, newline="") as f:
        for r in csv.DictReader(f):
            out[r["LOINC_NUM"]] = r.get("LONG_COMMON_NAME", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--loinc-list", default=None,
                    help="file of LOINC codes (one per line) to process instead of every code in "
                         "data/labelspace/labelspace.csv")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)

    used = ([l.strip() for l in open(args.loinc_list) if l.strip()]
            if args.loinc_list else load_used_loincs(LABELSPACE_CSV))
    canon = load_canonical_names(LOINC_TABLE)
    print(f"used LOINC: {len(used)}   canonical names loaded: {len(canon)}")
    missing_canon = [l for l in used if not canon.get(l)]
    if missing_canon:
        print(f"  WARNING: {len(missing_canon)} used LOINC lack a canonical name (e.g. {missing_canon[:5]})")

    removal_rows: list[tuple[str, str, str]] = []
    totals = {"loinc": 0, "lines_in": 0, "lines_out": 0}
    by_reason: dict[str, int] = {}
    per_loinc_kept: dict[str, int] = {}
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)

    for loinc in used:
        path = POOL_DIR / f"{loinc}.txt"
        with open(path) as f:
            lines = f.readlines()
        kept, removals = clean_loinc(lines, canon.get(loinc))
        totals["loinc"] += 1
        totals["lines_in"] += len(lines)
        totals["lines_out"] += len(kept)
        per_loinc_kept[loinc] = len(kept)
        for line, reason in removals:
            removal_rows.append((loinc, line, reason))
            by_reason[reason] = by_reason.get(reason, 0) + 1
        if not args.no_write:
            with open(out_dir / f"{loinc}.txt", "w") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))

    # Coverage assert.
    floor = min(per_loinc_kept.values())
    worst = min(per_loinc_kept, key=per_loinc_kept.get)
    print(f"\nstructural hygiene: {totals['lines_in']:,} -> {totals['lines_out']:,} lines "
          f"(removed {totals['lines_in']-totals['lines_out']:,})")
    print(f"  removals by reason: {dict(sorted(by_reason.items(), key=lambda kv: -kv[1]))}")
    print(f"  coverage: min {floor} paraphrases ({worst}); median "
          f"{sorted(per_loinc_kept.values())[len(per_loinc_kept)//2]}")
    assert floor >= COVERAGE_FLOOR, f"coverage floor violated: {worst} has only {floor} (< {COVERAGE_FLOOR})"

    # Leakage assert: re-run the real train/val split on the cleaned pool and check that no
    # string is in both.
    if not args.no_write:
        sys.path.insert(0, FHIR_UTILS_DIR)
        from paraphrase_utils import load_paraphrases_train_val_split
        train_para, val_para, _ = load_paraphrases_train_val_split(
            str(out_dir), test_fraction=0.2, seed=42, verbose=False)
        leaks = 0
        for loinc in train_para:
            inter = {normalize(s) for s in train_para[loinc]} & {normalize(s) for s in val_para.get(loinc, [])}
            leaks += len(inter)
        print(f"  leakage check (train∩val paraphrase strings): {leaks} (expected 0)")
        assert leaks == 0, f"FAIL: {leaks} paraphrase strings appear in both train and val"

        manifest = {
            "pool_in": str(POOL_DIR), "partA_pool_out": str(out_dir),
            "used_loinc": len(used), "lines_in": totals["lines_in"], "lines_out": totals["lines_out"],
            "removed_by_reason": by_reason, "coverage_floor": COVERAGE_FLOOR,
            "min_paraphrases": floor, "min_loinc": worst,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(out_dir / "partA_removals.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["loinc", "line", "reason"])
            w.writerows(removal_rows)
        print(f"\nWrote partA pool + removals + manifest to {out_dir}/")


if __name__ == "__main__":
    main()
