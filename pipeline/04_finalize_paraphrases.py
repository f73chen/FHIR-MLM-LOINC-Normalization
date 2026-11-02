"""Apply the garbage removals confirmed by the LLM review (part B) to the part A pool and write the
final cleaned paraphrase pool (thesis Chapter 4 and Appendix C).

A removal is applied only if line `line_no` (1-based) of the part A pool still equals the reported
verbatim `text`. This guards against a miscounted line number. Unmatched removals are logged and
skipped.

Then:
  - coverage check: every code keeps at least COVERAGE_FLOOR paraphrases;
  - recall cross-check: how many of the documented garbage strings (shared_paraphrase_labels.csv,
    is_garbage == 1) part B caught, and how many still survive in the final pool.

Inputs:  --removals-json (the review's result, {loinc_results: [{loinc, removed: [{line_no, text}]}]},
         or a directory of per-code files), data/paraphrases/partA_pool/,
         UPSTREAM_DIR/shared_paraphrase_labels.csv
Outputs: data/paraphrases/clean/{loinc}.txt, manifest.json and removals.csv (unmatched.csv when a
         removal did not match). No patient data.

  python pipeline/04_finalize_paraphrases.py --removals-json <review result>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

PARTA_POOL_DIR = locations.REPO / "data/paraphrases/partA_pool"
GARBAGE_LABELS = locations.UPSTREAM / "shared_paraphrase_labels.csv"
OUTPUT_DIR = locations.REPO / "data/paraphrases/clean"
COVERAGE_FLOOR = 20


# --- Pure, unit-testable logic ----------------------------------------------
def apply_removals(lines: list[str], removals: list[dict]):
    """Remove a line only if lines[line_no-1] still matches the reported text (stripped).
    Returns (kept, matched, unmatched)."""
    remove_idx: set[int] = set()
    matched, unmatched = [], []
    for r in removals:
        i = int(r["line_no"]) - 1
        if 0 <= i < len(lines) and lines[i].strip() == str(r["text"]).strip():
            remove_idx.add(i)
            matched.append(r)
        else:
            unmatched.append(r)
    kept = [l for j, l in enumerate(lines) if j not in remove_idx]
    return kept, matched, unmatched


def normalize(s: str) -> str:
    import re
    s = s.strip().strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", s).lower()


# --- Real-data IO (main only) -----------------------------------------------
def load_removals(removals_path: Path) -> dict[str, list[dict]]:
    """Accept either a directory of per-LOINC {loinc, removed} json files (the partB_removals
    directory from part B) or a single json containing {loinc_results: [...]}. Directory mode
    tolerates a partial or interrupted run and skips malformed files."""
    p = Path(removals_path)
    if p.is_dir():
        out: dict[str, list[dict]] = {}
        n_bad = 0
        for f in sorted(p.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                n_bad += 1
                continue
            if d.get("loinc"):
                out[d["loinc"]] = d.get("removed", [])
        if n_bad:
            print(f"  WARNING: skipped {n_bad} malformed per-LOINC result files")
        return out
    data = json.loads(p.read_text())
    results = data.get("loinc_results", data if isinstance(data, list) else [])
    return {r["loinc"]: r.get("removed", []) for r in results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--removals-json", required=True)
    ap.add_argument("--parta-pool-dir", default=str(PARTA_POOL_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()
    parta_pool = Path(args.parta_pool_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    removals_by_loinc = load_removals(Path(args.removals_json))
    files = sorted(parta_pool.glob("*.txt"))
    totals = {"loinc": 0, "lines_in": 0, "lines_out": 0, "matched": 0, "unmatched": 0}
    per_loinc_kept = {}
    removal_rows = []
    unmatched_rows = []

    for path in files:
        loinc = path.stem
        lines = [l for l in path.read_text().split("\n") if l != ""]  # partA pool has no blanks
        kept, matched, unmatched = apply_removals(lines, removals_by_loinc.get(loinc, []))
        totals["loinc"] += 1
        totals["lines_in"] += len(lines)
        totals["lines_out"] += len(kept)
        totals["matched"] += len(matched)
        totals["unmatched"] += len(unmatched)
        per_loinc_kept[loinc] = len(kept)
        for r in matched:
            removal_rows.append((loinc, r["line_no"], r["text"]))
        for r in unmatched:
            unmatched_rows.append((loinc, r.get("line_no"), r.get("text")))
        (out_dir / f"{loinc}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))

    floor = min(per_loinc_kept.values())
    worst = min(per_loinc_kept, key=per_loinc_kept.get)
    print(f"finalize: {totals['lines_in']:,} -> {totals['lines_out']:,} lines "
          f"(removed {totals['matched']:,} matched; {totals['unmatched']:,} unmatched/skipped)")
    print(f"  coverage: min {floor} ({worst}); median {sorted(per_loinc_kept.values())[len(per_loinc_kept)//2]}")
    assert floor >= COVERAGE_FLOOR, f"coverage floor violated: {worst} has {floor} (< {COVERAGE_FLOOR})"

    # Recall cross-check vs the documented garbage list.
    recall = {"documented_garbage": 0, "still_in_pool": 0, "removed_by_partB": 0}
    if GARBAGE_LABELS.exists():
        known = {normalize(r["paraphrase"]) for r in csv.DictReader(open(GARBAGE_LABELS))
                 if r.get("is_garbage") == "1"}
        final_strings = set()
        for path in out_dir.glob("*.txt"):
            final_strings |= {normalize(l) for l in path.read_text().split("\n") if l.strip()}
        recall["documented_garbage"] = len(known)
        recall["still_in_pool"] = len(known & final_strings)
        recall["removed_by_partB"] = len(known - final_strings)
        print(f"  recall x-check: of {len(known)} documented-garbage strings, "
              f"{recall['removed_by_partB']} now absent, {recall['still_in_pool']} still present")

    manifest = {
        "parta_pool_dir": str(parta_pool), "clean_dir": str(out_dir),
        "lines_in": totals["lines_in"], "lines_out": totals["lines_out"],
        "removed_matched": totals["matched"], "removed_unmatched_skipped": totals["unmatched"],
        "coverage_floor": COVERAGE_FLOOR, "min_paraphrases": floor, "min_loinc": worst,
        "recall_cross_check": recall,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(out_dir / "removals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loinc", "line_no", "text"])
        w.writerows(removal_rows)
    if unmatched_rows:
        with open(out_dir / "unmatched.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["loinc", "line_no", "text"])
            w.writerows(unmatched_rows)
    print(f"\nWrote cleaned pool + manifest + removals to {out_dir}/")


if __name__ == "__main__":
    main()
