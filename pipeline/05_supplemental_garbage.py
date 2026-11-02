"""Remove the paraphrases that are garbage for any LOINC code, after the LLM review (thesis Chapter 4
and Appendix C).

The LLM review keeps a line when in doubt, so a few strings from the cross-file garbage list
(UPSTREAM_DIR/shared_paraphrase_labels.csv, is_garbage == 1) survive it. This step removes only the
categories that are garbage in any LOINC context, by exact normalized match ("MCnc" alone is dropped,
"Sodium MCnc" is untouched):
  - axis-code    (acnc, mcnc, mfr, ...)        LOINC property codes, never a standalone description
  - vocab-id     (mimic, ciel)                 dataset and vocabulary names
  - prompt-leak  (mimic ldh, name nfr, ...)    artifacts of the generation prompt

Context-dependent strings are kept and left for human review: single characters ("k" for potassium)
and manual-exact fragments (body, free, color, ...), which can be valid for specific codes.

Edits data/paraphrases/clean/ in place and updates the manifest of the pool. It also writes
supplemental_removals.csv there, so every removal can be audited or reversed. Fails if a code would keep
fewer than COVERAGE_FLOOR paraphrases.

  python pipeline/05_supplemental_garbage.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

CLEAN_DIR = locations.REPO / "data/paraphrases/clean"
GARBAGE_LABELS = locations.UPSTREAM / "shared_paraphrase_labels.csv"
SAFE_PREFIXES = ("axis-code", "vocab-id", "prompt-leak")
COVERAGE_FLOOR = 20


def normalize(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", s).lower()


def filter_safe_garbage(lines: list[str], safe_set: set[str]):
    """Remove lines whose normalized form equals a safe-garbage string. Returns
    (kept, removed). Only exact matches count, so substrings of real descriptions are untouched."""
    kept, removed = [], []
    for l in lines:
        (removed if normalize(l) in safe_set else kept).append(l)
    return kept, removed


def load_safe_set(path: Path) -> set[str]:
    out = set()
    for r in csv.DictReader(open(path)):
        if r.get("is_garbage") == "1" and r.get("reason", "").startswith(SAFE_PREFIXES):
            out.add(normalize(r["paraphrase"]))
    return out


def main() -> None:
    safe = load_safe_set(GARBAGE_LABELS)
    print(f"safe-garbage strings (axis-code/vocab-id/prompt-leak): {len(safe)}")
    rows, n_in, n_out = [], 0, 0
    per_loinc_kept = {}
    for f in sorted(glob.glob(str(CLEAN_DIR / "*.txt"))):
        loinc = Path(f).stem
        lines = [l for l in Path(f).read_text().split("\n") if l != ""]
        kept, removed = filter_safe_garbage(lines, safe)
        n_in += len(lines)
        n_out += len(kept)
        per_loinc_kept[loinc] = len(kept)
        for r in removed:
            rows.append((loinc, r))
        if removed:
            Path(f).write_text("\n".join(kept) + ("\n" if kept else ""))

    floor = min(per_loinc_kept.values())
    worst = min(per_loinc_kept, key=per_loinc_kept.get)
    print(f"supplemental: {n_in:,} -> {n_out:,} lines (removed {n_in - n_out})")
    print(f"  coverage: min {floor} ({worst})")
    assert floor >= COVERAGE_FLOOR, f"coverage floor violated: {worst}={floor}"
    with open(CLEAN_DIR / "supplemental_removals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["loinc", "line"])
        w.writerows(rows)
    # update manifest
    mp = CLEAN_DIR / "manifest.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    m["supplemental_safe_garbage_removed"] = n_in - n_out
    m["lines_out_final"] = n_out
    m["min_paraphrases_final"] = floor
    mp.write_text(json.dumps(m, indent=2))
    print(f"  wrote supplemental_removals.csv; final pool = {n_out:,} lines")


if __name__ == "__main__":
    main()
