"""Build the RELMA local-name pools from the real local test names in RELMA's Community Mappings.
The names replace the paraphrases in the out-of-distribution evaluation (thesis Chapters 4 and 6).

Writes data/relma/pool_<strategy>.json, {LOINC: {"names", "clean_same_loinc", "clean_global"}}, and a
manifest with coverage statistics. The pool has the same shape as the paraphrase pool, so the
paraphrase substitution code (`assign_paraphrases_per_patient`, `apply_paraphrases_to_resources`)
applies it unchanged.

Two naming strategies:
  s1  the local test name                       "HCT"
  s2  the local test name and its battery       "HCT | CBC"   (s1 when there is no battery)

Names that the models may have seen in training are kept and flagged. `clean_same_loinc` marks a name
that is not a training paraphrase of the same code. `clean_global` marks a name that is not a training
paraphrase of any code. Accuracy on unseen names comes from filtering the evaluation output by the flag.
This measures accuracy given that the drawn name was clean, which is the case for about 81% of
observations. Codes whose names collide more often contribute fewer observations, so report the macro
average next to the micro one.

The language match is case-insensitive because the source spells English both as 'English (US)' and
as 'eNGLISH (us)'. An exact match would drop 74 rows.

Codes without a RELMA name are left out of the pool. Their observations keep the MIMIC display text
and are not scored, but they still give context to neighbouring observations.

Reads the RELMA export and the paraphrase pool (no patient data) and prints aggregate counts only.

  python pipeline/17_relma_pool.py --strategy s1 --write
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

REPO = locations.REPO
RELMA_CSV = locations.UPSTREAM / "relma_custom_codes.csv"
LABELSPACE_CSV = REPO / "data" / "labelspace" / "labelspace.csv"
OUT_DIR = REPO / "data" / "relma"

STRATEGIES = ("s1", "s2")


def norm(s: str) -> str:
    """Normalise a name for the collision check only. The pool keeps names as written."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_english(language: str) -> bool:
    """Case-insensitive, because the source has both 'English (US)' and 'eNGLISH (us)'."""
    return "english" in (language or "").lower()


def build_name(row: dict, strategy: str) -> str | None:
    """The name for a row. s1 uses the local test name. s2 uses 'local test name | battery name', or
    the s1 name when there is no battery."""
    name = (row.get("Local Test Name") or "").strip()
    if not name:
        return None
    if strategy == "s1":
        return name
    battery = (row.get("Battery Name") or "").strip()
    return f"{name} | {battery}" if battery else name


def build_pool(rows, strategy: str, label_space: set[str]) -> dict[str, list[str]]:
    """{LOINC: sorted unique English names} for the codes in the label space. The names are sorted so
    that each patient's draw does not depend on the row order of the CSV."""
    pool = defaultdict(set)
    for r in rows:
        if not is_english(r.get("Language", "")):
            continue
        loinc = (r.get("LOINC Code") or "").strip()
        if loinc not in label_space:
            continue
        nm = build_name(r, strategy)
        if nm:
            pool[loinc].add(nm)
    return {k: sorted(v) for k, v in sorted(pool.items())}


def mark_clean(pool: dict[str, list[str]], train_paraphrases: dict[str, list[str]]) -> dict[str, dict]:
    """Flag each name with two checks. `clean_same_loinc` is the main check and is true when the name
    is not a training paraphrase of the same code. `clean_global` is stricter and is true when the name
    is not a training paraphrase of any code."""
    all_train = {norm(p) for v in train_paraphrases.values() for p in v}
    out = {}
    for loinc, names in pool.items():
        same = {norm(p) for p in train_paraphrases.get(loinc, [])}
        out[loinc] = {
            "names": names,
            "clean_same_loinc": [norm(n) not in same for n in names],
            "clean_global": [norm(n) not in all_train for n in names],
        }
    return out


def coverage_stats(marked: dict, label_space: set[str]) -> dict:
    n_names = sum(len(v["names"]) for v in marked.values())
    n_clean = sum(sum(v["clean_same_loinc"]) for v in marked.values())
    n_clean_g = sum(sum(v["clean_global"]) for v in marked.values())
    per = [len(v["names"]) for v in marked.values()]
    all_collide = [l for l, v in marked.items() if not any(v["clean_same_loinc"])]
    single = [l for l, v in marked.items() if len(v["names"]) == 1]
    return {
        "n_loinc_in_pool": len(marked),
        "n_loinc_label_space": len(label_space),
        "code_coverage_pct": round(100.0 * len(marked) / max(len(label_space), 1), 2),
        "n_names_total": n_names,
        "names_per_loinc_mean": round(n_names / max(len(marked), 1), 2),
        "names_per_loinc_min": min(per) if per else 0,
        "names_per_loinc_max": max(per) if per else 0,
        "n_loinc_single_name": len(single),
        "n_clean_same_loinc": n_clean,
        "pct_clean_same_loinc": round(100.0 * n_clean / max(n_names, 1), 2),
        "n_clean_global": n_clean_g,
        "pct_clean_global": round(100.0 * n_clean_g / max(n_names, 1), 2),
        "n_loinc_all_names_collide": len(all_collide),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "pipeline"))
    import importlib.util
    s = importlib.util.spec_from_file_location("d9", REPO / "pipeline" / "09_mlm_dataset.py")
    d9 = importlib.util.module_from_spec(s); s.loader.exec_module(d9)

    label_space = {r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV))
                   if r["in_primary_ge10occ"] == "1"}
    rows = list(csv.DictReader(open(RELMA_CSV)))
    n_en = sum(1 for r in rows if is_english(r.get("Language", "")))
    print(f"RELMA rows: {len(rows):,} | English (case-insensitive): {n_en:,}")

    # the clean flags are checked against the training paraphrases the models saw
    train_para, _, _, pstats = d9.load_paraphrases_3way(d9.PARAPHRASE_DIR, verbose=False)
    print(f"train paraphrase pool: {pstats['n_train']:,} strings over {pstats['n_loincs']} LOINC "
          f"(split_sha256={pstats['split_sha256'][:16]}…)")

    pool = build_pool(rows, args.strategy, label_space)
    marked = mark_clean(pool, train_para)
    stats = coverage_stats(marked, label_space)

    print(f"\n=== RELMA pool [{args.strategy}] ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"pool_{args.strategy}.json").write_text(json.dumps(marked, indent=1))
        (OUT_DIR / f"manifest_{args.strategy}.json").write_text(json.dumps({
            "strategy": args.strategy, "source_csv": str(RELMA_CSV),
            "label_space": "in_primary_ge10occ (679)",
            "paraphrase_split_sha256": pstats["split_sha256"],
            "english_match": "case-insensitive substring 'english' (recovers 'eNGLISH (us)')",
            "set_b": "clean_same_loinc = name is NOT a TRAIN paraphrase for the SAME LOINC (primary); "
                     "clean_global = not a TRAIN paraphrase for ANY LOINC (stricter sensitivity)",
            "stats": stats,
        }, indent=2))
        print(f"\nwrote {OUT_DIR/f'pool_{args.strategy}.json'} and manifest")


if __name__ == "__main__":
    main()
