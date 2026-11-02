#!/usr/bin/env python
"""Count how many real RELMA names a MedGemma paraphrase reproduces exactly: a realism check on the
paraphrases (thesis Chapter 4).

The question is what MedGemma produced, so the headline compares the names with every split of the
paraphrase pool and leaves out matches with the grounding strings MedGemma was given. The
training-split rows give the narrower comparison that a leakage check needs. They show which names the
models saw in training, including the grounding strings.

Method:
    A match is exact for the same LOINC code after lower-casing and collapsing whitespace (the
    normalisation of `17_relma_pool.norm`). The RELMA names are `data/relma/pool_s1.json`. The pool is
    every split of `data/paraphrases/clean` (train + val + test). A match is "grounding" when the string
    is one of the code's "Verified real descriptions" in `data/paraphrases/concepts/<loinc>.txt`, and
    "generated only" otherwise. The headline is generated-only matches against the whole pool.

Writes aggregate counts only, to results/analysis/ch4/relma_paraphrase_overlap.json, and refuses to
overwrite it without --force. RELMA names and paraphrases are terminology strings, with no patient
data.

  python analysis/ch4_relma_paraphrase_overlap.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = REPO / "results" / "analysis" / "ch4" / "relma_paraphrase_overlap.json"

_s = importlib.util.spec_from_file_location("d9", REPO / "pipeline" / "09_mlm_dataset.py")
d9 = importlib.util.module_from_spec(_s); _s.loader.exec_module(d9)


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())


def grounding(loinc: str) -> set[str]:
    """The code's grounding strings: the bullets under "Verified real descriptions" in its concept file."""
    p = REPO / "data" / "paraphrases" / "concepts" / f"{loinc}.txt"
    if not p.exists():
        return set()
    out, on = set(), False
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("Verified real descriptions"):
            on = True
            continue
        if on and ln.startswith("- "):
            out.add(norm(ln[2:]))
        elif on and ln.strip() and not ln.startswith("- "):
            on = False
    return out


def count() -> Counter:
    pool = json.load(open(REPO / "data" / "relma" / "pool_s1.json"))
    train, val, test, _ = d9.load_paraphrases_3way(d9.PARAPHRASE_DIR, verbose=False)
    own = {}
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r["LOINC_NUM"] in pool:
                own[r["LOINC_NUM"]] = {norm(r.get(k, "")) for k in
                                       ("SHORTNAME", "DisplayName", "LONG_COMMON_NAME", "CONSUMER_NAME") if r.get(k)}
    c = Counter()
    for loinc, v in pool.items():
        g = grounding(loinc)
        tr = {norm(x) for x in train.get(loinc, [])}
        whole = tr | {norm(x) for x in val.get(loinc, []) + test.get(loinc, [])}
        names = own.get(loinc, set())
        for name in v["names"]:
            n = norm(name)
            c["names"] += 1
            for scope, ref in (("train", tr), ("pool", whole)):
                if n in ref:
                    c[f"{scope}_match"] += 1
                    kind = "grounding" if n in g else "generated_only"
                    c[f"{scope}_{kind}"] += 1
                    c[f"{scope}_{kind}_loinc_own_name"] += n in names
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()
    c = count()
    n = c["names"]
    res = {"relma_names": n, "paraphrase_dir": str(d9.PARAPHRASE_DIR), "scopes": {}}
    for scope in ("train", "pool"):
        res["scopes"][scope] = {k: {"n": c[f"{scope}_{k}"], "pct": round(100 * c[f"{scope}_{k}"] / n, 2)}
                                for k in ("match", "grounding", "generated_only")}
        res["scopes"][scope]["loinc_own_name_matches"] = (c[f"{scope}_grounding_loinc_own_name"]
                                                          + c[f"{scope}_generated_only_loinc_own_name"])
    head = res["scopes"]["pool"]["generated_only"]
    res["headline"] = f"{head['n']:,} of {n:,} RELMA names ({head['pct']:.2f}%) exactly match a generated paraphrase"
    print(json.dumps(res, indent=2))
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; pass --force to overwrite")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
