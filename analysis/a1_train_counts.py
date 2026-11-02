#!/usr/bin/env python
"""Write the per-LOINC training observation counts for the 679-code label space,
results/analysis/a1/train_loinc_counts.tsv.

v3_observation and v3_singleobs read the file to stratify codes by how often they occur in training
(T-11, T-34, F-02; thesis Chapter 6). The counts come from the label space itself. They are the
`train_occ` column of data/labelspace/labelspace.csv, written by pipeline/02_derive_labelspace.py,
for the codes in the primary space (`in_primary_ge10occ`).

The file has one line per code, `<count>\\t<loinc>`, sorted by count and then by LOINC, both
descending. It holds aggregate counts only. The script will not overwrite the file without --force.
With --check it compares the file instead and writes nothing.

    python analysis/a1_train_counts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELSPACE = REPO / "data" / "labelspace" / "labelspace.csv"
OUT = REPO / "results" / "analysis" / "a1" / "train_loinc_counts.tsv"


def render() -> str:
    ls = pd.read_csv(LABELSPACE)
    prim = ls[ls.in_primary_ge10occ.astype(bool)]
    rows = sorted(zip(prim.train_occ.astype(int), prim.loinc.astype(str)), reverse=True)
    return "".join(f"{n}\t{loinc}\n" for n, loinc in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the existing file and write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()
    text = render()
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print(f"{OUT}: {'identical' if same else 'DIFFERS'} ({text.count(chr(10))} codes)")
        return 0 if same else 1
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} exists; pass --force to overwrite, or --check to compare")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({text.count(chr(10))} codes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
