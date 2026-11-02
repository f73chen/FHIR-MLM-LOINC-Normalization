#!/usr/bin/env python
"""Count the observations that the first BCM window covers with RELMA names and with paraphrases, on
the v3 representation (thesis Chapter 4).

RELMA names tokenize shorter than paraphrases, so the text models cover slightly more observations
with them. RELMA names exist on the perturbed arm only, so the comparison is perturbed RELMA names
against perturbed paraphrases. The clean paraphrase set is counted too.

The count is the labelled positions (labels != -100) in the first window of each BCM test dataset.
This is the quantity behind the "First 8,192 tokens" column of T-26.

Reads the BCM datasets in WORK_DIR/datasets. Writes aggregate counts to
results/analysis/ch4/relma_coverage_v3.json and refuses to overwrite it.

The datasets were written with the `datasets` library of the deep-models environment. The older version
in the data-analysis environment cannot read that format, so the script runs in the deep-models
environment:

  python analysis/ch4_relma_coverage_v3.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

BASE = str(locations.WORK / "datasets")
OUT = Path(__file__).resolve().parent.parent / "results" / "analysis" / "ch4" / "relma_coverage_v3.json"
SETS = {"v3_clean_para": "bcm__para_xsite__clean",
        "v3_pert_para": "bcm__para_xsite__pert", "v3_pert_relma": "bcm__relma_s1_a0__para_xsite__pert"}


def labelled_positions(name):
    import datasets
    import pyarrow.compute as pc
    t = datasets.load_from_disk(f"{BASE}/{name}")["test"].data.table
    flat = t.column("labels").combine_chunks().flatten()
    return int(pc.sum(pc.not_equal(flat, -100)).as_py()), t.num_rows


def main(write=True):
    res = {"method": "labelled positions in BCM's first window, test split (see docstring)", "counts": {}}
    for key, name in SETS.items():
        n, rows = labelled_positions(name)
        res["counts"][key] = {"dataset": name, "test_patients": rows, "labelled_positions": n}
        print(f"{key:14s} {name:38s} patients {rows:>6,}  positions {n:>9,}")
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        assert not OUT.exists(), f"{OUT} exists (not overwriting)"
        OUT.write_text(json.dumps(res, indent=2))
        print("wrote", OUT)
    return res


if __name__ == "__main__":
    main()
