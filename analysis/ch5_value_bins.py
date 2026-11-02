#!/usr/bin/env python
"""Count the lab items FHIR-MLM's tokenizer writes a value token for, and the distinct value tokens it
can produce (thesis Chapter 5).

The tokenizer writes a lab value as a percentile token. It uses a per-item bin table built from
MIMIC-IV, stored in `labevent_bins.pkl` with the other tokenizer files. Like the base checkpoint, this
file is PhysioNet-only. An item without a bin table emits no value token. The script counts the lab
items in the model's lab-code head, how many of them have a bin table (by number of bins), and how many
distinct percentile tokens those tables can produce.

Reads labevent_bins.pkl and codes_to_tokens.pkl from FHIR_MLM_TOKENIZER_DIR. Prints the counts and
writes them to results/analysis/ch5/value_bins.json.

  python analysis/ch5_value_bins.py
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

TOKENIZER_DIR = locations.TOKENIZER
LAB_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
OUT = Path(__file__).resolve().parent.parent / "results" / "analysis" / "ch5" / "value_bins.json"


def percentile_token(bucket: int, n_edges: int) -> int:
    """The percentile token for a value in `bucket`, by the tokenizer's percentile_bins rule (bucket centre, clipped)."""
    scale = 100 / n_edges
    return round(min(max(bucket * scale - scale / 2, 0), 100) + 1e-6)


def main():
    with open(TOKENIZER_DIR / "labevent_bins.pkl", "rb") as fh:
        bins = pickle.load(fh)
    with open(TOKENIZER_DIR / "codes_to_tokens.pkl", "rb") as fh:
        head = pickle.load(fh)[LAB_SYSTEM]
    in_head = set(bins) & set(head)                     # a binned item need not be a lab code the model predicts
    edges = Counter(len(v) for v in bins.values())
    tokens = {percentile_token(b, n) for n in edges for b in range(n + 1)}     # np.digitize gives buckets 0..n
    out = {"lab_items_in_head": len(head), "items_with_bins": len(bins), "head_items_with_bins": len(in_head),
           "head_items_without_bins": len(head) - len(in_head),
           "items_by_number_of_bins": {str(n + 1): c for n, c in sorted(edges.items())},
           "distinct_percentile_tokens": len(tokens)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
