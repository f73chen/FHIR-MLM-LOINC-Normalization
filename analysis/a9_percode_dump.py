#!/usr/bin/env python
"""The rank of the true LOINC in each local code's summed probability mass, for the catalogue
analysis (thesis Chapter 6).

analysis/v3_catalogue.py imports `gold_rank`. This module has no command line of its own.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_8 = importlib.util.spec_from_file_location("a8", HERE/"a8_catalogue.py")
A8 = importlib.util.module_from_spec(_8); _8.loader.exec_module(A8)
_9 = importlib.util.spec_from_file_location("a9", HERE/"a9_catalogue_deployment.py")
A9 = importlib.util.module_from_spec(_9); _9.loader.exec_module(A9)


def gold_rank(mass: np.ndarray, gold: np.ndarray, space: np.ndarray) -> np.ndarray:
    """1-based rank of the true LOINC in each code's summed-mass row, or len(space)+1 if absent.

    Kept apart from A9.code_scores, whose other callers want only the argmax. Zero-mass rows are
    set to rank last, to agree with A9.code_scores, which counts them wrong at top-1.
    """
    pos = {c: i for i, c in enumerate(space)}
    gi = np.array([pos.get(g, -1) for g in gold])
    out = np.full(len(gold), len(space) + 1, dtype=np.int32)
    rows = np.flatnonzero(gi >= 0)
    if len(rows):
        gm = mass[rows, gi[rows]]
        out[rows] = (mass[rows] > gm[:, None]).sum(axis=1) + 1
        out[rows[mass[rows].sum(axis=1) <= 0]] = len(space) + 1
    return out
