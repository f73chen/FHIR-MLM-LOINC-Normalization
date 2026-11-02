#!/usr/bin/env python
"""The Wilson interval and the training-frequency strata shared by the observation-level analyses
(thesis Chapter 6).

analysis/v3_observation.py, v3_offshelf.py and v3_singleobs.py import them. This module has no
command line of its own. The strata are cut on training frequency. Test frequency is not used,
because stratifying test accuracy by it would be circular.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score interval. Used instead of the normal approximation because the rare strata have
    small n, where the normal interval is badly wrong and can leave [0, 1]."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

HERE = Path(__file__).resolve().parent
_A = importlib.util.spec_from_file_location("a2", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)
_D = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
D = importlib.util.module_from_spec(_D); _D.loader.exec_module(D)


# log-spaced training-frequency strata, spanning the observed range of 10 to 1.15M
STRATA_EDGES = [10, 32, 100, 316, 1000, 3162, 10000, 31623, 100000, 10**7]
