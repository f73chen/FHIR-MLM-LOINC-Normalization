"""Tests for resources/variability/source_tables.py: the transcribed tables are internally consistent.

  python -m pytest tests/test_source_tables.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "resources" / "variability"))
import source_tables as st  # noqa: E402


def test_friedberg_grid_is_internally_consistent():
    """Every published limit vector must be non-decreasing, with low < high at the median. A
    transcription slip from the PDF would almost certainly break one of these."""
    for key, (label, unit, n_labs, lows, highs) in st.FRIEDBERG2007_T5.items():
        assert len(lows) == len(highs) == len(st.FRIEDBERG2007_T5_PCTS), key
        assert list(lows) == sorted(lows), f"{key} low limits not monotonic"
        assert list(highs) == sorted(highs), f"{key} high limits not monotonic"
        assert lows[3] < highs[3], f"{key} median low >= median high"
        assert 100 < n_labs <= 163, key
