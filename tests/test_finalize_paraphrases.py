"""Tests for the removal matching in pipeline/04_finalize_paraphrases.py: a removal applies only when
both its line_no and its text match.

  python -m pytest tests/test_finalize_paraphrases.py -q
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "finalize", Path(__file__).resolve().parents[1] / "pipeline" / "04_finalize_paraphrases.py"
)
fz = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fz)

LINES = ["Sodium, Serum", "Glucose}}}", "CBC", "the output is well-formed"]


def test_apply_removals_exact_match():
    kept, matched, unmatched = fz.apply_removals(
        LINES, [{"line_no": 2, "text": "Glucose}}}"}, {"line_no": 4, "text": "the output is well-formed"}])
    assert kept == ["Sodium, Serum", "CBC"]
    assert len(matched) == 2 and len(unmatched) == 0


def test_apply_removals_text_mismatch_is_skipped():
    # line_no points at a different line than the reported text, so nothing is removed (safety).
    kept, matched, unmatched = fz.apply_removals(LINES, [{"line_no": 1, "text": "Glucose}}}"}])
    assert kept == LINES                      # nothing removed
    assert len(matched) == 0 and len(unmatched) == 1


def test_apply_removals_out_of_range_is_skipped():
    kept, matched, unmatched = fz.apply_removals(LINES, [{"line_no": 99, "text": "whatever"}])
    assert kept == LINES
    assert len(unmatched) == 1


def test_apply_removals_whitespace_tolerant():
    kept, matched, _ = fz.apply_removals(LINES, [{"line_no": 3, "text": "  CBC  "}])
    assert kept == ["Sodium, Serum", "Glucose}}}", "the output is well-formed"]
    assert len(matched) == 1
