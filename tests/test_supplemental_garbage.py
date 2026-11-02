"""Tests for the exact, normalized-match filtering in pipeline/05_supplemental_garbage.py."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "supp", Path(__file__).resolve().parents[1] / "pipeline" / "05_supplemental_garbage.py"
)
sg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sg)

SAFE = {"mcnc", "mimic", "mimic ldh", "name, nfr"}


def test_exact_match_only():
    lines = [
        "MCnc",                 # exact axis-code -> remove
        "Sodium MCnc",          # contains mcnc but not exactly -> keep
        "MIMIC",                # vocab-id -> remove
        "MIMIC LDH",            # prompt-leak -> remove
        "Lactate dehydrogenase",  # real -> keep
        "name, nfr",            # prompt-leak -> remove
    ]
    kept, removed = sg.filter_safe_garbage(lines, SAFE)
    assert kept == ["Sodium MCnc", "Lactate dehydrogenase"]
    assert sorted(removed) == sorted(["MCnc", "MIMIC", "MIMIC LDH", "name, nfr"])


def test_normalize_handles_quotes_ws():
    assert sg.normalize('  "MCnc"  ') == "mcnc"
    kept, removed = sg.filter_safe_garbage(['"MCnc"', "K"], SAFE)  # K not in SAFE -> kept
    assert kept == ["K"]
    assert removed == ['"MCnc"']
