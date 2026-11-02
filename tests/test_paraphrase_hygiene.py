"""Tests for pipeline/03_paraphrase_hygiene.py, the structural clean-up of the paraphrase pool (Part A),
on synthetic strings.

  python -m pytest tests/test_paraphrase_hygiene.py -q
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "hygiene", Path(__file__).resolve().parents[1] / "pipeline" / "03_paraphrase_hygiene.py"
)
hy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hy)


def test_normalize():
    assert hy.normalize('  "Sodium, Serum"  ') == "sodium, serum"
    assert hy.normalize("HEMOGLOBIN   A1c") == "hemoglobin a1c"


def test_structural_garbage_flags():
    assert hy.is_structural_garbage("") == "empty"
    assert hy.is_structural_garbage("   ") == "empty"
    assert hy.is_structural_garbage("----") == "punct_only"
    assert hy.is_structural_garbage("[[[[====") == "punct_only"
    assert hy.is_structural_garbage("血红蛋白") == "non_latin"
    assert hy.is_structural_garbage("x" * 151) == "too_long"


def test_partA_is_conservative_defers_to_partB():
    # Part A removes only garbage with zero false positives. Everything else is kept here and
    # left to the Part B LLM net: obvious code, markdown/template/HTML cruft, corrupted but
    # recognizable terms and valid '%'-prefixed lab terms.
    for keep in ["%Hct", "%RBC", "%A1c", "%Hb A1c",            # valid percent lab terms: a regression guard
                 "C<sub>3</sub> complement assay",              # valid concept w/ HTML formatting
                 "def get_component():", "```python",           # obvious code -> Part B
                 "Testost (Serum)}}", "**Sodium**",             # corrupted/markdown -> Part B
                 "component = loinc_axes.get('Component')"]:     # subtle code -> Part B
        assert hy.is_structural_garbage(keep) is None, keep


def test_structural_garbage_keeps_valid():
    # realistic lab descriptions must not be flagged
    for good in ["Sodium [Moles/volume] in Serum", "CBC", "Hemoglobin A1c", "Glucose, CSF",
                 "Potassium level (blood)", "lytes"]:
        assert hy.is_structural_garbage(good) is None, good


def test_clean_loinc_dedup_canonical_garbage():
    canonical = "Sodium [Moles/volume] in Serum or Plasma"
    lines = [
        "Sodium, Serum\n",
        "sodium,  serum\n",                       # dup of above (normalized)
        "Sodium [Moles/volume] in Serum or Plasma\n",  # == canonical -> strip
        "def f():\n",                              # obvious code -> kept (deferred to Part B)
        "Salt level in blood\n",
        "----\n",                                  # punct only -> removed
    ]
    kept, removals = hy.clean_loinc(lines, canonical)
    assert kept == ["Sodium, Serum", "def f():", "Salt level in blood"]
    reasons = sorted(r for _, r in removals)
    assert reasons == ["duplicate", "equals_canonical", "punct_only"]


def test_clean_loinc_no_canonical():
    kept, removals = hy.clean_loinc(["Glucose\n", "Glucose\n"], None)
    assert kept == ["Glucose"]
    assert [r for _, r in removals] == ["duplicate"]
