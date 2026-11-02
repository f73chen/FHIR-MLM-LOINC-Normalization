"""Tests for analysis/a20_unit_code_concentration.py.

Two things are checked, on cases with known answers:
  * concentration() counts distinct codes. A row count would make every high-volume unit look
    like a shortcut.
  * The LOINC-wide usage control separates "this unit names the analyte" from "this unit is
    unique only inside the 679-code label space". The per-unit verdicts depend on it.

  python -m pytest tests/test_a20_unit_concentration.py -q
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "pipeline"))
_s = importlib.util.spec_from_file_location(
    "a20", ROOT / "analysis" / "a20_unit_code_concentration.py")
a20 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(a20)
from input_preprocessing_utils import load_unit_alias_map  # noqa: E402

AMAP = load_unit_alias_map(str(locations.UPSTREAM / "unit_synonym_map.json"))


def _df(rows):
    """rows: (unit, loinc, item, pid)"""
    return pd.DataFrame(rows, columns=["std", "loinc", "item_code", "pid"])


def test_concentration_counts_distinct_codes_not_rows():
    """A unit with 1,000 observations of one code is a one-code unit. Counting rows instead of
    distinct codes would hide the case that the analysis is looking for."""
    d = _df([("pg", "785-6", "51248", i) for i in range(1000)] +   # MCH RBC Qn Auto
            [("mg/dL", "2160-0", "50912", i) for i in range(5)] +  # Creat SerPl-mCnc
            [("mg/dL", "2345-7", "50931", i) for i in range(5)])   # Glucose SerPl-mCnc
    t = a20.concentration(d, "std").set_index("unit")
    assert t.loc["pg", "n_loinc"] == 1 and t.loc["pg", "n_obs"] == 1000
    assert t.loc["mg/dL", "n_loinc"] == 2 and t.loc["mg/dL", "n_obs"] == 10


def test_concentration_counts_distinct_patients():
    d = _df([("pg", "785-6", "51248", p) for p in [1, 1, 1, 2, 2, 3]])
    t = a20.concentration(d, "std").set_index("unit")
    assert t.loc["pg", "n_obs"] == 6 and t.loc["pg", "n_patients"] == 3


def test_concentration_separates_loinc_from_mimic_item():
    """Several MIMIC items can map to one LOINC. The two label spaces must not be conflated."""
    # 787-2 MCV RBC Auto; 21000-5 RDW RBC Auto
    d = _df([("fL", "787-2", "51250", 1), ("fL", "787-2", "51000", 2), ("fL", "21000-5", "51277", 3)])
    t = a20.concentration(d, "std").set_index("unit")
    assert t.loc["fL", "n_loinc"] == 2 and t.loc["fL", "n_mimic_item"] == 3


# ------------------- the label-space-artefact control -------------------
def test_loinc_wide_usage_is_populated_and_discriminating():
    """The control must separate rare units from common ones. If every unit came back with a
    similar count, it could not support any per-unit verdict."""
    wide = a20.loinc_wide_unit_usage()
    assert len(wide) > 200, "LOINC example-unit index looks empty"
    assert len(wide.get("mg/dL", ())) > 50, "mg/dL should be used by many LOINC codes"
    assert 0 < len(wide.get("pg", ())), "pg should appear in LOINC's example units at all"
    # and it must discriminate: a common unit far outnumbers a rare one
    assert len(wide["mg/dL"]) > len(wide.get("pg", ())) * 2


def test_kind_delegates_to_a19_and_is_stable():
    """a20 takes the kind rule from a19, so its partition matches the one the models see."""
    assert a20._kind("mg/dL") == a20._kind("mg/dL")
    assert a20._kind("g/dL") == a20._kind("mg/dL")          # same kind, different scale
    assert a20._kind("mmol/L") != a20._kind("mg/dL")        # mass vs molar
    assert a20._kind("[IU]/L") != a20._kind("/L")           # the IU trap
    assert a20._kind("not-a-unit") is None


def test_kind_table_is_built_once():
    a20._kind("mg/dL")
    first = a20._UTAB
    a20._kind("g/dL")
    assert a20._UTAB is first, "UnitTable rebuilt per call, which would slow the analysis down"


# ------------------- external attestation -------------------
def test_external_attestation_covers_the_label_space_and_has_verdicts():
    ext = a20.external_attestation(AMAP)
    assert len(ext) == 679, f"expected the 679 label space, got {len(ext)}"
    assert set(ext.verdict) <= {"kind_agrees", "kind_differs", "no_external_unit", "no_mimic_unit"}
    # the control has to tell units apart, so it must give more than one verdict
    assert ext.verdict.nunique() >= 2
    comp = ext[ext.verdict.isin(["kind_agrees", "kind_differs"])]
    assert len(comp) > 100, "too few comparable codes for the claim to mean anything"


# ------------------- disagreement classification -------------------
Z = (0, 0, 0, 0, 0, 0, 0)
MASS_VOL = (-3, 0, 1, 0, 0, 0, 0)      # mg/dL
MOLE_VOL = (-3, 0, 0, 0, 0, 0, 0)      # mmol/L


@pytest.mark.parametrize("a,b,expect", [
    (((MASS_VOL), 0.0, ""), ((MASS_VOL), 0.0, ""), "identical"),
    (((MOLE_VOL), 0.0, ""), ((MOLE_VOL), 0.0, "[IU]"), "atom class only"),
    (((MASS_VOL), 0.0, ""), ((MOLE_VOL), 1.0, ""), "mass vs molar"),
    (((Z), 0.0, ""), ((MASS_VOL), 0.0, ""), "dimensionless vs dimensioned"),
])
def test_classify_pair(a, b, expect):
    assert a20.classify_pair(a, b) == expect


def test_classify_pair_names_the_changed_dimension():
    a = ((0, 0, 1, 0, 0, 0, 0), 0.0, "")      # pg, a mass
    b = ((3, 0, 0, 0, 0, 0, 0), 0.0, "")      # fL, a volume
    assert a20.classify_pair(a, b) == "dimension differs: length+mass"


def test_best_pair_picks_the_closest_not_the_first():
    """A code with several attested units must be classified by its closest match. Otherwise a code
    that differs only in its analyte-dependent unit is reported as a dimensional disagreement,
    because some other attested unit happens to be far away."""
    mine = {((MOLE_VOL), 0.0, ""): {"10*3/uL"}}
    theirs = {((MASS_VOL), 0.0, ""): {"mg/dL"},          # far
              ((MOLE_VOL), 0.0, "[IU]"): {"[IU]/L"}}     # near
    label, ka, kb = a20.best_pair(mine, theirs)
    assert label == "atom class only" and kb[2] == "[IU]"


def test_external_self_agreement():
    k1 = ((MASS_VOL), 0.0, "")
    k2 = ((MOLE_VOL), 1.0, "")
    assert a20.external_self_agreement({"loinc": {k1: 1}, "ucum": {k1: 1}}) == "all_agree"
    assert a20.external_self_agreement({"loinc": {k1: 1}, "ucum": {k2: 1}}) == "disjoint"
    assert a20.external_self_agreement({"loinc": {k1: 1, k2: 1}, "ucum": {k1: 1}}) == "overlap"
    assert a20.external_self_agreement({"loinc": {k1: 1}}) == "single_system"
    assert a20.external_self_agreement({}) == "no_system"


def test_attestation_carries_a_disagreement_class_for_every_disagreement():
    ext = a20.external_attestation(AMAP)
    dis = ext[ext.verdict == "kind_differs"]
    assert len(dis) > 0
    assert dis.disagreement_class.notna().all(), "a disagreement with no class is unexplained"
    assert (ext[ext.verdict == "kind_agrees"].disagreement_class.isna()).all()
