"""Tests for the LOINC six-axis decomposition in analysis/lib/loinc_axes.py, including:
  - a METHOD_TYP missing on both sides compares equal (a naive NaN != NaN would inflate the rarest axis);
  - severity classes are mutually exclusive and exhaustive;
  - the axis table stays row-aligned to the label space.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_S = importlib.util.spec_from_file_location(
    "loinc_axes", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "loinc_axes.py")
L = importlib.util.module_from_spec(_S)
_S.loader.exec_module(L)


def mk_table(rows):
    """rows: list of dicts keyed by axis; missing keys become NaN then the sentinel."""
    df = pd.DataFrame(rows)
    for c in L.AXES:
        if c not in df:
            df[c] = np.nan
        df[c] = df[c].fillna(L.MISSING).astype(str).str.strip().str.upper()
    df["loinc"] = [f"X{i}" for i in range(len(df))]
    return df


# -- the missing-value rule ---------------------------------------------------

def test_both_missing_method_compares_EQUAL():
    """NaN != NaN is True, so a naive comparison would read methods absent on both sides as 'method
    differs' on every such pair. That would inflate the rarest and most interesting axis."""
    t = mk_table([{"COMPONENT": "GLUCOSE"}, {"COMPONENT": "GLUCOSE"}])   # METHOD_TYP absent in both
    d = L.axis_differences(np.array([0]), np.array([1]), t)
    assert not d["METHOD_TYP"].iloc[0]


def test_one_missing_method_compares_DIFFERENT():
    t = mk_table([{"COMPONENT": "GLUCOSE", "METHOD_TYP": "AUTOMATED COUNT"},
                  {"COMPONENT": "GLUCOSE"}])
    d = L.axis_differences(np.array([0]), np.array([1]), t)
    assert d["METHOD_TYP"].iloc[0]


def test_axis_comparison_is_case_and_whitespace_insensitive():
    t = mk_table([{"COMPONENT": " glucose "}, {"COMPONENT": "GLUCOSE"}])
    d = L.axis_differences(np.array([0]), np.array([1]), t)
    assert not d["COMPONENT"].iloc[0]


# -- severity taxonomy --------------------------------------------------------

def test_severity_is_mutually_exclusive_first_rule_wins():
    """An error that differs on component and system counts as a component error only, never twice."""
    d = pd.DataFrame({"COMPONENT": [True], "SYSTEM": [True], "PROPERTY": [True],
                      "TIME_ASPCT": [True], "SCALE_TYP": [True], "METHOD_TYP": [True]})
    assert L.severity_class(d)[0] == "1_wrong_component"


def test_severity_ladder_each_rung():
    d = pd.DataFrame({
        "COMPONENT":  [True,  False, False, False],
        "SYSTEM":     [False, True,  False, False],
        "PROPERTY":   [False, False, True,  False],
        "TIME_ASPCT": [False, False, False, True],
        "SCALE_TYP":  [False, False, False, False],
        "METHOD_TYP": [False, False, False, False]})
    assert list(L.severity_class(d)) == ["1_wrong_component", "2_wrong_system",
                                         "3_wrong_property", "4_wrong_time_scale_method"]


def test_severity_is_exhaustive_over_real_error_rows():
    """Every row with at least one differing axis must receive a real class, never _unclassified."""
    rng = np.random.default_rng(0)
    m = rng.random((500, 6)) < 0.4
    m[~m.any(axis=1), 0] = True                      # make every row differ somewhere
    d = pd.DataFrame(m, columns=L.AXES)
    out = L.severity_class(d)
    assert "_unclassified" not in set(out)
    assert set(out) <= set(L.SEVERITY_LABELS)


def test_identical_on_all_axes_is_unclassified():
    d = pd.DataFrame({c: [False] for c in L.AXES})
    assert L.severity_class(d)[0] == "_unclassified"


def test_n_axes_differing_counts_correctly():
    d = pd.DataFrame({c: [i < 3] for i, c in enumerate(L.AXES)})
    assert L.n_axes_differing(d)[0] == 3


# -- table alignment (real file) ----------------------------------------------

def test_axis_table_is_row_aligned_to_the_label_space():
    """Downstream code indexes this table by class index, so a reorder would mislabel every error."""
    if not L.LOINC_CSV.exists():
        pytest.skip("Loinc.csv not present")
    # 2160-0 Creat SerPl-mCnc; 2345-7 Glucose SerPl-mCnc; 718-7 Hgb Bld-mCnc
    space = ["2160-0", "2345-7", "718-7"]
    t = L.load_axis_table(space)
    assert list(t["loinc"]) == space
    assert len(t) == len(space)


def test_axis_table_fills_missing_axes_with_the_sentinel():
    if not L.LOINC_CSV.exists():
        pytest.skip("Loinc.csv not present")
    t = L.load_axis_table(["2160-0", "NOT-A-REAL-LOINC"])
    assert t.loc[1, "COMPONENT"] == L.MISSING.upper()      # the unmatched row is still there
    assert list(t["loinc"]) == ["2160-0", "NOT-A-REAL-LOINC"]


# -- specimen sub-classification ----------------------------------------------

def test_serum_vs_blood_is_a_blood_compartment_synonym():
    t = mk_table([{"SYSTEM": "Ser/Plas"}, {"SYSTEM": "Bld"}])
    assert L.system_error_subclass(np.array([0]), np.array([1]), t)[0] == "blood_compartment_synonym"


def test_serum_vs_urine_is_a_different_specimen():
    t = mk_table([{"SYSTEM": "Ser/Plas"}, {"SYSTEM": "Urine"}])
    assert L.system_error_subclass(np.array([0]), np.array([1]), t)[0] == "different_specimen"


def test_body_fluid_pairs_are_different_specimens():
    t = mk_table([{"SYSTEM": "Body fld"}, {"SYSTEM": "Periton fld"}])
    assert L.system_error_subclass(np.array([0]), np.array([1]), t)[0] == "different_specimen"
