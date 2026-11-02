"""Tests for pipeline/17_relma_pool.py on synthetic fixtures (no patient data, no real CSV).

They guard the choices that could corrupt the RELMA evaluation:
  * case-insensitive English matching (an exact match drops 74 real rows labelled 'eNGLISH (us)');
  * a deterministic name order (the draw for a patient must not depend on the row order of the CSV);
  * the clean flags, for same-code versus any-code collisions with training paraphrases;
  * the restriction to the label space.
"""
import importlib.util
from pathlib import Path

import pytest

_S = importlib.util.spec_from_file_location(
    "relma_pool", Path(__file__).resolve().parents[1] / "pipeline" / "17_relma_pool.py")
rp = importlib.util.module_from_spec(_S)
_S.loader.exec_module(rp)


def _row(loinc="1234-5", name="HCT", battery="", lang="English (US)"):
    return {"LOINC Code": loinc, "Local Test Name": name, "Battery Name": battery, "Language": lang}


# -- English matching -------------------------------------------------------
@pytest.mark.parametrize("lang", ["English (US)", "eNGLISH (us)", "ENGLISH", "english (us)"])
def test_is_english_case_insensitive(lang):
    assert rp.is_english(lang)


@pytest.mark.parametrize("lang", ["Spanish (ES)", "Estonian (EE)", "German (DE)", "", None])
def test_is_english_rejects_others(lang):
    assert not rp.is_english(lang)


def test_case_variant_rows_are_not_dropped():
    """An exact `== 'English (US)'` match would drop 74 real rows."""
    rows = [_row(name="A", lang="English (US)"), _row(name="B", lang="eNGLISH (us)")]
    pool = rp.build_pool(rows, "s1", {"1234-5"})
    assert pool["1234-5"] == ["A", "B"]


# -- name construction ------------------------------------------------------
def test_s1_is_name_only():
    assert rp.build_name(_row(name="HCT", battery="CBC"), "s1") == "HCT"


def test_s2_appends_battery():
    assert rp.build_name(_row(name="HCT", battery="CBC"), "s2") == "HCT | CBC"


def test_s2_falls_back_to_name_when_no_battery():
    assert rp.build_name(_row(name="HCT", battery=""), "s2") == "HCT"
    assert rp.build_name(_row(name="HCT", battery=None), "s2") == "HCT"


def test_blank_name_is_dropped():
    assert rp.build_name(_row(name=""), "s1") is None
    assert rp.build_name(_row(name="   "), "s1") is None


# -- pool construction ------------------------------------------------------
def test_pool_restricted_to_label_space():
    rows = [_row(loinc="1111-1", name="A"), _row(loinc="9999-9", name="B")]
    pool = rp.build_pool(rows, "s1", {"1111-1"})
    assert set(pool) == {"1111-1"}


def test_pool_dedups_and_sorts_deterministically():
    """The assignment draw depends on list order, so the order must not follow the CSV's row order."""
    rows = [_row(name="Zeta"), _row(name="Alpha"), _row(name="Zeta"), _row(name="Mid")]
    pool = rp.build_pool(rows, "s1", {"1234-5"})
    assert pool["1234-5"] == ["Alpha", "Mid", "Zeta"]
    shuffled = [_row(name="Mid"), _row(name="Zeta"), _row(name="Alpha")]
    assert rp.build_pool(shuffled, "s1", {"1234-5"}) == pool


# -- clean flags ------------------------------------------------------------
def test_clean_same_loinc_flags_only_same_code_collisions():
    pool = {"A": ["seen", "novel"], "B": ["seen"]}
    train = {"A": ["SEEN"]}                      # collides with A only, case-insensitively
    m = rp.mark_clean(pool, train)
    assert m["A"]["clean_same_loinc"] == [False, True]
    assert m["B"]["clean_same_loinc"] == [True], "same string under a different LOINC is not same-LOINC leakage"


def test_clean_global_is_stricter_than_same_loinc():
    pool = {"B": ["seen"]}
    train = {"A": ["seen"]}
    m = rp.mark_clean(pool, train)
    assert m["B"]["clean_same_loinc"] == [True]
    assert m["B"]["clean_global"] == [False]


def test_collision_detection_normalises_whitespace_and_case():
    m = rp.mark_clean({"A": ["  Hema  Tocrit "]}, {"A": ["hema tocrit"]})
    assert m["A"]["clean_same_loinc"] == [False]


def test_norm_does_not_alter_emitted_names():
    """Normalisation is for the collision check only. The substituted string must stay verbatim."""
    pool = rp.build_pool([_row(name="  HCT (auto) ")], "s1", {"1234-5"})
    assert pool["1234-5"] == ["HCT (auto)"], "outer strip only; internal casing/spacing preserved"


# -- coverage stats ---------------------------------------------------------
def test_coverage_stats_counts():
    marked = rp.mark_clean({"A": ["x", "y"], "B": ["z"]}, {"A": ["x"]})
    st = rp.coverage_stats(marked, {"A", "B", "C", "D"})
    assert st["n_loinc_in_pool"] == 2
    assert st["code_coverage_pct"] == 50.0
    assert st["n_names_total"] == 3
    assert st["n_clean_same_loinc"] == 2          # y and z
    assert st["n_loinc_single_name"] == 1         # B
    assert st["n_loinc_all_names_collide"] == 0


def test_all_names_collide_is_detected():
    marked = rp.mark_clean({"A": ["x"]}, {"A": ["x"]})
    assert rp.coverage_stats(marked, {"A"})["n_loinc_all_names_collide"] == 1
