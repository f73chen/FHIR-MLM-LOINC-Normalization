"""Tests for pipeline/unit_standardisation.py, the unit and number rendering of the v2 and v3
representations.

The module converts a unit to the standard unit of its kind (g/dL to mg/dL, for example) and leaves
other units as they are. Many tests below are refusals, each a case where a purely dimensional rule
would convert and be wrong, as with [IU]/L to /L.

The shipped table is a hand-frozen constant. test_frozen_table_matches_the_kind_rule re-derives
every entry from the kind rule and compares.

  python -m pytest tests/test_unit_standardisation.py -q
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import unit_standardisation as us  # noqa: E402
from input_preprocessing_utils import load_unit_alias_map  # noqa: E402

AMAP = load_unit_alias_map(str(locations.UPSTREAM / "unit_synonym_map.json"))


# ----------------------- the frozen table vs the rule that made it -----------------------
def test_frozen_table_matches_the_kind_rule():
    """Re-derive every entry from the kind rule and assert the shipped table agrees."""
    spec = importlib.util.spec_from_file_location(
        "a19", ROOT / "analysis" / "a19_scale_only_standardisation.py")
    a19 = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "analysis"))
    spec.loader.exec_module(a19)
    import pandas as pd
    kinds = pd.read_csv(ROOT / "results" / "analysis" / "a19" / "a19_kinds.csv")
    U = a19.a16.UnitTable()
    # the table is built from corrected canonical units, so the re-derivation must correct too
    import collections
    counts = collections.Counter()
    for u, n in zip(kinds.unit, kinds.n_obs):
        counts[us.correct_canonical(u)] += int(n)
    counts = dict(counts)
    targets, _ = a19.choose_targets(U, counts, "modal")

    assert set(counts) == set(us.STANDARD_UNIT), "table covers a different unit set than a19 saw"
    for unit in counts:
        tgt = targets[a19.unit_kind(U, unit)]
        fac = a19.scale_factor(U, unit, tgt)
        if fac is None:
            tgt, fac = unit, 1.0
        got_t, got_f = us.STANDARD_UNIT[unit]
        assert got_t == tgt, f"{unit}: frozen target {got_t!r} != derived {tgt!r}"
        # rel=1e-6 because the frozen table snaps exact powers of ten: the raw magnitude ratio for
        # mL -> fL is 999999999999.9999 and the published table must say 1e12
        assert got_f == pytest.approx(fac, rel=1e-6), f"{unit}: frozen factor {got_f} != {fac}"


def test_table_is_internally_consistent():
    for unit, (tgt, fac) in us.STANDARD_UNIT.items():
        assert fac > 0
        if us.MAX_LOG10_FACTOR is not None:
            assert abs(math.log10(fac)) <= us.MAX_LOG10_FACTOR + 1e-9, f"{unit} exceeds the cap"
        assert tgt in us.STANDARD_UNIT, f"{unit} -> {tgt}, which is not itself a known unit"
        assert us.STANDARD_UNIT[tgt][0] == tgt, f"{tgt} is a target but does not map to itself"
        if tgt == unit:
            assert fac == 1.0


# ----------------------- conversions that must happen -----------------------
@pytest.mark.parametrize("raw,target,factor", [
    ("g/dL", "mg/dL", 1000), ("mg/dL", "mg/dL", 1), ("ug/dL", "mg/dL", 0.001),
    ("10*12/L", "10*3/uL", 1000), ("10*3/uL", "10*3/uL", 1), ("/uL", "10*3/uL", 0.001),
    ("u[IU]/mL", "[IU]/L", 0.001), ("[IU]/mL", "[IU]/L", 1000),
    ("umol/L", "mmol/L", 0.001), ("mL/min", "L/min", 0.001),
])
def test_allowed_scale_changes(raw, target, factor):
    t, f = us.standardise_unit(raw, AMAP)
    assert t == target and f == pytest.approx(factor, rel=1e-9)


def test_raw_surface_forms_resolve_through_the_alias_map():
    """The input is the raw MIMIC string, so mixed case and spaces must work."""
    for raw in ("G/DL", "g/dl", "mm hg", "K/uL"):
        t, f = us.standardise_unit(raw, AMAP)
        assert t in us.STANDARD_TARGETS or t == raw, f"{raw!r} -> {t!r}"


# ----------------------- refusals -----------------------
@pytest.mark.parametrize("raw,why", [
    ("[IU]/L", "IU is dimensionless; a dimensional rule would fold it into a plain count"),
    ("U/L", "enzyme units are arbitrary"),
    ("meq/L", "equivalents depend on valence, which the string does not carry"),
    ("%", "dimensionless"),
])
def test_kinds_that_must_stay_in_their_own_family(raw, why):
    t, _ = us.standardise_unit(raw, AMAP)
    assert t != "/L" and t != "mmol/L", f"{raw} -> {t}: {why}"


def test_mass_never_becomes_molar():
    for raw in ("mg/dL", "g/dL", "ng/mL", "pg/mL"):
        t, _ = us.standardise_unit(raw, AMAP)
        assert "mol" not in t, f"{raw} -> {t} crosses mass to molar (a different LOINC code)"


def test_large_rescales_are_ALLOWED_the_cap_was_dropped():
    """There is no cap on the rescale factor (MAX_LOG10_FACTOR is None). A cap of 6 on its log10
    raises unit-alone identifiability by 0.10 pp and adds one single-code unit. Its readability
    rationale also does not hold under fixed-width scientific rendering, where 5.000e+14 has the
    same token shape as 5.000e+00."""
    assert us.standardise_unit("mL", AMAP) == ("fL", 1e12)
    assert us.standardise_unit("pg/mL", AMAP) == ("mg/dL", 1e-07)


def test_power_of_ten_factors_are_exact():
    """A published table that a site implements must say 1e12. 999999999999.9999 has to be snapped."""
    for unit in us.STANDARD_UNIT:
        f = us.STANDARD_UNIT[unit][1]
        lg = math.log10(f)
        if abs(lg - round(lg)) < 1e-6:
            assert f == 10.0 ** round(lg), f"{unit}: factor {f!r} is not an exact power of ten"


def test_non_decimal_factors_are_NOT_snapped():
    """min -> s is x60 and h -> s is x3600. These are exact integers that are not powers of ten, so
    the power-of-ten snap must leave them alone."""
    assert us.STANDARD_UNIT["min"] == ("s", 60.0)
    assert us.STANDARD_UNIT["h"] == ("s", 3600.0)


def test_log_scale_units_are_never_rescaled():
    for raw in ("log10 IU/mL", "log10 copies/mL", "LOG10 IU/ML", "ln U/L"):
        assert us.is_log_scale(raw)
        t, f = us.standardise_unit(raw, AMAP)
        assert (t, f) == (raw, 1.0), f"{raw} was rescaled"


def test_unknown_and_empty_units_pass_through():
    # `ratio` is in the alias map (-> "RATIO") but is not a real UCUM unit, so it has no entry in
    # STANDARD_UNIT and must pass through unscaled. The canonical spelling still applies, as it
    # does in v1.
    assert us.standardise_unit("ratio", AMAP) == ("RATIO", 1.0)
    assert us.standardise_unit("n/a", AMAP)[1] == 1.0
    assert us.standardise_unit("", AMAP) == ("", 1.0)
    assert us.standardise_unit(None, AMAP) == ("", 1.0)


# ----------------------- rendering -----------------------
@pytest.mark.parametrize("v,expect", [
    (1.1, "1.100e+00"), (140, "1.400e+02"), (0.0000003, "3.000e-07"),
    (7.0, "7.000e+00"), (0, "0.000e+00"), (-4.567, "-4.567e+00"), (12.345, "1.235e+01"),   # float 12.345 sits just above the tie
])
def test_render_value(v, expect):
    assert us.render_value(v) == expect


def test_every_rendered_value_matches_the_gate_pattern():
    for v in (1.1, 140, 0.0000003, 7e9, 0, -4.567, 1e-30, 1e30):
        assert us.VALUE_RE.match(us.render_value(v)), us.render_value(v)


def test_render_value_rejects_non_numeric():
    for v in (None, "", "positive", "NEGATIVE", float("nan"), float("inf")):
        assert us.render_value(v) == ""


def test_rendering_is_lossless_to_four_sig_figs():
    for v in (1.1, 140.0, 1.126, 0.0250, 1.015, 98765.0):
        assert float(us.render_value(v)) == pytest.approx(us.round_sig(v), rel=1e-12)


def test_urine_specific_gravity_survives():
    """Why SIG_FIGS is 4. At 3 s.f. these six values give only four distinct strings:
    1.010 and 1.015 both render as 1.01e+00, and 1.020 and 1.025 as 1.02e+00."""
    vals = [1.005, 1.010, 1.015, 1.020, 1.025, 1.030]
    assert len({us.render_value(v) for v in vals}) == len(vals)
    assert len({us.render_value(v, nsf=3) for v in vals}) < len(vals)


# ----------------------- the whole transform -----------------------
def test_standardise_quantity_scales_value_and_unit_together():
    txt, unit = us.standardise_quantity(12.5, "g/dL", AMAP)
    assert unit == "mg/dL" and txt == "1.250e+04"          # 12.5 g/dL == 12500 mg/dL


def test_standardise_quantity_round_trips():
    for value, raw in [(12.5, "g/dL"), (7.3, "10*3/uL"), (140, "meq/L"), (1.015, "")]:
        txt, unit = us.standardise_quantity(value, raw, AMAP)
        _, factor = us.standardise_unit(raw, AMAP)
        assert float(txt) / factor == pytest.approx(us.round_sig(value), rel=1e-9)


def test_standardise_quantity_passes_qualitative_text_through():
    txt, unit = us.standardise_quantity("POSITIVE", "", AMAP)
    assert txt == "POSITIVE" and unit == ""


def test_unitless_values_are_still_rendered():
    """16.3% of observations carry no unit, and they must still get the uniform numeric grammar."""
    txt, unit = us.standardise_quantity(1.2, "", AMAP)
    assert txt == "1.200e+00" and unit == ""


# ----------------------- idempotence -----------------------
def test_the_alias_map_is_NOT_idempotent_on_its_own_outputs():
    """Documents why the gate cannot check idempotence. These are canonical forms that the alias
    map re-canonicalises to something else, so re-feeding a correctly emitted unit changes it."""
    for unit, recanon in [("h", "H"), ("{#}/[HPF]", "/[HPF]")]:
        assert unit in us.STANDARD_UNIT
        assert us.canonical_unit(unit, AMAP) == recanon != unit


def test_every_output_of_a_real_raw_unit_is_legitimate():
    """The invariant that the gate checks. Whatever the module emits for a real MIMIC surface form
    must pass is_legitimate_output."""
    raws = ["g/dL", "mg/dL", "K/uL", "10*3/uL", "sec", "mm hg", "%", "meq/L", "IU/L", "pg",
            "fL", "ratio", "log10 cop/mL", "ng/ml feu", "n/a", "min", "h", "mL"]
    for raw in raws:
        out, _ = us.standardise_unit(raw, AMAP)
        assert us.is_legitimate_output(out, AMAP), f"{raw!r} -> {out!r} judged illegitimate"


def test_is_legitimate_output_still_catches_a_real_miss():
    """It must not be vacuously true. An unconverted g/dL is the failure that the gate exists for,
    and it must be rejected."""
    assert not us.is_legitimate_output("g/dL", AMAP)
    assert not us.is_legitimate_output("10*12/L", AMAP)
    assert us.is_legitimate_output("mg/dL", AMAP)
    assert us.is_legitimate_output("log10 cop/mL", AMAP)
    assert us.is_legitimate_output("RATIO", AMAP)
    assert us.is_legitimate_output("[S]", AMAP)


def test_log10_viral_load_is_not_rescaled_despite_canonicalising_into_the_table():
    """The alias map maps `log10 cop/mL` to `/mL`, discarding the log10 marker, and `/mL` is in the
    table as `-> 10*3/uL x1e-06`. Without the log-scale guard a log10 viral load would be
    relabelled a cell count and multiplied by a millionth."""
    assert us.canonical_unit("log10 cop/mL", AMAP) == "/mL"
    assert us.STANDARD_UNIT["/mL"] == ("10*3/uL", 1e-06)
    assert us.standardise_unit("log10 cop/mL", AMAP) == ("log10 cop/mL", 1.0)
    txt, unit = us.standardise_quantity(3.5, "log10 cop/mL", AMAP)
    assert unit == "log10 cop/mL" and txt == "3.500e+00"


# ----------------------- the alias-map corrections -----------------------
def test_AU_is_corrected_to_the_real_UCUM_arbitrary_unit():
    """MIMIC writes `AU` to mean arbitrary units. UCUM has no bare `AU`, and lab2clean parses it as
    the astronomical unit (1.496e11 m, a length) on 433,001 observations. It is corrected to
    `[arb'U]`, which is dimensionless as it should be."""
    assert us.ALIAS_MAP_CORRECTIONS["AU"] == "[arb'U]"
    assert "AU" not in us.STANDARD_UNIT, "the raw AU key should be gone after correction"
    assert us.STANDARD_UNIT["[arb'U]"] == ("[arb'U]", 1.0)
    assert us.standardise_unit("units", AMAP) == ("[arb'U]", 1.0)


def test_sec_is_corrected_from_the_SVEDBERG_to_the_SECOND():
    """`[S]` is the svedberg (1e-13 s, a sedimentation coefficient). The alias map sends MIMIC's
    `sec` there, on 921,788 PT/aPTT/thrombin observations, while LOINC, UCUM and CDISC all attest
    `s`/`sec` for those codes."""
    assert us.ALIAS_MAP_CORRECTIONS["[S]"] == "s"
    assert "[S]" not in us.STANDARD_UNIT
    for form in ("sec", "second", "seconds", "SEC"):
        assert us.standardise_unit(form, AMAP) == ("s", 1.0), form


def test_the_correction_unifies_the_time_family():
    """With `[S]` corrected, seconds, minutes and hours share one kind: PT, aPTT, thrombin time and
    urine collection duration land on one unit instead of two."""
    assert us.STANDARD_UNIT["s"] == ("s", 1.0)
    assert us.STANDARD_UNIT["min"] == ("s", 60.0)
    assert us.STANDARD_UNIT["h"] == ("s", 3600.0)


def test_corrections_do_not_touch_anything_else():
    for u in ("mg/dL", "g/dL", "10*3/uL", "%", "meq/L", "[IU]/L", "pg", "fL"):
        assert us.correct_canonical(u) == u


# ----------------------- v3: the reference range travels with the value -----------------------
# v2 has no reference range, so only the value is standardised. v3 shows the range. In the corpus,
# 12.25% of observations carry a unit whose factor is not 1, and nearly all of them also carry a
# range. An unstandardised bound would print the haemoglobin value at 1000x its own range, and that
# mismatch would itself identify the code.
RR_HB = [{"low": {"value": 12.0, "unit": "g/dL"}, "high": {"value": 16.0, "unit": "g/dL"}}]


def test_range_is_rescaled_by_the_SAME_factor_as_the_value():
    val, vunit = us.standardise_quantity(14.0, "g/dL", AMAP)
    bounds = us.standardise_reference_range(RR_HB, "g/dL", AMAP)
    assert vunit == "mg/dL"
    assert [(l, t, u) for l, t, u in bounds] == [
        ("Low", "1.200e+04", "mg/dL"), ("High", "1.600e+04", "mg/dL")]
    assert us.range_is_consistent(vunit, bounds)
    # value and range stay on one scale
    assert float(val) / float(bounds[0][1]) == pytest.approx(14.0 / 12.0)


def test_a_bound_with_NO_unit_inherits_the_value_unit():
    """2.23% of bounds carry no unit. Without the fallback they take factor 1.0 and are emitted
    unscaled beside a scaled value, the same defect this module exists to prevent."""
    bare = [{"low": {"value": 12.0}, "high": {"value": 16.0}}]
    bounds = us.standardise_reference_range(bare, "g/dL", AMAP)
    assert [t for _l, t, _u in bounds] == ["1.200e+04", "1.600e+04"]
    assert all(u == "mg/dL" for _l, _t, u in bounds)


def test_range_consistency_check_actually_catches_the_v2_defect():
    """The check must reject a bound left in its raw unit beside a standardised value."""
    _val, vunit = us.standardise_quantity(14.0, "g/dL", AMAP)
    unstandardised = [("Low", "12.0", "g/dL"), ("High", "16.0", "g/dL")]
    assert not us.range_is_consistent(vunit, unstandardised)


def test_factor_one_units_pass_through_unchanged():
    val, vunit = us.standardise_quantity(140.0, "mmol/L", AMAP)
    rr = [{"low": {"value": 135.0, "unit": "mmol/L"}, "high": {"value": 145.0, "unit": "mmol/L"}}]
    bounds = us.standardise_reference_range(rr, "mmol/L", AMAP)
    assert vunit == "mmol/L" and us.range_is_consistent(vunit, bounds)
    assert [t for _l, t, _u in bounds] == ["1.350e+02", "1.450e+02"]


def test_empty_and_missing_ranges_are_not_errors():
    assert us.standardise_reference_range([], "mg/dL", AMAP) == []
    assert us.standardise_reference_range(None, "mg/dL", AMAP) == []
    assert us.standardise_reference_range([{}], "mg/dL", AMAP) == []
    # a one-sided range emits only the bound it has
    one = us.standardise_reference_range([{"high": {"value": 5.0, "unit": "mmol/L"}}],
                                         "mmol/L", AMAP)
    assert [l for l, _t, _u in one] == ["High"]


def test_every_emitted_range_unit_is_a_legitimate_output():
    """The same rule the build gate applies to values must hold for bounds."""
    for raw in ("g/dL", "mmol/L", "mEq/L", "IU/L", "K/uL", "sec", "mg/dL"):
        rr = [{"low": {"value": 1.0, "unit": raw}, "high": {"value": 2.0, "unit": raw}}]
        for _l, text, unit in us.standardise_reference_range(rr, raw, AMAP):
            assert us.VALUE_RE.match(text), (raw, text)
            assert us.is_legitimate_output(unit, AMAP), (raw, unit)
