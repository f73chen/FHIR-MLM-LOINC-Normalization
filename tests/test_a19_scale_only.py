"""Tests for analysis/a19_scale_only_standardisation.py.

The rule allows a change of scale, refuses a change of kind and reads only the unit string. Each
refusal below is a case where a purely dimensional rule would allow a conversion and be wrong, so
the refusals are tested at least as carefully as the conversions.

  python -m pytest tests/test_a19_scale_only.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
_s = importlib.util.spec_from_file_location("a19", ROOT / "analysis" / "a19_scale_only_standardisation.py")
a19 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(a19)
from lib.units import UnitTable  # noqa: E402

U = UnitTable()


# --------------------------- analyte-dependent units ---------------------------
@pytest.mark.parametrize("unit,expect", [
    ("[IU]/L", "[IU]"), ("u[IU]/mL", "[IU]"), ("m[IU]/mL", "[IU]"),
    ("[arb'U]/mL", "[arb'U]"), ("[GPL'U]", "[GPL'U]"),
    ("{#}/[HPF]", "[HPF]"), ("/[HPF]", "[HPF]"), ("{score}", "{score}"), ("AU", "AU"),
    ("U/L", "U"), ("mU/L", "U"),
    ("meq/L", "eq"), ("eq/L", "eq"),
    ("10*3/uL", ""), ("10*9/L", ""), ("/uL", ""), ("g/dL", ""), ("mmol/L", ""), ("%", ""),
    ("mg/g", ""),
])
def test_atom_class(unit, expect):
    assert a19.atom_class(unit) == expect


def test_every_bracketed_atom_gets_its_own_class():
    """This includes atoms that are not named in advance. [GPL'U] parses as dimensionless and would
    otherwise merge into the percent kind."""
    for u in ["[GPL'U]", "[MPL'U]", "[APL'U]", "[IU]", "[arb'U]"]:
        assert a19.atom_class(u) == u
        assert a19.atom_class(u) != a19.atom_class("%")


def test_atom_class_does_not_confuse_U_inside_a_bracket():
    assert a19.atom_class("[IU]/L") == "[IU]"
    assert "U" not in a19.atom_class("[IU]/L").replace("[IU]", "")


# --------------------------- the rule ---------------------------
@pytest.mark.parametrize("src,dst,factor", [
    ("10*3/uL", "10*9/L", 1.0),        # the canonical zero-risk relabel
    ("g/dL", "g/L", 10.0),
    ("mg/dL", "g/L", 0.01),
    ("ng/mL", "ug/L", 1.0),
    ("10*3/uL", "/uL", 1000.0),        # both plain counts per volume: a real scale change
])
def test_allowed_scale_changes(src, dst, factor):
    f = a19.scale_factor(U, src, dst)
    assert f is not None, f"{src} -> {dst} was refused but is a pure scale change"
    assert f == pytest.approx(factor, rel=1e-9)


@pytest.mark.parametrize("src,dst,why", [
    ("[IU]/L", "/L", "deletes the IU atom, which a purely dimensional rule would allow at factor 1"),
    ("[GPL'U]", "%", "a bracketed arbitrary unit is not a percentage"),
    ("[IU]/L", "10*9/L", "same"),
    ("U/L", "/L", "deletes the enzyme-unit atom"),
    ("meq/L", "mmol/L", "equivalents depend on valence, which the unit string does not carry"),
    ("mg/dL", "mmol/L", "mass to molar is a different LOINC code"),
    ("g/dL", "mmol/L", "mass to molar"),
    ("%", "/L", "different dimension"),
    ("fL", "pg", "volume to mass"),
])
def test_refused_kind_changes(src, dst, why):
    assert a19.scale_factor(U, src, dst) is None, f"{src} -> {dst} must be refused: {why}"


def test_the_IU_trap_is_real():
    """Documents why the kind includes the analyte-dependent unit. The dimensional table alone
    says these are the same."""
    assert U.dim.get("[IU]/L") == U.dim.get("/L")
    assert U.mole.get("[IU]/L") == U.mole.get("/L")
    assert U.factor("[IU]/L", "/L") == pytest.approx(1.0)     # a16's dimensional rule allows it
    assert a19.scale_factor(U, "[IU]/L", "/L") is None        # a19's kind rule does not


def test_kind_is_symmetric_and_reflexive():
    for u in ["g/dL", "[IU]/L", "10*3/uL", "meq/L", "%"]:
        assert a19.scale_factor(U, u, u) == pytest.approx(1.0)
    for a, b in [("g/dL", "g/L"), ("10*3/uL", "10*9/L")]:
        fa, fb = a19.scale_factor(U, a, b), a19.scale_factor(U, b, a)
        assert fa * fb == pytest.approx(1.0)


# --------------------------- notation ---------------------------
@pytest.mark.parametrize("v,n", [(7.0, 1), (7.5, 2), (0.025, 2), (1.13, 3), (12.5, 3),
                                 (0.0001, 1), (140.0, 2), (1000.0, 1), (1.01, 3)])
def test_sig_figs_is_minimal_lossless(v, n):
    """sig_figs counts the fewest digits that reproduce the value. 140 is reproduced exactly by
    1.4e2, which gives 2. Writing 1.40e2 would assert a third measured digit that the source never
    established."""
    assert int(a19.sig_figs(np.array([v]))[0]) == n


def test_sig_figs_sentinel_regression():
    """One significant figure is a possible answer and must stay distinguishable from the
    'not yet determined' initial state. Confusing the two returns 2 for every such value."""
    assert int(a19.sig_figs(np.array([7.0]))[0]) == 1
    assert int(a19.sig_figs(np.array([0.0001]))[0]) == 1
    assert int(a19.sig_figs(np.array([200.0]))[0]) == 1


def test_sig_figs_handles_zero_and_nan():
    out = a19.sig_figs(np.array([0.0, np.nan]))
    assert out[0] == 0 and out[1] == 0


def test_scientific_preserves_significant_figures():
    """Rescaling and then rendering must not add or lose a single significant digit."""
    native = np.array([7.0, 7.5, 140.0, 0.025, 12.5])
    sf = a19.sig_figs(native)
    scaled = native * 1e9                       # e.g. 10*3/uL -> /L
    sci = a19.render_scientific(scaled, sf)
    assert list(sci) == ["7e+09", "7.5e+09", "1.4e+11", "2.5e+07", "1.25e+10"]
    # and the value round-trips exactly
    assert np.allclose([float(x) for x in sci], scaled, rtol=1e-12)


def test_scientific_is_shorter_than_decimal_for_a_rescaled_count():
    sf = a19.sig_figs(np.array([7.0]))
    dec = a19.render_decimal(np.array([7.0e9]), np.array([0]))[0]
    sci = a19.render_scientific(np.array([7.0e9]), sf)[0]
    assert dec == "7000000000" and len(sci) < len(dec)


def test_render_decimal_respects_places():
    assert list(a19.render_decimal(np.array([1.126, 140.0]), np.array([2, 0]))) == ["1.13", "140"]


def test_hybrid_keeps_short_values_decimal_and_rescues_long_ones():
    v = np.array([7.3, 140.0, 0.0000003, 7.0e9])
    sf = a19.sig_figs(v)
    dec = np.array([1, 0, 7, 0])
    out = a19.render_hybrid(v, sf, dec, max_chars=8)
    assert out[0] == "7.3" and out[1] == "140"          # already compact -> left alone
    assert "e" in out[2] and "e" in out[3]              # long -> scientific
    assert all(len(x) <= 11 for x in out)


def test_hybrid_is_lossless():
    rng = np.random.default_rng(0)
    v = np.round(rng.uniform(0.001, 5000, 5000), 3)
    sf = a19.sig_figs(v)
    out = a19.render_hybrid(v, sf, np.full(len(v), 3), max_chars=8)
    assert np.allclose([float(x) for x in out], v, rtol=1e-12)


# --------------------------- fixed-width scientific ---------------------------
def test_fixed_sci_is_one_shape_for_every_value():
    """The fixed render gives the same structure at every magnitude."""
    import re as _re
    v = np.array([7.3, 140.0, 0.0000003, 7.0e9, 1.0, 0.5])
    out = a19.render_fixed_sci(v, 3)
    pat = _re.compile(r"^\d\.\d{2}e[+-]\d{2}$")
    assert all(pat.match(x) for x in out), list(out)


def test_fixed_sci_preserves_value_to_three_sig_figs():
    v = np.array([7.3, 140.0, 1.126, 0.0250])
    out = a19.render_fixed_sci(v, 3)
    assert list(out) == ["7.30e+00", "1.40e+02", "1.13e+00", "2.50e-02"]
    assert np.allclose([float(x) for x in out], a19.round_sig(v, 3), rtol=1e-12)


def test_round_sig_matches_a15_convention():
    """Must agree with the multiply-round-divide rule a15 uses, or the rungs are not comparable."""
    v = np.array([1.2345, 987.65, 0.0012345, 0.0, -4.567])
    assert np.allclose(a19.round_sig(v, 3), [1.23, 988.0, 0.00123, 0.0, -4.57])


def test_round_sig_is_lossless_on_values_already_at_that_precision():
    v = np.array([7.3, 140.0, 1.13, 0.025])
    assert np.allclose(a19.round_sig(v, 3), v)


def test_fixed_sci_handles_zero_and_nan():
    out = a19.render_fixed_sci(np.array([0.0, np.nan]), 3)
    assert out[0] == "0.00e+00" and out[1] == ""


# --------------- annotations are decoration, except when bare ---------------
def test_annotation_is_decoration_when_a_real_atom_is_present():
    """UCUM defines `{...}` as carrying no meaning. `{#}/[HPF]` and `/[HPF]` are the same unit and
    must not land in different kinds because one is labelled."""
    assert a19.atom_class("{#}/[HPF]") == a19.atom_class("/[HPF]") == "[HPF]"
    assert a19.atom_class("{#}/[LPF]") == a19.atom_class("/[LPF]") == "[LPF]"
    assert a19.scale_factor(U, "{#}/[HPF]", "/[HPF]") == pytest.approx(1.0)


def test_a_bare_annotation_IS_the_unit_and_must_not_be_dropped():
    """This exception keeps the rule safe. `{score}` has no other atom, so dropping the annotation
    would leave it dimensionless and merge it with `%` and every other ratio."""
    assert a19.atom_class("{score}") == "{score}"
    assert a19.atom_class("{score}") != a19.atom_class("%")
    assert a19.scale_factor(U, "{score}", "%") is None


def test_HPF_and_LPF_stay_apart():
    """High-power and low-power field are different magnifications, with no factor between them."""
    assert a19.scale_factor(U, "{#}/[HPF]", "{#}/[LPF]") is None


def test_AU_has_its_own_atom_class_so_it_can_never_meet_a_length_unit():
    """lab2clean parses the MIMIC unit `AU` (arbitrary units) as the astronomical unit (1.496e11 m,
    a length) on 433,001 observations. Nothing else in the corpus has a length dimension. Even so,
    treating `AU` as its own analyte-dependent unit keeps it apart by construction, without relying
    on that accident."""
    assert a19.atom_class("AU") == "AU"
    assert a19.unit_kind(U, "AU") != a19.unit_kind(U, "m")
    assert a19.scale_factor(U, "AU", "m") is None
    assert a19.scale_factor(U, "AU", "AU") == pytest.approx(1.0)


def test_AU_does_not_match_inside_other_units():
    for u in ("[arb'U]", "[IU]/L", "U/L", "mg/dL"):
        assert "AU" not in a19.atom_class(u), f"{u} wrongly classed as AU"
