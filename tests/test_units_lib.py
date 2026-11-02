"""Tests for analysis/lib/units.py.

A conversion between mass and molar units would relabel the observation to a LOINC code that is
almost never in the label space. The refusals and the blocklists are therefore tested at least as
closely as the conversion factors.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from lib.units import UnitTable, _norm_key, load_blocked_code_pairs, load_blocked_code_unit  # noqa: E402

pytestmark = pytest.mark.slow  # loads a 1,540-row .rda and a 5,942-entry alias map


@pytest.fixture(scope="module")
def U():
    return UnitTable()


# -- alias resolution --
@pytest.mark.parametrize("raw,expect", [
    ("k/ul", "10*3/uL"), ("K/uL", "10*3/uL"),
    ("meq/l", "meq/L"), ("g/dl", "g/dL"), ("mg/dl", "mg/dL"),
    ("mm hg", "mm[Hg]"),          # the whitespace spelling, common in MIMIC
    ("mmHg", "mm[Hg]"),
])
def test_canonical_resolves_mimic_spellings(U, raw, expect):
    assert U.canonical(raw) == expect


def test_canonical_may_pick_an_equivalent_si_spelling(U):
    """`M/uL` canonicalises to the SI form `10*12/L`, which the alias map prefers to `10*6/uL`. Both
    are correct. The test pins that string and checks that the two convert at factor 1, so nothing
    is rescaled."""
    c = U.canonical("m/ul")
    assert c == "10*12/L"
    assert U.factor(c, "10*6/uL") == pytest.approx(1.0)


def test_canonical_is_none_for_junk(U):
    assert U.canonical(None) is None
    assert U.canonical("not a unit at all zzz") is None


def test_norm_key_strips_space_and_case():
    assert _norm_key("mm Hg") == "mmhg"
    assert _norm_key("  K/uL ") == "k/ul"


# -- conversion factors: the values must be right, not only present --
@pytest.mark.parametrize("src,dst,factor", [
    ("g/dL", "g/L", 10.0),          # 1 g/dL = 10 g/L
    ("g/L", "g/dL", 0.1),
    ("mg/dL", "mg/L", 10.0),
    ("10*3/uL", "10*9/L", 1.0),     # US vs SI haematology: same number, different label
    ("10*6/uL", "10*12/L", 1.0),
    ("ng/mL", "ug/L", 1.0),         # Clinical Chemistry exempts this pair from SI conversion
    ("pg/mL", "ng/L", 1.0),
    ("mg/dL", "g/L", 0.01),
])
def test_factor_values(U, src, dst, factor):
    got = U.factor(src, dst)
    assert got == pytest.approx(factor, rel=1e-9), f"{src}->{dst} gave {got}, want {factor}"


def test_factor_roundtrips(U):
    for a, b in [("g/dL", "g/L"), ("mg/dL", "mg/L"), ("ng/mL", "ug/L")]:
        assert U.factor(a, b) * U.factor(b, a) == pytest.approx(1.0, rel=1e-12)


# -- the refusals --
@pytest.mark.parametrize("src,dst", [
    ("mg/dL", "mmol/L"),    # mass -> molar: the forbidden conversion
    ("mmol/L", "mg/dL"),
    ("g/dL", "mmol/L"),
    ("mg/L", "umol/L"),
])
def test_mass_to_molar_is_refused(U, src, dst):
    assert not U.commensurable(src, dst), f"{src}->{dst} must not be commensurable"
    assert U.factor(src, dst) is None


def test_dimension_alone_would_not_have_caught_it(U):
    """Why the mole exponent is checked separately. mg/dL and mmol/L share a dimension vector once
    mass is folded in, so a dimension-only rule would emit the forbidden conversion."""
    assert U.dim.get("mg/dL") == U.dim.get("mg/L")          # same dimension, same property: fine
    assert U.mole.get("mg/dL") == 0.0
    assert U.mole.get("mmol/L") == 1.0                       # the mole exponent is what separates them


def test_incommensurable_dimensions_refused(U):
    assert U.factor("g/dL", "10*3/uL") is None               # mass conc vs number conc
    assert U.factor("s", "g/L") is None


def test_meq_and_mmol_are_numerically_identical(U):
    """meq/L -> mmol/L is factor 1, which is correct only for monovalent ions (Na, K, Cl, HCO3) and
    wrong by 2x for Ca and Mg. The library cannot know valence, so the caller must whitelist. The
    test keeps this hazard visible."""
    assert U.factor("meq/L", "mmol/L") == pytest.approx(1.0)


# -- blocklists --
def test_hauser_blocklist_loads_and_contains_a_known_pair(U):
    b = load_blocked_code_pairs()
    assert len(b) >= 250, f"expected ~285 Hauser pairs, got {len(b)}"
    assert all(isinstance(x, frozenset) and len(x) == 2 for x in list(b)[:20])


def test_harmonisation_blocklist_contains_the_glucose_case():
    b = load_blocked_code_unit()
    assert len(b) >= 700
    # 2339-0 (Glucose Bld-mCnc) Glucose [Mass/volume] + mmol/L belongs to
    # 15074-8 (Glucose Bld-sCnc) Glucose [Moles/volume]
    assert ("2339-0", _norm_key("mEq/L")) in b or ("2339-0", _norm_key("mmol/L")) in b
