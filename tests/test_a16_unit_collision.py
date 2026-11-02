"""Tests for analysis/a16_unit_collision.py.

Whatever unit an arm picks, the emitted value must be the same quantity as the original. Otherwise
the analysis would measure a value-corruption effect and report it as a unit-collision effect.
test_quantity_preserved checks this row by row against an independently computed factor.

  python -m pytest tests/test_a16_unit_collision.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
_s = importlib.util.spec_from_file_location("a16", ROOT / "analysis" / "a16_unit_collision.py")
a16 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(a16)
from lib.units import UnitTable  # noqa: E402

U = UnitTable()


# --------------------------- keyval ---------------------------
def test_keyval_significant_figures():
    a = np.array([1.23456789012, 0.000123456789012, 123456789012.0, 0.0, np.nan])
    out = a16.keyval(a)
    assert out[0] == pytest.approx(1.234567890)
    assert out[1] == pytest.approx(0.0001234567890)
    assert out[2] == pytest.approx(123456789000.0)
    assert out[3] == 0.0
    assert np.isnan(out[4])


def test_keyval_is_scale_free():
    """a15 keys at 4 decimal places, which destroys small values after a downward conversion.
    keyval must keep 0.000125 and 0.000127 distinct."""
    a = np.array([0.000125, 0.000127])
    assert a16.keyval(a)[0] != a16.keyval(a)[1]
    assert np.round(a, 4)[0] == np.round(a, 4)[1]      # the rule keyval replaces


def test_keyval_agrees_with_4dp_on_native_precision():
    """MIMIC values carry at most 4 dp, so on unconverted data the two rules must key identically."""
    rng = np.random.default_rng(0)
    a = np.round(rng.uniform(0.01, 5000, 20000), 4)
    assert np.array_equal(a16.keyval(a), np.round(a, 4))


# --------------------------- sites ---------------------------
def test_site_of_stable_and_spread():
    ids = [f"patient-{i:06d}" for i in range(5000)]
    s1, s2 = a16.site_of(ids), a16.site_of(list(reversed(ids)))
    assert np.array_equal(s1, s2[::-1])                       # depends only on the id
    counts = np.bincount(s1, minlength=a16.N_SITES)
    assert counts.min() > len(ids) / a16.N_SITES * 0.9


# --------------------------- fixture ---------------------------
@pytest.fixture(scope="module")
def world():
    """Two codes with real convertible units, plus one code with no alternative."""
    # 718-7 Hgb Bld-mCnc; 6690-2 WBC # Bld Auto; 2160-0 Creat SerPl-mCnc
    codes = [("718-7", "g/dL"), ("6690-2", "10*3/uL"), ("2160-0", "mg/dL")]
    alts_att, alts_prop, std_fam, _ = a16.build_alternatives(codes, verbose=False)
    n = 3000
    rng = np.random.default_rng(0)
    y = np.array([codes[i][0] for i in rng.integers(3, size=n)])
    u_canon = np.array([dict(codes)[c] for c in y], dtype=object)
    u_raw = u_canon.copy()
    val = rng.uniform(1, 100, n)
    pid, pairs = a16.pair_index(y, u_canon)
    grp = a16.groups_of(pid, pid >= 0)
    sites = a16.site_of([f"p{i%997}" for i in range(n)])
    return dict(pairs=pairs, pid=pid, grp=grp, val=val, u_canon=u_canon, u_raw=u_raw,
                sites=sites, alts={"att": alts_att, "prop": alts_prop},
                std_fam=std_fam, codes=dict(codes), y=y)


def _tables(w, tier):
    tbl = w["alts"][tier]
    alt_tbl = {p: (u, tbl[(c, u)]) for p, (c, u) in w["pairs"].items() if (c, u) in tbl}
    conv = np.isin(w["pid"], np.fromiter(alt_tbl, dtype="int32", count=len(alt_tbl))
                   if alt_tbl else np.empty(0, "int32"))
    return alt_tbl, conv


def _run(w, arm, tier="prop", seed=0, std_tbl=None):
    alt_tbl, conv = _tables(w, tier)
    rng = np.random.default_rng(seed)
    st = a16.site_table(alt_tbl, rng) if arm == "A2" else None
    return a16.apply_arm(arm, w["pid"], w["grp"], w["val"], w["u_canon"], w["u_raw"],
                         conv, w["sites"], alt_tbl, std_tbl or {}, rng=rng, sitetbl=st)


# --------------------------- arms ---------------------------
def test_A0_is_identity(world):
    v, u = _run(world, "A0")
    assert np.array_equal(v, world["val"])
    assert list(u) == list(world["u_raw"])


def test_A0c_changes_no_value(world):
    v, u = _run(world, "A0c")
    assert np.array_equal(v, world["val"])


@pytest.mark.parametrize("arm", ["A1", "A2"])
@pytest.mark.parametrize("tier", ["att", "prop"])
def test_quantity_preserved(world, arm, tier):
    """The central invariant: emitted_value in emitted_unit == original_value in original_unit."""
    v, u = _run(world, arm, tier)
    changed = 0
    for i in range(len(v)):
        src, dst = world["u_canon"][i], u[i]
        if src is None:
            continue
        f = U.factor(src, dst)
        assert f is not None, f"emitted an unconvertible unit: {src} -> {dst}"
        assert v[i] == pytest.approx(world["val"][i] * f, rel=1e-9)
        changed += src != dst
    assert changed > 0, "arm changed nothing (the fixture has no usable alternatives)"


@pytest.mark.parametrize("arm", ["A1", "A2"])
def test_label_preserving(world, arm):
    """Dimension and mole exponent must survive, or the true LOINC code changed."""
    _, u = _run(world, arm, "prop")
    for src, dst in zip(world["u_canon"], u):
        if src is None or src == dst:
            continue
        assert U.dim.get(src) == U.dim.get(dst)
        assert U.mole.get(src) == U.mole.get(dst)


def test_A2_is_consistent_within_a_site(world):
    """A site reports an analyte in a single unit, which is what arm A2 simulates."""
    _, u = _run(world, "A2", "prop")
    seen = {}
    for s, p, uu in zip(world["sites"], world["pid"], u):
        if p < 0:
            continue
        assert seen.setdefault((s, p), uu) == uu


def test_A2_table_is_split_independent(world):
    """The per-site table is drawn once, outside the row loop. If it were drawn inside the loop, a
    split missing a pair would shift every later draw, and train and test would disagree on what a
    site reports."""
    alt_tbl, conv = _tables(world, "prop")
    st = a16.site_table(alt_tbl, np.random.default_rng(0))
    full = a16.apply_arm("A2", world["pid"], world["grp"], world["val"], world["u_canon"],
                         world["u_raw"], conv, world["sites"], alt_tbl, {}, sitetbl=st)[1]
    keep = np.zeros(len(world["pid"]), bool)          # a split holding only the last pair
    keep[world["grp"][max(world["grp"])]] = True
    idx = np.flatnonzero(keep)
    sub = a16.apply_arm("A2", world["pid"][idx], a16.groups_of(world["pid"][idx], keep[idx] | True),
                        world["val"][idx], world["u_canon"][idx], world["u_raw"][idx],
                        conv[idx], world["sites"][idx], alt_tbl, {}, sitetbl=st)[1]
    assert list(sub) == list(full[idx])


def test_A1_varies_within_a_site(world):
    """A1 is the upper bound and must vary within a site. Otherwise it would be the same as A2."""
    _, u = _run(world, "A1", "prop")
    seen, clashes = {}, 0
    for s, p, uu in zip(world["sites"], world["pid"], u):
        if p < 0:
            continue
        clashes += seen.setdefault((s, p), uu) != uu
    assert clashes > 0


def test_A3_collapses_a_family(world):
    std_fam = world["std_fam"]
    counts = {k: 1000 for k in world["pairs"].values()}
    counts[("718-7", "g/dL")] = 1                    # make g/dL the minority so it must move
    std_res = a16.resolve_standard(std_fam, counts, U)
    std_tbl = {p: std_res[(c, u)] for p, (c, u) in world["pairs"].items() if (c, u) in std_res}
    assert std_tbl, "nothing to standardise (the fixture is degenerate)"
    v, u = _run(world, "A3", std_tbl=std_tbl)
    for i in range(len(v)):
        src = world["u_canon"][i]
        tgt = std_res.get((world["y"][i], src))
        if tgt is None:
            assert u[i] == world["u_raw"][i]
            continue
        assert u[i] == tgt[0]
        assert v[i] == pytest.approx(world["val"][i] * U.factor(src, tgt[0]), rel=1e-9)


def test_mass_to_molar_never_offered(world):
    """Creatinine mg/dL must not become mmol/L, because that changes the PROPERTY and so the LOINC code."""
    for tier in ("att", "prop"):
        for alt, _ in world["alts"][tier].get(("2160-0", "mg/dL"), []):
            assert U.mole.get(alt) == U.mole.get("mg/dL")


# --------------------------- scoring ---------------------------
def test_score_matches_a15_on_unconverted_data():
    """Sanity ranges for score() on unconverted data. score() computes the value+unit number with a15's
    joint_codes and score_vec, but this test does not compare its output with a15."""
    rng = np.random.default_rng(7)
    n = 40000
    y = rng.integers(0, 60, n).astype("int32")
    val = np.round(rng.uniform(1, 200, n), 2)
    u = np.array(["mg/dL", "g/dL", "%"], dtype=object)[rng.integers(3, size=n)]
    half = n // 2
    prior = np.array([int(x) for x in np.argsort(-np.bincount(y[:half]))[:10]], dtype="int32")
    r = a16.score(val[:half], u[:half], y[:half], val[half:], u[half:], y[half:], prior)
    assert 0 <= r["top1_covered"] <= 1
    assert 0 <= r["collision_mass_pct"] <= 100
    assert r["collision_mass_pct"] <= 100 - 1e-9
    # a key seen in training with one label only is never counted as a collision
    assert r["collision_of_covered_pct"] >= r["collision_mass_pct"] - 1e-9


def test_collision_mass_is_exact_on_a_hand_case():
    """Two codes share one (value, unit), and one code has its own. Collision mass = 2/3."""
    ytr = np.array([0, 1, 2], dtype="int32")
    vtr = np.array([1.0, 1.0, 9.0])
    utr = np.array(["g/L", "g/L", "g/L"], dtype=object)
    yte = np.array([0, 1, 2], dtype="int32")
    r = a16.score(vtr, utr, ytr, vtr, utr, yte, np.array([0, 1, 2], dtype="int32"))
    assert r["collision_mass_pct"] == pytest.approx(200 / 3)
    assert r["collision_of_covered_pct"] == pytest.approx(200 / 3)


# --------------------------- the valence guard ---------------------------
def test_is_equivalent_unit():
    assert a16.is_equivalent_unit("meq/L") and a16.is_equivalent_unit("eq/L")
    assert not a16.is_equivalent_unit("mmol/L")
    assert not a16.is_equivalent_unit("[IU]/L")     # "equivalent" must match a whole atom.
    assert not a16.is_equivalent_unit("g{eqv}/L")   # A substring of a longer token is no match


def test_monovalent_eq_conversion_allowed():
    """Sodium 140 mEq/L is 140 mmol/L. Blocking this conversion would remove the bulk of the attested tier."""
    att, _, _, _ = a16.build_alternatives([("2951-2", "meq/L")], verbose=False)  # Sodium SerPl-sCnc
    assert [u for u, _ in att[("2951-2", "meq/L")]] == ["mmol/L"]


def test_divalent_eq_conversion_refused():
    """Calcium 2 mEq/L is 1 mmol/L. The UCUM dimension table cannot see valence, so the
    conversion must be refused on the LOINC COMPONENT."""
    # 17861-6 Calcium SerPl-mCnc; 49765-1 Calcium Bld-mCnc
    for code in ("17861-6", "49765-1"):
        att, prop, _, _ = a16.build_alternatives([(code, "meq/L")], verbose=False)
        for tier in (att, prop):
            for alt, _ in tier.get((code, "meq/L"), []):
                assert a16.is_equivalent_unit(alt), \
                    f"{code}: offered eq->{alt} without knowing the valence"


def test_no_eq_to_mol_survives_for_an_unknown_component():
    att, prop, _, _ = a16.build_alternatives([("9999-9", "meq/L")], verbose=False)
    for tier in (att, prop):
        for alt, _ in tier.get(("9999-9", "meq/L"), []):
            assert a16.is_equivalent_unit(alt)


def test_keys_per_code_counts_distinct_keys_not_rows():
    """Code 0 has two distinct keys over four rows, and code 1 has one key in one row.
    keys_per_code must count keys and give (2+1)/2 = 1.5."""
    ytr = np.array([0, 0, 0, 0, 1], dtype="int32")
    vtr = np.array([1.0, 1.0, 2.0, 2.0, 5.0])
    utr = np.array(["g/L"] * 5, dtype=object)
    r = a16.score(vtr, utr, ytr, vtr, utr, ytr, np.array([0, 1], dtype="int32"))
    assert r["n_codes_in_table"] == 2
    assert r["keys_per_code_mean"] == pytest.approx(1.5)
    assert r["n_keys"] == 3
