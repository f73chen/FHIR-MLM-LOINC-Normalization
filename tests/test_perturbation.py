"""Tests for pipeline/perturbation.py, the simulated second site: value and reference-range
perturbation and unit-notation randomisation.

Every guard (the G numbers in pipeline/perturbation.py) is tested on a case that a straightforward
implementation would get wrong. Without the guards, a perturbed record could be clinically impossible
or carry a new shortcut. A model could then spot perturbed rows by checking whether MCHC still equals
Hb/Hct, or whether the differential still sums to 100.

  python -m pytest tests/test_perturbation.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import perturbation as P  # noqa: E402

SPEC = P.load_spec()
CODE = lambda r: r["code"]["coding"][0]["code"]          # noqa: E731


def obs(code, value, unit="", lo=None, hi=None, t="2130-01-01T00:00:00"):
    o = {"resourceType": "Observation", "code": {"coding": [{"code": code}]},
         "effectiveDateTime": t, "valueQuantity": {"value": value, "unit": unit}}
    if lo is not None:
        o["referenceRange"] = [{"low": {"value": lo, "unit": unit},
                                "high": {"value": hi, "unit": unit}}]
    return o


# --------------------------- determinism ---------------------------
def test_draws_are_stable_and_independent():
    # 2951-2 Sodium SerPl-sCnc; 2823-3 Potassium SerPl-sCnc
    a = P.site_z("pt-1", "2951-2", "value")
    assert a == P.site_z("pt-1", "2951-2", "value")          # same everything
    assert a != P.site_z("pt-2", "2951-2", "value")          # a different site
    assert a != P.site_z("pt-1", "2823-3", "value")          # a different analyte
    assert a != P.site_z("pt-1", "2951-2", "range")          # value vs range: rho = 0 (a26)


def test_one_draw_per_patient_and_analyte_not_per_observation():
    """A site's calibration does not change between two measurements on the same patient."""
    res = [obs("2951-2", 140.0), obs("2951-2", 140.0), obs("2951-2", 141.0)]
    P.perturb_patient(res, "pt-1", SPEC, CODE)
    assert res[0]["valueQuantity"]["value"] == res[1]["valueQuantity"]["value"]


# --------------------------- G0 / G1 / G2 ---------------------------
def test_G0_tier0_codes_are_never_touched():
    tier0 = [c for c, s in SPEC.items() if s["tier0"]]
    assert tier0, "expected some tier-0 codes in the spec"
    res = [obs(c, 50.0) for c in tier0[:5]]
    before = [r["valueQuantity"]["value"] for r in res]
    assert P.perturb_patient(res, "pt-1", SPEC, CODE) == 0
    assert [r["valueQuantity"]["value"] for r in res] == before


def test_G1_negative_is_clamped_except_where_it_is_legitimate():
    ordinary = next(c for c, s in SPEC.items()
                    if not s["allow_negative"] and s["kind"] == "relative_pct" and not s["tier0"])
    s = SPEC[ordinary]
    assert P.perturb_number(1.0, -50.0, s, s["value_cv"] or 5.0, severity=10.0) >= 0
    base_excess = next(c for c, s in SPEC.items() if s["allow_negative"])
    s = SPEC[base_excess]
    assert P.perturb_number(-2.0, 0.0, s, s["value_cv"]) < 0, "base excess must stay negative"


def test_G2_percentages_cannot_exceed_100():
    frac = next(c for c, s in SPEC.items() if s["clamp_0_100"])
    s = SPEC[frac]
    assert P.perturb_number(99.8, 40.0, s, s["value_cv"], severity=10.0) <= 100.0
    assert P.perturb_number(0.2, -40.0, s, s["value_cv"], severity=10.0) >= 0.0


# --------------------------- G3: recompute, do not perturb ---------------------------
# 718-7 Hgb Bld-mCnc; 4544-3 Hct VFr Bld Auto; 789-8 RBC # Bld Auto;
# 787-2 MCV RBC Auto; 785-6 MCH RBC Qn Auto; 786-4 MCHC RBC Auto-EntMCnc
RED_CELL = {"Hb": "718-7", "Hct": "4544-3", "RBC": "789-8",
            "MCV": "787-2", "MCH": "785-6", "MCHC": "786-4"}


def _red_cell_panel():
    return [obs(RED_CELL["Hb"], 14.0, "g/dL"), obs(RED_CELL["Hct"], 42.0, "%"),
            obs(RED_CELL["RBC"], 4.7, "m/uL"), obs(RED_CELL["MCV"], 89.0, "fL"),
            obs(RED_CELL["MCH"], 29.8, "pg"), obs(RED_CELL["MCHC"], 33.3, "g/dL")]


def test_G3_identities_still_hold_after_perturbation():
    """The test that independent perturbation would fail. MIMIC obeys these identities to a median
    0.12-0.28%, so a model could detect perturbed rows by recomputing the ratio."""
    res = _red_cell_panel()
    P.perturb_patient(res, "pt-rc", SPEC, CODE)
    v = {k: next(r["valueQuantity"]["value"] for r in res if CODE(r) == c)
         for k, c in RED_CELL.items()}
    assert v["MCV"] == pytest.approx(v["Hct"] / v["RBC"] * 10.0, rel=0.01)
    assert v["MCH"] == pytest.approx(v["Hb"] / v["RBC"] * 10.0, rel=0.01)
    assert v["MCHC"] == pytest.approx(v["Hb"] / v["Hct"] * 100.0, rel=0.01)


def test_G3_parents_actually_moved():
    """Guards against a vacuous pass: the identities would also hold if nothing changed."""
    res = _red_cell_panel()
    P.perturb_patient(res, "pt-rc", SPEC, CODE)
    assert any(r["valueQuantity"]["value"] != o for r, o in
               zip(res, [14.0, 42.0, 4.7, 89.0, 29.8, 33.3]))


def test_G3_incomplete_panel_leaves_the_derived_value_alone():
    """With no Hb or Hct in the record, MCHC is left unperturbed."""
    res = [obs(RED_CELL["MCHC"], 33.3, "g/dL")]          # no Hb, no Hct
    P.perturb_patient(res, "pt-rc", SPEC, CODE)
    assert res[0]["valueQuantity"]["value"] == 33.3


# --------------------------- G4: the differential still sums ---------------------------
# 770-8 Neutrophils NFr Bld Auto; 736-9 Lymphocytes NFr Bld Auto; 5905-5 Monocytes NFr Bld Auto;
# 713-8 Eosinophil NFr Bld Auto; 706-2 Basophils NFr Bld Auto
DIFF = ["770-8", "736-9", "5905-5", "713-8", "706-2"]


def test_G4_panel_sum_is_preserved():
    vals = [55.0, 30.0, 8.0, 4.0, 2.6]
    res = [obs(c, v, "%") for c, v in zip(DIFF, vals)]
    P.perturb_patient(res, "pt-d", SPEC, CODE, severity=8.0)
    after = sum(r["valueQuantity"]["value"] for r in res)
    assert after == pytest.approx(sum(vals), abs=0.2)


def test_G4_is_not_a_no_op():
    """Computing the target sum after perturbing gives a factor of 1.0, which changes nothing."""
    vals = [70.0, 20.0, 5.0, 3.0, 2.0]
    unnorm = []
    for c, v in zip(DIFF, vals):
        s = SPEC[c]
        unnorm.append(P.perturb_number(v, P.site_z("pt-g4", c, "value"), s, s["value_cv"], 8.0))
    assert abs(sum(unnorm) - 100.0) > 1.0, "fixture no longer exercises the drift"
    res = [obs(c, v, "%") for c, v in zip(DIFF, vals)]
    P.perturb_patient(res, "pt-g4", SPEC, CODE, severity=8.0)
    assert sum(r["valueQuantity"]["value"] for r in res) == pytest.approx(100.0, abs=0.2)


# --------------------------- G6: scale-specific rules ---------------------------
def test_G6_blood_pH_takes_an_absolute_offset_not_a_percentage():
    """Blood pH takes an absolute offset in pH units. A relative CV applied as that offset would turn
    pH 7.40 into 12.5, and a 16% relative change would turn it into 8.6. Both are outside the range
    compatible with life."""
    s = SPEC["11558-4"]                                  # pH Bld
    assert s["kind"] == "absolute"
    for z in (-3.0, -1.0, 1.0, 3.0):
        assert 6.5 < P.perturb_number(7.40, z, s, s["value_cv"]) < 8.2


def test_G6_ladders_move_by_whole_rungs_only():
    # 5804-0 Prot Ur Strip-mCnc
    s = SPEC["5804-0"]                                   # urine protein test strip
    assert s["kind"] == "rung" and s["ladder"]
    for z in (-2.0, -1.0, 1.0, 2.0):
        assert P.perturb_number(30, z, s, s["value_cv"]) in s["ladder"]


def test_G6_specific_gravity_perturbs_the_excess_over_one():
    s = SPEC["5811-5"]                                   # Sp Gr Ur Strip
    assert s["kind"] == "excess_over_1"
    for z in (-2.0, 2.0):
        assert 1.0 < P.perturb_number(1.015, z, s, s["value_cv"]) < 1.06


def test_G6_microscopy_counts_use_poisson_and_stay_integral():
    # 13945-1 RBC #/area UrnS HPF
    s = SPEC["13945-1"]                                  # urine erythrocytes /HPF
    assert s["kind"] == "poisson"
    out = P.perturb_number(3, 1.0, s, s["value_cv"])
    assert out == int(out) and out >= 0


# --------------------------- G7 and the reference range ---------------------------
def test_G7_never_emits_more_precision_than_the_input_or_the_code():
    # 2160-0 Creat SerPl-mCnc
    s = SPEC["2160-0"]                                   # serum creatinine, 1 d.p. habitually
    assert P.perturb_number(0.9, 1.0, s, s["value_cv"]) == \
        round(P.perturb_number(0.9, 1.0, s, s["value_cv"]), 1)
    # an input written with more precision than the code's habit keeps its own
    out = P.perturb_number(0.93, 1.0, s, s["value_cv"])
    assert out == round(out, 2)


def test_G7_does_not_round_a_reported_value_away_to_zero():
    """Re-rounding to the `native_decimals` of the code alone destroys the minority precision of a
    code whose site varies it. a30 computes `native_decimals` as the modal decimal count. 29% of
    nucleated-erythrocyte ratios and 1,106 TSH results would round to 0.0, which is a false
    clinical value."""
    # 58413-6 nRBC Bld Auto-Rto; 3016-3 TSH SerPl-aCnc; 54218-3 CD3+CD4+ Cells/CD3+CD8+ Cll Bld
    for code, value in (("58413-6", 0.5), ("3016-3", 0.012), ("54218-3", 0.04)):
        s = SPEC[code]
        for z in (-2.0, -0.5, 0.5, 2.0):
            out = P.perturb_number(value, z, s, s["value_cv"])
            assert out != 0.0, f"{code}: {value} -> 0.0 at z={z}"


def test_G7_does_not_under_perturb_a_value_that_lands_on_a_round_number():
    """The other direction. This is also why G7 does not use the precision of the input alone.
    `repr(12.0)` strips to zero decimals, so a haemoglobin reference low of 12.0 would round
    every perturbation back to 12, and a creatinine of 1.0 could never reach 1.1."""
    s = SPEC["2160-0"]
    assert P.emit_decimals(1.0, s) >= 1
    moved = {P.perturb_number(1.0, z, s, s["value_cv"]) for z in (-2.0, -1.0, 1.0, 2.0)}
    assert moved != {1.0}, "a round creatinine never moves"
    hb = SPEC["718-7"]
    assert P.emit_decimals(12.0, hb) >= 1


def test_both_range_bounds_move_by_the_SAME_factor():
    """The variation of the two bounds is correlated (measured r(LRL, URL) = 0.926), so they share
    one multiplier. This also makes an inverted interval impossible by construction."""
    res = [obs("718-7", 14.0, "g/dL", 12.0, 16.0)]
    P.perturb_patient(res, "pt-rr", SPEC, CODE, perturb_values=False)
    rr = res[0]["referenceRange"][0]
    lo, hi = rr["low"]["value"], rr["high"]["value"]
    assert lo < hi
    assert lo / 12.0 == pytest.approx(hi / 16.0, rel=0.02)


def test_arms_can_be_switched_independently():
    base = lambda: [obs("718-7", 14.0, "g/dL", 12.0, 16.0)]
    r = base(); P.perturb_patient(r, "pt-a", SPEC, CODE, perturb_values=False)
    assert r[0]["valueQuantity"]["value"] == 14.0
    assert r[0]["referenceRange"][0]["low"]["value"] != 12.0
    r = base(); P.perturb_patient(r, "pt-a", SPEC, CODE, perturb_ranges=False)
    assert r[0]["referenceRange"][0]["low"]["value"] == 12.0
    r = base(); n = P.perturb_patient(r, "pt-a", SPEC, CODE,
                                      perturb_values=False, perturb_ranges=False)
    assert n == 0 and r == base()


# ----------------------- magnitude kinds, switches and ladders -----------------------
def test_absolute_kind_codes_have_an_ABSOLUTE_range_magnitude_too():
    """`magnitude_kind` is one field that governs both value and range, but the two magnitudes come
    from different places. The value magnitude of blood pH is an absolute 0.014 pH units. If its
    range magnitude fell back to the register percentage (16.2), that percentage would be applied
    as an absolute offset and would move a reference range of 7.35-7.45 to 16.13-16.23.

    A test on a code whose value and range magnitudes are both percentages cannot catch this."""
    for code, s in SPEC.items():
        if s["kind"] == "absolute" and s["range_pctV"] is not None:
            assert s["range_pctV"] < 1.0, (
                f"{code}: kind=absolute but range magnitude {s['range_pctV']} looks like a "
                "percentage. It would be applied as an offset in the value's own units")


def test_blood_pH_range_stays_physiological():
    s = SPEC["11558-4"]
    for z in (-3.0, -1.0, 1.0, 3.0):
        assert 7.0 < P.perturb_number(7.35, z, s, s["range_pctV"]) < 7.8
        assert 7.0 < P.perturb_number(7.45, z, s, s["range_pctV"]) < 7.8


def test_no_code_mixes_a_percentage_magnitude_with_a_non_relative_kind():
    """The general form of the same defect, across every scale that is not relative_pct."""
    for code, s in SPEC.items():
        if s["tier0"] or s["kind"] in ("relative_pct", "excess_over_1"):
            continue
        if s["kind"] == "absolute":
            for m in (s["value_cv"], s["range_pctV"]):
                assert m is None or m < 1.0, f"{code}: absolute magnitude {m} looks like a percent"
        # `rung` and `poisson` ignore the magnitude, so a percentage there has no effect. This is
        # asserted so that a change to it makes the test fail
        if s["kind"] in ("rung", "poisson"):
            v = P.perturb_number(10.0, 1.0, s, 999999.0)
            assert v == P.perturb_number(10.0, 1.0, s, 1.0), \
                f"{code}: kind={s['kind']} must ignore the magnitude"


def test_perturbation_fully_disabled_is_a_byte_level_no_op():
    """The `xsite` arm must be able to isolate the perturbation from the representation change.
    With both switches off, perturb_patient must change nothing, so that a difference between the
    arms can only come from the site simulation."""
    import copy
    res = [obs("718-7", 14.0, "g/dL", 12.0, 16.0), obs("2160-0", 1.1, "mg/dL", 0.5, 1.2),
           # 5803-2 pH Ur Strip
           obs("11558-4", 7.40, "pH", 7.35, 7.45), obs("5803-2", 6.0, "pH", 5.0, 8.0),
           obs("13945-1", 4.0, "#/hpf", 0.0, 5.0), obs("786-4", 33.3, "g/dL", 32.0, 36.0)]
    before = copy.deepcopy(res)
    n = P.perturb_patient(res, "pt-noop", SPEC, CODE,
                          perturb_values=False, perturb_ranges=False)
    assert n == 0
    assert res == before


def test_severity_zero_is_also_a_no_op_for_every_relative_kind():
    """Severity scales the magnitude, so zero must leave every relative rule untouched. `rung`
    and `poisson` are excluded by construction, because neither reads the magnitude. The kind
    tests above assert this."""
    import copy
    res = [obs("718-7", 14.0, "g/dL", 12.0, 16.0), obs("2160-0", 1.1, "mg/dL", 0.5, 1.2),
           obs("11558-4", 7.40, "pH", 7.35, 7.45)]
    before = copy.deepcopy(res)
    P.perturb_patient(res, "pt-sev0", SPEC, CODE, severity=0.0)
    assert res == before


def test_a_rung_code_leaves_a_bound_that_is_not_on_its_ladder_alone():
    """The ladder is built from the observed values of a code, and a reference bound need not lie
    on that grid. The urine protein ladder starts at 10 mg/dL, while its stated reference interval
    is 0-0 ("negative"). Snapping to the nearest rung would turn a reference range of 0-0 into
    15-15 mg/dL. Six of the fifteen rung codes state a bound off their own ladder, and in all six
    it is the same zero."""
    s = SPEC["5804-0"]                                   # urine protein by test strip
    assert 0.0 not in s["ladder"] and min(s["ladder"]) >= 10
    for z in (-2.0, -0.5, 0.5, 2.0):
        assert P.perturb_number(0.0, z, s, s["range_pctV"]) == 0.0
    # a value that is on the ladder still steps
    assert P.perturb_number(30.0, 1.0, s, s["value_cv"]) != 30.0


def test_a_rung_value_near_a_rung_still_steps():
    """The guard looks at the local rung spacing. A number that would round onto a rung is treated
    as on the ladder even when it does not equal the rung exactly."""
    s = SPEC["5803-2"]                                   # urine pH strip, half-step ladder
    assert P.perturb_number(6.1, 1.0, s, s["value_cv"]) != 6.1
    assert P.perturb_number(6.0, -1.0, s, s["value_cv"]) == 5.5


# ----------------------- unit-notation randomisation -----------------------
RTAB = P.load_randomisation()


def test_randomisation_covers_the_electrolytes_and_excludes_pH():
    assert len(RTAB) == 25
    # 2075-0 Chloride SerPl-sCnc; 1863-0 Anion Gap SerPl Calculated.4Ions-sCnc
    for c in ("2951-2", "2823-3", "2075-0", "1863-0"):        # Na, K, Cl, anion gap
        assert RTAB[c] == ("mEq/L", "mmol/L")
    for c in ("5803-2", "11558-4"):                            # urine pH, blood pH
        assert c not in RTAB, "pH is excluded (see build_unit_randomisation.py)"


def test_randomisation_rewrites_the_value_AND_its_bounds_to_the_same_spelling():
    """Randomising the value to mmol/L while leaving its range in mEq/L would leave a visible
    artefact, which a model could learn to spot."""
    seen = set()
    for uid in [f"pt-{i}" for i in range(40)]:
        r = [obs("2951-2", 140.0, "mEq/L", 133.0, 145.0)]
        P.randomise_units(r, uid, RTAB, CODE)
        u = r[0]["valueQuantity"]["unit"]
        seen.add(u)
        assert r[0]["valueQuantity"]["code"] == u
        for k in ("low", "high"):
            assert r[0]["referenceRange"][0][k]["unit"] == u
            assert r[0]["referenceRange"][0][k]["code"] == u
    assert seen == {"mEq/L", "mmol/L"}, f"both spellings must occur across sites, saw {seen}"


def test_randomisation_never_changes_the_value():
    """Every one of the 25 codes is monovalent (a31), so the factor is 1 both ways."""
    for uid in [f"pt-{i}" for i in range(20)]:
        r = [obs("2823-3", 4.1, "mEq/L", 3.3, 5.1)]
        P.randomise_units(r, uid, RTAB, CODE)
        assert r[0]["valueQuantity"]["value"] == 4.1
        assert r[0]["referenceRange"][0]["low"]["value"] == 3.3
        assert r[0]["referenceRange"][0]["high"]["value"] == 5.1


def test_the_whole_electrolyte_panel_uses_ONE_notation_per_patient():
    """The notation is drawn per patient, while the perturbation magnitudes are drawn per
    (patient, LOINC). Notation is a property of the laboratory information system. A per-code draw
    gives most patients a mixed panel (62.8% on a 400-patient build), with chloride in mmol/L
    beside potassium in mEq/L on the same specimen."""
    # 1963-8 HCO3 SerPl-sCnc
    panel = ["2951-2", "2823-3", "2075-0", "1963-8", "1863-0"]   # Na, K, Cl, HCO3, anion gap
    both = set()
    for uid in [f"pt-{i}" for i in range(40)]:
        rs = [obs(c, 100.0, "mEq/L", 90.0, 110.0) for c in panel]
        P.randomise_units(rs, uid, RTAB, CODE)
        units = {r["valueQuantity"]["unit"] for r in rs}
        assert len(units) == 1, f"{uid}: mixed panel {units}"
        both |= units
    assert both == {"mEq/L", "mmol/L"}, f"both notations must occur across sites, saw {both}"


def test_randomisation_is_deterministic_and_independent_of_the_other_draws():
    """It draws under its own `kind`, so it cannot shift the value or range draws. Shifting them
    would change every dataset already generated."""
    a = [obs("2951-2", 140.0, "mEq/L", 133.0, 145.0)]
    b = [obs("2951-2", 140.0, "mEq/L", 133.0, 145.0)]
    P.randomise_units(a, "pt-x", RTAB, CODE)
    P.randomise_units(b, "pt-x", RTAB, CODE)
    assert a == b
    assert P.site_z("pt-x", "mEq/L|mmol/L", "unit") not in (
        P.site_z("pt-x", "2951-2", "value"), P.site_z("pt-x", "2951-2", "range"))


def test_randomisation_is_not_disabled_by_the_perturbation_switches():
    """Randomisation is part of the representation, on in every arm. The perturbation module's
    no-op guarantee is about perturbation only."""
    r = [obs("2951-2", 140.0, "mEq/L", 133.0, 145.0)]
    P.perturb_patient(r, "pt-y", SPEC, CODE, perturb_values=False, perturb_ranges=False)
    assert r[0]["valueQuantity"]["unit"] == "mEq/L"          # perturbation alone touches nothing
    assert P.randomise_units(r, "pt-y", RTAB, CODE) > 0      # but randomisation still fires


def test_a_rare_code_still_gets_its_PROPERTY_based_scale_rule():
    """a30 skips any code with fewer than MIN_OBS = 200 observations, because below that the value
    distribution says little. But the a30 precedence puts PROPERTY first, and PROPERTY is published
    LOINC metadata that needs no observations. Take 41284-1 (Non-sq Epi Cells #/area UrnS HPF,
    "Epithelial cells.non-squamous [#/area] in Urine sediment", 64 observations, PROPERTY=Naric).
    Falling through to plain `ratio` would give it a 71% relative CV instead of Poisson, so a count
    of 1 would clamp to 0. Its siblings 13945-1 (RBC #/area UrnS HPF) and 5787-7 (Epi Cells #/area
    UrnS HPF) clear the threshold and are Poisson."""
    import csv as _csv
    from pathlib import Path as _P
    spec_csv = _P(__file__).resolve().parents[1] / "data" / "variability" / "perturbation_spec.csv"
    rows = list(_csv.DictReader(open(spec_csv)))
    loinc_csv = str(locations.UPSTREAM / "Loinc.csv")
    with open(loinc_csv, newline="", encoding="utf-8", errors="replace") as fh:
        prop = {r["LOINC_NUM"]: r["PROPERTY"] for r in _csv.DictReader(fh)}
    LOG = {"LsCnc", "LnCnc", "LaCnc", "LogRto"}
    # Only for codes a30 never saw. Where a30 did measure, its empirical branches are meant to
    # outrank the PROPERTY ones (`ladder` beats `count_poisson` and `integer_grid` beats
    # `bounded_frac`). Asserting otherwise would flag the documented a30 precedence as a defect.
    bad = []
    for r in rows:
        if r["tier0_no_perturb"] == "1" or r["scale_source"].startswith("a30"):
            continue
        p = prop.get(r["loinc"], "")
        if p == "Naric" and r["scale_class"] != "count_poisson":
            bad.append((r["loinc"], p, r["scale_class"]))
        if p in LOG and r["scale_class"] != "log":
            bad.append((r["loinc"], p, r["scale_class"]))
    assert not bad, f"PROPERTY-implied scale rule not applied to a code a30 never saw: {bad[:8]}"


def test_every_fraction_of_a_whole_is_clamped_to_0_100_whatever_its_scale_class():
    """G2 applies to every fraction of a whole, whatever its scale class. A fraction cannot exceed
    100, however precisely it is reported. Deriving the flag from `scale_class == "bounded_frac"`
    would lose it on every fraction that a30 classes as `integer_grid`, because the a30 precedence
    puts the >=99%-integer test first. Oxygen saturation would then reach 113% at z=+3."""
    import csv as _csv
    from pathlib import Path as _P
    rows = list(_csv.DictReader(open(_P(__file__).resolve().parents[1]
                                     / "data" / "variability" / "perturbation_spec.csv")))
    with open(str(locations.UPSTREAM / "Loinc.csv"), newline="",
              encoding="utf-8", errors="replace") as fh:
        L = {r["LOINC_NUM"]: r for r in _csv.DictReader(fh)}
    FRAC = {"NFr", "MFr", "VFr", "CFr", "SFr", "AFr"}
    bad = [r["loinc"] for r in rows
           if L.get(r["loinc"], {}).get("PROPERTY", "") in FRAC
           and L.get(r["loinc"], {}).get("EXAMPLE_UCUM_UNITS", "").strip() in ("%", "% of total", "")
           and r["clamp_0_100"] != "1"]
    assert not bad, f"fraction codes with no [0,100] clamp: {bad[:8]}"
    # and the guard binds
    # 20564-1 SaO2 % Bld
    sat = SPEC["20564-1"]                                    # oxygen saturation
    assert sat["clamp_0_100"]
    assert P.perturb_number(98.0, 3.0, sat, sat["value_cv"]) <= 100.0


def test_every_spec_row_records_where_its_scale_came_from():
    for code, s in SPEC.items():
        pass
    import csv as _csv
    from pathlib import Path as _P
    rows = list(_csv.DictReader(open(_P(__file__).resolve().parents[1]
                                     / "data" / "variability" / "perturbation_spec.csv")))
    assert all(r["scale_source"] for r in rows)
    n_prop = sum(1 for r in rows if not r["scale_source"].startswith("a30"))
    assert n_prop > 0, "no code fell back to PROPERTY, so the fallback is untested"
