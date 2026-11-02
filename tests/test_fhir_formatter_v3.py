"""Tests for the v2/v3 unit standardisation in common/fhir_formatter.py, which applies to the lab
call sites only.

`_quantity_text` has six callers: the lab valueQuantity, the reference range, and four medication
fields (dispense quantity, administration dose, dosage doseQuantity, and the ratio helper). Moving
the v2 transform into `_quantity_text` is the obvious one-line implementation, but it would
multiply heparin doses by 1000 on their way to a laboratory unit. These tests catch that.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import unit_standardisation as us                                    # noqa: E402
from fhir_formatter import FHIRFormatter, LAB_CODE_SYSTEM            # noqa: E402
from input_preprocessing_utils import load_unit_alias_map            # noqa: E402

CHART_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items"
SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"


@pytest.fixture(scope="module")
def umap():
    return load_unit_alias_map(str(locations.UPSTREAM / "unit_synonym_map.json"))


@pytest.fixture(scope="module")
def m06():
    """Loads 06 itself, so that the parity test uses the real TF-IDF renderer."""
    spec = importlib.util.spec_from_file_location("m06", SCRIPTS / "06_tfidf_dataset.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def fmts(umap):
    """The units are the standardised targets, emitted verbatim. They are not mapped back to a
    MIMIC surface form.

    The FHIR-MLM unit tokens are keyed on raw surface forms, so a target like `10*3/uL` might seem
    to emit [UNK]. It does not, because the MLM builder (09) tokenises through NdjsonTokenizer,
    which does not see rendered text. It bins the value into the percentile table of the item and
    looks the unit up from the untouched FHIR `Quantity.code`. The two builders that use
    FHIRFormatter, Llama (11) and BCM (10), feed BPE text tokenizers that can tokenise any string.
    Remapping there would only make the deep-model text disagree with the TF-IDF text.
    test_value_and_units_match_the_tfidf_path_exactly exists to prevent that."""
    def quantity_fmt(value, raw_unit):
        return us.standardise_quantity(value, raw_unit, umap)

    def range_fmt(rrs, value_raw_unit):
        return us.standardise_reference_range(rrs, value_raw_unit, umap)
    return quantity_fmt, range_fmt


def obs(code, value, unit, lo=None, hi=None, system=LAB_CODE_SYSTEM, rr_unit=...):
    ru = unit if rr_unit is ... else rr_unit
    o = {"resourceType": "Observation", "id": f"o-{code}",
         "code": {"coding": [{"system": system, "code": code, "display": "d"}]},
         "valueQuantity": {"value": value, "unit": unit, "code": unit}}
    if lo is not None:
        q = (lambda v: {"value": v} if ru == "" else {"value": v, "unit": ru, "code": ru})
        o["referenceRange"] = [{"low": q(lo), "high": q(hi)}]
    return o


def pair(resources, fmts, bins):
    v1 = FHIRFormatter({}, bins, mask_token="[MASK]")
    v3 = FHIRFormatter({}, bins, mask_token="[MASK]",
                       quantity_fmt=fmts[0], range_fmt=fmts[1])
    return v1.format_resources(resources)[0], v3.format_resources(resources)[0]


def lines(block, prefix):
    return [l.strip() for l in block.splitlines() if l.strip().startswith(prefix)]


# -------------------------- the lab path transforms --------------------------
def test_lab_value_and_range_are_rescaled_together(fmts):
    o = obs("51222", 14.0, "g/dL", 12.0, 16.0)          # g/dL -> mg/dL, factor 1000
    _v1, v3 = pair([o], fmts, {"51222": [0.0, 1.0]})
    assert lines(v3, "ValueQuantity:") == ["ValueQuantity: 1.400e+04 mg/dL"]
    assert lines(v3, "ReferenceRange:") == \
        ["ReferenceRange: Low: 1.200e+04 mg/dL, High: 1.600e+04 mg/dL"]


def test_a_bound_with_no_unit_of_its_own_inherits_the_values(fmts):
    """2.25% of MIMIC's bounds state no unit. Left alone they would print unscaled beside a
    rescaled value."""
    o = obs("51222", 14.0, "g/dL", 12.0, 16.0, rr_unit="")
    _v1, v3 = pair([o], fmts, {"51222": [0.0, 1.0]})
    assert lines(v3, "ReferenceRange:") == \
        ["ReferenceRange: Low: 1.200e+04 mg/dL, High: 1.600e+04 mg/dL"]


CASES = [("51222", 14.0, "g/dL", 12.0, 16.0), ("50983", 140.0, "mEq/L", 133.0, 145.0),
         ("51301", 7.4, "K/uL", 4.0, 11.0), ("50861", 30.0, "IU/L", 0.0, 40.0),
         ("51493", 4.0, "#/hpf", 0.0, 5.0), ("51274", 12.5, "sec", 9.4, 12.5),
         ("50813", 1.2, "mmol/L", 0.5, 2.0), ("51256", 60.0, "%", 50.0, 70.0)]


def test_value_and_units_match_the_tfidf_path_exactly(fmts, m06, umap):
    """Path parity. The deep models and TF-IDF must state the same number in the same unit for
    the same observation, or the two model families are not asked the same question. Both go
    through unit_standardisation, so the test checks that the two call sites agree, because that
    is where they can diverge. It does not test the shared transform."""
    bins = {c: [0.0, 1.0] for c, *_ in CASES}
    for code, v, u, lo, hi in CASES:
        o = obs(code, v, u, lo, hi)
        _v1, deep = pair([o], fmts, bins)
        flat = m06.format_observation_as_prompt(
            o, "d", unit_norm=lambda x: x, qual_norm=lambda x: x,
            quantity_fmt=fmts[0], include_reference_range=True, range_fmt=fmts[1])
        got_v = lines(deep, "ValueQuantity:")[0][len("ValueQuantity: "):]
        want_v = next(l for l in flat.splitlines() if l.startswith("Value: "))[len("Value: "):]
        assert got_v == want_v, f"{code}: deep {got_v!r} vs tfidf {want_v!r}"
        got_r = lines(deep, "ReferenceRange:")[0][len("ReferenceRange: "):]
        want_r = next(l for l in flat.splitlines()
                      if l.startswith("ReferenceRange: "))[len("ReferenceRange: "):]
        assert got_r == want_r, f"{code}: deep {got_r!r} vs tfidf {want_r!r}"


# ----------------------- everything else is left alone -----------------------
MCC = {"coding": [{"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication",
                   "code": "HEPA5I"}]}
LAB_UNIT_DOSE = {"value": 2.0, "unit": "g/dL", "code": "g/dL"}


def test_medication_quantities_are_untouched(fmts):
    """The six-call-site problem in its most visible form. Every one of these doses is stated in
    g/dL, a unit that the lab transform rescales by 1000. None of them may change."""
    res = [
        {"resourceType": "MedicationDispense", "id": "d1",
         "medicationCodeableConcept": MCC, "quantity": dict(LAB_UNIT_DOSE)},
        {"resourceType": "MedicationAdministration", "id": "a1",
         "medicationCodeableConcept": MCC, "dosage": {"dose": dict(LAB_UNIT_DOSE, value=5.0)}},
        {"resourceType": "MedicationRequest", "id": "r1", "status": "active",
         "medicationCodeableConcept": MCC,
         "dosageInstruction": [{"doseAndRate": [
             {"doseQuantity": dict(LAB_UNIT_DOSE, value=7.0)}]}]},
    ]
    v1, v3 = pair(res, fmts, {"dose": [0.0, 1.0]})
    assert v1 == v3
    for expect in ("2.0 g/dL", "5.0 g/dL", "7.0 g/dL"):
        assert expect in v3, f"{expect} missing, so the test would pass vacuously"


def test_a_non_lab_observation_is_untouched(fmts):
    """The transform is gated on the LOINC-backed lab code system. A chart observation with a
    valueQuantity stays as it was."""
    o = obs("220045", 14.0, "g/dL", 12.0, 16.0, system=CHART_SYSTEM)
    v1, v3 = pair([o], fmts, {"220045": [0.0, 1.0]})
    assert v1 == v3
    assert "14.0 g/dL" in v3


def test_passing_neither_formatter_is_byte_identical_to_v1(fmts):
    res = [obs("51222", 14.0, "g/dL", 12.0, 16.0),
           {"resourceType": "MedicationDispense", "id": "d1",
            "medicationCodeableConcept": MCC, "quantity": dict(LAB_UNIT_DOSE)}]
    bins = {"51222": [0.0, 1.0], "dose": [0.0, 1.0]}
    a = FHIRFormatter({}, bins, mask_token="[MASK]").format_resources(res)[0]
    b = FHIRFormatter({}, bins, mask_token="[MASK]",
                      quantity_fmt=None, range_fmt=None).format_resources(res)[0]
    assert a == b


def test_c11_suppresses_the_value_but_not_the_range(fmts):
    """This v1 behaviour is kept. ReferenceRange is not gated on numeric_bins, so 1.5% of ranged
    lab observations emit a bound whose value line C11 removed. The bound must still be rescaled
    with the unit of the suppressed value."""
    o = obs("99999", 14.0, "g/dL", 12.0, 16.0)
    _v1, v3 = pair([o], fmts, {})                       # empty bin table
    assert lines(v3, "ValueQuantity:") == []
    assert lines(v3, "ReferenceRange:") == \
        ["ReferenceRange: Low: 1.200e+04 mg/dL, High: 1.600e+04 mg/dL"]


def test_the_fhir_comparator_survives_the_v3_transform(fmts, m06, umap):
    """`_quantity_text` does not read `comparator`. Without separate handling in the lab branch, a
    censored urine microscopy result would render as `37` on the deep path and `>37` on the
    TF-IDF path. A censored result is not the same measurement as an uncensored one.

    Only the v2/v3 branch emits the comparator. v1 rendering is kept as it is so that v1 datasets
    and models stay comparable. This is why the v1 half of this test asserts the form without the
    comparator."""
    o = obs("51516", 37.0, "#/hpf", 0.0, 5.0)
    o["valueQuantity"]["comparator"] = ">"
    v1, v3 = pair([o], fmts, {"51516": [0.0, 1.0]})
    assert lines(v3, "ValueQuantity:") == ["ValueQuantity: >3.700e+01 {#}/[HPF]"]
    assert lines(v1, "ValueQuantity:") == ["ValueQuantity: 37.0 #/hpf"]      # v1, unchanged

    flat = m06.format_observation_as_prompt(
        o, "d", unit_norm=lambda x: x, qual_norm=lambda x: x,
        quantity_fmt=fmts[0], include_reference_range=True, range_fmt=fmts[1])
    want = next(l for l in flat.splitlines() if l.startswith("Value: "))[len("Value: "):]
    assert lines(v3, "ValueQuantity:")[0][len("ValueQuantity: "):] == want
