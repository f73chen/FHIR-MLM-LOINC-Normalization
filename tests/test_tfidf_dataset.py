"""Tests for pipeline/06_tfidf_dataset.py on synthetic observations."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "tfds", Path(__file__).resolve().parents[1] / "pipeline" / "06_tfidf_dataset.py"
)
td = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(td)

LAB = td.LAB_CODE_SYSTEM
ID = lambda r: r                       # identity unit normalizer
QUAL = lambda r: (r or None)           # identity qual normalizer (blank -> None)


def lab_obs(code, **kw):
    o = {"resourceType": "Observation", "code": {"coding": [{"system": LAB, "code": code}]}}
    o.update(kw)
    return o


def test_is_lab_obs():
    assert td.is_lab_obs(lab_obs("50912"))
    assert not td.is_lab_obs({"resourceType": "Condition", "code": {"coding": [{"system": LAB, "code": "x"}]}})
    assert not td.is_lab_obs({"resourceType": "Observation", "code": {"coding": [{"system": "other", "code": "x"}]}})
    assert not td.is_lab_obs({"resourceType": "Observation", "code": {"coding": [{"system": LAB}]}})  # no code


def test_extract_loinc_code():
    m = {"50912": "2160-0"}                                                # Creat SerPl-mCnc
    assert td.extract_loinc_code(lab_obs("50912"), m) == "2160-0"          # mimic itemid -> loinc
    assert td.extract_loinc_code(lab_obs("99999"), m) is None              # unmapped
    # direct loinc-system coding returns the code itself
    o = {"code": {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]}}
    assert td.extract_loinc_code(o, m) == "2160-0"
    assert td.extract_loinc_code({"code": {"coding": []}}, m) is None


def test_format_prompt_quantity():
    o = lab_obs("50912",
                valueQuantity={"value": 1.2, "unit": "mg/dL", "comparator": ">"},
                referenceRange=[{"low": {"value": 0.5, "unit": "mg/dL"}, "high": {"value": 1.0, "unit": "mg/dL"}}],
                extension=[{"valueString": "STAT"}])
    got = td.format_observation_as_prompt(o, "Creatinine, serum", ID, QUAL)
    assert got == ("Description: Creatinine, serum\n"
                   "Value: >1.2 mg/dL\n"
                   "ReferenceRange: Low: 0.5 mg/dL, High: 1.0 mg/dL\n"
                   "Priority: STAT")


def test_format_prompt_qual_and_blank():
    o = lab_obs("x", valueString="positive")
    assert td.format_observation_as_prompt(o, "Test", ID, QUAL) == "Description: Test\nValue: positive"
    # qual normalizer returning None (blank) omits the Value line entirely
    o2 = lab_obs("x", valueString="")
    assert td.format_observation_as_prompt(o2, "Test", ID, QUAL) == "Description: Test"


def test_tally_filters_to_label_set_and_paraphrases():
    import random
    res = [lab_obs("a", valueQuantity={"value": 1, "unit": "x"}),   # LA in set + para -> keep
           lab_obs("b", valueQuantity={"value": 2, "unit": "x"}),   # LB not in set -> drop
           {"resourceType": "Procedure"},                          # not an obs -> drop
           lab_obs("c", valueQuantity={"value": 3, "unit": "x"})]   # LC in set but no para -> drop
    m = {"a": "LA", "b": "LB", "c": "LC"}
    out = td.tally_patient_samples(res, m, {"LA", "LC"}, {"LA": ["pA"]}, random.Random(0), ID, QUAL)
    assert len(out) == 1 and out[0][1] == "LA"
    assert out[0][0].startswith("Description: pA")
