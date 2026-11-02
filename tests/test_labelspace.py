"""Tests for pipeline/02_derive_labelspace.py on synthetic fixtures. No MIMIC data is needed, nor
ndjson or tqdm, which scan_split imports lazily.

  python -m pytest tests/test_labelspace.py -q
"""
import importlib.util
from collections import Counter
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "labelspace", Path(__file__).resolve().parents[1] / "pipeline" / "02_derive_labelspace.py"
)
ls = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ls)

LSYS = {"http://lab"}  # stand-in for DEFAULT_LAB_SYSTEMS


def obs(system, code):
    return {"resourceType": "Observation", "code": {"coding": [{"system": system, "code": code}]}}


def test_iter_lab_codes_filters():
    resources = [
        obs("http://lab", "A"),                                   # lab -> yield
        obs("http://lab", "A"),                                   # second occurrence -> yield again
        obs("http://other", "X"),                                 # non-lab system -> skip
        {"resourceType": "Condition", "code": {"coding": [{"system": "http://lab", "code": "Z"}]}},  # not Observation
        {"resourceType": "Observation", "code": {"coding": []}},  # empty coding -> skip
        {"resourceType": "Observation", "code": {"coding": [{"system": "http://lab"}]}},  # no code -> skip
        obs("http://lab", "B"),                                   # lab -> yield
    ]
    assert ls.iter_lab_codes(resources, LSYS) == ["A", "A", "B"]


def test_tally_patient_head_and_mapped():
    head = {"A", "B", "C"}              # A, B, C are in the model's 1623-code head; OUT is not
    code_to_loinc = {"A": "LA", "B": "LB"}  # C in head but not LOINC-mapped
    resources = [obs("http://lab", "A"), obs("http://lab", "A"),
                 obs("http://lab", "B"), obs("http://lab", "C"),
                 obs("http://lab", "OUT")]   # a labitems code outside the head, like 53153
    occ, n_lab, n_head, n_mapped = ls.tally_patient(resources, LSYS, head, code_to_loinc)
    assert occ == Counter({"LA": 2, "LB": 1})
    assert n_lab == 5      # all 5 labitems obs
    assert n_head == 4     # A,A,B,C (OUT not in head)
    assert n_mapped == 3   # A,A,B (C in head but unmapped)


def test_compose_code_to_loinc_drops_unmappable_classes():
    mimic_code_to_class = {"x": 0, "y": 1, "z": 2}
    class_to_loinc = {0: "LA", 2: "LC"}  # class 1 (code y) not LOINC-mappable
    assert ls.compose_code_to_loinc(mimic_code_to_class, class_to_loinc) == {"x": "LA", "z": "LC"}


def _fixture_counts():
    # D: 10 occ but only 2 patients (occurrence-inflated); C: 9 occ (below threshold).
    train_occ = Counter({"A": 100, "B": 50, "D": 10, "C": 9})
    train_pat = Counter({"A": 100, "B": 50, "D": 2, "C": 9})
    test_occ = Counter({"A": 80, "B": 5, "D": 1})
    test_pat = Counter({"A": 60, "B": 5, "D": 1})
    return train_occ, train_pat, test_occ, test_pat


def test_primary_filter_and_ranking():
    r = ls.build_labelspace(*_fixture_counts(), min_occ=10, topn_candidates=[2])
    assert r["primary"] == ["A", "B", "D"]           # C (9 occ) excluded; ranked by train_occ desc
    assert r["distinct_loinc_in_train"] == ["A", "B", "D", "C"]


def test_distinct_patient_aside():
    r = ls.build_labelspace(*_fixture_counts(), min_occ=10, topn_candidates=[2])
    assert r["by_patients"] == ["A", "B"]            # D has only 2 patients -> excluded
    assert r["only_in_occ"] == ["D"]                 # >=10 occ but <10 patients
    assert r["only_in_pat"] == []                    # impossible (patients <= occurrences)


def test_macro_core_test_floor():
    # A=80, B=5, D=1 test obs; core bar >=20 -> only A qualifies (and only among primary).
    r = ls.build_labelspace(*_fixture_counts(), min_occ=10, min_core_test_obs=20, topn_candidates=[2])
    assert r["macro_core"] == ["A"]
    # core is always a subset of primary
    assert set(r["macro_core"]) <= set(r["primary"])


def test_floor_table():
    r = ls.build_labelspace(*_fixture_counts(), min_occ=10, topn_candidates=[2])
    f2 = r["floor_rows"][2]                           # top-2 = {A, B}
    assert f2["floor_train_occ"] == 50               # B is the 2nd
    assert f2["floor_test_occ"] == 5                 # min(test A=80, B=5)
    f3 = r["floor_rows"][3]                           # full primary set {A,B,D}
    assert f3["floor_test_occ"] == 1                 # D has 1 test obs
    # the cumulative train coverage is over all mapped train obs, including the below-threshold C (9):
    # (100+50)/(100+50+10+9) = 150/169
    assert abs(f2["cum_train_coverage"] - 150 / 169) < 1e-9
