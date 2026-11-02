"""Tests for pipeline/16_tfidf_eval_dump.py on synthetic fixtures (no GPU, patient data or dataset).

Script 16 re-implements the 07 downsampling so that it can select training rows without loading every
training prompt. `test_downsample_*_identical_to_07` checks that the two pick the same rows. If they
diverged, the refit would not match 07, and the --verify check (16 against 07, same cell, same seed)
would compare two different fits.
"""
import importlib.util
import random
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"


def _load(name, mod):
    s = importlib.util.spec_from_file_location(mod, _SCRIPTS / name)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d16 = _load("16_tfidf_eval_dump.py", "d16")
d07 = _load("07_tfidf_train_eval.py", "d07")


# -- fidelity to 07 ---------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 4])
def test_downsample_indices_identical_to_07(seed):
    rng = random.Random(123)
    labels = [f"L{rng.randint(0, 40)}" for _ in range(20000)]
    a = d16.downsample_indices(labels, 5000, 50, seed)
    b = d07.downsample_indices(labels, 5000, 50, seed)
    assert a == b, "16's downsample diverged from 07, so the refit is no longer faithful"


def test_threshold_targets_identical_to_07():
    rng = random.Random(7)
    counts = {f"L{i}": rng.randint(1, 5000) for i in range(60)}
    assert d16.threshold_targets(counts, 20000, 500) == d07.threshold_targets(counts, 20000, 500)


def test_pinned_recipe_identical_to_07():
    assert d16.PINNED == d07.PINNED
    assert d16.SUBSET_SIZE == d07.SUBSET_SIZE
    assert d16.DOWNSAMPLE_MIN == d07.DOWNSAMPLE_MIN
    assert d16.TOPK == d07.TOPK
    assert d16.EVAL_BATCH == d07.EVAL_BATCH


def test_downsample_respects_budget_and_floor():
    labels = ["big"] * 10000 + ["small"] * 20
    idx = d16.downsample_indices(labels, 1000, 100, 0)
    got = {}
    for i in idx:
        got[labels[i]] = got.get(labels[i], 0) + 1
    assert got["small"] == 20, "classes at/below the floor must be taken whole"
    assert got["big"] >= 100, "downsampled classes keep the floor"


# -- lab-observation filter -------------------------------------------------
def _obs(system=d16.LAB_CODE_SYSTEM, code="50912", rtype="Observation"):
    return {"resourceType": rtype, "code": {"coding": [{"system": system, "code": code}]}}


def test_is_lab_obs_accepts_lab_system():
    assert d16.is_lab_obs(_obs())


def test_is_lab_obs_rejects_non_observation_and_other_systems():
    assert not d16.is_lab_obs(_obs(rtype="Condition"))
    assert not d16.is_lab_obs(_obs(system="http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-microbiology-test"))
    assert not d16.is_lab_obs({"resourceType": "Observation", "code": {"coding": []}})
    assert not d16.is_lab_obs(_obs(code=None))


# -- obs_ordinal semantics: the join key contract ---------------------------
def test_obs_ordinal_counts_mappable_not_scored():
    """An observation that is LOINC-mappable but outside the 679-code TF-IDF label space still takes an
    ordinal. Otherwise the ordinals drift out of step with the deep-model dumps, and every paired
    comparison compares different observations."""
    mimic_code_to_class = {"A": 1, "B": 2, "C": 3}
    mappable = {1, 2, 3}
    mimic_to_loinc = {"A": "1000-1", "B": "2000-2", "C": "3000-3"}
    label_679 = {"1000-1", "3000-3"}                 # B is mappable but not scorable
    para = {"1000-1": ["pa"], "2000-2": ["pb"], "3000-3": ["pc"]}
    resources = [_obs(code=c) for c in ("A", "B", "C")]

    import sys, types
    sys.modules.setdefault("ndjson", types.ModuleType("ndjson"))
    fake = types.ModuleType("fhir_formatter")
    fake.get_encounters = lambda r: {}
    fake.sort_key = lambda r, e: "0"              # stable -> preserves input order
    sys.modules["fhir_formatter"] = fake

    rows = _run_build(resources, ["u1"], label_679, mimic_to_loinc, mimic_code_to_class, mappable, para)

    ords = [(r[0], r[1], r[3]) for r in rows]
    assert [o for _, o, _ in ords] == [0, 2], (
        f"expected ordinals [0,2] (B consumes ordinal 1 without being scored), got {ords}")
    assert [l for _, _, l in ords] == ["1000-1", "3000-3"]
    assert d16.build_test_rows.stats["n_mappable"] == 3
    assert d16.build_test_rows.stats["n_emitted"] == 2


def _run_build(resources, uuids, label_set, m2l, m2c, mappable, pool, **kw):
    """Drive build_test_rows against synthetic resources (no ndjson on disk)."""
    import sys, types, io, builtins
    sys.modules.setdefault("ndjson", types.ModuleType("ndjson"))
    fake = types.ModuleType("fhir_formatter")
    fake.get_encounters = lambda r: {}
    fake.sort_key = lambda r, e: "0"                 # stable -> preserves input order
    sys.modules["fhir_formatter"] = fake
    sys.modules["ndjson"].load = lambda f: resources
    real_open, real_exists = builtins.open, Path.exists
    Path.exists = lambda self: True
    builtins.open = lambda *a, **k: io.StringIO("")
    try:
        return list(d16.build_test_rows(uuids, label_set, m2l, m2c, mappable, pool,
                                        lambda x: "", lambda x: None,
                                        lambda o, p, un, qn: f"P:{p}", random.Random(0), **kw))
    finally:
        builtins.open, Path.exists = real_open, real_exists


# -- RELMA mode -------------------------------------------------------------
def _relma_fixture():
    m2c = {"A": 1, "B": 2}
    mappable = {1, 2}
    m2l = {"A": "1000-1", "B": "2000-2"}
    label_679 = {"1000-1", "2000-2"}
    resources = [_obs(code="A"), _obs(code="B")]
    return resources, m2c, mappable, m2l, label_679


def test_relma_mode_skips_codes_with_no_local_name():
    """Observations of codes with no RELMA name are dropped but still take an ordinal. Otherwise the
    join key falls out of step with the deep-model dumps."""
    resources, m2c, mappable, m2l, label_679 = _relma_fixture()
    pool = {"2000-2": ["HCT"]}                        # A has no RELMA name
    rows = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                      relma_salt="relma-s1-a0", clean_lookup={("2000-2", "HCT"): (True, True)})
    assert len(rows) == 1
    assert rows[0][1] == 1, "surviving observation keeps ordinal 1 (A consumed ordinal 0)"
    assert rows[0][3] == "2000-2" and rows[0][4] == "HCT"
    assert d16.build_test_rows.stats["n_skipped_no_relma_name"] == 1


def test_relma_mode_emits_name_and_clean_flags():
    resources, m2c, mappable, m2l, label_679 = _relma_fixture()
    pool = {"1000-1": ["SEEN"], "2000-2": ["NOVEL"]}
    clean = {("1000-1", "SEEN"): (False, False), ("2000-2", "NOVEL"): (True, True)}
    rows = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                      relma_salt="relma-s1-a0", clean_lookup=clean)
    got = {r[3]: (r[4], r[5], r[6]) for r in rows}
    assert got["1000-1"] == ("SEEN", False, False)
    assert got["2000-2"] == ("NOVEL", True, True)


def test_relma_assignment_is_per_patient_not_per_observation():
    """Every observation of one LOINC code in a patient gets the same name. This matches the name the
    deep models see and keeps paired tests comparing like with like."""
    m2c = {"A": 1}; mappable = {1}; m2l = {"A": "1000-1"}; label_679 = {"1000-1"}
    resources = [_obs(code="A"), _obs(code="A"), _obs(code="A")]
    pool = {"1000-1": [f"name{i}" for i in range(20)]}
    rows = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                      relma_salt="relma-s1-a0", clean_lookup={})
    names = {r[4] for r in rows}
    assert len(rows) == 3 and len(names) == 1, f"expected one shared name, got {names}"


def test_relma_different_salt_changes_the_name():
    m2c = {"A": 1}; mappable = {1}; m2l = {"A": "1000-1"}; label_679 = {"1000-1"}
    resources = [_obs(code="A")]
    pool = {"1000-1": [f"name{i}" for i in range(50)]}
    a = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                   relma_salt="relma-s1-a0", clean_lookup={})[0][4]
    b = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                   relma_salt="relma-s1-a1", clean_lookup={})[0][4]
    a2 = _run_build(resources, ["u1"], label_679, m2l, m2c, mappable, pool,
                    relma_salt="relma-s1-a0", clean_lookup={})[0][4]
    assert a == a2, "same salt must be deterministic"
    assert a != b, "different assignment seeds must draw different names"


# -- train index mapping ----------------------------------------------------
def test_global_index_maps_to_file_and_line(tmp_path):
    """load_selected_prompts must map a global index back to the right (file, line)."""
    lab, sam = tmp_path / "train" / "labels", tmp_path / "train" / "samples"
    lab.mkdir(parents=True); sam.mkdir(parents=True)
    import json as _json
    files, counts, expect = [], [], {}
    g = 0
    for fi, (name, n) in enumerate([("a", 3), ("b", 2), ("c", 4)]):
        (lab / f"{name}.txt").write_text("\n".join(["L"] * n) + "\n")
        with open(sam / f"{name}.jsonl", "w") as f:
            for li in range(n):
                f.write(_json.dumps({"prompt": f"{name}-{li}", "label": "L"}) + "\n")
                expect[g] = f"{name}-{li}"; g += 1
        files.append(str(lab / f"{name}.txt")); counts.append(n)
    idx = [0, 4, 8, 2]
    got = d16.load_selected_prompts(str(tmp_path), files, counts, idx, lambda p, e, c=None: p, None)
    assert got == [expect[i] for i in idx]
    # both the exclude set and the coarsen setting must reach the extractor unchanged
    seen = []

    def _spy(prompt, excl, coarse):
        seen.append((excl, coarse))
        return prompt

    d16.load_selected_prompts(str(tmp_path), files, counts, idx, _spy, {"description"}, "1sf")
    assert len(seen) == len(idx)
    assert all(e == {"description"} and c == "1sf" for e, c in seen), seen


# -- --verify expectation: the 07 JSON, same cell, same seed ----------------
def _write07(tmp_path, cond_arm, pinned, per_seed):
    import json as _json
    (tmp_path / f"tfidf_{cond_arm}.json").write_text(
        _json.dumps({"pinned_params": pinned, "per_seed": per_seed}))


def test_expected_top1_reads_07_per_seed_and_cell(tmp_path):
    per = [{"seed": 0, "micro": {"top1": 0.91}, "eval_sets": {"pert": {"micro": {"top1": 0.81}}}},
           {"seed": 1, "micro": {"top1": 0.92}, "eval_sets": {"pert": {"micro": {"top1": 0.82}}}}]
    _write07(tmp_path, "para_xsite__clean", d16.PINNED, per)
    f = d16.expected_top1_from_07
    assert f("para_xsite", "__clean", "", 1, tmp_path)[0] == 0.92          # home cell, seed 1
    assert f("para_xsite", "__clean", "__onpert", 0, tmp_path)[0] == 0.81  # cross cell, seed 0
    assert f("para_xsite", "__clean", "", 7, tmp_path)[0] is None          # seed absent
    assert f("para_xsite", "__clean", "__onclean", 0, tmp_path)[0] is None  # eval set absent
    assert f("para_xsite", "__pert", "", 0, tmp_path)[0] is None            # 07 never ran this arm


def test_expected_top1_refuses_a_07_result_from_another_recipe(tmp_path):
    """A 07 JSON from another recipe, left in place, must not become the target a dump is checked against."""
    _write07(tmp_path, "para_xsite__clean", {**d16.PINNED, "stop_words": "english"},
             [{"seed": 0, "micro": {"top1": 0.97}}])
    v, why = d16.expected_top1_from_07("para_xsite", "__clean", "", 0, tmp_path)
    assert v is None and "different recipe" in why


def test_verify_tolerance_and_source():
    assert d16.VERIFY_TOL == 0.001
    assert d16.RESULTS_07 == d07.RESULTS_DIR
    assert not hasattr(d16, "EXPECTED_TOP1"), "the hand-copied table must not come back"
