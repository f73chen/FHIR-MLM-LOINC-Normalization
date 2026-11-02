"""Tests for the `code_last` option of common/fhir_formatter.py, on synthetic resources.

Llama is causal and predicts each lab code at its mask. For Llama, the `Code:` line therefore moves
to the end of the lines of its Observation. The tests check that:
  - off (the default) keeps the usual line order, so BCM, TF-IDF and every analysis are unaffected.
  - on moves one line per Observation. The lines, the codes and the order of the codes stay the
    same, and the masks in the text still appear in the order of the returned codes.
    tokenize_and_align zips them in that order.
  - inlined hasMember children stay after the code of their parent.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import unit_standardisation as us                                    # noqa: E402
from fhir_formatter import FHIRFormatter, LAB_CODE_SYSTEM            # noqa: E402
from input_preprocessing_utils import load_unit_alias_map            # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"
CHART_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items"
BINS = {"50983": [130.0, 135.0, 140.0, 145.0], "50902": [95.0, 100.0, 105.0]}
MASK = "[MASK]"


def q(v, unit="mEq/L"):
    return {"value": v, "unit": unit, "code": unit}


def lab(oid, code, display, edt, value=None, lo=None, hi=None, members=(), note=None, interp=None,
        system=LAB_CODE_SYSTEM):
    o = {"resourceType": "Observation", "id": oid, "effectiveDateTime": edt,
         "code": {"coding": [{"system": system, "code": code, "display": display}]}}
    if value is not None:
        o["valueQuantity"] = q(value)
    if lo is not None:
        o["referenceRange"] = [{"low": q(lo), "high": q(hi)}]
    if members:
        o["hasMember"] = [{"reference": f"Observation/{m}"} for m in members]
    if note:
        o["note"] = [{"text": note}]
    if interp:
        o["interpretation"] = [{"coding": [{"code": interp}]}]
    return o


SODIUM = lab("o-na", "50983", "Sodium", "2154-03-02T09:41:00-05:00", value=139, lo=133, hi=145,
             note="hemolyzed", interp="N")


def fmt(**kw):
    return FHIRFormatter({}, BINS, mask_token=MASK, include_description=kw.pop("include_description", True), **kw)


def test_default_is_off_and_keeps_the_historical_order():
    f = fmt()
    assert f.code_last is False
    text, codes = f.format_resources([SODIUM])
    assert (text, codes) == fmt(code_last=False).format_resources([SODIUM])
    lines = text.split("\n")
    # the default order: Resource, Description, Code, then the rest
    assert lines[:3] == ["Resource: Observation", "Description: Sodium", f"Code: {MASK}"]
    assert codes == ["50983"]


def test_code_last_moves_exactly_one_line():
    off_text, off_codes = fmt().format_resources([SODIUM])
    on_text, on_codes = fmt(code_last=True).format_resources([SODIUM])
    off, on = off_text.split("\n"), on_text.split("\n")
    expected = [ln for ln in off if ln != f"Code: {MASK}"] + [f"Code: {MASK}"]
    assert on == expected
    assert on_codes == off_codes == ["50983"]
    # with code_last, every field the causal model needs comes before the mask
    for field in ("Description:", "EffectiveDateTime:", "ValueQuantity:", "ReferenceRange:",
                  "Interpretation:", "Note:"):
        assert any(ln.startswith(field) for ln in on[:-1]), field


def test_two_observations_keep_their_codes_in_order():
    k = lab("o-cl", "50902", "Chloride", "2154-03-02T10:00:00-05:00", value=101, lo=98, hi=107)
    off_text, off_codes = fmt().format_resources([SODIUM, k])
    on_text, on_codes = fmt(code_last=True).format_resources([SODIUM, k])
    assert on_codes == off_codes == ["50983", "50902"]
    blocks = on_text.split("\n\n")
    assert len(blocks) == 2 and all(b.split("\n")[-1] == f"Code: {MASK}" for b in blocks)
    assert sorted(on_text.split("\n")) == sorted(off_text.split("\n"))


def test_inlined_children_follow_the_parents_code():
    # the parent sorts first, so the child is inlined (and marked seen) before its own top-level turn
    child = lab("o-child", "50902", "Chloride", "2154-03-02T09:50:00-05:00", value=101, lo=98, hi=107)
    parent = lab("o-parent", "50983", "Sodium", "2154-03-02T09:41:00-05:00", value=139, lo=133, hi=145,
                 members=("o-child",))
    off_text, off_codes = fmt().format_resources([parent, child])
    on_text, on_codes = fmt(code_last=True).format_resources([parent, child])
    assert on_codes == off_codes == ["50983", "50902"]
    lines = on_text.split("\n")
    own = [i for i, ln in enumerate(lines) if not ln.startswith("  ")]
    nested = [i for i, ln in enumerate(lines) if ln.startswith("  ")]
    assert nested, "the child was not inlined"
    assert lines[own[-1]] == f"Code: {MASK}"            # the parent's code closes its own lines
    assert own[-1] < nested[0]                           # ... and comes before the child's block
    assert lines[nested[-1]] == f"  Code: {MASK}"        # the child's code closes the child's lines
    # the masks appear in the text in the order of the returned codes
    mask_owner = ["parent" if not ln.startswith("  ") else "child" for ln in lines if ln.endswith(f"Code: {MASK}")]
    assert mask_owner == ["parent", "child"]


def test_a_non_lab_code_moves_too_and_is_not_a_label():
    o = lab("o-chart", "220045", "Heart Rate", "2154-03-02T09:41:00-05:00", system=CHART_SYSTEM)
    on_text, on_codes = fmt(code_last=True).format_resources([o])
    assert on_text.split("\n")[-1] == "Code: 220045"
    assert on_codes == []


def test_without_a_description():
    off_text, _ = fmt(include_description=False).format_resources([SODIUM])
    on_text, codes = fmt(include_description=False, code_last=True).format_resources([SODIUM])
    assert not any(ln.startswith("Description:") for ln in on_text.split("\n"))
    assert on_text.split("\n")[-1] == f"Code: {MASK}" and codes == ["50983"]
    assert sorted(on_text.split("\n")) == sorted(off_text.split("\n"))


def test_a_code_only_observation_is_the_same_either_way():
    o = {"resourceType": "Observation", "id": "o-bare", "effectiveDateTime": "2154-03-02T09:41:00-05:00",
         "code": {"coding": [{"system": LAB_CODE_SYSTEM, "code": "50983"}]}}
    o.pop("effectiveDateTime")
    a = fmt(include_description=False).format_resources([o])
    b = fmt(include_description=False, code_last=True).format_resources([o])
    assert a == b == (f"Resource: Observation\nCode: {MASK}", ["50983"])


def test_composes_with_the_v3_representation():
    umap = load_unit_alias_map(str(locations.UPSTREAM / "unit_synonym_map.json"))
    qf = lambda v, u: us.standardise_quantity(v, u, umap)
    rf = lambda rrs, u: us.standardise_reference_range(rrs, u, umap)
    off_text, off_codes = fmt(quantity_fmt=qf, range_fmt=rf).format_resources([SODIUM])
    on_text, on_codes = fmt(quantity_fmt=qf, range_fmt=rf, code_last=True).format_resources([SODIUM])
    assert on_codes == off_codes
    assert on_text.split("\n") == [ln for ln in off_text.split("\n") if ln != f"Code: {MASK}"] + [f"Code: {MASK}"]
    assert any("e+02" in ln for ln in on_text.split("\n"))     # the v3 rendering is really on


def test_from_data_dirs_passes_the_flag():
    assert FHIRFormatter.from_data_dirs(mask_token=MASK).code_last is False
    assert FHIRFormatter.from_data_dirs(mask_token=MASK, code_last=True).code_last is True


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_the_llama_builder_turns_it_on_and_bcm_does_not():
    old_argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        m11, m10 = _load("11_llama_dataset"), _load("10_bcm_dataset")
    finally:
        sys.argv = old_argv
    assert m11.CODE_LAST is True
    assert m11._build_llama_tokenizer_formatter(True, True, "v3")[1].code_last is True
    assert m10._build_bcm_tokenizer_formatter(True, True, "v3")[1].code_last is False
