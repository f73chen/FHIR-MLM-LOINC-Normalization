"""
Tests for the FHIR-MLM reference-range ablation (`*_norr`) and the perturbed `xsite` arm in
pipeline/09_mlm_dataset.py. norr drops `referenceRange` from `relevant_keys` in the same way that
`nodesc` drops `code:display`.

Checked here:

  1. The default path is byte-identical to the output pinned before the flag existed. Every
     reported FHIR-MLM number came from it.
  2. For a patient under the 8192 cap, norr is a strict subsequence of full (tokens removed, none
     added, none reordered), and the lab-code tokens (the labels) are untouched.
  3. For a patient at the cap, norr is not expected to be a subsequence. Shorter observations pack
     more of the record into the same window, so norr reaches observations that full never saw.
     This is the same denominator effect as for `nodesc` (+32% observations), and the analysis has
     to handle it.

Slow (~3 min), because each tokenizer build loads the FHIR static references. Marked so it can be
deselected.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
_s = importlib.util.spec_from_file_location("d9", ROOT / "pipeline" / "09_mlm_dataset.py")
d9 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d9)

from conditions import (MLM_CONDITIONS, has_description, includes_reference_range,  # noqa: E402
                        refrange_mode, representation)

N_PATIENTS = 40
pytestmark = pytest.mark.slow


def _strip_pad(ids, am):
    return [t for t, a in zip(ids, am) if a == 1]


def _is_subseq(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


@pytest.fixture(scope="module")
def rows():
    uuids = d9.read_uuid_list(d9.SPLITS_DIR / "test_patients.txt")[:N_PATIENTS]
    m2l = d9._load_seen_codes_mimic_to_loinc(d9.SEEN_CODES_PATH)
    out = {}
    for key, (wd, wr) in {"full": (True, True), "nodesc": (False, True), "norr": (True, False)}.items():
        tok = d9._build_fhir_tokenizer(wd, wr)
        lo, hi = d9._lab_token_range(tok)
        rs = []
        for u in uuids:
            r = d9._process_one_patient(u, tok, None, m2l, lo, hi, False, d9._new_stats(1))
            rs.append(None if r is None else
                      (r["input_ids"], r["token_type_ids"], r["attention_mask"], r["special_tokens_mask"]))
        out[key] = rs
    out["_lab_range"] = (lo, hi)
    return out


def test_condition_slugs_reject_unsupported_coarsening():
    """rr1sf/rrmag have no surface string to coarsen in FHIR-MLM, so 09 must refuse them.

    `norr_v2` is excluded for a different reason and must stay excluded. The v2 value transform
    cannot produce a distinct arm for an encoder that bins each value into the percentile table of
    its item. Done coherently, it is a no-op, because a percentile is invariant under a monotone
    rescale. Done naively, it corrupts the values. In this code it is never reached, because the
    transform is in the renderer. `xsite` is included because perturbation adds per-patient noise.
    The noisy values are binned against the same fixed tables the model was trained on."""
    assert MLM_CONDITIONS == ["para", "para_norr", "nodesc", "nodesc_norr",
                              "para_xsite", "nodesc_xsite"]
    for c in MLM_CONDITIONS:
        assert refrange_mode(c) in ("full", "norr", "xsite")
        assert representation(c) in ("v1", "v3")
    assert not any("norr_v2" in c for c in MLM_CONDITIONS)
    assert has_description("para_norr") and not has_description("nodesc_norr")


def test_xsite_keeps_its_reference_range():
    """`refrange_mode(c) == "full"` is False for `xsite`, so a builder or gate that decides by
    that comparison drops the reference range from xsite, the arm that is built around the perturbed
    reference range."""
    for c in ("para_xsite", "nodesc_xsite"):
        assert includes_reference_range(c), c
        assert refrange_mode(c) != "full"          # the trap: not "full", yet the range is kept
    for c in ("para_norr", "nodesc_norr", "para_norr_v2", "nodesc_norr_v2"):
        assert not includes_reference_range(c), c


def test_no_builder_decides_the_reference_range_by_string_comparison():
    """The rule is defined in conditions.REFRANGES and read through includes_reference_range. A call
    site that derives it as `refrange_mode(...) == "full"` is wrong for xsite. A builder and its
    inspector that are wrong in the same way agree with each other, so only a search of the source
    catches the error."""
    import re
    root = Path(__file__).resolve().parents[1]
    # blank out docstrings and comments first, because several of these files discuss this rule in
    # prose. A test that fired on its own explanation would soon be deleted
    tq = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.S)
    offenders = []
    for f in sorted((root / "pipeline").glob("*.py")) + sorted((root / "analysis").glob("*.py")):
        src = tq.sub(lambda m: "\n" * m.group(0).count("\n"), f.read_text())
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"refrange_mode\([^)]*\)\s*==", line):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, ("decide the reference range with includes_reference_range(), not a "
                           f"comparison on refrange_mode: {offenders}")


def test_default_path_is_byte_identical(rows):
    """sha256 pinned from a snapshot taken before with_refrange was added."""
    assert hashlib.sha256(json.dumps(rows["full"]).encode()).hexdigest()[:32] == "67bd7fe41ca5ba7269148a530666ac90"
    assert hashlib.sha256(json.dumps(rows["nodesc"]).encode()).hexdigest()[:32] == "7e5b2b9adc907f36a464b1cddafd7e3a"


def test_norr_removes_only_percentile_and_unit_tokens(rows):
    lo, hi = rows["_lab_range"]
    n_uncapped = 0
    for f, n in zip(rows["full"], rows["norr"]):
        if f is None or n is None:
            assert (f is None) == (n is None)
            continue
        fi, ni = _strip_pad(f[0], f[2]), _strip_pad(n[0], n[2])
        if len(fi) >= d9.MAX_LEN:
            continue                                   # capped: covered by the packing test
        n_uncapped += 1
        assert len(ni) < len(fi), "norr must be shorter when the record fits the window"
        assert _is_subseq(ni, fi), "norr must be a subsequence: nothing added or reordered"
        # the labels themselves must survive untouched
        fl = [t for t in fi if lo <= t <= hi]
        nl = [t for t in ni if lo <= t <= hi]
        assert fl == nl, "norr changed the lab-code tokens, which are the labels"
    assert n_uncapped >= 8, f"only {n_uncapped} uncapped patients, too few to be meaningful"


def test_capped_patients_pack_more_observations(rows):
    """At the cap, shorter observations let norr reach records that full never saw. This is the
    same denominator effect as for nodesc (+32%), and the analysis must account for it."""
    lo, hi = rows["_lab_range"]
    capped = 0
    for f, n in zip(rows["full"], rows["norr"]):
        if f is None or n is None:
            continue
        fi, ni = _strip_pad(f[0], f[2]), _strip_pad(n[0], n[2])
        if len(fi) < d9.MAX_LEN:
            continue
        capped += 1
        assert len(ni) == d9.MAX_LEN, "a capped record should still fill the window"
        n_lab_full = sum(1 for t in fi if lo <= t <= hi)
        n_lab_norr = sum(1 for t in ni if lo <= t <= hi)
        assert n_lab_norr >= n_lab_full, "norr should fit at least as many lab observations"
    assert capped >= 3, f"only {capped} capped patients, too few to be meaningful"


# ----------------------------- v3 / xsite, end to end -----------------------------
XSITE_CFG = {"salt": "xsite-v3", "severity": 1.0,
             "perturb_values": True, "perturb_ranges": True}


@pytest.fixture(scope="module")
def xsite_rows():
    """The same patients tokenised clean, perturbed (twice), with the perturbation switched off,
    and with everything off.

    This is the only place where the v3 claim for the MLM can be checked. The perturbation acts on
    the raw float before `NdjsonTokenizer` bins it into the percentile table of the item. Only a
    comparison of token ids shows that it survived the binning."""
    uuids = d9.read_uuid_list(d9.SPLITS_DIR / "test_patients.txt")[:N_PATIENTS]
    m2l = d9._load_seen_codes_mimic_to_loinc(d9.SEEN_CODES_PATH)
    tok = d9._build_fhir_tokenizer(True, True)
    lo, hi = d9._lab_token_range(tok)
    out = {}
    for key, cfg in (("clean", None), ("xsite", XSITE_CFG), ("xsite_again", XSITE_CFG),
                     ("switched_off", dict(XSITE_CFG, perturb_values=False,
                                           perturb_ranges=False)),
                     ("all_off", dict(XSITE_CFG, perturb_values=False, perturb_ranges=False,
                                      randomise_units=False))):
        rs, st = [], d9._new_stats(len(uuids))
        for u in uuids:
            r = d9._process_one_patient(u, tok, None, m2l, lo, hi, False, st, perturb_cfg=cfg)
            rs.append(None if r is None else _strip_pad(r["input_ids"], r["attention_mask"]))
        out[key], out[key + "_stats"] = rs, st
    return out


def test_xsite_actually_changes_what_the_MLM_sees(xsite_rows):
    """The perturbation must move the emitted percentile tokens (a32 measures how often it does).
    If the binning absorbed the perturbation, the token sequences would be identical and the arm
    would be `para` under another name."""
    pairs = [(a, b) for a, b in zip(xsite_rows["clean"], xsite_rows["xsite"])
             if a is not None and b is not None]
    assert len(pairs) >= 10, "too few patients tokenised to conclude anything"
    changed = sum(1 for a, b in pairs if a != b)
    assert changed == len(pairs), f"only {changed}/{len(pairs)} patients changed"
    assert xsite_rows["xsite_stats"]["total_fields_perturbed"] > 0


def test_xsite_is_deterministic_for_the_same_patient_and_salt(xsite_rows):
    assert xsite_rows["xsite"] == xsite_rows["xsite_again"]


def test_turning_everything_off_is_byte_identical_to_clean(xsite_rows):
    """The arm has to be able to isolate the site simulation from everything else about v3."""
    assert xsite_rows["all_off"] == xsite_rows["clean"]
    assert xsite_rows["all_off_stats"]["total_fields_perturbed"] == 0
    assert xsite_rows["all_off_stats"]["total_units_randomised"] == 0


def test_unit_randomisation_is_NOT_governed_by_the_perturbation_switches(xsite_rows):
    """Unit randomisation is outside the ablation. It is part of the v3 representation and is on in
    every arm. `--no-perturb-values --no-perturb-ranges` must therefore still differ from clean.
    Only adding `--no-randomise-units` to those returns to clean."""
    assert xsite_rows["switched_off_stats"]["total_fields_perturbed"] == 0
    assert xsite_rows["switched_off_stats"]["total_units_randomised"] > 0
    assert xsite_rows["switched_off"] != xsite_rows["clean"], \
        "unit randomisation should still have moved the unit tokens"


def test_the_MLM_perturbs_exactly_the_observations_the_tfidf_path_does():
    """The mimic->LOINC map in 09 is unfiltered, while the one in 06 keeps only
    `observation_type == "lab"`. The MLM notion of "lab" also includes chartevents. If each builder
    used its own map, 14 chartevents codes would be perturbed in the MLM arm only, which would break
    the paired comparison. Both therefore go through perturbation.lab_loinc_of."""
    import perturbation as pert
    unfiltered = d9._load_seen_codes_mimic_to_loinc(d9.SEEN_CODES_PATH)
    spec = pert.load_spec()
    chart = {"resourceType": "Observation",
             "code": {"coding": [{"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/"
                                            "mimic-chartevents-d-items", "code": "220045"}]}}
    assert pert.lab_loinc_of(chart, unfiltered) is None
    lab_code = next(c for c, l in unfiltered.items() if l in spec)
    lab = {"resourceType": "Observation",
           "code": {"coding": [{"system": pert.LAB_CODE_SYSTEM, "code": lab_code}]}}
    assert pert.lab_loinc_of(lab, unfiltered) == unfiltered[lab_code]
