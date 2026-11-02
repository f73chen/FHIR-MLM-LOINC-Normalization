"""
Tests for the condition axes in pipeline/conditions.py: the reference-range setting, the
representation, and the train and test arms of the perturbed `xsite` conditions.

Three things must hold:

  1. The default path is byte-identical to the two-argument call. Every reported number was
     produced by that call. If adding a parameter changed its output, no existing number could be
     reproduced.
  2. The coarsening applied at training time is the same transformation that a15 measures. If the
     two diverge, the arm that is trained is not the arm whose floor is reported.
  3. Coarsening changes only the reference-range bounds. It does not touch the description, the
     value or the units, which in this corpus can contain digits (10*3/uL, log10 IU/mL).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from conditions import CONDITIONS, LEGACY_CONDITIONS, parse_condition, refrange_mode, has_description  # noqa: E402
from utils import (extract_structured_features_single as F,  # noqa: E402
                   _coarsen_reference_range, _coarsen_bound, _fmt_num)

_a15_spec = importlib.util.spec_from_file_location("a15", ROOT / "analysis" / "a15_field_identifiability.py")
a15 = importlib.util.module_from_spec(_a15_spec)
_a15_spec.loader.exec_module(a15)

PROMPT = ("Description: Potassium [Moles/volume] in Serum or Plasma\n"
          "Value: 4.2 meq/L\n"
          "ReferenceRange: Low: 3.5 meq/L, High: 5.1 meq/L\n"
          "Priority: STAT")


# -- 1. the default path is untouched ---------------------------------------
def test_default_is_byte_identical_to_two_arg_call():
    assert F(PROMPT) == F(PROMPT, None) == F(PROMPT, None, None)


def test_legacy_conditions_resolve_to_the_pre_axis_arguments():
    assert parse_condition("para") == (None, None)
    assert parse_condition("nodesc") == ({"description"}, None)
    for c in LEGACY_CONDITIONS:
        assert refrange_mode(c) == "full"


def test_coarsen_is_ignored_when_the_field_is_excluded():
    base = F(PROMPT, {"reference_range"})
    for how in ("1sf", "mag"):
        assert F(PROMPT, {"reference_range"}, how) == base


# -- 2. train-time coarsening == the a15 measurement ------------------------
@pytest.mark.parametrize("how,ref", [
    # The vectorised path (_sigfig_vec) produced the reported ladder numbers. It is the reference
    # that the train-time transformation has to match, and the scalar copy is only a convenience.
    ("1sf", lambda v: float(a15._sigfig_vec(__import__("numpy").array([v]), 1)[0])),
    ("mag", lambda v: float(a15._mag_vec(__import__("numpy").array([v]))[0])),
])
def test_coarsening_matches_a15(how, ref):
    """Every real reference-range bound in MIMIC, through both implementations."""
    import pandas as pd
    cache = locations.WORK / "analysis_cache/a15_obs_fields/train.parquet"
    if not cache.exists():
        pytest.skip("a15 cache not built")
    df = pd.read_parquet(cache, columns=["rr_low", "rr_high"])
    vals = pd.concat([df.rr_low, df.rr_high]).dropna().unique()
    assert len(vals) > 200, "expected the full ~264 distinct bounds"
    for v in vals:
        assert _coarsen_bound(float(v), how) == pytest.approx(ref(float(v)), rel=0, abs=0), v


def test_coarsen_bound_edge_cases():
    assert _coarsen_bound(0.0, "1sf") == 0.0
    assert _coarsen_bound(0.0, "mag") == -99.0          # same sentinel a15 uses
    assert _coarsen_bound(-250.0, "mag") == 2.0         # magnitude uses abs
    assert _coarsen_bound(9.99, "mag") == 0.0 and _coarsen_bound(10.0, "mag") == 1.0
    assert _coarsen_bound(0.004567, "1sf") == pytest.approx(0.005)
    with pytest.raises(ValueError):
        _coarsen_bound(1.0, "2sf")


def test_fmt_num_never_uses_scientific_notation():
    assert _fmt_num(100000.0) == "100000"
    assert _fmt_num(4.0) == "4"
    assert _fmt_num(0.0046) == "0.0046"
    assert _fmt_num(-90.0) == "-90"


# -- 3. coarsening touches only the bounds ----------------------------------
@pytest.mark.parametrize("rr,expect_1sf,expect_mag", [
    ("Low: 3.5 meq/L, High: 5.1 meq/L",           "Low: 4 meq/L, High: 5 meq/L",
                                                   "Low: 0 meq/L, High: 0 meq/L"),
    # units containing digits must survive untouched
    ("Low: 3.5 10*3/uL, High: 5.1 10*3/uL",       "Low: 4 10*3/uL, High: 5 10*3/uL",
                                                   "Low: 0 10*3/uL, High: 0 10*3/uL"),
    ("Low: 0.5 log10 IU/mL, High: 7 log10 IU/mL", "Low: 0.5 log10 IU/mL, High: 7 log10 IU/mL",
                                                   "Low: -1 log10 IU/mL, High: 0 log10 IU/mL"),
    ("Low: 1.005, High: 1.035",                   "Low: 1, High: 1",
                                                   "Low: 0, High: 0"),          # unit absent
    ("Low: -2.5 mmol/L, High: 2.5 mmol/L",        "Low: -2 mmol/L, High: 2 mmol/L",
                                                   "Low: 0 mmol/L, High: 0 mmol/L"),
])
def test_only_the_bounds_change(rr, expect_1sf, expect_mag):
    assert _coarsen_reference_range(rr, "1sf") == expect_1sf
    assert _coarsen_reference_range(rr, "mag") == expect_mag


def test_other_fields_are_untouched_by_coarsening():
    """Coarsening must rewrite only the reference clause of the feature string. `priority-STAT`,
    which is emitted after the reference clause, must stay as it was."""
    base = F(PROMPT)
    clause = "Low: 3.5 meq/L, High: 5.1 meq/L"
    assert clause in base and base.endswith("priority-STAT")
    assert F(PROMPT, None, "1sf") == base.replace(clause, "Low: 4 meq/L, High: 5 meq/L")
    assert F(PROMPT, None, "mag") == base.replace(clause, "Low: 0 meq/L, High: 0 meq/L")
    assert F(PROMPT, {"reference_range"}) == base.replace(f" reference {clause}", "")


def test_norr_equals_the_pre_existing_exclude_behaviour():
    excl, coarse = parse_condition("para_norr")
    assert F(PROMPT, excl, coarse) == F(PROMPT, {"reference_range"})
    excl, coarse = parse_condition("nodesc_norr")
    assert F(PROMPT, excl, coarse) == F(PROMPT, {"description", "reference_range"})


# -- slug grammar -----------------------------------------------------------
def test_slug_grammar_is_total_and_closed():
    # 2 descriptions x 6 refrange settings (none, norr, rr1sf, rrmag, norr_v2, xsite)
    assert len(CONDITIONS) == 12 and len(set(CONDITIONS)) == 12
    for c in CONDITIONS:
        excl, coarse = parse_condition(c)
        assert has_description(c) == c.startswith("para")
        assert (excl is None) or excl <= {"description", "reference_range"}
        assert coarse in (None, "1sf", "mag")
        # a slug says nodesc iff description is excluded
        assert (not has_description(c)) == bool(excl and "description" in excl)
    with pytest.raises(ValueError):
        parse_condition("para_rr2sf")
    with pytest.raises(ValueError):
        parse_condition("nodesc_")


def test_every_slug_produces_a_distinct_feature_string_WITHIN_a_representation():
    """The invariant holds within each representation.

    Every v1 slug is a train-time transform of one dataset. Two v1 slugs that extracted the same
    string would be the same experiment under two names. `norr_v2` is different. It reads a
    different dataset, with standardised units, 4 s.f. scientific values and no ReferenceRange, and
    that text is what makes it distinct. It is parsed the same way, so given the same v1 prompt it
    must extract the same string as `norr`, which the next test asserts."""
    from conditions import representation
    for rep in ("v1", "v2"):
        slugs = [c for c in CONDITIONS if representation(c) == rep]
        outs = {c: F(PROMPT, *parse_condition(c)) for c in slugs}
        assert len(set(outs.values())) == len(slugs), (rep, outs)


def test_norr_v2_extracts_identically_to_norr_given_the_same_prompt():
    """Documents why the test above is scoped per representation. The v2 difference is in the
    dataset. If this fails, a train-time transform has been added to norr_v2, and the dataset is not
    the only source of the v2 representation."""
    for desc in ("para", "nodesc"):
        assert F(PROMPT, *parse_condition(f"{desc}_norr")) == \
               F(PROMPT, *parse_condition(f"{desc}_norr_v2"))


def test_representation_axis_is_wired_through():
    from conditions import representation, dataset_suffix
    assert [c for c in CONDITIONS if representation(c) == "v2"] == \
           ["para_norr_v2", "nodesc_norr_v2"]
    assert all(dataset_suffix(c) == "" for c in CONDITIONS if representation(c) == "v1")
    assert all(dataset_suffix(c) == "_v2" for c in CONDITIONS if representation(c) == "v2")
    with pytest.raises(ValueError):
        representation("para_rr2sf")


# -- v3 / xsite -------------------------------------------------------------
def test_xsite_is_v3_and_KEEPS_the_reference_range():
    """`xsite` is the opposite of `norr_v2` on the reference range.

    v2 removes the reference range and v3 keeps it, because the range is what gets perturbed. A
    lookup on the MIMIC ranges works on MIMIC and fails at a shifted site. Anything that decides
    "drop the range" by string-matching "v2" must not match "xsite"."""
    from conditions import representation, refrange_mode
    for slug in ("para_xsite", "nodesc_xsite"):
        excl, coarse = parse_condition(slug)
        assert coarse is None
        assert not (excl and "reference_range" in excl), \
            f"{slug} must keep the reference range"
        assert representation(slug) == "v3"
        assert refrange_mode(slug) == "xsite"
    # and the description axis still works independently
    assert parse_condition("para_xsite")[0] is None
    assert parse_condition("nodesc_xsite")[0] == {"description"}


def test_v3_datasets_cannot_collide_with_v1_or_v2_on_disk():
    """A suffix rule of the form `"" if v1 else "_v2"` would file every v3 dataset under `_v2`."""
    from conditions import representation, dataset_suffix
    seen = {}
    for c in CONDITIONS:
        seen.setdefault(representation(c), set()).add(dataset_suffix(c))
    assert seen == {"v1": {""}, "v2": {"_v2"}, "v3": {"_v3"}}


def test_legacy_slugs_are_untouched_by_the_v3_addition():
    """Every reported number came from these slugs, so the way they resolve must not change."""
    assert parse_condition("para") == (None, None)
    assert parse_condition("nodesc") == ({"description"}, None)
    assert parse_condition("para_norr_v2") == ({"reference_range"}, None)
    from conditions import representation, dataset_suffix
    assert representation("para_norr_v2") == "v2" and dataset_suffix("para_norr_v2") == "_v2"
    assert dataset_suffix("para") == ""


# ------------------- train-arm naming -------------------
def test_the_two_v3_training_arms_get_DIFFERENT_paths():
    """The 2x2 of training and test conditions (thesis Chapter 5) needs a train-clean and a
    train-perturbed dataset. Both use the same condition slug and representation and differ only
    in the perturbation switches. Without an arm segment they would share a directory, and one
    would overwrite the other."""
    from conditions import arm_suffix, train_arm
    assert train_arm("para_xsite", True, True) == "pert"
    assert train_arm("para_xsite", False, False) == "clean"
    assert arm_suffix("para_xsite", True, True) != arm_suffix("para_xsite", False, False)


def test_a_half_disabled_v3_arm_is_still_perturbed():
    """--no-perturb-values alone gives the value/range ablation, which is still a `pert` arm."""
    from conditions import train_arm
    assert train_arm("para_xsite", False, True) == "pert"
    assert train_arm("para_xsite", True, False) == "pert"


def test_v1_and_v2_paths_are_completely_unchanged():
    """Every dataset already on disk must keep its name."""
    from conditions import CONDITIONS, arm_suffix, representation
    for c in CONDITIONS:
        if representation(c) != "v3":
            assert arm_suffix(c, True, True) == "" and arm_suffix(c, False, False) == "", c


def test_every_condition_taking_script_handles_or_refuses_v3():
    """A script that accepts a slug it cannot honour is worse than one that refuses it.

    Scripts share the same `choices=` list of conditions. A script with no perturbation code that
    is given `--condition para_xsite` would build an unperturbed dataset under a perturbed name.

    So if a script exposes `--condition`, it must build v3 (import `perturbation`), read it
    arm-aware (`resolve_arm_suffix`), or call `refuse_unsupported_v3`."""
    import re
    root = Path(__file__).resolve().parents[1] / "pipeline"
    tq = re.compile(r'("""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')')
    offenders = []
    for f in sorted(root.glob("*.py")):
        src = tq.sub(lambda m: "\n" * m.group(0).count("\n"), f.read_text())
        if not re.search(r'add_argument\(\s*["\']--condition["\']', src):
            continue
        builds_v3 = re.search(r"^\s*import perturbation", src, re.M) is not None
        consumes_v3 = "resolve_arm_suffix(" in src      # reads a v3 dataset / model, arm-aware
        refuses = "refuse_unsupported_v3(" in src
        if not (builds_v3 or consumes_v3 or refuses):
            offenders.append(f.name)
    assert not offenders, (
        "these scripts accept --condition but do not build v3 (import perturbation), consume it "
        "arm-aware (resolve_arm_suffix) or refuse it, so they would use the wrong data under an "
        f"xsite name: {offenders}")


# --------------- the train-arm axis, on the training and evaluation side ---------------
def test_resolve_arm_suffix_refuses_to_guess_for_v3():
    """A v3 condition names two datasets and two trained models that differ only in whether the
    training data was perturbed. A default could train or evaluate the wrong arm. There is therefore
    no default, and v3 requires an explicit --train-arm."""
    from conditions import resolve_arm_suffix
    assert resolve_arm_suffix("para", None, "t") == ""
    assert resolve_arm_suffix("nodesc_norr_v2", None, "t") == ""
    assert resolve_arm_suffix("para_xsite", "pert", "t") == "__pert"
    assert resolve_arm_suffix("para_xsite", "clean", "t") == "__clean"
    for slug, arm in (("para_xsite", None), ("para_xsite", "both"), ("nodesc_xsite", "")):
        with pytest.raises(SystemExit):
            resolve_arm_suffix(slug, arm, "t")
    # a non-v3 slug has no arms, so an arm given with one is refused
    with pytest.raises(SystemExit):
        resolve_arm_suffix("para", "pert", "t")


def test_no_consumer_builds_a_condition_path_without_the_arm():
    """The same collision as for datasets, on the expensive side. A clean-trained and a
    perturbed-trained `mlm__para_xsite__seed0` would be the same directory, and training into it
    would overwrite one arm with the other.

    Every path that a consumer derives from a condition must carry the arm. This covers the
    dataset, the model output dir (including the dry-run and short variants), the sidecar log, the
    manifest filename and the eval run name. 15_eval_runner.resolve_best_checkpoint reads the
    manifest by its filename, so a manifest without the arm would load the checkpoint of the wrong
    model."""
    import re
    root = Path(__file__).resolve().parents[1] / "pipeline"
    tq = re.compile(r'("""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')')
    pat = re.compile(r"(DATASET_BASE|MODELS_BASE|DATA_DIR_BASE|TRAIN_RESULTS\[model\]|RESULTS_DIR)"
                     r"[^=\n]*\{(?:args\.)?condition\}")
    offenders = []
    for name in ("07_tfidf_train_eval.py", "12_mlm_train.py", "13_bcm_train.py",
                 "14_llama_train.py", "15_eval_runner.py", "16_tfidf_eval_dump.py"):
        src = tq.sub(lambda m: "\n" * m.group(0).count("\n"), (root / name).read_text())
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            # `{test_arm}` counts too. A model path takes the train arm (which checkpoint) and a
            # dataset path takes the test arm (which data). 15 uses both, on different lines. A
            # condition-derived path must carry at least one.
            if pat.search(line) and not any(t in line for t in ("{arm}", "+ arm", "{test_arm}")):
                offenders.append(f"{name}:{i}")
    assert not offenders, f"condition-derived path with no train-arm segment: {offenders}"


def test_the_two_v3_arms_cannot_share_a_model_directory():
    """Checks directly, without a grep, that the two arms get different model directories."""
    from conditions import resolve_arm_suffix
    mk = lambda arm: ("/models/mlm__para_xsite"
                      + resolve_arm_suffix("para_xsite", arm, "t") + "__seed0")
    assert mk("pert") != mk("clean")
    assert "pert" in mk("pert") and "clean" in mk("clean")


# ------------------- the test arm is a second, separate axis -------------------
def test_the_four_2x2_cells_are_all_distinguishable():
    """The 2x2 (scenarios A-D of thesis Chapter 5) is the cross product of two axes. --train-arm
    picks the model and --test-arm picks the data. One flag cannot express this, because C and D
    would become indistinguishable and B and C unreachable."""
    from conditions import resolve_arm_suffix, resolve_test_arm
    cells = {}
    for label, (tr, te) in {"A": ("clean", None), "B": ("clean", "pert"),
                            "C": ("pert", "clean"), "D": ("pert", None)}.items():
        arm = resolve_arm_suffix("para_xsite", tr, "t")
        ta, tag = resolve_test_arm("para_xsite", arm, te, "t")
        cells[label] = (f"model{arm}__seed0{tag}", f"data{ta}")
    # every cell has a distinct run name and the right dataset
    assert len({v[0] for v in cells.values()}) == 4, cells
    assert cells["A"][1] == cells["C"][1] == "data__clean"   # both scored on clean data
    assert cells["B"][1] == cells["D"][1] == "data__pert"    # both scored on perturbed data
    assert cells["A"][0] != cells["B"][0]   # same model, different data -> different run name
    assert cells["D"][0] != cells["C"][0]


def test_the_diagonal_keeps_its_old_name_so_v1_and_v2_runs_are_unchanged():
    from conditions import resolve_test_arm
    for slug in ("para", "nodesc_norr_v2", "para_norr"):
        assert resolve_test_arm(slug, "", None, "t") == ("", "")


def test_test_arm_is_refused_where_it_means_nothing():
    from conditions import resolve_test_arm
    for slug in ("para", "nodesc", "para_norr_v2"):
        with pytest.raises(SystemExit):
            resolve_test_arm(slug, "", "pert", "t")


# ----------- every perturbation config gets its own dataset name -----------
def test_no_two_perturbation_configs_share_a_dataset_name():
    """`arm_suffix` only tells whether anything was perturbed. Without a perturbation tag,
    severity 0.5, severity 2.0, value-only and range-only would all resolve to `__pert`, which is
    the dataset the pert arm trains on. Building the severity sweep or the value/range ablation
    would then overwrite the training set."""
    from conditions import arm_suffix, perturbation_tag
    seen = {}
    for pv in (True, False):
        for pr in (True, False):
            for sev in (0.5, 1.0, 2.0):
                for rep in (None, 1, 5):
                    if not (pv or pr) and (sev != 1.0 or rep is not None):
                        continue                       # refused; covered separately
                    name = (arm_suffix("para_xsite", pv, pr)
                            + perturbation_tag(sev, pv, pr, rep))
                    key = (pv, pr, sev, rep)
                    assert name not in seen, f"{key} collides with {seen[name]} on {name!r}"
                    seen[name] = key
    assert len(seen) >= 20, f"only {len(seen)} configs exercised"


def test_the_canonical_config_and_the_replicates_keep_the_names_already_on_disk():
    """The tag must be empty for the canonical config and `_rN` for a replicate, or every
    existing dataset path breaks."""
    from conditions import perturbation_tag
    assert perturbation_tag() == ""
    assert perturbation_tag(1.0, True, True, None) == ""
    for n in (1, 2, 3, 4, 5):
        assert perturbation_tag(1.0, True, True, n) == f"_r{n}"


def test_severity_and_replicate_are_refused_on_the_clean_arm():
    """Nothing is perturbed there, so a severity would be a config that looks real and is not."""
    from conditions import perturbation_tag
    assert perturbation_tag(1.0, False, False, None) == ""
    for bad in ({"severity": 2.0}, {"replicate": 1}):
        with pytest.raises(SystemExit):
            perturbation_tag(perturb_values=False, perturb_ranges=False, **bad)


def test_every_built_eval_set_name_is_accepted_by_the_eval_runner():
    """The builder and the consumer must agree on the vocabulary, or a set can be built and then
    not be selectable."""
    from conditions import arm_suffix, perturbation_tag, resolve_test_arm
    for pv, pr, sev, rep in [(True, True, 1.0, None), (True, True, 1.0, 3),
                             (True, True, 0.5, None), (True, True, 2.0, None),
                             (True, False, 1.0, None), (False, True, 1.0, None),
                             (False, False, 1.0, None)]:
        suffix = arm_suffix("para_xsite", pv, pr) + perturbation_tag(sev, pv, pr, rep)
        arm_name = suffix.lstrip("_")
        got, _tag = resolve_test_arm("para_xsite", "__pert", arm_name, "t")
        assert got == suffix, f"{arm_name!r} -> {got!r}, expected {suffix!r}"
