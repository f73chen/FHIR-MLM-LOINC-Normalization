"""
The condition-slug grammar, shared by every script that takes --condition (thesis Chapters 4 and 5).

A condition has two axes. One is the description, where `para` means a paraphrased description and
`nodesc` means none. The other is the reference-range setting, which also fixes the numeric
representation. New settings extend the condition token and leave the path structure alone, because
paths already use `__` as the field separator and single `_` within a field
(`mlm__singleobs_trim__para`, `relma_{strategy}_a{seed}`):

    condition := {desc}[_{refrange}]
      desc     : para | nodesc
      refrange : <omitted> = full range | norr | rr1sf | rrmag | norr_v2 | xsite

    para · nodesc · para_norr · nodesc_norr · para_rr1sf · nodesc_rr1sf · para_rrmag · nodesc_rrmag
    para_norr_v2 · nodesc_norr_v2 · para_xsite · nodesc_xsite

`norr_v2` is a separate input representation that also drops the reference range, with units
standardised code-blind and values rendered as fixed-width 4 s.f. scientific. It therefore reads
from a different dataset, and its numbers are not comparable to any v1 condition.
`representation()` is the axis that says so, and every summary JSON records it.

An omitted segment means the full reference range, so `para` and `nodesc` keep their bare names.

`rrmag` is defined so it can be run without a code change.
"""
from __future__ import annotations

import re

DESCS = ("para", "nodesc")
REFRANGES = {
    None:      dict(exclude=None,              coarsen=None, representation="v1"),  # full range
    "norr":    dict(exclude="reference_range", coarsen=None, representation="v1"),  # field removed
    "rr1sf":   dict(exclude=None,              coarsen="1sf", representation="v1"), # bounds to 1 s.f.
    "rrmag":   dict(exclude=None,              coarsen="mag", representation="v1"), # bounds to decade
    # v2: no reference range + standardised units + fixed-width 4 s.f. scientific values.
    # `exclude` is a safeguard. The v2 builder writes no ReferenceRange line, so this costs nothing
    # and would catch a leak if one were introduced.
    "norr_v2": dict(exclude="reference_range", coarsen=None, representation="v2"),
    # v3 (`xsite`): v2's standardised units and 4 s.f. scientific values, plus unit-notation
    # randomisation, with the reference range present because it is what gets perturbed. `exclude`
    # is None here, the opposite of norr_v2, so includes_reference_range reads `exclude` to decide
    # whether to drop the range.
    "xsite":   dict(exclude=None,              coarsen=None, representation="v3"),
}

#: every legal slug, in a stable order, for argparse `choices=`
CONDITIONS = [d + (f"_{r}" if r else "")
              for d in DESCS for r in (None, "norr", "rr1sf", "rrmag", "norr_v2", "xsite")]

#: the two conditions with no reference-range segment (the full range)
LEGACY_CONDITIONS = ("para", "nodesc")


def parse_condition(slug: str):
    """Map a slug such as 'nodesc_rr1sf' to (exclude_fields, coarsen_reference_range) for
    utils.extract_structured_features_single.

    When nothing is excluded, `exclude_fields` is None, so `para` calls the extractor with its
    default arguments.

    >>> parse_condition("para")
    (None, None)
    >>> parse_condition("nodesc")
    ({'description'}, None)
    >>> parse_condition("para_norr")
    ({'reference_range'}, None)
    >>> parse_condition("nodesc_rr1sf") == ({"description"}, "1sf")
    True
    >>> parse_condition("para_norr_v2")
    ({'reference_range'}, None)
    >>> parse_condition("nodesc_norr_v2") == ({"description", "reference_range"}, None)
    True
    """
    if slug not in CONDITIONS:
        raise ValueError(f"unknown condition {slug!r}; expected one of {CONDITIONS}")
    desc, _, rr = slug.partition("_")
    spec = REFRANGES[rr or None]
    excl = set()
    if desc == "nodesc":
        excl.add("description")
    if spec["exclude"]:
        excl.add(spec["exclude"])
    return (excl or None), spec["coarsen"]


def has_description(slug: str) -> bool:
    """True when the paraphrased description is present (the `para` conditions)."""
    return slug.split("_")[0] == "para"


def refrange_mode(slug: str) -> str:
    """'full' | 'norr' | 'rr1sf' | 'rrmag' | 'norr_v2' | 'xsite', for manifests and tables.

    >>> refrange_mode("para"), refrange_mode("nodesc_norr_v2"), refrange_mode("para_xsite")
    ('full', 'norr_v2', 'xsite')
    """
    _, _, rr = slug.partition("_")
    return rr or "full"


def includes_reference_range(slug: str) -> bool:
    """True when this condition's prompt carries a reference range.

    Do not test `refrange_mode(slug) == "full"` for this. refrange_mode returns the slug's suffix, so
    the comparison is False for `xsite` and would drop the range from the condition that is about the
    perturbed reference range. The rule is a property of REFRANGES, so it is read from there.

    `norr` and `norr_v2` remove the field. Every other condition keeps it, either coarsened
    (`rr1sf`, `rrmag`), standardised and perturbed (`xsite`), or verbatim.

    >>> [s for s in CONDITIONS if not includes_reference_range(s)]
    ['para_norr', 'para_norr_v2', 'nodesc_norr', 'nodesc_norr_v2']
    >>> includes_reference_range("para_xsite"), includes_reference_range("nodesc_xsite")
    (True, True)
    """
    if slug not in CONDITIONS:
        raise ValueError(f"unknown condition {slug!r}; expected one of {CONDITIONS}")
    _, _, rr = slug.partition("_")
    return REFRANGES[rr or None]["exclude"] != "reference_range"


def representation(slug: str) -> str:
    """'v1' | 'v2' | 'v3': the input representation the slug needs, which also decides the dataset.

    v1  native units, plain-decimal values, reference range present in the text
    v2  units standardised code-blind, values as fixed-width 4 s.f. scientific, no reference range
    v3  as v2 but the reference range is present and standardised, unit notation randomised, and
        values and range bounds perturbed to simulate a second clinical site

    Numbers from the three are not comparable. Callers must use this to pick the dataset directory
    and must record it in every output.

    >>> representation("para"), representation("para_norr_v2"), representation("nodesc_xsite")
    ('v1', 'v2', 'v3')
    """
    if slug not in CONDITIONS:
        raise ValueError(f"unknown condition {slug!r}; expected one of {CONDITIONS}")
    _, _, rr = slug.partition("_")
    return REFRANGES[rr or None]["representation"]


#: representation -> dataset-directory suffix. The suffixes are listed explicitly, because a rule
#: such as `"" if v1 else "_v2"` would file every v3 dataset under `_v2`.
_SUFFIX = {"v1": "", "v2": "_v2", "v3": "_v3"}


def resolve_arm_suffix(slug: str, train_arm: "str | None", script: str) -> str:
    """The arm path segment for a consumer of a dataset: '' for v1/v2, '__pert'/'__clean' for v3.

    v3 requires an explicit choice and has no default. A v3 condition names two datasets and two
    trained models that differ only in whether the training data was perturbed. Defaulting to either
    one would train or evaluate the wrong arm whenever the other was meant.

    The builders derive the arm from their perturbation switches (`arm_suffix`). Consumers
    cannot, because nothing in the request says which arm is wanted. Hence `--train-arm`.
    """
    if representation(slug) != "v3":
        if train_arm:
            raise SystemExit(f"{script}: --train-arm {train_arm!r} does not apply to {slug!r}, "
                             f"which is representation {representation(slug)}, not v3")
        return ""
    if train_arm not in ("pert", "clean"):
        raise SystemExit(
            f"{script}: condition {slug!r} is a v3 arm, so --train-arm {{pert,clean}} is required. "
            "'pert' = trained on perturbed data, 'clean' = trained on the v3 representation with "
            "no perturbation (the 2x2 design needs both). There is no default.")
    return f"__{train_arm}"


def perturbation_tag(severity: float = 1.0, perturb_values: bool = True,
                     perturb_ranges: bool = True, replicate: "int | None" = None,
                     script: str = "?") -> str:
    """The part of a dataset name that distinguishes one perturbation config from another.

    Without it, different configs share one name. `arm_suffix` answers only "was anything
    perturbed". Severity 0.5, severity 2.0, value-only and range-only would all resolve to `__pert`,
    which is the dataset the `pert` arm trains on. Building the severity sweep or the value/range
    ablation would then overwrite that training set.

    Returns '' for the canonical config (both halves, severity 1.0, no replicate), so the main
    datasets carry no tag:

    >>> perturbation_tag()                                   # the primary pert arm
    ''
    >>> perturbation_tag(replicate=3)                        # a replicate perturbation draw
    '_r3'
    >>> perturbation_tag(severity=0.5), perturbation_tag(severity=2.0)
    ('_sev0.5', '_sev2.0')
    >>> perturbation_tag(perturb_ranges=False)               # value-only ablation
    '_value'
    >>> perturbation_tag(perturb_values=False)               # range-only ablation
    '_range'
    >>> perturbation_tag(severity=2.0, perturb_values=False, replicate=2)
    '_range_sev2.0_r2'
    """
    if not (perturb_values or perturb_ranges):
        # nothing is perturbed, so this is the `clean` arm, where severity and replicate do not
        # apply. Ignoring them would let `--severity 2 --no-perturb-*` look like a real config, so
        # giving either one raises SystemExit.
        if severity != 1.0 or replicate is not None:
            raise SystemExit(
                f"{script}: severity={severity} and replicate={replicate} do not apply with both "
                "perturbation switches off. That is the clean arm, in which nothing is perturbed.")
        return ""
    parts = []
    if not (perturb_values and perturb_ranges):
        parts.append("value" if perturb_values else "range")
    if severity != 1.0:
        # str(float(...)), because :g renders 2.0 as "2" but 0.5 as "0.5", and the two severities
        # would be named in two different styles.
        parts.append(f"sev{float(severity)}")
    if replicate is not None:
        parts.append(f"r{replicate}")
    return ("_" + "_".join(parts)) if parts else ""


def resolve_test_arm(slug: str, train_arm_suffix: str, test_arm: "str | None",
                     script: str) -> "tuple[str, str]":
    """(test_arm_suffix, cross_tag) for the test axis, which is separate from the train axis.

    `train_arm` selects the model (which checkpoint, via its manifest). `test_arm` selects the data
    (which copy of the test split it is scored on). The 2x2 of scenarios A–D is their cross product:

        train clean / test clean -> A      train clean / test pert  -> B
        train pert  / test clean -> C      train pert  / test pert  -> D

    Collapsing them into one flag makes C and D indistinguishable and B and C unreachable.

    `cross_tag` is empty on the diagonal and `__on{arm}` off it, so B and C cannot overwrite the A
    and D results they share a model with.

    >>> resolve_test_arm("para_xsite", "__clean", None, "t")
    ('__clean', '')
    >>> resolve_test_arm("para_xsite", "__clean", "pert", "t")
    ('__pert', '__onpert')
    >>> resolve_test_arm("para_xsite", "__pert", "clean", "t")
    ('__clean', '__onclean')
    >>> resolve_test_arm("para", "", None, "t")
    ('', '')
    >>> resolve_test_arm("para_xsite", "__clean", "pert_r3", "t")
    ('__pert_r3', '__onpert_r3')
    >>> resolve_test_arm("para_xsite", "__pert", "pert_sev2.0", "t")
    ('__pert_sev2.0', '__onpert_sev2.0')
    >>> resolve_test_arm("para_xsite", "__pert", "pert_range", "t")
    ('__pert_range', '__onpert_range')
    """
    # mirrors perturbation_tag's output: pert[_value|_range][_sevS][_rN]
    _ARM_RX = r"clean|pert(_value|_range)?(_sev[0-9]+(\.[0-9]+)?)?(_r[1-9][0-9]?)?"
    if test_arm and not re.fullmatch(_ARM_RX, test_arm):
        raise SystemExit(
            f"{script}: --test-arm {test_arm!r} is not a known arm. Use 'clean', 'pert', or a "
            "perturbed variant named by perturbation_tag: a replicate ('pert_r3'), "
            "a severity ('pert_sev2.0') or an ablation ('pert_value' or 'pert_range').")
    if test_arm and representation(slug) != "v3":
        raise SystemExit(f"{script}: --test-arm applies only to a v3 (xsite) condition, "
                         f"not {slug!r} (representation {representation(slug)})")
    suffix = f"__{test_arm}" if test_arm else train_arm_suffix
    return suffix, (f"__on{suffix.lstrip('_')}" if suffix != train_arm_suffix else "")


def refuse_unsupported_v3(slug: str, script: str, *, consumer: bool = False) -> None:
    """Exit if `slug` is a v3 condition and the caller has not implemented v3.

    A script that takes the shared `choices=` list but has no perturbation code would build an
    unperturbed dataset under a perturbed name for `--condition para_xsite`.

    Call this right after `parse_args`, and remove the call once the script implements v3.
    `test_every_condition_taking_script_handles_or_refuses_v3` (tests/test_conditions_refrange.py)
    checks that every script taking --condition builds, consumes or refuses v3.
    """
    if representation(slug) != "v3":
        return
    if consumer:
        raise SystemExit(
            f"{script}: condition {slug!r} is a v3 (xsite) arm. A v3 dataset path has a "
            "train-arm segment (`__pert` or `__clean`), and this script has no --train-arm "
            "option, so it would look for a directory that does not exist.")
    raise SystemExit(
        f"{script}: condition {slug!r} is a v3 (xsite) arm, and this script does not implement "
        "the v3 representation, so it would build an unperturbed dataset under a perturbed "
        "name. v3 is implemented in 06_tfidf_dataset, 09_mlm_dataset, 10_bcm_dataset and "
        "11_llama_dataset.")


def train_arm(slug: str, perturb_values: bool = True, perturb_ranges: bool = True) -> str:
    """The train-arm axis: 'pert' or 'clean' for a v3 slug, and '' for anything else.

    Without it the two v3 training arms collide on disk. The 2x2 of scenarios needs a train-clean
    and a train-perturbed dataset. Both have the same condition slug (`para_xsite`) and the same
    representation, and only the perturbation switches differ. Writing both to `…__para_xsite`
    would overwrite one with the other.

    "clean" means the v3 representation with no perturbation, which is not raw MIMIC. It has
    standardised units, 4 s.f. scientific values, the reference range and randomised unit notation.
    Only the site perturbation is off. This makes scenario C interpretable, because the task is the
    same and only the site simulation differs. Scenario C is trained perturbed and tested clean, and
    it measures the cost at the original site.

    Empty for v1 and v2, whose dataset paths have no arm segment.

    >>> train_arm("para"), train_arm("para_norr_v2")
    ('', '')
    >>> train_arm("para_xsite"), train_arm("para_xsite", False, False)
    ('pert', 'clean')
    >>> train_arm("nodesc_xsite", True, False)
    'pert'
    """
    if representation(slug) != "v3":
        return ""
    return "clean" if not (perturb_values or perturb_ranges) else "pert"


def arm_suffix(slug: str, perturb_values: bool = True, perturb_ranges: bool = True) -> str:
    """`train_arm` as a `__` path segment: '__pert', '__clean' or ''."""
    a = train_arm(slug, perturb_values, perturb_ranges)
    return f"__{a}" if a else ""


def dataset_suffix(slug: str) -> str:
    """Return "" for v1, "_v2" for v2 and "_v3" for v3, so no two representations can collide on disk.

    >>> dataset_suffix("para"), dataset_suffix("nodesc_norr_v2"), dataset_suffix("para_xsite")
    ('', '_v2', '_v3')
    """
    return _SUFFIX[representation(slug)]


#: Conditions FHIR-MLM can express. The deep-model scripts take this list as their --condition
#: choices. The tokenizer digitises reference-range bounds into the item's percentile table, so
#: there is no surface string to coarsen. rr1sf/rrmag would have to act on the raw float before
#: binning, inside fhir_tokenizer. That is a different intervention from TF-IDF's, and it is not
#: implemented.
MLM_CONDITIONS = ["para", "para_norr", "nodesc", "nodesc_norr", "para_xsite", "nodesc_xsite"]

#: `xsite` is included because 09, 10 and 11 all call `perturbation.perturb_patient` on the
#: patient's resources before formatting or tokenising. Unlike rr1sf/rrmag it needs no surface
#: string, since the perturbation acts on the raw float before the tokenizer bins it. At the
#: register's magnitudes it moves the emitted percentile token for many values and reference-range
#: bounds (analysis/a32_bin_change_final.py). The arm is therefore meaningful as well as expressible.
#:
#: `norr_v2` is left out because v2's value transform cannot produce a distinct FHIR-MLM arm. Done
#: coherently, with the bins recomputed from the rescaled values, it is a no-op, since a percentile
#: is invariant under a monotone rescale. Done naively, with the shipped bins left alone, it corrupts
#: the input and collapses every haemoglobin onto one bucket. In the shipped code it is never
#: reached, because the transform lives in the renderer and this path reads `Quantity.value` and
#: `Quantity.code` off the resource. For FHIR-MLM, v3 means the perturbation plus the unit
#: randomisation.
