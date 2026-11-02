"""Write per-observation TF-IDF predictions on the test split, keyed to join the evaluation dumps of the
deep models (thesis Chapters 5 and 6).

07_tfidf_train_eval.py reports aggregate metrics only, and its test rows cannot be joined to the
deep-model dumps. 06_tfidf_dataset.py walks each patient's raw ndjson without sorting, while the deep
models number observations chronologically. A JSONL line index is therefore not an obs_ordinal. This
script regenerates the test prompts in chronological order with (patient_uuid, obs_ordinal), refits
the model (or loads a cached fit) and dumps the top-K predictions.

The training side is the 06 dataset, read unchanged. TF-IDF is a bag-of-words model, so the order of
the observations does not matter to the fit. Reusing the files keeps the refit faithful to 07, with
the same files in the same order, the same `downsample_indices` and the same recipe. The pipeline
comes from tfidf_recipe.make_pipeline, the definition 07 and 08 use. A cached model is refused
unless its pipeline_signature equals a freshly built one, because the cache path encodes condition,
arm and seed but not the recipe.

The test side is regenerated chronologically, with descriptions from the 70/15/15 paraphrase pools
of 09 assigned per (patient, LOINC) as for the deep models. The only expected difference from 07 is
which paraphrase an observation draws in the regenerated split, so accuracy matches 07 only within
noise. --verify checks each seed against the value from 07 for the same cell and seed.

obs_ordinal counts the LOINC-mappable (828-class) lab observations, which are the ones the deep
models label. TF-IDF scores only the 679-code label space, so a mappable observation outside the 679
codes still takes an ordinal but is not scored. --test-only checks the alignment against a BCM dump
before any fit.

Writes WORK_DIR/eval/tfidf__{condition}{arm}__seed{n}__{evalset}.../part-0000.parquet with a
manifest, a copy of the manifest to results/16_tfidf_eval_dump/, and fitted models to
WORK_DIR/models/tfidf__{condition}{arm}__seed{n}/. Patient-level rows stay in WORK_DIR. Stdout shows
only aggregates. The Chapter 6 analyses need full logits (--dump-full-logits) in every main
evaluation dump.

  python pipeline/16_tfidf_eval_dump.py --condition para_xsite --train-arm pert --seeds 0 1 2 3 4 \
      --write --dump-full-logits
"""
from __future__ import annotations

import argparse
import functools
import csv
import glob
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (CONDITIONS, dataset_suffix, parse_condition,  # noqa: E402
                        resolve_arm_suffix, representation,
                        includes_reference_range, resolve_test_arm)
from tfidf_recipe import PINNED, make_pipeline, pipeline_signature  # noqa: E402

COMMON_DIR = str(locations.COMMON)
for _p in (COMMON_DIR, str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO = locations.REPO
#: v1, v2 and v3 are different datasets. The suffix is derived from the condition slug, so a v2 dump
#: cannot read v1 training text.
DATA_DIR_BASE = str(locations.WORK / "tfidf/dataset")
LABELSPACE_CSV = REPO / "data" / "labelspace" / "labelspace.csv"
PARAPHRASE_DIR = REPO / "data" / "paraphrases" / "clean"
SPLITS_DIR = REPO / "data" / "splits"
NDJSON_DIR = locations.PATIENTS
SEEN_CODES = str(locations.UPSTREAM / "seen_codes.csv")
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
EVAL_BASE = str(locations.WORK / "eval")
# Fitted TF-IDF pipelines are saved here, as the deep models are (models/{model}__{condition}{arm}__seed{n}/).
# 07_tfidf_train_eval.py does not save its models, which is why this script refits. A cached model
# skips both the fit and the slow load of the training labels.
MODELS_BASE = str(locations.WORK / "models")
RESULTS_DIR = REPO / "results" / "16_tfidf_eval_dump"
RELMA_POOL_DIR = REPO / "data" / "relma"

# Normalizer inputs: the same files as 06_tfidf_dataset.py, because the regenerated prompts must have
# the same form that 07 was trained on
QUAL_NORMALIZE_PATH = f"{locations.UPSTREAM}/pcornet_qual_normalize.json"
QUAL_PASSTHROUGH_PATH = f"{locations.UPSTREAM}/pcornet_qual_passthrough.json"
UNIT_SYNONYM_PATH = f"{locations.UPSTREAM}/unit_synonym_map.json"

# Must equal 07_tfidf_train_eval.py (tests/test_tfidf_eval_dump.py asserts it). The pinned recipe is
# not repeated here. It is tfidf_recipe.PINNED, imported above.
SUBSET_SIZE = 2_000_000
DOWNSAMPLE_MIN = 500
SEED = 42
TOPK = [1, 3, 5, 10]
TOP_K_MAX = max(TOPK)
EVAL_BATCH = 50_000
# --verify target: the micro top-1 from 07 for the same cell and seed, read from the JSON that 07
# writes. Per seed, this script refits the same downsample with the same recipe and scores the same
# observations. The only difference is which paraphrase an observation draws in the regenerated
# split. Over 30 v3 seed-cells under an earlier recipe, |delta| was at most 7.2e-5 for para and
# exactly 0 for nodesc. VERIFY_TOL is about 14 times the largest difference seen.
RESULTS_07 = REPO / "results" / "07_tfidf_train_eval"
VERIFY_TOL = 0.001
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"


# --- pure logic (the same as in 07, which tests/test_tfidf_eval_dump.py checks) ---
def threshold_targets(counts: dict, subset_size: int, downsample_min: int) -> dict:
    take_all = {c: n for c, n in counts.items() if n <= downsample_min}
    big = {c: n for c, n in counts.items() if n > downsample_min}
    remaining = subset_size - sum(take_all.values())
    total_big = sum(big.values())
    target = dict(take_all)
    for c, n in big.items():
        target[c] = min(max(int((n / total_big) * remaining), downsample_min), n) if total_big > 0 else n
    return target


def downsample_indices(labels: list, subset_size: int, downsample_min: int, seed: int) -> list:
    counts = Counter(labels)
    target = threshold_targets(counts, subset_size, downsample_min)
    by_label = defaultdict(list)
    for i, l in enumerate(labels):
        by_label[l].append(i)
    rng = random.Random(seed)
    out = []
    for l, idxs in by_label.items():
        t = target.get(l, len(idxs))
        out.extend(idxs if len(idxs) <= t else rng.sample(idxs, t))
    rng.shuffle(out)
    return out


def expected_top1_from_07(condition: str, arm: str, cross_tag: str, seed: int,
                          results_dir: Path = RESULTS_07):
    """Return the micro top-1 from 07 for this cell and seed as (value, source), or (None, reason).

    The home cell (cross_tag '') is the test split of 07, and '__on<arm>' is the eval set <arm> of 07.
    A 07 JSON produced under a different recipe is not a valid expectation, so it is reported as a
    reason. Under --verify that counts as a failure, so an unattended run cannot pass a check it never
    made."""
    f = Path(results_dir) / f"tfidf_{condition}{arm}.json"
    if not f.exists():
        return None, f"{f} not found (run 07 for this arm first)"
    j = json.loads(f.read_text())
    if j.get("pinned_params") != json.loads(json.dumps(PINNED)):
        return None, f"{f.name} was produced under a different recipe than tfidf_recipe.PINNED"
    other = cross_tag[len("__on"):] if cross_tag else None
    for r in j.get("per_seed", []):
        if r.get("seed") != seed:
            continue
        if other is None:
            return float(r["micro"]["top1"]), f"{f.name} seed {seed}"
        es = (r.get("eval_sets") or {}).get(other)
        if es is None:
            return None, f"{f.name} has no eval set {other!r} for seed {seed}"
        return float(es["micro"]["top1"]), f"{f.name} seed {seed} on[{other}]"
    return None, f"seed {seed} not in {f.name}"


def is_lab_obs(obs: dict) -> bool:
    if obs.get("resourceType") != "Observation":
        return False
    c = obs.get("code", {}).get("coding", [])
    return bool(c) and c[0].get("system") == LAB_CODE_SYSTEM and bool(c[0].get("code"))


# --- train side: labels first, then only the selected prompts ---------------
def load_train_labels(data_dir):
    """Labels in the same global order that load_split in 07 produces (files sorted and concatenated in
    order). Returns (labels, file_paths, per-file line counts), so a global index maps back to (file, line)."""
    files = sorted(glob.glob(f"{data_dir}/train/labels/*.txt"))
    labels, counts = [], []
    for fp in files:
        with open(fp) as f:
            ls = [l.strip() for l in f if l.strip()]
        labels.extend(ls)
        counts.append(len(ls))
    return labels, files, counts


def load_selected_prompts(data_dir, files, counts, idx, extract_fn, exclude, coarsen=None):
    """Load only the prompts at global indices `idx`. Loading tens of millions of training prompts to
    keep about 5% would be wasteful. This reads each file once and takes the lines needed."""
    sample_files = [fp.replace("/labels/", "/samples/").replace(".txt", ".jsonl") for fp in files]
    starts = np.cumsum([0] + counts[:-1])
    want = defaultdict(list)
    for gi in idx:
        fi = int(np.searchsorted(starts, gi, side="right") - 1)
        want[fi].append(gi - int(starts[fi]))
    out = {}
    for fi, lines in want.items():
        need = set(lines)
        with open(sample_files[fi]) as f:
            for li, line in enumerate(f):
                if li in need:
                    out[int(starts[fi]) + li] = extract_fn(json.loads(line)["prompt"], exclude, coarsen)
    return [out[gi] for gi in idx]


# --- test side: regenerate chronologically with join keys -------------------
def build_test_rows(uuids, label_set, mimic_to_loinc, mimic_code_to_class, mappable,
                    test_para, unit_norm, qual_norm, fmt_fn, rng,
                    relma_salt=None, clean_lookup=None, perturb=None):
    """Per patient, chronological. Yields (uuid, obs_ordinal, prompt, loinc, name, clean_same, clean_global).
    obs_ordinal counts the LOINC-mappable (828-class) observations, so it matches the deep-model dumps.
    Only observations whose LOINC is in the 679-code label space of TF-IDF are emitted.

    With relma_salt set (RELMA mode):
      * the description pool holds RELMA local names;
      * names are assigned per (patient, LOINC) with the same `assign_paraphrases_per_patient` and salt
        as the deep models. An observation gets the same name in every model, so paired tests compare
        like with like. TF-IDF has no patient context, so this costs it nothing;
      * observations whose LOINC has no RELMA name are skipped. TF-IDF has no history, so skipping
        them is equivalent to the deep models dropping them at scoring time. The ordinal is still
        consumed, so the join keys stay aligned."""
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    n_mappable = n_emitted = n_no_relma = 0
    for u in uuids:
        p = NDJSON_DIR / f"{u}.ndjson"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                resources = ndjson.load(f)
        except Exception:
            continue
        # v3 only: the second site, applied to the whole patient before anything is formatted, with
        # the same hook and order as `perturb(resources, uuid)` in 06. It works per patient because
        # MCHC is recomputed from Hb and Hct in the same panel. Without it this function would
        # regenerate raw MIMIC numbers and score a v3 model on v1 data.
        if perturb is not None:
            perturb(resources, u)
        # always per (patient, LOINC), as in 06 and the deep models. salt=None gives the default
        # patient_rng(uuid) stream used for para, and a salt selects a RELMA assignment draw.
        from paraphrase_utils import patient_rng, assign_paraphrases_per_patient
        _rng = patient_rng(u, salt=relma_salt) if relma_salt else patient_rng(u)
        mapping = assign_paraphrases_per_patient(resources, test_para, mimic_to_loinc, _rng)
        enc = get_encounters(resources)
        resources = sorted(resources, key=lambda r: sort_key(r, enc))
        ordinal = 0
        for r in resources:
            if not is_lab_obs(r):
                continue
            code = str(r["code"]["coding"][0].get("code"))
            cls = mimic_code_to_class.get(code)
            if cls is None or cls not in mappable:
                continue                       # not in the scored space of the deep models, so no ordinal
            this_ord = ordinal
            ordinal += 1                       # consumed even when skipped below, so the join stays aligned
            n_mappable += 1
            loinc = mimic_to_loinc.get(code)
            if not loinc or loinc not in label_set:
                continue                       # mappable-828 but outside TF-IDF's 679 space
            name = mapping.get(loinc)
            if not name:
                if relma_salt:
                    n_no_relma += 1            # no RELMA name for this code, so it is dropped
                continue
            cs, cg = ((clean_lookup or {}).get((loinc, name), (False, False))
                      if relma_salt else (None, None))
            yield u, this_ord, fmt_fn(r, name, unit_norm, qual_norm), loinc, name, cs, cg
            n_emitted += 1
    build_test_rows.stats = {"n_mappable": n_mappable, "n_emitted": n_emitted,
                             "n_skipped_no_relma_name": n_no_relma}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=CONDITIONS, required=True,
                    help="see pipeline/conditions.py for the slug grammar")
    ap.add_argument("--train-arm", choices=["pert", "clean"], default=None,
                    help="v3 (xsite) only, and required there: which training arm's model to "
                         "evaluate. See --test-arm for the other axis.")
    ap.add_argument("--test-arm", default=None,
                    help="v3 (xsite) only: which copy of the test split to score on (--train-arm "
                         "selects the model). Defaults to the train arm (the 2x2 diagonal: A and "
                         "D). Set it to the other arm for the off-diagonal cells: train clean + "
                         "test pert = B, train pert + test clean = C. The same flag as in "
                         "15_eval_runner, except that the test copy is regenerated here instead of "
                         "read from a directory.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: perturbation severity multiplier for the test arm (0.5 / 1 / 2)")
    ap.add_argument("--perturb-salt", default=None,
                    help="v3 only: override the base perturbation salt")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fit-jobs", type=int, default=8)
    ap.add_argument("--subset", type=int, default=SUBSET_SIZE,
                    help="training budget. Change it only for smoke runs: any other value caches "
                         "the model under a __subset<N> path, so it is not loaded as the canonical model")
    ap.add_argument("--n-test-patients", type=int, default=None, help="limit test patients (smoke)")
    ap.add_argument("--verify", action="store_true",
                    help="compare each seed's micro top-1 with 07's value for the same cell and seed "
                         f"(tolerance {VERIFY_TOL}). A missing 07 result, or one from another recipe, "
                         "counts as a failure")
    ap.add_argument("--test-only", action="store_true",
                    help="regenerate the test split, check obs_ordinal alignment against the "
                         "deep-model dumps, then exit. Checks the join before any fit.")
    ap.add_argument("--write", action="store_true", help="write per-observation parquet + manifest")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--evalset", choices=["mimictest", "relma"], default="mimictest",
                    help="relma = R2: substitute real local lab names for descriptions")
    ap.add_argument("--relma-strategy", choices=["s1", "s2"], default="s1")
    ap.add_argument("--assignment-seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="val = held-out patients used to fit the temperature for A2. They are "
                         "disjoint from test and get the val paraphrase pool of 09's 70/15/15 "
                         "split, as in 06's val split (the train pool would make val look like "
                         "training data and bias T downward).")
    ap.add_argument("--dump-full-logits", action="store_true",
                    help="write the complete per-observation distribution as centred log-probs "
                         "(float16) + row_max (float32). TF-IDF is OvR-LogReg and has no native "
                         "softmax logit vector, so log(predict_proba) stands in for it. Since "
                         "softmax(log p) == p at T=1, temperature scaling applies unchanged.")
    ap.add_argument("--refit", action="store_true",
                    help="ignore a cached fitted model and refit from scratch")
    args = ap.parse_args()

    arm = resolve_arm_suffix(args.condition, args.train_arm, "16_tfidf_eval_dump")
    from utils import extract_structured_features_single
    from input_preprocessing_utils import (load_qual_normalizer, load_qual_passthrough,
                                           load_unit_alias_map, normalize_qual_value, normalize_unit)
    from fhir_data_utils import build_lab_code_mappings
    from sklearn.preprocessing import LabelEncoder
    import importlib.util
    _s = importlib.util.spec_from_file_location("d6", Path(__file__).resolve().parent / "06_tfidf_dataset.py")
    d6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(d6)
    _s9 = importlib.util.spec_from_file_location("d9", Path(__file__).resolve().parent / "09_mlm_dataset.py")
    d9 = importlib.util.module_from_spec(_s9); _s9.loader.exec_module(d9)

    exclude, coarsen = parse_condition(args.condition)
    print(f"condition {args.condition}: exclude={sorted(exclude) if exclude else None}, "
          f"coarsen_reference_range={coarsen!r}")
    primary679 = {r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV)) if r["in_primary_ge10occ"] == "1"}
    mimic_to_loinc = d6.load_mimic_to_loinc(SEEN_CODES)
    mimic_code_to_class, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES)
    mappable = set(class_to_loinc)
    # the 70/15/15 split from 09, with the pools the deep models and 06 use
    _tr, _va, _te, _pst = d9.load_paraphrases_3way(d9.PARAPHRASE_DIR, verbose=False)
    test_para = {"train": _tr, "val": _va, "test": _te}[args.split]
    print(f"  paraphrase pool [{args.split}] from 09: {sum(len(v) for v in test_para.values()):,} "
          f"strings | split sha256={_pst['split_sha256'][:16]}…")
    print(f"label space 679={len(primary679)} | LOINC-mappable(828)={len(mappable)} | "
          f"paraphrase LOINC={len(test_para)}")

    # -- description source: MedGemma test paraphrases (mimictest) or RELMA local names (relma) --
    relma_salt = clean_lookup = None
    desc_pool = test_para
    if args.evalset == "relma":
        marked = json.loads((RELMA_POOL_DIR / f"pool_{args.relma_strategy}.json").read_text())
        desc_pool = {l: list(v["names"]) for l, v in marked.items()}
        clean_lookup = {(l, n): (bool(cs), bool(cg))
                        for l, v in marked.items()
                        for n, cs, cg in zip(v["names"], v["clean_same_loinc"], v["clean_global"])}
        relma_salt = f"relma-{args.relma_strategy}-a{args.assignment_seed}"
        print(f"RELMA pool [{args.relma_strategy}]: {len(desc_pool)} LOINC, "
              f"{sum(len(v) for v in desc_pool.values()):,} names | salt='{relma_salt}' "
              "(per-patient assignment, the same draw as the deep models)")

    qual_v2c = load_qual_normalizer(QUAL_NORMALIZE_PATH)
    qual_lit, qual_rx = load_qual_passthrough(QUAL_PASSTHROUGH_PATH)
    unit_map = load_unit_alias_map(UNIT_SYNONYM_PATH)
    unit_norm = lambda raw: (normalize_unit(raw, unit_map)[0] if raw else "")
    def qual_norm(raw):
        outcome, val = normalize_qual_value(raw, qual_v2c, qual_lit, qual_rx)
        return None if outcome == "blank" else val

    # The test side is regenerated here, so it must be formatted the same way as the training side,
    # or the model is scored on text it never saw. For v2 that means the same standardisation and
    # fixed-width rendering and no ReferenceRange.
    _rep = representation(args.condition)
    # DATA_DIR is the directory of the train arm. It feeds `load_train_labels` and
    # `load_selected_prompts` for the fit. The test arm is not a directory here, because the test
    # side is regenerated from the raw ndjson below. It is a different `perturb` closure.
    DATA_DIR = DATA_DIR_BASE + dataset_suffix(args.condition) + arm
    if not Path(DATA_DIR).exists():
        sys.exit(f"dataset for representation {_rep} not found at {DATA_DIR}")
    _perturb = None
    if _rep in ("v2", "v3"):
        import unit_standardisation as us
        # v2 drops the ReferenceRange, and v3 keeps it and standardises it the same way, as main() in
        # 06_tfidf_dataset.py does. If the two disagree, the model is scored on text it was never
        # trained on.
        _inc_rr = _rep == "v3"
        _want_rr = includes_reference_range(args.condition)
        if _inc_rr != _want_rr:
            sys.exit(f"representation/condition mismatch: representation {_rep} emits "
                     f"{'a' if _inc_rr else 'no'} ReferenceRange, but condition {args.condition!r} "
                     f"says it should {'carry' if _want_rr else 'omit'} one.")
        _kw = dict(quantity_fmt=lambda v, u: us.standardise_quantity(v, u, unit_map),
                   include_reference_range=_inc_rr)
        if _inc_rr:
            _kw["range_fmt"] = lambda rrs, vru: us.standardise_reference_range(rrs, vru, unit_map)
        _fmt = functools.partial(d6.format_observation_as_prompt, **_kw)
        print(f"  representation {_rep}: values at {us.SIG_FIGS} s.f. scientific, units standardised, "
              f"{'ReferenceRange standardised the same way' if _inc_rr else 'no ReferenceRange line'}")
    else:
        _fmt = d6.format_observation_as_prompt

    if _rep == "v3":
        import perturbation as pert
        # The test arm (which copy of the test split to score on), built as a closure. Unit-notation
        # randomisation sits outside the perturbation switches, because it is part of the v3
        # representation and on in every arm.
        _test_suffix, _cross_tag = resolve_test_arm(args.condition, arm, args.test_arm,
                                                    "16_tfidf_eval_dump")
        _spec = pert.load_spec()
        _rand = None if args.no_randomise_units else pert.load_randomisation()
        _salt = args.perturb_salt or pert.DEFAULT_SALT
        # The arm name is parsed explicitly, because it decides which numbers move. A name that does
        # not parse stops the run. Falling back to a default could score the model of one cell on
        # the data of another. When the name carries a severity, a --severity that disagrees with it
        # is refused.
        _tname = args.test_arm or _test_suffix.lstrip("_")
        _sev = args.severity
        _m = re.fullmatch(r"(clean|pert)(?:_(value|range))?(?:_r(\d+))?(?:_sev([0-9.]+))?", _tname)
        if _m is None:
            sys.exit(f"unrecognised --test-arm {_tname!r}. Expected clean | pert | pert_value | "
                     "pert_range | pert_r<N> | pert_sev<X>.")
        _base, _only, _rep_n, _sev_s = _m.groups()
        if _sev_s is not None:
            if args.severity != 1.0 and float(_sev_s) != args.severity:
                sys.exit(f"--test-arm {_tname!r} says severity {_sev_s} but --severity says "
                         f"{args.severity}. Pick one.")
            _sev = float(_sev_s)
        if _rep_n is not None:                       # a replicate is a different site, so a new salt
            _salt = f"{_salt}-r{_rep_n}"
        _do_v = _base == "pert" and _only != "range"
        _do_r = _base == "pert" and _only != "value"

        def _perturb(resources, uuid):
            if _rand is not None:
                pert.randomise_units(resources, uuid, _rand,
                                     lambda r: pert.lab_loinc_of(r, mimic_to_loinc), salt=_salt)
            pert.perturb_patient(resources, uuid, _spec,
                                 lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
                                 severity=_sev, perturb_values=_do_v,
                                 perturb_ranges=_do_r, salt=_salt)
        print(f"  test arm {_tname!r}: severity {_sev:g}, values {'on' if _do_v else 'off'}, "
              f"ranges {'on' if _do_r else 'off'}, units randomised "
              f"{'on' if _rand is not None else 'off'}, salt {_salt!r}")
    else:
        _cross_tag = ""

    # -- Test side: regenerate chronologically --
    t0 = time.time()
    test_uuids = d6._load_uuids(SPLITS_DIR / f"{args.split}_patients.txt")
    if args.n_test_patients:
        test_uuids = test_uuids[: args.n_test_patients]
    rng = random.Random(SEED)
    rows = list(build_test_rows(test_uuids, primary679, mimic_to_loinc, mimic_code_to_class, mappable,
                                desc_pool, unit_norm, qual_norm, _fmt, rng,
                                relma_salt=relma_salt, clean_lookup=clean_lookup,
                                perturb=_perturb))
    st = build_test_rows.stats
    print(f"test regenerated chronologically in {time.time()-t0:.0f}s: {len(rows):,} scorable obs "
          f"({st['n_mappable']:,} LOINC-mappable enumerated -> {100*len(rows)/max(st['n_mappable'],1):.1f}% "
          "in the 679 space)")
    if relma_salt:
        print(f"  RELMA: skipped {st['n_skipped_no_relma_name']:,} obs with no local name "
              f"({100*st['n_skipped_no_relma_name']/max(st['n_mappable'],1):.1f}% of mappable)")
    te_uuid = [r[0] for r in rows]; te_ord = np.array([r[1] for r in rows], dtype=np.int32)
    Xte = [extract_structured_features_single(r[2], exclude, coarsen) for r in rows]
    yte = [r[3] for r in rows]
    te_name = [r[4] for r in rows]
    te_clean = [r[5] for r in rows]
    te_clean_g = [r[6] for r in rows]

    le = LabelEncoder().fit(sorted(primary679))
    yte_enc = le.transform(yte)

    if args.test_only:
        # Arm gate: a digest of the regenerated prompt text. The join gate below checks the set of
        # positions and ignores the numbers, so a --test-arm that had no effect would still pass it.
        # Run --test-only for two arms and compare these lines. The counts must match (the same
        # observations), and the digests must differ (different numbers).
        #
        # The digest ignores the condition. For TF-IDF the description axis is applied at feature
        # extraction (`exclude={'description'}`), after the prompt is written. para_xsite and
        # nodesc_xsite therefore give the same digest, and they also share one dataset directory.
        # Equal digests across two conditions are expected. Equal digests across two arms are a bug.
        import hashlib
        _dig = hashlib.sha256("\n".join(r[2] for r in rows).encode()).hexdigest()[:16]
        _shown = _tname if _rep == "v3" else "n/a"
        print(f"\nARM GATE | test arm {_shown!r} | {len(rows):,} prompts | sha256={_dig}")
        # Join gate: the regenerated obs_ordinal must identify the same observation the deep models
        # scored, so compare the true LOINC on shared (uuid, obs_ordinal) keys with a BCM dump.
        import pandas as pd
        print("\n" + "=" * 64 + "\nJOIN GATE vs BCM dump\n" + "=" * 64)
        bcm = pd.read_parquet(
            f"{EVAL_BASE}/bcm__nodesc__seed0__mimictest/part-0000.parquet",
            columns=["patient_uuid", "obs_ordinal", "true_class"])
        c2l = np.array([class_to_loinc.get(i, f"?{i}") for i in range(1623)], dtype=object)
        bcm["loinc"] = c2l[bcm.true_class.values]
        theirs = {(u, int(o)): l for u, o, l in
                  zip(bcm.patient_uuid, bcm.obs_ordinal, bcm.loinc)}
        mine = {(u, int(o)): l for u, o, l in zip(te_uuid, te_ord, yte)}
        shared = set(mine) & set(theirs)
        agree = sum(mine[k] == theirs[k] for k in shared)
        print(f"my scorable obs: {len(mine):,} | BCM rows: {len(theirs):,} | shared keys: {len(shared):,}")
        print(f"true-LOINC agreement on shared keys: {agree:,}/{len(shared):,} = "
              f"{100*agree/max(len(shared),1):.2f}%")
        print("VERDICT:", "✓ PASS (ordinals align, so TF-IDF joins the paired tests)"
              if shared and agree == len(shared) else "✗ FAIL (ordinals do not align)")
        return

    # -- Training side: loaded lazily, so a cached model skips the slow label load --
    _train_cache = {}

    def get_train():
        if not _train_cache:
            t = time.time()
            ytr, files, counts = load_train_labels(DATA_DIR)
            print(f"train labels loaded in {time.time()-t:.0f}s: {len(ytr):,} observations")
            assert set(ytr) <= primary679, "train labels escape the 679 space"
            _train_cache.update(ytr=ytr, files=files, counts=counts)
        return _train_cache["ytr"], _train_cache["files"], _train_cache["counts"]

    pretrain4096 = set()
    if args.dump_full_logits:
        pretrain4096 = {l.strip() for l in open(REPO / "data" / "pretrain" /
                                                "pretrain_4096_patients.txt") if l.strip()}
        print(f"pretrain-4096 contamination set: {len(pretrain4096):,} patients")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    verify_failures = []
    for seed in args.seeds:
        mdir = Path(MODELS_BASE) / f"tfidf__{args.condition}{arm}__seed{seed}"
        if args.subset != SUBSET_SIZE:       # a smoke-sized model must not sit at the canonical path
            mdir = mdir.with_name(mdir.name + f"__subset{args.subset}")
        mpath = mdir / "pipeline.joblib"

        if mpath.exists() and not args.refit:
            import joblib
            t0 = time.time()
            pipe = joblib.load(mpath)
            # Recipe guard: the cache path does not encode the recipe, so a model fitted under another
            # recipe would otherwise be loaded and scored as if it were current.
            _got = pipeline_signature(pipe)
            _want = pipeline_signature(make_pipeline(PINNED, args.fit_jobs))
            if _got != _want:
                _diff = {k: (_got.get(k), _want.get(k)) for k in sorted(set(_got) | set(_want))
                         if _got.get(k) != _want.get(k)}
                sys.exit(f"cached model {mpath} was fitted under a different recipe "
                         f"(cached, current): {_diff}. Delete it or pass --refit.")
            print(f"[seed {seed}] loaded cached model from {mpath} in {time.time()-t0:.0f}s "
                  "(recipe signature matches, so the fit and train load were skipped)")
        else:
            import joblib
            ytr, files, counts = get_train()
            t0 = time.time()
            idx = downsample_indices(ytr, args.subset, DOWNSAMPLE_MIN, seed)
            Xs = load_selected_prompts(DATA_DIR, files, counts, idx,
                                       extract_structured_features_single, exclude, coarsen)
            ys = le.transform([ytr[i] for i in idx])
            print(f"[seed {seed}] downsampled+loaded {len(Xs):,} train obs in {time.time()-t0:.0f}s")

            t0 = time.time()
            pipe = make_pipeline(PINNED, args.fit_jobs)
            pipe.fit(Xs, ys)
            fit_sec = time.time() - t0
            print(f"[seed {seed}] fit in {fit_sec:.0f}s")

            mdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            joblib.dump(pipe, mpath, compress=3)
            (mdir / "manifest.json").write_text(json.dumps({
                "model": "tfidf", "condition": args.condition, "seed": seed,
                "pinned": PINNED, "signature": pipeline_signature(pipe),
                "subset_size": args.subset, "downsample_min": DOWNSAMPLE_MIN,
                "n_train_obs_fitted": len(Xs), "label_space_679_size": len(le.classes_),
                "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
                "fit_sec": round(fit_sec, 1), "sklearn_env": "data-analysis",
                "note": ("Persisted so later evaluations never refit; 07_tfidf_train_eval.py does not "
                         "save its models."),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))
            print(f"[seed {seed}] saved model -> {mpath} "
                  f"({mpath.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")

        # -- predict + reduce to top-K, batched --
        t0 = time.time()
        clf_classes = pipe.named_steps["clf"].classes_
        n = len(Xte)
        topk_idx = np.zeros((n, TOP_K_MAX), dtype=np.int16)
        topk_lp = np.zeros((n, TOP_K_MAX), dtype=np.float32)
        true_lp = np.zeros(n, dtype=np.float32)
        true_rank_all = np.zeros(n, dtype=np.int32)
        logit_chunks, rowmax_chunks = [], []
        hits = {k: 0 for k in TOPK}
        for s in range(0, n, EVAL_BATCH):
            sl = slice(s, min(s + EVAL_BATCH, n))
            pm = pipe.predict_proba(Xte[sl])
            full = np.zeros((pm.shape[0], len(le.classes_)), dtype=np.float64)
            full[:, clf_classes] = pm
            part = np.argpartition(-full, TOP_K_MAX - 1, axis=1)[:, :TOP_K_MAX]
            ordr = np.argsort(-np.take_along_axis(full, part, 1), axis=1)
            tk = np.take_along_axis(part, ordr, 1)
            topk_idx[sl] = tk.astype(np.int16)
            if args.dump_full_logits:
                with np.errstate(divide="ignore"):
                    lp = np.log(full)                      # (b, 679) log-probabilities
                lp = np.maximum(lp, -60.0)                 # floor -inf (p=0) so float16 stays finite
                rmx = lp.max(axis=1, keepdims=True)
                logit_chunks.append((lp - rmx).astype(np.float16))
                rowmax_chunks.append(rmx.squeeze(1).astype(np.float32))
                _yb = yte_enc[sl]      # yb is bound later in the loop, so use the slice directly here
                true_rank_all[sl] = (full > full[np.arange(full.shape[0]), _yb][:, None]).sum(axis=1) + 1
            yb = yte_enc[sl]
            with np.errstate(divide="ignore"):   # p=0 gives a -inf logprob, which is correct
                topk_lp[sl] = np.log(np.take_along_axis(full, tk, 1)).astype(np.float32)
                true_lp[sl] = np.log(full[np.arange(full.shape[0]), yb]).astype(np.float32)
            tp = full[np.arange(full.shape[0]), yb]
            ng = (full > tp[:, None]).sum(axis=1)
            for k in TOPK:
                hits[k] += int((ng < k).sum())
        micro = {f"top{k}": hits[k] / n for k in TOPK}
        print(f"[seed {seed}] predicted in {time.time()-t0:.0f}s | " +
              " ".join(f"{k}={v:.5f}" for k, v in micro.items()))

        # The expectation is the number from 07 for this (condition, train arm, test arm, seed).
        _exp_key = f"{args.condition}{arm}{_cross_tag}"
        exp = None
        if args.split == "test" and args.evalset == "mimictest":
            exp, _exp_src = expected_top1_from_07(args.condition, arm, _cross_tag, seed)
        if args.verify and args.split != "test":
            print(f"[seed {seed}] --verify skipped: 07's numbers are test-set numbers")
        elif args.verify and args.evalset != "mimictest":
            print(f"[seed {seed}] --verify skipped: 07's numbers are mimictest numbers, "
                  f"not {args.evalset!r}")
        elif args.verify and exp is None:
            # counted as a failure. Under --verify, a check that could not be made must not read as
            # passed in an unattended run.
            print(f"✗ VERIFY FAIL [seed {seed}] [{_exp_key}]: no expected value: {_exp_src}")
            verify_failures.append((seed, _exp_key, micro["top1"], float("nan"), float("nan")))
        elif args.verify:
            d = abs(micro["top1"] - exp)
            ok = d < VERIFY_TOL
            print("=" * 64)
            print(f"{'✓ VERIFY PASS' if ok else '✗ VERIFY FAIL'}: micro top1={micro['top1']:.5f} "
                  f"vs 07 {exp:.5f} (Δ={d:.5f}, tolerance {VERIFY_TOL})   [{_exp_key}; {_exp_src}]")
            print("   (Δ expected to be tiny but can be nonzero: regenerating the split can change which\n"
                  "    paraphrase an observation draws from the same test pool.)")
            print("=" * 64)
            if not ok:
                # recorded here, without raising. The run still dumps every seed, and the non-zero
                # exit comes after all the writes, so a batch runner logs the failure and nothing is
                # lost. A printed FAIL alone is easy to miss in a long unattended run.
                verify_failures.append((seed, _exp_key, micro["top1"], exp, d))

        if args.write:
            import pandas as pd
            tag = ("mimictest" if args.evalset != "relma"
                   else f"relma_{args.relma_strategy}_a{args.assignment_seed}")
            # `_cross_tag` is '' on the 2x2 diagonal and '__on{arm}' off it, so cells B and C cannot
            # overwrite the A and D dumps they share a model with.
            run = f"tfidf__{args.condition}{arm}__seed{seed}__{tag}{_cross_tag}"
            # Guardrails: a partial run must not occupy a canonical path. The split goes in the run
            # name, or a val run would overwrite the test dump. --n-test-patients (a smoke-scale flag,
            # like --limit elsewhere) and a non-default --subset add their own suffixes.
            if args.split != "test":
                run += f"__{args.split}"
            if args.n_test_patients:
                run += f"__smoke{args.n_test_patients}"
                print(f"  [PARTIAL RUN] --n-test-patients set -> '{run}' (not the canonical path)")
            if args.subset != SUBSET_SIZE:   # same rule for a smoke-sized training budget
                run += f"__subset{args.subset}"
                print(f"  [PARTIAL RUN] --subset {args.subset} -> '{run}' (not the canonical path)")
            out = Path(EVAL_BASE) / run
            if out.exists() and not args.force:
                raise FileExistsError(f"{out} exists (pass --force to overwrite)")
            out.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "patient_uuid": te_uuid, "obs_ordinal": te_ord,
                # `desc_shown`: the description this observation received, a MedGemma paraphrase in
                # mimictest or a real local name in relma
                "desc_shown": te_name, "clean_same_loinc": te_clean, "clean_global": te_clean_g,
                "true_rank": true_rank_all,
                "in_pretrain_4096": [u in pretrain4096 for u in te_uuid],
                "true_loinc": yte, "true_idx": yte_enc.astype(np.int16),
                "topk_idx": list(topk_idx), "topk_logprob": list(topk_lp),
                "true_logprob": true_lp, "row_logsumexp": np.zeros(n, dtype=np.float32),
                "model": "tfidf", "condition": args.condition, "seed": seed, "evalset": args.evalset,
            })
            if args.dump_full_logits:
                L = np.concatenate(logit_chunks, axis=0)
                RM = np.concatenate(rowmax_chunks, axis=0)
                # alignment check: npy row i must be parquet row i (as in 15_eval_runner.py)
                if not (len(L) == len(RM) == len(df) == n):
                    raise AssertionError(f"misalignment: logits={len(L)} rowmax={len(RM)} "
                                         f"parquet={len(df)} n={n}")
                mx = L.max(axis=1)
                if not np.allclose(mx, 0, atol=1e-2):
                    raise AssertionError("centring broken: max(centred) range "
                                         f"[{mx.min():.4f}, {mx.max():.4f}]")
                np.save(out / "logits_centered_f16.npy", L)
                np.save(out / "row_max_f32.npy", RM)
                print(f"[seed {seed}] wrote logits_centered_f16.npy {L.shape} "
                      f"({L.nbytes/1e9:.2f} GB, alignment checked)")
            df.to_parquet(out / "part-0000.parquet", index=False)
            man = {"run_name": run, "model": "tfidf", "condition": args.condition, "seed": seed,
                   "evalset": args.evalset, "evalset_tag": tag, "split": args.split,
                   # what was captured, so a resumable runner can tell a full-logit run from one
                   # without logits and does not re-run it needlessly
                   "full_logits": bool(args.dump_full_logits),
                   "full_logits_dtype": "float16 centred log-probs (log p - row_max); row_max float32",
                   "extra_columns": ["true_rank", "in_pretrain_4096", "desc_shown"],
                   "relma_strategy": args.relma_strategy if args.evalset == "relma" else None,
                   "assignment_seed": args.assignment_seed if args.evalset == "relma" else None,
                   "n_skipped_no_relma_name": st.get("n_skipped_no_relma_name", 0),
                   "n_scored": int(n),
                   "n_mappable_enumerated": int(st["n_mappable"]),
                   "metrics": micro, "pinned": PINNED, "subset_size": args.subset,
                   "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
                   # the number from 07 for this cell and seed (None for val and RELMA, which 07
                   # does not score)
                   "expected_top1_from_07": exp,
                   "label_space_679": sorted(le.classes_.tolist()),
                   "note": ("probabilities are already normalised: row_logsumexp=0 so the standard "
                            "p = exp(logit - lse) reconstruction still holds"),
                   "description_source": (f"RELMA local names, pool_{args.relma_strategy}"
                                          if args.evalset == "relma" else
                                          f"paraphrases: the '{args.split}' pool of 09's 70/15/15 "
                                          "split, the same pools the deep models use"),
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (out / "manifest.json").write_text(json.dumps(man, indent=2))
            (RESULTS_DIR / f"{run}.json").write_text(json.dumps({k: v for k, v in man.items()
                                                                 if k != "label_space_679"}, indent=2))
            print(f"[seed {seed}] wrote {out/'part-0000.parquet'} "
                  f"({(out/'part-0000.parquet').stat().st_size/1e6:.1f} MB)")

    if verify_failures:
        print("\n" + "=" * 64)
        print(f"✗ {len(verify_failures)} VERIFY FAILURE(S): 16's refit does not reproduce 07")
        for seed, key, got, exp, d in verify_failures:
            print(f"   seed {seed} [{key}]: got {got:.5f}, 07 reported {exp:.5f}, Δ={d:.5f}")
        print("The dumps above were written. TF-IDF's catalogue numbers would come from a")
        print("different model than its observation-level ones, so investigate before using them.")
        print("Exiting non-zero, so a driver records this as a failure.")
        print("=" * 64)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
