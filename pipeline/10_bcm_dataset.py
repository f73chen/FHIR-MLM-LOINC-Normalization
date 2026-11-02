"""Build the BioClinical ModernBERT (BCM) masked-history datasets (thesis Chapter 5).

BCM is trained as token classification, while FHIR-MLM is trained as a masked LM. `FHIRFormatter`
turns the FHIR resources into text, which the ModernBERT BPE tokenizer tokenizes. `tokenize_and_align`
adds explicit `labels`: the MIMIC class id at [MASK] positions and -100 elsewhere. Row schema:
{input_ids, attention_mask, labels, uuid}.

Conditions (--condition; see pipeline/conditions.py):
  para*   : FHIRFormatter(include_description=True) and per-(patient, LOINC) paraphrase patching.
  nodesc* : FHIRFormatter(include_description=False), no patching (no description line).
The v3 conditions (`xsite`) also standardise units, randomise unit notation and, unless switched
off, perturb values and reference ranges to simulate a second site.

Cross-model controls: `load_paraphrases_3way` and `read_uuid_list` come from
pipeline/09_mlm_dataset.py, so the 70/15/15 paraphrase pools are identical to FHIR-MLM's. The same
`patient_rng` (shared salt) gives every (patient, LOINC) the same paraphrase as in FHIR-MLM.
Paraphrase sampling therefore does not differ between the models.

The FHIRFormatter used for BCM needs no encounter anchor (t0), unlike FHIR-MLM, so it keeps patients
without an encounter (lab-only patients) and uses nearly the whole split. The build covers the full
split. Restricting it to the FHIR-MLM subset of patients with an encounter is a later UUID-subset
decision.

--jobs parallelises across processes and gives the same dataset as the sequential path. Error logs
print the exception type only, never a patient UUID. Runs in the deep-models environment.

Writes WORK_DIR/datasets/bcm__<condition><arm>/{train,val,test}/ and a manifest in
results/10_bcm_dataset/.

  python pipeline/10_bcm_dataset.py --condition para_xsite --jobs 16
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, has_description, arm_suffix, includes_reference_range,  # noqa: E402
                        perturbation_tag, train_arm,
                        refrange_mode, representation)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -- shared helpers and the FHIR-MLM code --
for p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Import the 70/15/15 split + UUID reader from 09 (its name starts with a digit, so importlib).
#    The paraphrase pools are then identical to FHIR-MLM's (cross-model control). --
_spec09 = importlib.util.spec_from_file_location(
    "md09", Path(__file__).resolve().parent / "09_mlm_dataset.py")
md09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(md09)
load_paraphrases_3way = md09.load_paraphrases_3way
read_uuid_list = md09.read_uuid_list

# -- Paths: repository --
REPO = locations.REPO
SPLITS_DIR = REPO / "data" / "splits"
PARAPHRASE_DIR = REPO / "data" / "paraphrases" / "clean"
RESULTS_DIR = REPO / "results" / "10_bcm_dataset"

# -- Paths: static inputs --
NDJSON_DIR = locations.PATIENTS
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
UNIT_SYNONYM_PATH = str(locations.UPSTREAM / "unit_synonym_map.json")
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"

# -- Output / model --
OUTPUT_BASE = str(locations.WORK / "datasets")
MODERNBERT_MODEL = "thomas-sounack/BioClinical-ModernBERT-large"

# -- Config --
MAX_LEN = 8192
SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)
EXPECTED_COLS = {"input_ids", "attention_mask", "labels", "uuid"}
CACHE_DIR = ""  # set in main


# --- stats + shared per-patient logic ---------------------------------------
def _new_stats(n_input):
    return {"n_input": n_input, "n_missing_file": 0, "n_format_error": 0, "n_no_codes": 0,
            "n_other_error": 0, "n_no_labels": 0, "n_ok": 0, "total_obs_patched": 0,
            # v3 only. Declared here because _merge_stats sums over the keys of
            # _new_stats. A key that a parallel worker creates on first use would be
            # dropped when the shards are merged.
            "total_fields_perturbed": 0,
            "total_units_randomised": 0}


def _merge_stats(stats_list, n_input):
    out = _new_stats(n_input)
    for s in stats_list:
        for k in out:
            if k != "n_input":
                out[k] += s.get(k, 0)
    return out


def _load_seen_codes_mimic_to_loinc(seen_codes_path):
    """{mimic_code -> loinc} for paraphrase patching."""
    out = {}
    with open(seen_codes_path) as f:
        for row in csv.DictReader(f):
            if row["mimic_code"] and row["loinc_code"]:
                out[row["mimic_code"]] = row["loinc_code"]
    return out


#: loaded once per process (the parallel path runs worker processes)
_PERT_SPEC = None
_RAND_TABLE = None


def _apply_perturbation(uuid, resources, mimic_to_loinc, cfg, stats):
    """v3 (`xsite`): simulate a second site on this patient's raw resources, before formatting.

    `cfg` is a plain dict so that it survives pickling to the joblib workers, which a callable might
    not. The LOINC predicate is perturbation.lab_loinc_of, the same one 06 and 09 use, so all four
    builders perturb the same set of observations."""
    global _PERT_SPEC
    import perturbation as pert
    if _PERT_SPEC is None:
        _PERT_SPEC = pert.load_spec()
    # unit-notation randomisation is part of the v3 representation. It is on in every arm, and
    # the perturbation switches do not control it.
    global _RAND_TABLE
    if cfg.get("randomise_units", True):
        if _RAND_TABLE is None:
            _RAND_TABLE = pert.load_randomisation()
        stats["total_units_randomised"] += pert.randomise_units(
            resources, uuid, _RAND_TABLE, lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
            salt=cfg.get("salt", pert.DEFAULT_SALT))
    stats["total_fields_perturbed"] += pert.perturb_patient(
        resources, uuid, _PERT_SPEC, lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
        severity=cfg.get("severity", 1.0), perturb_values=cfg.get("perturb_values", True),
        perturb_ranges=cfg.get("perturb_ranges", True), salt=cfg.get("salt", pert.DEFAULT_SALT))


def _process_one_patient(uuid, formatter, tokenizer, mimic_code_to_class, class_to_loinc,
                         mask_token_id, max_len, paraphrase_pool, mimic_to_loinc, patch, stats, paraphrase_salt=None,
                         perturb_cfg=None):
    """Per-patient processing for both the sequential and the parallel path. Returns a row dict or
    None and updates `stats`. Logs the exception type only."""
    import ndjson
    from paraphrase_utils import patient_rng, assign_paraphrases_per_patient, apply_paraphrases_to_resources
    from fhir_data_utils import tokenize_and_align
    path = NDJSON_DIR / f"{uuid}.ndjson"
    if not path.exists():
        stats["n_missing_file"] += 1
        return None
    try:
        with open(path) as f:
            resources = ndjson.load(f)
    except Exception as e:
        stats["n_other_error"] += 1
        print(f"  read error: {type(e).__name__}")
        return None
    # runs before paraphrase patching and before formatting, on the resource list as it
    # came off disk. perturb_patient needs whole-patient context to recompute derived
    # indices (G3) and renormalise fraction panels (G4).
    if perturb_cfg:
        _apply_perturbation(uuid, resources, mimic_to_loinc, perturb_cfg, stats)
    if patch:
        rng = patient_rng(uuid, salt=paraphrase_salt) if paraphrase_salt else patient_rng(uuid)
        mapping = assign_paraphrases_per_patient(resources, paraphrase_pool, mimic_to_loinc, rng)
        stats["total_obs_patched"] += apply_paraphrases_to_resources(resources, mapping, mimic_to_loinc)
    try:
        prompt, codes = formatter.format_resources(resources)
    except Exception as e:
        stats["n_format_error"] += 1
        print(f"  format error: {type(e).__name__}")
        return None
    if not codes:
        stats["n_no_codes"] += 1
        return None
    try:
        example = tokenize_and_align(prompt, codes, tokenizer, mimic_code_to_class, class_to_loinc,
                                     mask_token_id=mask_token_id, max_len=max_len)
    except Exception as e:
        stats["n_other_error"] += 1
        print(f"  align error: {type(e).__name__}")
        return None
    if example is None:
        stats["n_no_labels"] += 1
        return None
    stats["n_ok"] += 1
    example["uuid"] = uuid
    return example


def _build_bcm_tokenizer_formatter(with_desc, with_refrange=True, rep="v1"):
    """Build the ModernBERT tokenizer + FHIRFormatter (once per worker in parallel, once in main for jobs=1)."""
    import transformers
    from transformers import AutoTokenizer
    transformers.logging.set_verbosity_error()  # silence the benign ">max length" warning, since rows are truncated to MAX_LEN (the first window)
    from fhir_formatter import FHIRFormatter
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise RuntimeError("ModernBERT tokenizer has no mask_token")
    import unit_standardisation as us
    from input_preprocessing_utils import load_unit_alias_map
    # v2/v3 numeric representation, applied at the lab call sites only. Built here,
    # inside the worker, because these are closures and the parallel path pickles its
    # arguments. `rep` is a plain string and survives the trip.
    qfmt, rfmt = us.deep_model_fmts(rep, load_unit_alias_map(UNIT_SYNONYM_PATH))
    formatter = FHIRFormatter.from_data_dirs(mask_token=tokenizer.mask_token, include_description=with_desc,
                                             include_reference_range=with_refrange,
                                             quantity_fmt=qfmt, range_fmt=rfmt)
    return tokenizer, formatter


def _process_chunk(uuids_chunk, paraphrase_pool, mimic_code_to_class, class_to_loinc, mimic_to_loinc,
                   with_desc, patch, max_len, shard_path, paraphrase_salt=None,
                   with_refrange=True, rep="v1", perturb_cfg=None):
    """Parallel worker (module level, so it can be pickled). Builds a tokenizer and formatter in the
    worker, runs the same _process_one_patient over a contiguous chunk of patients, writes a shard
    and returns (stats, path)."""
    import datasets
    tokenizer, formatter = _build_bcm_tokenizer_formatter(with_desc, with_refrange, rep)
    mask_token_id = tokenizer.mask_token_id
    stats = _new_stats(len(uuids_chunk))
    rows = []
    for uuid in uuids_chunk:
        row = _process_one_patient(uuid, formatter, tokenizer, mimic_code_to_class, class_to_loinc,
                                   mask_token_id, max_len, paraphrase_pool, mimic_to_loinc, patch, stats,
                                   paraphrase_salt=paraphrase_salt, perturb_cfg=perturb_cfg)
        if row is not None:
            rows.append(row)
    if rows:
        datasets.Dataset.from_list(rows).save_to_disk(shard_path)
    return stats, (shard_path if rows else None)


def _build_split_dataset(uuids, paraphrase_pool, mimic_code_to_class, class_to_loinc, mimic_to_loinc,
                         formatter, tokenizer, mask_token_id, with_desc, max_len, desc, patch, jobs,
                         paraphrase_salt=None, with_refrange=True, rep="v1",
                         perturb_cfg=None):
    """jobs == 1: sequential from_generator (the reference path). jobs > 1: split the patients into
    contiguous chunks, process them across processes and concatenate the shards in order. This gives
    an identical dataset and merged stats. Both paths call the same _process_one_patient. Returns
    (ds, stats, shard_root_or_None)."""
    import datasets
    if not uuids:
        return datasets.Dataset.from_dict({c: [] for c in EXPECTED_COLS}), _new_stats(0), None
    if jobs == 1:
        from tqdm import tqdm
        stats = _new_stats(len(uuids))

        def _gen(uuid_list):
            for uuid in tqdm(uuid_list, desc=desc):
                row = _process_one_patient(uuid, formatter, tokenizer, mimic_code_to_class, class_to_loinc,
                                           mask_token_id, max_len, paraphrase_pool,
                                           mimic_to_loinc, patch, stats,
                                           paraphrase_salt=paraphrase_salt,
                                           perturb_cfg=perturb_cfg)
                if row is not None:
                    yield row

        ds = datasets.Dataset.from_generator(_gen, gen_kwargs={"uuid_list": tuple(uuids)}, cache_dir=CACHE_DIR)
        return ds, stats, None

    from joblib import Parallel, delayed
    chunks = [list(c) for c in np.array_split(np.array(uuids, dtype=object), min(jobs, len(uuids))) if len(c)]
    shard_root = f"{CACHE_DIR}_shards_{desc}"
    shutil.rmtree(shard_root, ignore_errors=True)
    results = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_process_chunk)(ch, paraphrase_pool, mimic_code_to_class, class_to_loinc, mimic_to_loinc,
                                with_desc, patch, max_len, f"{shard_root}/{i}",
                                paraphrase_salt=paraphrase_salt, with_refrange=with_refrange,
                                rep=rep, perturb_cfg=perturb_cfg)
        for i, ch in enumerate(chunks))
    stats = _merge_stats([s for s, _ in results], len(uuids))
    shard_paths = [p for _, p in results if p]
    assert shard_paths, f"{desc}: every chunk empty"
    ds = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in shard_paths])
    return ds, stats, shard_root


def _print_stats(name, stats, ds, sample_n=1000):
    print(f"\n  {name} stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    n = len(ds)
    if n == 0:
        print("    (empty dataset)")
        return
    sample = min(sample_n, n)
    seq_lens, n_labels = [], []
    for i in range(sample):
        row = ds[i]
        seq_lens.append(len(row["input_ids"]))
        n_labels.append(sum(1 for l in row["labels"] if l != -100))
    print(f"    --- sampled {sample}/{n} sequences ---")
    print(f"    seq_len:           mean={np.mean(seq_lens):.0f} p50={np.percentile(seq_lens,50):.0f} "
          f"p90={np.percentile(seq_lens,90):.0f} max={max(seq_lens)}")
    print(f"    labeled positions: mean={np.mean(n_labels):.1f} p50={np.percentile(n_labels,50):.0f} "
          f"p90={np.percentile(n_labels,90):.0f} max={max(n_labels)}")


# --- main -------------------------------------------------------------------
fn_tag = "10_bcm_dataset"


def main():
    global CACHE_DIR
    import datasets
    from fhir_data_utils import build_lab_code_mappings, N_LAB_CLASSES
    from paraphrase_utils import check_paraphrase_coverage

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=MLM_CONDITIONS, required=True,
                    help="see pipeline/conditions.py for the slug grammar")
    ap.add_argument("--smoke", type=int, default=0, help="process only first N patients/split, write to __smoke")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for tokenization (1=sequential, audited)")
    # v3 (xsite) settings, ignored for every other condition. Each changes numbers, so each is
    # recorded in the manifest.
    ap.add_argument("--replicate", type=int, default=None,
                    help="v3 only: build perturbation replicate N, an independent site draw "
                         "with its own derived salt and its own directory `..._r{N}`.")
    ap.add_argument("--splits", default="train,val,test",
                    help="which splits to build. For a v3 replicate, build only 'test': its "
                         "train and val are not read, and train is 10x the size of test "
                         "(100,000 patients against 10,000).")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: multiplies every perturbation magnitude (sensitivity arm)")
    ap.add_argument("--no-perturb-values", action="store_true",
                    help="v3 only: leave measured values alone, perturb the reference range only")
    ap.add_argument("--no-perturb-ranges", action="store_true",
                    help="v3 only: leave the reference range alone, perturb values only")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--perturb-salt", default=None,
                    help="v3 only: overrides the site-draw salt; two arms differing only by salt "
                         "are two different simulated sites")
    args = ap.parse_args()
    # has_description is used here. `condition == "para"` is False for `para_norr`, so that test
    # would build a description-free dataset under a para name.
    with_desc = has_description(args.condition)
    with_refrange = includes_reference_range(args.condition)
    rep = representation(args.condition)
    perturb_cfg = None
    if rep == "v3":
        import perturbation as _pert
        # a replicate is a different site, so it needs a different salt. The replicates measure
        # the variance across perturbation draws. The salt is derived from the base salt, so each
        # replicate is reproducible from its number alone.
        _salt = args.perturb_salt or _pert.DEFAULT_SALT
        if args.replicate is not None:
            _salt = f"{_salt}-r{args.replicate}"
        perturb_cfg = {"salt": _salt,
                       "severity": args.severity,
                       "perturb_values": not args.no_perturb_values,
                       "perturb_ranges": not args.no_perturb_ranges,
                       "randomise_units": not args.no_randomise_units}
        print(f"  perturbation: severity {args.severity:g}, "
              f"values {'on' if perturb_cfg['perturb_values'] else 'off'}, "
              f"ranges {'on' if perturb_cfg['perturb_ranges'] else 'off'}, "
              f"salt {perturb_cfg['salt']!r}, spec {_pert.SPEC_PATH.name}")
    random.seed(SEED)        # kept for parity with 09 (the data path uses a local SHA1-based RNG)
    np.random.seed(SEED)

    # Every setting that changes numbers goes in the path as well as in the manifest. The train-arm
    # segment is there because v3 clean and perturbed builds share a condition slug and would
    # otherwise collide on disk. perturbation_tag adds the severity, the value-only or range-only
    # ablation and the replicate number. Replicates differ only by a salt that the name does not
    # record, so without the number they would be indistinguishable on disk. `arm_suffix` alone
    # only says whether anything was perturbed.
    # Without the tag, severity 0.5, severity 2.0, value-only and range-only would all land on
    # `__pert`, the dataset the perturbed arm trains on, and building the severity sweep or the
    # ablation would overwrite it.
    _rep_tag = perturbation_tag(args.severity, not args.no_perturb_values,
                                not args.no_perturb_ranges, args.replicate, fn_tag)
    out_path = (f"{OUTPUT_BASE}/bcm__{args.condition}"
                + arm_suffix(args.condition, not args.no_perturb_values,
                             not args.no_perturb_ranges) + _rep_tag
                + ("__smoke" if args.smoke else ""))
    CACHE_DIR = out_path + ".cache"
    if Path(out_path).exists():
        if args.smoke:
            shutil.rmtree(out_path)
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        else:
            raise FileExistsError(f"{out_path} exists (rm -rf '{out_path}' '{CACHE_DIR}' to rebuild)")

    print(f"=== 10 BCM dataset | condition={args.condition} (include_description={with_desc}, "
          f"include_reference_range={with_refrange}) jobs={args.jobs} ===")
    mimic_code_to_class, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    assert len(mimic_code_to_class) == N_LAB_CLASSES, f"expected {N_LAB_CLASSES} classes, got {len(mimic_code_to_class)}"
    mimic_to_loinc = _load_seen_codes_mimic_to_loinc(SEEN_CODES_PATH)
    print(f"  lab classes={len(mimic_code_to_class)} (={N_LAB_CLASSES})  LOINC-mappable={len(class_to_loinc)}  "
          f"mimic_to_loinc={len(mimic_to_loinc)}")

    print("\n=== Paraphrase split (70/15/15, same pools as 09/MLM) ===")
    train_para, val_para, test_para, para_stats = load_paraphrases_3way(PARAPHRASE_DIR)
    if with_desc:
        for lbl, pool in [("train", train_para), ("val", val_para), ("test", test_para)]:
            check_paraphrase_coverage(pool, mimic_to_loinc, label=lbl, raise_on_missing=True)

    print("\n=== Split UUIDs (read from the split files) ===")
    splits = {s: read_uuid_list(SPLITS_DIR / f"{s}_patients.txt") for s in ("train", "val", "test")}
    for s, u in splits.items():
        print(f"  {s}: {len(u)} patients")
    a, b, c = (set(splits["train"]), set(splits["val"]), set(splits["test"]))
    assert not (a & b) and not (a & c) and not (b & c), "FAIL: split UUID overlap"

    print("\n=== Tokenizer + Formatter ===")
    tokenizer, formatter = _build_bcm_tokenizer_formatter(with_desc, with_refrange, rep)
    print(f"  model={MODERNBERT_MODEL}  mask={tokenizer.mask_token!r}(id={tokenizer.mask_token_id})  "
          f"vocab={tokenizer.vocab_size}  include_description={formatter.include_description}")

    pools = {"train": train_para if with_desc else None,
             "val": val_para if with_desc else None,
             "test": test_para if with_desc else None}
    built, all_stats, shard_roots = {}, {}, []
    want_splits = tuple(x for x in ("val", "test", "train") if x in args.splits.split(","))
    if not want_splits:
        sys.exit(f"--splits {args.splits!r} selected nothing")
    for s in want_splits:                    # small to large, for earlier feedback
        uu = splits[s][:args.smoke] if args.smoke else splits[s]
        print(f"\n=== Building {s} ({len(uu)} patients, jobs={args.jobs}) ===")
        ds, st, sr = _build_split_dataset(uu, pools[s], mimic_code_to_class, class_to_loinc, mimic_to_loinc,
                                          formatter, tokenizer, tokenizer.mask_token_id, with_desc, MAX_LEN,
                                          desc=s, patch=with_desc, jobs=args.jobs,
                                          with_refrange=with_refrange, rep=rep,
                                          perturb_cfg=perturb_cfg)
        built[s], all_stats[s] = ds, st
        if sr:
            shard_roots.append(sr)

    print("\n=== Schema check ===")
    # a replicate builds `test` only, so check the split that was built. The schema is
    # the same across splits by construction.
    my_cols = set(built[want_splits[-1]].column_names)
    assert my_cols == EXPECTED_COLS, f"schema mismatch: mine={my_cols} expected={EXPECTED_COLS}"
    print(f"  columns match the spec: {sorted(my_cols)}")

    print("\n=== Saving ===")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    datasets.DatasetDict({k: built[k] for k in want_splits}).save_to_disk(out_path)
    print(f"  saved DatasetDict -> {out_path}")

    for s in want_splits:
        _print_stats(s, all_stats[s], built[s])

    for sr in shard_roots:                    # cleanup after stats prints (built[s] mmaps shards until here)
        shutil.rmtree(sr, ignore_errors=True)

    manifest = {
        "condition": args.condition, "include_description": with_desc,
        "include_reference_range": with_refrange, "refrange_mode": refrange_mode(args.condition),
        "representation": rep,
        "train_arm": train_arm(args.condition, not args.no_perturb_values,
                               not args.no_perturb_ranges) or None,
        # a v3 dataset is not reproducible without the salt, and two arms that differ only by salt
        # must be told apart later
        "v3_perturbation": perturb_cfg,
        "output_path": out_path,
        "max_len": MAX_LEN, "seed": SEED, "modernbert_model": MODERNBERT_MODEL, "smoke": args.smoke or None,
        "paraphrase_split": para_stats,
        "splits_built": list(want_splits), "replicate": args.replicate,
        "uuid_split": {s: len(splits[s]) for s in want_splits},
        "split_stats": all_stats, "schema": sorted(my_cols),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # the arm belongs in the manifest name too. Otherwise two arms of one condition would write
    # the same file, and the second would replace the first.
    mf = (RESULTS_DIR / f"manifest_{args.condition}"
          f"{arm_suffix(args.condition, not args.no_perturb_values, not args.no_perturb_ranges)}"
          f"{_rep_tag}{'_smoke' if args.smoke else ''}.json")
    mf.write_text(json.dumps(manifest, indent=2, default=str))
    Path(out_path + "/manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  manifest -> {mf}")


if __name__ == "__main__":
    main()
