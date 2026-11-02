"""Build the FHIR-MLM masked-history datasets for fine-tuning and evaluating FHIR-MLM (thesis Chapters 4
and 5). Each patient's FHIR resources are sorted chronologically and tokenized by the FHIR-MLM
tokenizer into one 8192-token row.

Rows have the columns of FHIR-MLM's fine-tuning data: input_ids, token_type_ids, attention_mask and
special_tokens_mask (each 8192 long), and uuid. There is no `labels` column, because the collator
masks the lab-code tokens at training time.

Conditions (--condition; see pipeline/conditions.py for the slug grammar):
  para      relevant_keys include 'code:display', and each (patient, LOINC) gets a paraphrase
  nodesc    relevant_keys drop 'code:display', with no paraphrases, so there are no description tokens
  *_norr    relevant_keys drop 'referenceRange'. This is the reference-range ablation, built the way
            nodesc is. Like nodesc it is a strict token subset of what the pretrained checkpoint saw,
            so it is safe against that checkpoint.
  *_xsite   v3: unit notation is randomised, and the patient's values and reference ranges are
            perturbed to simulate a second site (see _apply_perturbation). --no-perturb-values
            with --no-perturb-ranges gives the clean arm.
*_rr1sf and *_rrmag are not supported. FHIR-MLM never sees a reference-range bound as text, because
the tokenizer bins it into the 5- or 20-bucket percentile table of the item. Coarsening would
therefore have to act on the raw float before binning, inside the tokenizer. That is a different
intervention from the coarsening of the text in TF-IDF.

Each LOINC's paraphrases are split 70/15/15 into disjoint train, val and test pools. Train patients
draw from the train pool, val patients from val and test patients from test, so the test paraphrases
are held out from training and from the val-based early stopping.

As in FHIR-MLM's fine-tuning data, resources are sorted chronologically, rows are 8192 tokens, and a
patient whose tokenizer state has no t0 (KeyError 't0') or who has no lab token is skipped. For para
the relevant_keys match those of the pretrained checkpoint, and nodesc and *_norr only drop keys.

Reads the patient split (data/splits/{train,val,test}_patients.txt), the cleaned paraphrases
(data/paraphrases/clean), the per-patient ndjson files (PATIENT_NDJSON_DIR),
UPSTREAM_DIR/seen_codes.csv, the tokenizer files (FHIR_MLM_TOKENIZER_DIR) and the MIMIC medication and
location resources (MIMIC_FHIR_DIR). Writes a DatasetDict of the splits to
WORK_DIR/datasets/mlm__{condition}{arm}{tag} and a manifest to results/09_mlm_dataset/. Errors are
logged by exception type only, never with patient identifiers. Stdout shows only aggregates.

  python pipeline/09_mlm_dataset.py --condition para_xsite
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, has_description, arm_suffix, includes_reference_range,  # noqa: E402
                        perturbation_tag, train_arm,
                        refrange_mode, representation)

# Avoid HF-tokenizer / BLAS thread oversubscription when running many loky workers (--jobs N).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -- FHIR-MLM tokenizer (third_party/fhir_mlm) and shared FHIR utilities (common/) --
for p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Paths: repository data --
REPO = locations.REPO
SPLITS_DIR = REPO / "data" / "splits"
PARAPHRASE_DIR = REPO / "data" / "paraphrases" / "clean"
RESULTS_DIR = REPO / "results" / "09_mlm_dataset"

# -- Paths: patient files, seen codes and FHIR-MLM tokenizer inputs --
NDJSON_DIR = locations.PATIENTS
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
NOTEBOOKS_DIR = locations.TOKENIZER
TOKENIZER_JSON = NOTEBOOKS_DIR / "bert-base-uncased-modified-tokenizer.json"
CODES_TO_TOKENS_PATH = NOTEBOOKS_DIR / "codes_to_tokens.pkl"
LABEVENT_BINS_PATH = NOTEBOOKS_DIR / "labevent_bins.pkl"
MED_DOSE_BINS_PATH = NOTEBOOKS_DIR / "medication_dose_bins.pkl"
MED_DISP_BINS_PATH = NOTEBOOKS_DIR / "medication_dispense_max_dose_bins.pkl"
FHIR_CANONICAL_DIR = locations.MIMIC_FHIR

# -- Output --
OUTPUT_BASE = str(locations.WORK / "datasets")

# -- Config --
MAX_LEN = 8192
SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)        # train / val / test paraphrase pools
EXPECTED_COLS = {"input_ids", "token_type_ids", "attention_mask", "special_tokens_mask", "uuid"}


# --- Pure, testable logic ---------------------------------------------------
def load_paraphrases_3way(corpus_dir, fractions=FRACTIONS, seed=SEED, dedup=True, verbose=True):
    """Split each LOINC's paraphrases three ways (train/val/test). Uses the seed, dedup and shuffle
    logic of paraphrase_utils.load_paraphrases_train_val_split, extended to three disjoint pools. One
    random.Random(seed) is advanced across LOINCs in filename order. Duplicates within a LOINC are
    removed before the split. Val and test each get max(1, int(len*frac)), and train gets the rest.
    Asserts that the pools are pairwise disjoint per LOINC and that val and test are not empty.
    Returns (train, val, test, stats)."""
    f_val, f_test = fractions[1], fractions[2]
    rng = random.Random(seed)
    train, val, test = {}, {}, {}
    n_dedup_total = n_loincs_with_dups = 0
    min_test = min_train = 10 ** 9
    for fp in sorted(glob.glob(os.path.join(str(corpus_dir), "*.txt"))):
        loinc = Path(fp).stem
        with open(fp, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if dedup:
            seen, uniq = set(), []
            for ln in lines:
                if ln not in seen:
                    seen.add(ln)
                    uniq.append(ln)
            if len(uniq) < len(lines):
                n_dedup_total += len(lines) - len(uniq)
                n_loincs_with_dups += 1
            lines = uniq
        if not lines:
            continue
        rng.shuffle(lines)
        n_val = max(1, int(len(lines) * f_val))
        n_test = max(1, int(len(lines) * f_test))
        val[loinc] = lines[:n_val]
        test[loinc] = lines[n_val:n_val + n_test]
        train[loinc] = lines[n_val + n_test:]
        s_tr, s_v, s_te = set(train[loinc]), set(val[loinc]), set(test[loinc])
        assert not (s_tr & s_v) and not (s_tr & s_te) and not (s_v & s_te), f"pool overlap {loinc}"
        assert len(val[loinc]) >= 1 and len(test[loinc]) >= 1, f"degenerate val/test pool {loinc}"
        min_test = min(min_test, len(test[loinc]))
        min_train = min(min_train, len(train[loinc]))
    # Content hash of the realised split. The split is recomputed from `clean/` on every run and never
    # saved. One RNG stream runs across LOINCs in filename order, so adding or removing a single .txt
    # reshuffles every later LOINC, and train paraphrases can then become test paraphrases. The hash
    # goes in the manifest to catch this. If it differs between two dataset builds, their pools are
    # not comparable, and the datasets of every model family must be rebuilt together.
    _h = hashlib.sha256()
    for loinc in sorted(train):
        for tag, pool in (("train", train), ("val", val), ("test", test)):
            _h.update(f"{loinc}|{tag}|".encode())
            for s in sorted(pool.get(loinc, [])):
                _h.update(s.encode("utf-8", "replace"))
                _h.update(b"\x00")
    split_hash = _h.hexdigest()

    stats = {
        "n_loincs": len(train), "n_train": sum(map(len, train.values())),
        "n_val": sum(map(len, val.values())), "n_test": sum(map(len, test.values())),
        "n_dedup_removed": n_dedup_total, "n_loincs_with_dups": n_loincs_with_dups,
        "min_test_pool": min_test, "min_train_pool": min_train, "fractions": list(fractions),
        "split_sha256": split_hash, "seed": seed,
    }
    if verbose:
        print(f"  3-way paraphrase split: {stats['n_loincs']} LOINC | "
              f"train {stats['n_train']} / val {stats['n_val']} / test {stats['n_test']} | "
              f"min pools train={min_train} test={min_test}")
        print(f"  split_sha256={split_hash[:16]}…  (must match across the 3 model families. "
              "If it changes, the pools were reshuffled and all datasets must be rebuilt together.)")
    return train, val, test, stats


def read_uuid_list(path):
    """Read a split UUID file and strip the .ndjson suffix (split files store 'uuid.ndjson')."""
    with open(path) as f:
        return sorted({ln.strip().replace(".ndjson", "") for ln in f if ln.strip()})


def _load_seen_codes_mimic_to_loinc(seen_codes_path):
    """{mimic_code -> loinc} for paraphrase patching, one entry per seen_codes row."""
    out = {}
    with open(seen_codes_path) as f:
        for row in csv.DictReader(f):
            if row["mimic_code"] and row["loinc_code"]:
                out[row["mimic_code"]] = row["loinc_code"]
    return out


# --- Tokenizer construction (with_desc toggles 'code:display', with_refrange 'referenceRange') ------
def _build_fhir_tokenizer(with_desc: bool, with_refrange: bool = True):
    import ndjson
    from transformers import BertTokenizerFast
    import fhirclient_patch  # noqa: F401  (patches fhirclient 4.1.0 for MIMIC)
    from fhir_tokenizer import NdjsonTokenizer
    from fhirclient.models.medication import Medication
    from fhirclient.models.location import Location

    text_tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_JSON))
    text_tokenizer.bos_token, text_tokenizer.bos_token_id = "[BOS]", 104
    text_tokenizer.eos_token, text_tokenizer.eos_token_id = "[EOS]", 105

    with open(CODES_TO_TOKENS_PATH, "rb") as f:
        codes_to_tokens = pickle.load(f)
    max_token = max(itertools.chain.from_iterable(v.values() for v in codes_to_tokens.values()))
    codes_to_tokens["value_percentile"] = {f"p_{i}": i + max_token + 1 for i in range(101)}
    codes_to_tokens["time_log_seconds"] = {
        f"t_{sign}{ord_}": i for i, (sign, ord_) in
        enumerate(itertools.product(["-", "+"], range(25)), start=max_token + 102)
    }
    with open(LABEVENT_BINS_PATH, "rb") as f:
        numeric_bins = pickle.load(f)
    with open(MED_DOSE_BINS_PATH, "rb") as f:
        numeric_bins.update(pickle.load(f))
    with open(MED_DISP_BINS_PATH, "rb") as f:
        numeric_bins.update(pickle.load(f))

    obs_keys = ["code", "code:display", "effectiveDateTime", "valueString", "specimen:reference",
                "collection:collectedDateTime", "hasMember:reference", "valueCodeableConcept",
                "valueQuantity", "referenceRange", "interpretation", "note:text"]
    if not with_desc:
        obs_keys = [k for k in obs_keys if k != "code:display"]   # nodesc: drop description tokens
    if not with_refrange:
        obs_keys = [k for k in obs_keys if k != "referenceRange"]  # norr: drop reference-range tokens
    relevant_keys = {
        "Patient": ["gender", "maritalStatus", "communication:language", "extension",
                    "birthDate", "deceasedDateTime"],
        "Encounter": ["class_fhir", "type", "serviceType", "priority",
                      "hospitalization:admitSource", "hospitalization:dischargeDisposition",
                      "period:start", "period:end"],
        "Condition": ["code", "encounter:reference:period:end"],
        "Procedure": ["code", "encounter:reference:location:name", "performedDateTime"],
        "Observation": obs_keys,
        "MedicationRequest": ["medicationCodeableConcept", "dosageInstruction", "authoredOn", "status"],
        "MedicationDispense": ["medicationCodeableConcept", "quantity", "dosageInstruction", "effectiveDateTime"],
        "MedicationAdministration": ["medicationCodeableConcept", "dosage:text", "dosage:method",
                                     "dosage:dose", "effectiveDateTime"],
        "Dosage": ["text", "route", "doseQuantity", "timing:code", "maxDosePerPeriod"],
        "MedicationAdministrationDosage": ["text", "route", "doseQuantity", "timing:code", "maxDosePerPeriod"],
    }
    resource_types = ["Patient", "Encounter", "Condition", "Procedure", "Observation",
                      "MedicationRequest", "MedicationDispense", "MedicationAdministration"]
    token_type_id_mapping = {k: i for i, k in enumerate(resource_types, start=1)}

    static_refs = {}
    for fn, model in [("MimicMedication.ndjson", Medication), ("MimicMedicationMix.ndjson", Medication),
                      ("MimicLocation.ndjson", Location)]:
        with open(FHIR_CANONICAL_DIR / fn) as f:
            for r in ndjson.load(f):
                static_refs[r["id"]] = model(r)

    return NdjsonTokenizer(
        code_mapping=codes_to_tokens, numeric_bins=numeric_bins, relevant_keys=relevant_keys,
        static_references=static_refs, text_tokenizer=text_tokenizer,
        token_type_id_mapping=token_type_id_mapping, max_length=MAX_LEN,
    )


def _lab_token_range(tokenizer):
    from paraphrase_utils import MIMIC_LAB_SYSTEM
    lab_ids = list(tokenizer.code_mapping[MIMIC_LAB_SYSTEM].values())
    return min(lab_ids), max(lab_ids)


def _new_stats(n_input):
    return {"n_input": n_input, "n_missing_file": 0, "n_state_t0_error": 0, "n_other_error": 0,
            "n_no_lab_tokens": 0, "n_ok": 0, "total_obs_patched": 0, "total_lab_tokens": 0,
            # v3 only. Stays 0 for every other condition. Declared here because _merge_stats
            # sums over the keys of _new_stats. A key that a parallel worker creates on first
            # use would be dropped when the shards are merged.
            "total_fields_perturbed": 0,
            "total_units_randomised": 0}


def _merge_stats(stats_list, n_input):
    out = _new_stats(n_input)
    for s in stats_list:
        for k in out:
            if k != "n_input":
                out[k] += s.get(k, 0)
    return out


#: loaded once per process (the parallel path runs worker processes), so each process reads the
#: tables from disk only once.
_PERT_SPEC = None
_RAND_TABLE = None


def _apply_perturbation(uuid, resources, mimic_to_loinc, cfg, stats):
    """v3 (`xsite`): simulate a second site on this patient's raw resources, before tokenising.

    For FHIR-MLM, v3 means the perturbation and the unit randomisation. The v2 rendering, values
    written as 4 s.f. scientific in standardised units, cannot give a distinct arm here.

    In this code the transform is never reached. It lives in the renderer, and this path does not
    render. `NdjsonTokenizer` reads `Quantity.value` and `Quantity.code` straight off the resource.
    Every lab valueQuantity and reference bound carries `system = .../mimic-units`, so the unit is a
    code token and never meets the standardiser.

    Applied coherently, by rescaling the values and recomputing the bin tables from the rescaled
    values, the transform is a no-op, because a percentile does not change under a monotone rescale.
    On the bins of item 51222, every value maps to the same `p_k` before and after a x1000 rescale.
    Applied naively, by rescaling the values and keeping the shipped `labevent_bins.pkl`, it corrupts
    the input. Every haemoglobin from 8 to 16.5 g/dL collapses from p_3..p_97 onto p_97.

    Perturbation is different. It is per-patient noise, which is not a monotone rescale. It is binned
    against the same fixed tables the model was trained on, which is the deployment situation being
    simulated. It moves the raw float before binning, so it changes the emitted percentile token for
    many values and bounds (analysis/a32_bin_change_final.py measures how many).

    `cfg` is a plain dict so that it survives pickling to the joblib workers, which a callable might
    not.
    """
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
    n = pert.perturb_patient(resources, uuid, _PERT_SPEC,
                             lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
                             severity=cfg.get("severity", 1.0),
                             perturb_values=cfg.get("perturb_values", True),
                             perturb_ranges=cfg.get("perturb_ranges", True),
                             salt=cfg.get("salt", pert.DEFAULT_SALT))
    stats["total_fields_perturbed"] += n


def _process_one_patient(uuid, fhir_tokenizer, paraphrase_pool, mimic_to_loinc, lab_min, lab_max, patch, stats,
                         paraphrase_salt=None, perturb_cfg=None):
    """Run one patient through perturbation (v3), paraphrase patching, sort, tokenization and the
    skip rules. The sequential and the parallel path both call this. Returns a row dict or None and
    updates `stats`. Logs exception types only, never patient identifiers."""
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    from paraphrase_utils import patient_rng, assign_paraphrases_per_patient, apply_paraphrases_to_resources
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
    # runs before paraphrase patching and before the sort, on the resource list as read from disk.
    # perturbation.perturb_patient needs the whole patient, because it recomputes derived red-cell
    # indices from their parents and renormalises the differential across its parts.
    if perturb_cfg:
        _apply_perturbation(uuid, resources, mimic_to_loinc, perturb_cfg, stats)
    if patch:
        rng = patient_rng(uuid, salt=paraphrase_salt) if paraphrase_salt else patient_rng(uuid)
        mapping = assign_paraphrases_per_patient(resources, paraphrase_pool, mimic_to_loinc, rng)
        stats["total_obs_patched"] += apply_paraphrases_to_resources(resources, mapping, mimic_to_loinc)
    encounters_period = get_encounters(resources)
    resources = sorted(resources, key=lambda r: sort_key(r, encounters_period))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fhir_tokenizer(resources)
    except KeyError as e:
        if str(e).strip("'\"") == "t0":
            stats["n_state_t0_error"] += 1
        else:
            stats["n_other_error"] += 1
            print(f"  tokenizer KeyError: {type(e).__name__}")
        return None
    except Exception as e:
        stats["n_other_error"] += 1
        print(f"  tokenizer error: {type(e).__name__}")
        return None
    n_lab = sum(1 for t, a in zip(out["input_ids"], out["attention_mask"])
                if a == 1 and lab_min <= t <= lab_max)
    if n_lab == 0:
        stats["n_no_lab_tokens"] += 1
        return None
    stats["n_ok"] += 1
    stats["total_lab_tokens"] += n_lab
    return {"input_ids": out["input_ids"], "token_type_ids": out["token_type_ids"],
            "attention_mask": out["attention_mask"], "special_tokens_mask": out["special_tokens_mask"],
            "uuid": uuid}


def _process_chunk(uuids_chunk, paraphrase_pool, mimic_to_loinc, with_desc, patch, shard_path,
                   paraphrase_salt=None, with_refrange=True, perturb_cfg=None):
    """Parallel worker (module level, so it can be pickled). It builds a tokenizer inside the worker,
    which avoids pickling the fhirclient static references. It then runs _process_one_patient over a
    contiguous chunk of patients, writes an Arrow shard and returns (stats, shard_path or None)."""
    import datasets
    tok = _build_fhir_tokenizer(with_desc, with_refrange)
    lab_min, lab_max = _lab_token_range(tok)
    stats = _new_stats(len(uuids_chunk))
    rows = []
    for uuid in uuids_chunk:
        row = _process_one_patient(uuid, tok, paraphrase_pool, mimic_to_loinc, lab_min, lab_max, patch, stats,
                                   paraphrase_salt=paraphrase_salt, perturb_cfg=perturb_cfg)
        if row is not None:
            rows.append(row)
    if rows:
        datasets.Dataset.from_list(rows).save_to_disk(shard_path)
    return stats, (shard_path if rows else None)


def _build_split_dataset(uuids, paraphrase_pool, mimic_to_loinc, fhir_tokenizer, with_desc,
                         lab_min, lab_max, desc, patch, jobs, paraphrase_salt=None,
                         with_refrange=True, perturb_cfg=None):
    """jobs == 1: sequential `from_generator`. jobs > 1: split the patients into `jobs` contiguous
    chunks, tokenize them in parallel processes and concatenate the shards in chunk order, which gives
    the same dataset (the same rows in the same sorted-uuid order) and merged stats. Both paths call
    `_process_one_patient`. Returns (ds, stats, shard_root or None)."""
    import datasets
    if not uuids:                                       # unreachable at scale; guards array_split(.,0)
        return datasets.Dataset.from_dict({c: [] for c in EXPECTED_COLS}), _new_stats(0), None
    if jobs == 1:
        from tqdm import tqdm
        stats = _new_stats(len(uuids))

        def _gen(uuid_list):
            for uuid in tqdm(uuid_list, desc=desc):
                row = _process_one_patient(uuid, fhir_tokenizer, paraphrase_pool, mimic_to_loinc,
                                           lab_min, lab_max, patch, stats,
                                           paraphrase_salt=paraphrase_salt, perturb_cfg=perturb_cfg)
                if row is not None:
                    yield row

        ds = datasets.Dataset.from_generator(_gen, gen_kwargs={"uuid_list": tuple(uuids)}, cache_dir=CACHE_DIR)
        return ds, stats, None

    from joblib import Parallel, delayed
    chunks = [list(c) for c in np.array_split(np.array(uuids, dtype=object), min(jobs, len(uuids))) if len(c)]
    shard_root = f"{CACHE_DIR}_shards_{desc}"
    shutil.rmtree(shard_root, ignore_errors=True)
    results = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_process_chunk)(ch, paraphrase_pool, mimic_to_loinc, with_desc, patch, f"{shard_root}/{i}",
                                paraphrase_salt=paraphrase_salt, with_refrange=with_refrange,
                                perturb_cfg=perturb_cfg)
        for i, ch in enumerate(chunks))
    stats = _merge_stats([s for s, _ in results], len(uuids))
    shard_paths = [p for _, p in results if p]                       # in chunk order
    assert shard_paths, f"{desc}: every chunk is empty, so there is nothing to concatenate"
    ds = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in shard_paths])
    return ds, stats, shard_root


def _print_split_stats(name, stats, ds, lab_min, lab_max, sample_n=500):
    print(f"\n  {name} stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    n = len(ds)
    if n == 0:
        print("    (empty dataset)")
        return
    sample = min(sample_n, n)
    nreal, nlab = [], []
    for i in range(sample):
        row = ds[i]
        nreal.append(sum(row["attention_mask"]))
        nlab.append(sum(1 for t, a in zip(row["input_ids"], row["attention_mask"])
                        if a == 1 and lab_min <= t <= lab_max))
    print(f"    --- sampled {sample}/{n} rows ---")
    print(f"    real tokens/row: mean={np.mean(nreal):.0f} p50={np.percentile(nreal,50):.0f} "
          f"p90={np.percentile(nreal,90):.0f} max={max(nreal)}")
    print(f"    lab  tokens/row: mean={np.mean(nlab):.1f} p50={np.percentile(nlab,50):.0f} "
          f"p90={np.percentile(nlab,90):.0f} max={max(nlab)}")


# --- main -------------------------------------------------------------------
CACHE_DIR = ""  # set in main once OUTPUT_PATH is known


fn_tag = "09_mlm_dataset"


def main():
    global CACHE_DIR
    import datasets
    from paraphrase_utils import check_paraphrase_coverage

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=MLM_CONDITIONS, required=True,
                    help="see pipeline/conditions.py; *_rr1sf / *_rrmag are not "
                         "supported for FHIR-MLM (see the module docstring)")
    ap.add_argument("--smoke", type=int, default=0, help="process only first N patients/split, write to __smoke")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for tokenization (1=sequential, audited path)")
    # v3 (xsite) settings, ignored for every other condition. Each changes numbers, so each is
    # recorded in the manifest.
    ap.add_argument("--replicate", type=int, default=None,
                    help="v3 only: build perturbation replicate N, an independent site draw "
                         "with its own derived salt and its own directory `..._r{N}`.")
    ap.add_argument("--splits", default="train,val,test",
                    help="which splits to build. A v3 replicate is a perturbed copy of the test "
                         "split, so build only 'test': its train and val are not read "
                         "(100,000 train patients against 10,000 test).")
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
    with_desc = has_description(args.condition)
    with_refrange = includes_reference_range(args.condition)
    # v3 == `xsite`. For this path that means the perturbation and the unit randomisation, with no
    # rendering. _apply_perturbation explains why the rendering half of v2 cannot apply to a
    # percentile-binned encoder.
    perturb_cfg = None
    if representation(args.condition) == "v3":
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
        print(f"  replicate: {args.replicate}" if args.replicate is not None
              else "  replicate: none (primary arm)")
        print(f"  perturbation: severity {args.severity:g}, "
              f"values {'on' if perturb_cfg['perturb_values'] else 'off'}, "
              f"ranges {'on' if perturb_cfg['perturb_ranges'] else 'off'}, "
              f"salt {perturb_cfg['salt']!r}, spec {_pert.SPEC_PATH.name}")

    random.seed(SEED)
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
    out_path = (f"{OUTPUT_BASE}/mlm__{args.condition}"
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

    print(f"=== 09 MLM dataset | condition={args.condition} "
          f"(with_desc={with_desc}, with_refrange={with_refrange}) ===")
    mimic_to_loinc = _load_seen_codes_mimic_to_loinc(SEEN_CODES_PATH)
    print(f"  mimic_to_loinc: {len(mimic_to_loinc)} entries")

    print("\n=== Paraphrase split (70/15/15) ===")
    train_para, val_para, test_para, para_stats = load_paraphrases_3way(PARAPHRASE_DIR)
    if with_desc:  # coverage only matters when descriptions are injected
        for lbl, pool in [("train", train_para), ("val", val_para), ("test", test_para)]:
            check_paraphrase_coverage(pool, mimic_to_loinc, label=lbl, raise_on_missing=True)

    print("\n=== Split UUIDs (read from the split files) ===")
    splits = {s: read_uuid_list(SPLITS_DIR / f"{s}_patients.txt") for s in ("train", "val", "test")}
    for s, u in splits.items():
        print(f"  {s}: {len(u)} patients")
    a, b, c = (set(splits["train"]), set(splits["val"]), set(splits["test"]))
    assert not (a & b) and not (a & c) and not (b & c), "FAIL: split UUID overlap"

    print("\n=== Tokenizer ===")
    fhir_tokenizer = _build_fhir_tokenizer(with_desc, with_refrange)
    lab_min, lab_max = _lab_token_range(fhir_tokenizer)
    print(f"  MAX_LEN={fhir_tokenizer.max_length}  lab token range=[{lab_min},{lab_max}] "
          f"({lab_max - lab_min + 1} tokens)  obs_has_code:display={with_desc}  "
          f"obs_has_referenceRange={with_refrange}")

    pools = {"train": train_para if with_desc else None,
             "val": val_para if with_desc else None,
             "test": test_para if with_desc else None}
    built, all_stats, shard_roots = {}, {}, []
    # A replicate is a perturbed copy of the test split only (--splits test), and each replicate is
    # a different site draw. Its train and val splits would never be read, and the train split alone
    # has ten times as many patients as the test split.
    want_splits = tuple(x for x in ("val", "test", "train") if x in args.splits.split(","))
    if not want_splits:
        sys.exit(f"--splits {args.splits!r} selected nothing")
    for s in want_splits:                    # small to large, for earlier feedback
        uu = splits[s][:args.smoke] if args.smoke else splits[s]
        print(f"\n=== Building {s} ({len(uu)} patients, jobs={args.jobs}) ===")
        ds, st, sr = _build_split_dataset(uu, pools[s], mimic_to_loinc, fhir_tokenizer, with_desc,
                                          lab_min, lab_max, desc=s, patch=with_desc, jobs=args.jobs,
                                          with_refrange=with_refrange, perturb_cfg=perturb_cfg)
        built[s], all_stats[s] = ds, st
        if sr:
            shard_roots.append(sr)

    print("\n=== Schema check ===")
    my_cols = set(built[want_splits[-1]].column_names)
    assert my_cols == EXPECTED_COLS, f"schema mismatch: mine={my_cols} expected={EXPECTED_COLS}"
    print(f"  columns match the spec: {sorted(my_cols)}")

    print("\n=== Saving ===")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    datasets.DatasetDict({k: built[k] for k in want_splits}).save_to_disk(out_path)
    print(f"  saved DatasetDict -> {out_path}")

    for s in want_splits:
        _print_split_stats(s, all_stats[s], built[s], lab_min, lab_max)

    for sr in shard_roots:                    # cleanup after the stats prints (built[s] maps the shards until here)
        shutil.rmtree(sr, ignore_errors=True)

    manifest = {
        "condition": args.condition, "with_desc": with_desc,
        "representation": representation(args.condition),
        "train_arm": train_arm(args.condition, not args.no_perturb_values,
                               not args.no_perturb_ranges) or None,
        # a v3 dataset is not reproducible without the salt, and two arms that differ only by salt
        # must be told apart later
        "v3_perturbation": perturb_cfg, "replicate": args.replicate,
        "with_refrange": with_refrange, "refrange_mode": refrange_mode(args.condition),
        "output_path": out_path,
        "max_len": MAX_LEN, "seed": SEED, "smoke": args.smoke or None,
        "paraphrase_split": para_stats,
        "splits_built": list(want_splits),
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
