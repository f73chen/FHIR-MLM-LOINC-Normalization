#!/usr/bin/env python
"""Build the single-observation train, val and test datasets on which FHIR-MLM is retrained without
patient history (thesis Chapter 4).

pipeline/24_singleobs_dataset.py builds a single-observation test set for scoring the full-history
model. This script imports 24's helpers (`_encode_single`, `sorted_resources`, `_obs_t0`) and adds
only what retraining needs, so 24's datasets stay reproducible. 23 and 24 follow the same convention
with respect to 09. What it adds:
  1. Train, val and test, each with the per-split paraphrase pool that 09 uses. Train patients draw
     from the train pool, val patients from val and test patients from test. The pools are pairwise
     disjoint per LOINC.
  2. A self-computed cap. 24 reads each patient's observation count from the full-history evaluation
     dump (`--match-dump`). No dump exists for train, so the cap is computed the way the dump was
     produced. The script tokenizes the patient's full record at max_len 8192 and counts its mappable
     lab tokens. This also inherits 09's skip rule. A patient whose full record cannot be tokenized
     (no Encounter, so KeyError('t0')) is skipped here, as in 09. The training population is then
     the same for the single-observation and full-history models.
     The rule is validated against the dump on the test split (`--stage validate-cap`), where the
     self-computed cap must equal the dump's count for every patient. Only then is it trusted on
     train, where nothing can check it.
  3. --trim: rows are cut to their true length instead of padded to max_len.
     `DataCollatorForLabCodeMLM` calls `torch.tensor(v)` on the batch and cannot pad ragged rows, so
     trimming only pays off together with the padding wrapper (pipeline/padcollate.py) and
     `group_by_length`. Measured over 60k observations, fixed 320-token padding costs 8.54x as many
     tokens per step as the full-history run, and trimmed, length-grouped rows cost 0.62x. A
     `length` column is emitted for `group_by_length`.

Writes WORK_DIR/datasets/mlm__singleobs_trim__<condition><arm> as a DatasetDict(train, val, test).
Each split is built separately into a staging folder, and `--stage assemble` puts them together. The
build is restartable, and a failed split cannot corrupt the others. Manifests go to
results/25_singleobs_train_dataset/. Patient-level rows stay in WORK_DIR, and stdout is aggregate
only.

  python pipeline/25_singleobs_train_dataset.py --stage build --split train --condition para_xsite --jobs 12 --trim
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, arm_suffix, has_description,  # noqa: E402
                        includes_reference_range, perturbation_tag, representation)

HERE = Path(__file__).resolve().parent
for _p in (str(locations.THIRD_PARTY), str(locations.COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET_BASE = str(locations.WORK / "datasets")
RESULTS = HERE.parent / "results" / "25_singleobs_train_dataset"
STAGE_BASE = f"{DATASET_BASE}/_singleobs_trim_stage"
FULL_MAX_LEN = 8192          # for the cap pass; must equal 09's MAX_LEN
ONE_MAX_LEN = 320            # zero truncation: p100 over 60k observations was 280


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def modules():
    m9 = _load("m9", str(HERE / "09_mlm_dataset.py"))
    m23 = _load("m23", str(HERE / "23_sliding_dataset.py"))
    m24 = _load("m24", str(HERE / "24_singleobs_dataset.py"))
    return m9, m23, m24


def relma_tag(relma):
    """None for the plain para build, otherwise the `relma_{strategy}_a{seed}` name fragment."""
    return None if relma is None else f"relma_{relma[0]}_a{relma[1]}"


def stage_base_for(condition, relma, arm=""):
    """Isolated staging dir per (condition, variant).

    The condition must be in the path. With one shared staging directory, a nodesc build would
    write where para staged. A later `--stage assemble --condition para` would then read nodesc
    shards and save them as mlm__singleobs_trim__para. That would corrupt the dataset every para
    result depends on. With the suffix, a stale-staging mistake stops with "nothing staged".
    """
    t = relma_tag(relma)
    # the v3 train arm belongs here for the same reason as the condition. The two arms are the same
    # condition and differ only in whether the training data was perturbed. A shared staging dir
    # would let a `clean` build write where `pert` staged, and a later assemble would read the
    # wrong shards.
    base = f"{STAGE_BASE}_{condition}{arm}"
    return base if t is None else f"{base}_{t}"


def relma_pool(strategy):
    """Return (pool, clean lookup, M18). The RELMA name pool is loaded through 18_relma_dataset (M18),
    and the caller takes the salt from `M18.salt_for`.

    RELMA uses the same mechanism as the paraphrases. It is the same assign_paraphrases_per_patient
    with a different pool and a salt, which is why this can reuse the per-patient path of 24/25
    unchanged.
    """
    M18 = _load("m18", str(HERE / "18_relma_dataset.py"))
    pool, clean = M18.load_pool(strategy)
    return pool, clean, M18


def context(M9, M23, split, condition="para", relma=None):
    """Two tokenizers (8192 for the cap pass, 320 for the rows) + this split's pool.

    `relma=(strategy, assignment_seed)` swaps the MedGemma paraphrase pool for the RELMA local-name
    pool and returns the matching salt. RELMA substitutes the description, so it is `para`-only and
    test-only, as 18_relma_dataset states.
    """
    M9.MAX_LEN = FULL_MAX_LEN
    # derived from the slug, so every description condition (para_norr, para_xsite, ...) keeps its
    # description
    _wd, _wr = has_description(condition), includes_reference_range(condition)
    tok_full = M9._build_fhir_tokenizer(_wd, _wr)
    M9.MAX_LEN = ONE_MAX_LEN
    tok_one = M9._build_fhir_tokenizer(_wd, _wr)
    lab_min, lab_max = M9._lab_token_range(tok_one)
    mappable = M23.mappable_lab_tokens()
    pool, salt = None, None
    if relma is not None:
        strategy, aseed = relma
        if not has_description(condition):
            raise ValueError("RELMA substitutes the description; it is para-only (18_relma_dataset)")
        if split != "test":
            raise ValueError(f"RELMA is a test-split substitution (got split={split})")
        pool, _clean, M18 = relma_pool(strategy)
        salt = M18.salt_for(strategy, aseed)
    elif has_description(condition):
        tr, va, te, st = M9.load_paraphrases_3way(M9.PARAPHRASE_DIR, verbose=False)
        pool = {"train": tr, "val": va, "test": te}[split]     # 09's per-split pool rule
    m2l = M9._load_seen_codes_mimic_to_loinc(M9.SEEN_CODES_PATH)
    return tok_full, tok_one, pool, m2l, lab_min, lab_max, mappable, salt


def full_history_cap(tok_full, resources, lab_min, lab_max, mappable):
    """Mappable lab tokens in the patient's full-history 8192 sequence, which is what the
    full-history arm trained and scored on. None when the patient is untokenizable, which matches
    09's skip."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = tok_full(resources)
    except Exception:
        return None
    return sum(1 for t, a in zip(out["input_ids"], out["attention_mask"])
               if a == 1 and lab_min <= t <= lab_max and t in mappable)


def _trim_row(row):
    """Cut every parallel array to the row's true length, keeping the last n elements.

    `NdjsonTokenizer.padding_side == "left"`, so pads sit at the front and the real content at the
    end. The attention_mask is [0...0,1...1] and input_ids ends in 102 = [SEP]. Slicing `[:n]` would
    keep the padding and throw the observation away. The result would be a dataset of padding-only
    rows, and training on it would still run and report a loss. The BertTokenizerFast inside the
    collator reports padding_side "right", but it does not do the padding.
    """
    n = int(sum(row["attention_mask"]))
    for k in ("input_ids", "token_type_ids", "attention_mask", "special_tokens_mask"):
        row[k] = row[k][-n:]
    row["length"] = n
    return row


def rows_for_patient(M24, M9, uuid, tok_full, tok_one, pool, m2l, lab_min, lab_max, mappable,
                     salt,
                     patch, stats, trim, perturb_cfg=None):
    resources = M24.sorted_resources(M9, uuid, pool, m2l, patch, stats, salt=salt,
                                     perturb_cfg=perturb_cfg)
    if resources is None:
        return []
    cap = full_history_cap(tok_full, resources, lab_min, lab_max, mappable)
    if cap is None:
        stats["n_full_untokenizable"] += 1
        return []                                   # 09 skips this patient, and so does this build
    if cap == 0:
        stats["n_no_rows"] += 1
        return []
    rows, offset = [], 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        if offset >= cap:
            # Cap reached. `offset` only ever increases, so nothing later can fall under it. Checking
            # here, before encoding, skips the expensive tokenize for every discarded observation.
            # Val keeps about a third of its observations, so this saves about 3x the work, and more
            # on train. This is a `continue` so that `n_beyond_cap` counts observations. A `break`
            # would count patients.
            stats["n_beyond_cap"] += 1
            continue
        t0 = M24._obs_t0(r)
        if t0 is None:
            stats["n_no_t0"] += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = M24._encode_single(tok_one, r, t0)
        except Exception:
            stats["n_tokenize_error"] += 1
            continue
        n_lab = sum(1 for t, a in zip(out["input_ids"], out["attention_mask"])
                    if a == 1 and lab_min <= t <= lab_max and t in mappable)
        if n_lab == 0:
            stats["n_no_mappable"] += 1
            continue
        row = {"input_ids": out["input_ids"], "token_type_ids": out["token_type_ids"],
               "attention_mask": out["attention_mask"],
               "special_tokens_mask": out["special_tokens_mask"],
               "uuid": uuid, "obs_offset": offset}
        rows.append(_trim_row(row) if trim else {**row, "length": int(sum(out["attention_mask"]))})
        offset += n_lab
        stats["total_lab_tokens"] += n_lab
    if rows:
        stats["n_ok"] += 1
        stats["cap_sum"] += cap
    return rows


def _new_stats():
    return {k: 0 for k in ("n_ok", "n_no_rows", "n_full_untokenizable", "n_no_t0",
                           "n_tokenize_error", "n_no_mappable", "n_beyond_cap",
                           "total_lab_tokens", "total_obs_patched", "cap_sum",
                           # v3 only, and 0 otherwise. Declared here because 09's
                           # _apply_perturbation adds to them with +=.
                           "total_fields_perturbed", "total_units_randomised")}


def _merge(stats_list):
    out = _new_stats()
    for s in stats_list:
        for k, v in s.items():
            if isinstance(v, int):
                out[k] = out.get(k, 0) + v
    return out


def split_uuids(M9, split, limit=None):
    u = M9.read_uuid_list(M9.SPLITS_DIR / f"{split}_patients.txt")
    return u[:limit] if limit else u


# -- stage: validate the self-computed cap against the dump -------------------
def stage_validate_cap(limit, condition="para", perturb_cfg=None, train_seg=""):
    """Check the self-computed cap against the test split's full-history dump. Nothing else stops a
    wrong cap from producing a wrong training set, and nothing can check the cap on train. The test
    split has a ground truth, so the check runs there."""
    import pandas as pd, glob
    M9, M23, M24 = modules()
    tok_full, tok_one, pool, m2l, lab_min, lab_max, mappable, salt = context(M9, M23, "test", condition)
    # The dump has to belong to this condition and this train arm. Nodesc single-obs caps cannot be
    # compared with para full-history counts, because the conditions cover different observations
    # per patient. Nodesc rows are shorter, so more of them fit the window. The arm
    # matters too, since v3 dumps are named `…__clean__…` / `…__pert__…`.
    match_dump = M24.default_match_dump(condition, train_seg)
    parts = sorted(glob.glob(f"{match_dump}/*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no full-history dump to validate against: {match_dump}")
    print(f"  validating self-cap against {match_dump}")
    d = pd.concat([pd.read_parquet(f, columns=["patient_uuid", "obs_ordinal"]) for f in parts])
    dump_counts = d.groupby("patient_uuid").obs_ordinal.count().to_dict()

    uuids = split_uuids(M9, "test", limit)
    st = _new_stats()
    agree = mismatch = absent = 0
    tot_self = tot_dump = 0
    for u in uuids:
        resources = M24.sorted_resources(M9, u, pool, m2l, has_description(condition), st,
                                       perturb_cfg=perturb_cfg)
        cap = None if resources is None else full_history_cap(tok_full, resources, lab_min, lab_max, mappable)
        want = dump_counts.get(u)
        if want is None:
            # patient absent from the dump, so the cap rule must refuse it too (cap None or 0)
            if cap in (None, 0):
                absent += 1
            else:
                mismatch += 1
                print(f"  ✗ patient absent from dump but self-cap = {cap}")
            continue
        if cap == want:
            agree += 1
            tot_self += cap
            tot_dump += want
        else:
            mismatch += 1
            print(f"  ✗ cap mismatch: self {cap} vs dump {want}")
    print(f"\nvalidate-cap over {len(uuids)} test patients:")
    print(f"  in dump and cap agrees : {agree}")
    print(f"  absent from dump, correctly refused : {absent}")
    print(f"  mismatches : {mismatch}")
    print(f"  positions: self {tot_self:,} vs dump {tot_dump:,}")
    if mismatch or not agree:
        raise SystemExit("CAP RULE FAILED: do not build train with it")
    print("  ✓ self-computed cap reproduces the full-history dump exactly")


# -- stage: ordering gate on any split ----------------------------------------
def stage_gate(split, limit, condition="para", relma=None, perturb_cfg=None, arm=""):
    """24's ordering assumption, re-checked on this split's patients and paraphrase pool.

    The full-record mappable-lab-token sequence must be an exact prefix of the concatenated
    single-observation tokenizations. If a panel Observation pulled its `hasMember` children inline in
    the full bundle but not when alone, every obs_offset after it would be wrong.
    """
    M9, M23, M24 = modules()
    tok_full, tok_one, pool, m2l, lab_min, lab_max, mappable, salt = context(M9, M23, split, condition, relma)

    def labtoks(out):
        return [t for t, a in zip(out["input_ids"], out["attention_mask"])
                if a == 1 and lab_min <= t <= lab_max and t in mappable]

    agree = diverge = skipped = 0
    for u in split_uuids(M9, split, limit):
        st = _new_stats()
        res = M24.sorted_resources(M9, u, pool, m2l, has_description(condition), st,
                                       perturb_cfg=perturb_cfg)
        if res is None:
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                full = labtoks(tok_full(res))
            except Exception:
                skipped += 1
                continue
            if not full:
                skipped += 1
                continue
            single = []
            for r in res:
                if r.get("resourceType") != "Observation":
                    continue
                t0 = M24._obs_t0(r)
                if t0 is None:
                    continue
                try:
                    single += labtoks(M24._encode_single(tok_one, r, t0))
                except Exception:
                    pass
        if single[:len(full)] == full:
            agree += 1
        else:
            diverge += 1
            n = next((i for i, (a, b) in enumerate(zip(full, single)) if a != b),
                     min(len(full), len(single)))
            print(f"  ✗ DIVERGE at index {n}: full={len(full)} single={len(single)}")
    print(f"\ngate on {split}: {agree + diverge} patients checked ({skipped} skipped)")
    print(f"  exact prefix match: {agree}/{agree + diverge}")
    if diverge or not agree:
        raise SystemExit("GATE FAILED: obs_offset would be wrong on this split")
    print("  ✓ ordering holds")


# -- stage: build one split ---------------------------------------------------
def _chunk(uuids, split, condition, trim, shard_path, relma=None, perturb_cfg=None):
    import datasets
    M9, M23, M24 = modules()
    tok_full, tok_one, pool, m2l, lab_min, lab_max, mappable, salt = context(M9, M23, split, condition, relma)
    st = _new_stats()

    def _gen(ul):
        for u in ul:
            for row in rows_for_patient(M24, M9, u, tok_full, tok_one, pool, m2l,
                                        lab_min, lab_max, mappable, salt,
                                        has_description(condition), st, trim,
                                        perturb_cfg=perturb_cfg):
                yield row

    ds = datasets.Dataset.from_generator(_gen, gen_kwargs={"ul": tuple(uuids)},
                                         cache_dir=f"{shard_path}__cache")
    if len(ds) == 0:
        return st, None
    ds.save_to_disk(shard_path)
    return st, shard_path


def stage_build(split, condition, limit, jobs, trim, relma=None, perturb_cfg=None, arm=""):
    import datasets
    M9, M23, M24 = modules()
    uuids = split_uuids(M9, split, limit)
    SB = stage_base_for(condition, relma, arm)
    print(f"building single-obs {split} | condition={condition} | {len(uuids)} patients | "
          f"trim={trim} | max_len={ONE_MAX_LEN} | jobs={jobs}"
          + (f" | RELMA {relma_tag(relma)}" if relma else ""))
    t0 = time.time()
    if jobs == 1:
        st, path = _chunk(uuids, split, condition, trim, f"{SB}/_seq/{split}", relma,
                          perturb_cfg)
        ds = datasets.load_from_disk(path)
        stats = st
    else:
        from joblib import Parallel, delayed
        root = f"{SB}/_shards/{split}"
        shutil.rmtree(root, ignore_errors=True)
        chunks = [list(c) for c in np.array_split(np.array(uuids, dtype=object), min(jobs, len(uuids))) if len(c)]
        res = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_chunk)(ch, split, condition, trim, f"{root}/{i}", relma, perturb_cfg)
            for i, ch in enumerate(chunks))
        stats = _merge([s for s, _ in res])
        paths = [p for _, p in res if p]
        assert paths, "every chunk empty"
        ds = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in paths])
    out = Path(f"{SB}/{split}")
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    el = time.time() - t0

    L = np.array(ds.select(range(min(5000, len(ds))))["length"])
    print(f"\n  rows: {len(ds):,}   (cap_sum over contributing patients: {stats['cap_sum']:,})")
    ok = "✓" if len(ds) == stats["cap_sum"] else "✗"
    print(f"  {ok} rows == sum of per-patient caps")
    print(f"  patients ok {stats['n_ok']:,} | full-record untokenizable {stats['n_full_untokenizable']:,} "
          f"| no mappable token {stats['n_no_mappable']:,} | beyond cap {stats['n_beyond_cap']:,}")
    print(f"  length over first 5k rows: p50 {int(np.percentile(L,50))} p95 {int(np.percentile(L,95))} "
          f"max {int(L.max())} mean {L.mean():.1f}")
    print(f"  elapsed {el/60:.1f} min -> staged at {out}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"manifest_{condition}{arm}_{relma_tag(relma) + '_' if relma else ''}{split}.json").write_text(json.dumps({
        "split": split, "condition": condition, "trim": trim, "max_len": ONE_MAX_LEN,
        "n_patients_input": len(uuids), "n_rows": len(ds), "stats": stats,
        "elapsed_sec": round(el, 1)}, indent=2))


def stage_assemble(condition="para", relma=None, arm=""):
    import datasets
    parts = {}
    SB = stage_base_for(condition, relma, arm)
    # RELMA substitutes the description and is a test-split artefact only (18_relma_dataset).
    for split in (("test",) if relma else ("train", "val", "test")):
        p = Path(f"{SB}/{split}")
        if p.exists():
            parts[split] = datasets.load_from_disk(str(p))
            print(f"  {split}: {len(parts[split]):,} rows")
    if not parts:
        raise SystemExit("nothing staged")
    tag = relma_tag(relma)
    out = Path(f"{DATASET_BASE}/mlm__singleobs_trim__{condition}{arm}" if tag is None
               else f"{DATASET_BASE}/mlm__singleobs_trim_{tag}__{condition}{arm}")
    shutil.rmtree(out, ignore_errors=True)
    datasets.DatasetDict(parts).save_to_disk(str(out))
    print(f"\nwrote DatasetDict({list(parts)}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validate-cap", "gate", "build", "assemble"], required=True)
    ap.add_argument("--replicate", type=int, default=None,
                    help="v3 only: build perturbation replicate N, an independent site draw "
                         "with its own salt, staging dir and output dir.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: multiplies every perturbation magnitude")
    ap.add_argument("--no-perturb-values", action="store_true", help="v3 only")
    ap.add_argument("--no-perturb-ranges", action="store_true", help="v3 only")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--perturb-salt", default=None, help="v3 only: overrides the site-draw salt")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--condition", default="para", choices=MLM_CONDITIONS,
                    help="nodesc drops the description, so rows are shorter and more observations "
                         "fit the 8192-token cap: the nodesc full-history test dump holds 1,570,708 "
                         "observations against para's 1,189,162. The two conditions do not "
                         "cover the same observation set.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--trim", action="store_true")
    ap.add_argument("--relma-strategy", choices=["s1", "s2"], default=None,
                    help="build the RELMA-substituted variant instead of the MedGemma-paraphrase one, "
                         "with the same assign_paraphrases_per_patient but a different pool and salt "
                         "(para conditions, test split only)")
    ap.add_argument("--assignment-seed", type=int, default=0,
                    help="RELMA assignment seed (the salt namespace), per 18_relma_dataset.")
    a = ap.parse_args()
    # v3 (`xsite`): perturbation runs inside 24.sorted_resources via 09's _apply_perturbation, so
    # the single-observation training arm sees the same input as the full-history arm.
    perturb_cfg, arm, train_seg = None, "", ""
    if representation(a.condition) == "v3":
        import perturbation as _pert
        # a replicate is a different site and needs a different salt, derived from the base (as in 09/24).
        _salt = a.perturb_salt or _pert.DEFAULT_SALT
        if a.replicate is not None:
            _salt = f"{_salt}-r{a.replicate}"
        perturb_cfg = {"salt": _salt, "severity": a.severity,
                       "perturb_values": not a.no_perturb_values,
                       "perturb_ranges": not a.no_perturb_ranges,
                       "randomise_units": not a.no_randomise_units}
        # The whole path segment (see 24). It goes into the staging dir via stage_base_for and into
        # the assembled dataset via stage_assemble. The staging dir needs it too. A severity-2.0
        # build that staged where the primary arm staged would be read by a later plain
        # `--stage assemble` and saved under the primary name.
        # Build and assemble are separate invocations, so pass the same v3 flags to both. Otherwise
        # assemble looks in a staging dir that was never written and fails with "nothing staged".
        train_seg = arm_suffix(a.condition, perturb_cfg["perturb_values"],
                               perturb_cfg["perturb_ranges"])
        arm = train_seg + perturbation_tag(a.severity, perturb_cfg["perturb_values"],
                                           perturb_cfg["perturb_ranges"], a.replicate,
                                           "25_singleobs_train_dataset")
        print(f"  replicate: {a.replicate if a.replicate is not None else 'none (primary arm)'}")
        print(f"  v3 perturbation: {perturb_cfg}  -> path segment {arm!r}")
    # a --limit build is partial. As in 16, 18 and 23, the limit goes into its staging folder and
    # dataset path. Pass the same --limit to `--stage assemble` to assemble it.
    if a.limit and a.stage in ("build", "assemble"):
        arm += f"__smoke{a.limit}"
        print(f"  [PARTIAL BUILD] --limit set -> path segment {arm!r} (not the canonical path)")
    relma = (a.relma_strategy, a.assignment_seed) if a.relma_strategy else None
    if a.stage == "validate-cap":
        stage_validate_cap(a.limit or 300, a.condition, perturb_cfg, train_seg)
    elif a.stage == "gate":
        stage_gate(a.split or "train", a.limit or 40, a.condition, relma, perturb_cfg, arm)
    elif a.stage == "assemble":
        stage_assemble(a.condition, relma, arm)
    else:
        if not a.split:
            raise SystemExit("--split required for --stage build")
        stage_build(a.split, a.condition, a.limit, a.jobs, a.trim, relma, perturb_cfg, arm)


if __name__ == "__main__":
    main()
