#!/usr/bin/env python
"""Build FHIR-MLM's single-observation test set. Each row holds one lab Observation without the rest
of the patient's record, so a checkpoint can be scored with and without history on the same
observations. 25_singleobs_train_dataset.py reuses the per-patient preamble, the per-observation
encoding and the pairing with the full-history dump from this file to build the single-observation
sets of thesis Chapter 4.

Scoring a full-history checkpoint on this set is a test-time ablation. It measures how much accuracy
survives when history is removed at inference. It does not measure how well a model trained on single
observations does, and 25 builds the data for that model. The ablation is conclusive in one direction
only. A small drop is strong evidence that the model does not rely on history, because removing
context can only hurt. A large drop is ambiguous, since it mixes "history mattered" with "the input
no longer looks like training data".

The per-patient logic of 09 (tokenizer construction, paraphrase assignment, chronological sort) and
the mappable-token rule of 23 are imported from those files. Only the final tokenize step is replaced.

t0. The tokenizer takes `state["t0"]` from the `period.end` of an Encounter and raises KeyError('t0')
without one, which is why a bare Observation cannot normally be tokenized. `process_bundle` accepts
an explicit `t0` that overrides the Encounter lookup, so no Encounter is needed. `__call__` does not
forward it, so `_encode_single` repeats the four steps of `__call__` and the shared tokenizer stays
unpatched. t0 is the observation's effectiveDateTime. Every date in the resource is then a delta
from itself, log1p(0) = 0, which gives a constant `t_-0` token that carries no information. Taking t0
from the patient's real Encounter would encode how long before discharge the test happened. That is
record-level context, and it would bring history back into the history-free arm.

Join key. Each row carries `obs_offset`, the number of mappable lab tokens in all earlier
observations of the patient. 15_eval_runner.py emits obs_ordinal = range(offset, offset + n), so
(patient_uuid, obs_ordinal) matches the full-history dump position for position. The offset counts
mappable tokens, because the collator labels only LOINC-mappable lab tokens.
Counting raw tokens would leave gaps in the ordinal (see 23_sliding_dataset.mappable_lab_tokens). An
observation with no mappable token is skipped and does not advance the offset, as in the sliding
windows. `--stage gate` checks the assumption behind this, which is that the full record's mappable
lab tokens are an exact prefix of the concatenated single-observation tokenizations. If a panel
Observation pulled its `hasMember` children inline in the full bundle but not when alone, the order
would diverge and every later ordinal would be wrong.

Rows are 320 tokens long instead of 8192. Over 8,743 observations the median length was 13 tokens,
the 90th percentile 20 and the maximum 278, so 320 truncates nothing. At 8192 the test split would be
about 350 GB of padding, against about 14 GB at 320.

Reads the inputs of 09. With --match-dump it also reads a full-history evaluation dump in
WORK_DIR/eval, so that only the observations that run scored are built. Writes a test split to
WORK_DIR/datasets/mlm__singleobs__{condition}{arm}/ and a manifest to results/24_singleobs_dataset/.
stdout is aggregate only, with no patient content or identifiers.

  python pipeline/24_singleobs_dataset.py --stage build --condition para_xsite --match-dump --jobs 12
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import itertools
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
RESULTS = HERE.parent / "results" / "24_singleobs_dataset"
CACHE_DIR = str(locations.WORK / "_hf_cache_singleobs")
SINGLEOBS_MAX_LEN = 320


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:                      # 09/23 call sys.exit() under __main__ guards
        pass
    return mod


def modules():
    """09 (the builder's per-patient logic) and 23 (the mappable-token rule and the test-uuid list)."""
    m9 = _load("m9", str(HERE / "09_mlm_dataset.py"))
    m9.MAX_LEN = SINGLEOBS_MAX_LEN          # tokenizer reads this global at construction
    m23 = _load("m23", str(HERE / "23_sliding_dataset.py"))
    return m9, m23


def _encode_single(tok, resource, t0):
    """`NdjsonTokenizer.__call__`, with t0 forwarded to process_bundle.

    __call__ hard-codes `process_bundle(bundle)` with no t0, so a bare Observation raises
    KeyError('t0'). Repeating its four steps here leaves the shared tokenizer (third_party/fhir_mlm)
    unchanged.
    """
    bundle = tok._convert_to_bundle([resource])
    mi = tok.process_bundle(bundle, t0=t0)
    mi = {k: list(itertools.chain.from_iterable(v)) for k, v in mi.items()}
    return tok._process_for_mlm(tok._truncate(mi))


def _obs_t0(resource):
    """The observation's timestamp (see t0 in the module docstring), or None when unusable."""
    d = resource.get("effectiveDateTime")
    if not d:
        return None
    try:
        return datetime.datetime.fromisoformat(str(d).replace("Z", "+00:00"))
    except Exception:
        return None


def sorted_resources(M9, uuid, pool, m2l, patch, stats, salt=None, perturb_cfg=None):
    """09's per-patient preamble, unchanged: read, [perturb], paraphrase-patch, chronological sort.

    `perturb_cfg` is for v3 (`xsite`). It calls `M9._apply_perturbation`, the function in 09 itself,
    so the single-observation arms are perturbed in the same way as the full-history arm, with the
    same `lab_loinc_of` predicate and the same unit randomisation. The perturbation is applied where
    09 applies it, straight after the read and before paraphrase patching. It runs on the whole
    resource list, because recomputing the derived red-cell indices and renormalising the
    differential need the panel."""
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    from paraphrase_utils import patient_rng, assign_paraphrases_per_patient, apply_paraphrases_to_resources
    path = M9.NDJSON_DIR / f"{uuid}.ndjson"
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
    if perturb_cfg:
        M9._apply_perturbation(uuid, resources, m2l, perturb_cfg, stats)
    if patch:
        rng = patient_rng(uuid, salt=salt) if salt else patient_rng(uuid)
        mapping = assign_paraphrases_per_patient(resources, pool, m2l, rng)
        stats["total_obs_patched"] += apply_paraphrases_to_resources(resources, mapping, m2l)
    ep = get_encounters(resources)
    return sorted(resources, key=lambda r: sort_key(r, ep))


def process_patient(M9, uuid, tok, pool, m2l, lab_min, lab_max, mappable, patch, stats,
                    salt=None, cap=None, perturb_cfg=None):
    """One row per lab Observation. Returns a list of rows (possibly empty).

    `cap` = stop after that many mappable lab tokens, i.e. emit the prefix the full-history run
    scored for this patient. None = every observation.
    """
    resources = sorted_resources(M9, uuid, pool, m2l, patch, stats, salt, perturb_cfg)
    if resources is None:
        return []
    rows, offset = [], 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        t0 = _obs_t0(r)
        if t0 is None:
            stats["n_no_t0"] += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = _encode_single(tok, r, t0)
        except Exception as e:
            stats["n_tokenize_error"] += 1
            stats.setdefault("err_types", {})
            stats["err_types"][type(e).__name__] = stats["err_types"].get(type(e).__name__, 0) + 1
            continue
        n_lab = sum(1 for t, a in zip(out["input_ids"], out["attention_mask"])
                    if a == 1 and lab_min <= t <= lab_max and t in mappable)
        if n_lab == 0:
            stats["n_no_mappable"] += 1
            continue                        # contributes no offset, as in the sliding windows
        if cap is not None and offset >= cap:
            stats["n_beyond_cap"] += 1
            continue                        # past what the full-history run reached
        rows.append({"input_ids": out["input_ids"], "token_type_ids": out["token_type_ids"],
                     "attention_mask": out["attention_mask"],
                     "special_tokens_mask": out["special_tokens_mask"],
                     "uuid": uuid, "obs_offset": offset})
        offset += n_lab
        stats["total_lab_tokens"] += n_lab
    if rows:
        stats["n_ok"] += 1
    else:
        stats["n_no_rows"] += 1
    return rows


_EVAL_BASE = str(locations.WORK / "eval")
DEFAULT_MATCH_DUMP = f"{_EVAL_BASE}/mlm__para__seed0__mimictest"


def default_match_dump(condition: str, train_arm_suffix: str = "") -> str:
    """The full-history dump that this (condition, train arm) is paired with.

    The train arm is part of the name. The v3 full-history dumps are
    `mlm__para_xsite__clean__seed0__mimictest` and `…__pert__…`, and no dump exists under the name
    without the arm.

    Pass only the train arm (`__clean` or `__pert`), with no severity or replicate tag. The pairing
    target is the full-history run of the same model. A `__pert_sev2.0` test set is scored by the
    `__pert` model, so it pairs with the `__pert` dump. The full path segment would name a
    full-history dump that was never produced.

    >>> default_match_dump("para").endswith("mlm__para__seed0__mimictest")
    True
    >>> default_match_dump("para_xsite", "__clean").endswith("mlm__para_xsite__clean__seed0__mimictest")
    True
    """
    return f"{_EVAL_BASE}/mlm__{condition}{train_arm_suffix}__seed0__mimictest"


def load_match_counts(dump_dir):
    """{patient_uuid: n positions the full-history run scored}, the pairing target.

    Only observations the full run reached can be compared with it. The dump's ordinals are
    contiguous from 0, which is asserted here, so these are the first `n` in chronological order.
    Building the rest would roughly quadruple the cost for rows that can never be paired. It would
    also widen the population beyond the one that every other number in the thesis uses.
    """
    import pandas as pd, glob
    parts = sorted(glob.glob(f"{dump_dir}/*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet in {dump_dir}")
    d = pd.concat([pd.read_parquet(f, columns=["patient_uuid", "obs_ordinal"]) for f in parts])
    g = d.groupby("patient_uuid").obs_ordinal
    assert (g.min() == 0).all() and (g.max() == g.count() - 1).all(), \
        "dump ordinals are not contiguous from 0, so the prefix rule would be wrong"
    return g.count().to_dict()


def _new_stats():
    return {k: 0 for k in ("n_ok", "n_no_rows", "n_missing_file", "n_other_error", "n_no_t0",
                           "n_tokenize_error", "n_no_mappable", "n_beyond_cap",
                           "total_lab_tokens", "total_obs_patched",
                           # v3 only, and 0 otherwise. Declared here because 09's
                           # _apply_perturbation adds to them with +=.
                           "total_fields_perturbed", "total_units_randomised")}


# -- gate ---------------------------------------------------------------------
def stage_gate(limit, condition="para", perturb_cfg=None):
    """Assert the ordering assumption the join key depends on (see the module docstring).

    The context follows the condition, so the gate checks the same tokenization a build under that
    condition uses (for a norr arm, a context without the reference range)."""
    M9, M23 = modules()
    M9.MAX_LEN = 8192                                    # full-record side needs the real cap
    tok_full, pool, m2l, lab_min, lab_max, mappable = M23.build_context(M9, condition)
    M9.MAX_LEN = SINGLEOBS_MAX_LEN
    tok_one = M9._build_fhir_tokenizer(has_description(condition),
                                       includes_reference_range(condition))

    def labtoks(out):
        return [t for t, a in zip(out["input_ids"], out["attention_mask"])
                if a == 1 and lab_min <= t <= lab_max and t in mappable]

    uuids = M23.test_uuids(limit)
    agree = diverge = skipped = 0
    for uuid in uuids:
        st = _new_stats()
        resources = sorted_resources(M9, uuid, pool, m2l, True, st, None, perturb_cfg)
        if resources is None:
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                full = labtoks(tok_full(resources))       # the same call as in 09
            except Exception:
                skipped += 1                              # no t0, and 09 skips this patient too
                continue
            if not full:
                skipped += 1
                continue
            single = []
            for r in resources:
                if r.get("resourceType") != "Observation":
                    continue
                t0 = _obs_t0(r)
                if t0 is None:
                    continue
                try:
                    single += labtoks(_encode_single(tok_one, r, t0))
                except Exception:
                    pass
        if single[:len(full)] == full:
            agree += 1
        else:
            diverge += 1
            n = next((i for i, (a, b) in enumerate(zip(full, single)) if a != b),
                     min(len(full), len(single)))
            print(f"  ✗ DIVERGE at index {n}: full={len(full)} single={len(single)}")
    print(f"\ngate: checked {agree + diverge} patients ({skipped} skipped: no t0 / no lab tokens)")
    print(f"      full-record tokens are an exact prefix of concatenated single-obs: {agree}/{agree + diverge}")
    if diverge or not agree:
        raise SystemExit("GATE FAILED: obs_ordinal would not align with the full-history dump")
    print("      ✓ obs_ordinal will align with the full-history dump")


# -- build --------------------------------------------------------------------
def _process_chunk(uuids, condition, shard_path, caps=None, perturb_cfg=None):
    """Worker that builds one shard. It re-imports in-process, because loky pickles neither the
    module nor the tokenizer.

    Rows are streamed via from_generator. Accumulated, a chunk of ~225k observations at 320 tokens x
    4 arrays would be tens of GB as Python lists before Arrow sees it.
    """
    import datasets
    M9, M23 = modules()
    tok, pool, m2l, lab_min, lab_max, mappable = M23.build_context(M9, condition)
    st = _new_stats()

    def _gen(uuid_list):
        for u in uuid_list:
            for row in process_patient(M9, u, tok, pool, m2l, lab_min, lab_max, mappable,
                                       has_description(condition), st,
                                       cap=None if caps is None else caps.get(u),
                                       perturb_cfg=perturb_cfg):
                yield row

    ds = datasets.Dataset.from_generator(_gen, gen_kwargs={"uuid_list": tuple(uuids)},
                                         cache_dir=f"{shard_path}__cache")
    if len(ds) == 0:
        return st, None
    ds.save_to_disk(shard_path)
    return st, shard_path


def stage_build(condition, limit, jobs, match_dump=None, perturb_cfg=None, arm=""):
    import datasets
    M9, M23 = modules()
    uuids = M23.test_uuids(None)
    caps = None
    if match_dump:
        caps = load_match_counts(match_dump)
        uuids = [u for u in uuids if u in caps]
        print(f"matched to full-history dump: {len(caps):,} patients, "
              f"{sum(caps.values()):,} scored positions -> expected rows")
    if limit:
        uuids = uuids[:limit]
    print(f"building single-observation test split | condition={condition} | "
          f"{len(uuids)} patients | max_len={SINGLEOBS_MAX_LEN} | jobs={jobs}")
    t0 = time.time()

    if jobs == 1:
        tok, pool, m2l, lab_min, lab_max, mappable = M23.build_context(M9, condition)
        st = _new_stats()
        prog = {"n": 0}

        def _gen(uuid_list):
            for i, u in enumerate(uuid_list, 1):
                for row in process_patient(M9, u, tok, pool, m2l, lab_min, lab_max, mappable,
                                           has_description(condition), st,
                                           cap=None if caps is None else caps.get(u),
                                           perturb_cfg=perturb_cfg):
                    prog["n"] += 1
                    yield row
                if i % 25 == 0 or i == len(uuid_list):
                    el = time.time() - t0
                    print(f"  {i}/{len(uuid_list)} patients | {prog['n']:,} rows | "
                          f"{el:.0f}s | {prog['n']/max(el,1e-9):.0f} rows/s", flush=True)

        ds = datasets.Dataset.from_generator(_gen, gen_kwargs={"uuid_list": tuple(uuids)},
                                             cache_dir=CACHE_DIR)
        stats, shard_root = st, None
    else:
        from joblib import Parallel, delayed
        shard_root = f"{CACHE_DIR}_shards"
        shutil.rmtree(shard_root, ignore_errors=True)
        chunks = [list(c) for c in np.array_split(np.array(uuids, dtype=object), min(jobs, len(uuids))) if len(c)]
        res = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_process_chunk)(ch, condition, f"{shard_root}/{i}", caps, perturb_cfg)
            for i, ch in enumerate(chunks))
        stats = _new_stats()
        for s, _ in res:
            for k, v in s.items():
                if isinstance(v, int):
                    stats[k] = stats.get(k, 0) + v
        paths = [p for _, p in res if p]                  # chunk order preserved by Parallel
        assert paths, "every chunk is empty, so there is nothing to concatenate"
        ds = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in paths])

    out = Path(f"{DATASET_BASE}/mlm__singleobs__{condition}{arm}")
    if out.exists():
        shutil.rmtree(out)
    datasets.DatasetDict({"test": ds}).save_to_disk(str(out))
    if shard_root:
        shutil.rmtree(shard_root, ignore_errors=True)
    el = time.time() - t0

    lens = [int(sum(a)) for a in ds.select(range(min(2000, len(ds))))["attention_mask"]]
    print(f"\nwrote {out}")
    print(f"  rows (observations scored): {len(ds):,}")
    print(f"  patients contributing rows: {stats['n_ok']:,} | no rows: {stats['n_no_rows']:,}")
    print(f"  skipped observations: no t0 {stats['n_no_t0']:,} | "
          f"no mappable lab token {stats['n_no_mappable']:,} | tokenize error {stats['n_tokenize_error']:,}")
    print(f"  mappable lab tokens total: {stats['total_lab_tokens']:,}")
    if caps and not limit:
        want = sum(caps.values())
        ok = "✓" if len(ds) == want else "✗"
        print(f"  {ok} rows vs full-history scored positions: {len(ds):,} vs {want:,} "
              f"(diff {len(ds)-want:+,})")
    print(f"  observations past the full-history reach (skipped): {stats['n_beyond_cap']:,}")
    print(f"  seq len over first 2k rows: p50 {int(np.percentile(lens,50))} "
          f"p99 {int(np.percentile(lens,99))} max {max(lens)} (cap {SINGLEOBS_MAX_LEN})")
    print(f"  elapsed {el/60:.1f} min")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"manifest_{condition}{arm}.json").write_text(json.dumps({
        "condition": condition, "split": "test", "max_len": SINGLEOBS_MAX_LEN,
        "t0_policy": "observation's own effectiveDateTime (constant t_-0; carries no record context)",
        "n_patients_input": len(uuids), "n_rows": len(ds), "stats": stats,
        "output_path": str(out), "elapsed_sec": round(el, 1),
    }, indent=2))
    print(f"  manifest -> {RESULTS / f'manifest_{condition}{arm}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "build"], required=True)
    ap.add_argument("--condition", default="para", choices=MLM_CONDITIONS,
                    help="which condition to build; the set covers the observations its full-history model scored "
                         "(--match-dump)")
    ap.add_argument("--replicate", type=int, default=None,
                    help="v3 only: build perturbation replicate N, an independent site draw "
                         "with its own salt and its own directory. The salt is derived from the "
                         "base salt, so each of the five can be rebuilt from its number alone.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: multiplies every perturbation magnitude")
    ap.add_argument("--no-perturb-values", action="store_true",
                    help="v3 only: leave measured values alone")
    ap.add_argument("--no-perturb-ranges", action="store_true",
                    help="v3 only: leave the reference range alone")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--perturb-salt", default=None, help="v3 only: overrides the site-draw salt")
    ap.add_argument("--limit", type=int, default=None, help="first N test patients")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--match-dump", nargs="?", const="AUTO", default=None,
                    help="restrict to the (patient, ordinal) pairs the full-history run "
                         "scored, so the ablation is paired. Given without a value, it uses the "
                         "seed-0 dump of this condition and train arm.")
    a = ap.parse_args()
    # v3 (`xsite`): perturbation is applied inside 24.sorted_resources by 09's
    # _apply_perturbation, so the single-observation arms see the same input as the full-history arm.
    # The arm segment keeps the two training arms in separate directories.
    perturb_cfg, arm, train_seg = None, "", ""
    if representation(a.condition) == "v3":
        import perturbation as _pert
        # a replicate is a different site, so it needs a different salt. The salt is derived from the
        # base salt, so each replicate is reproducible from its number alone. The
        # rule is the one 09 uses, because 24 has to perturb the same values as 09.
        _salt = a.perturb_salt or _pert.DEFAULT_SALT
        if a.replicate is not None:
            _salt = f"{_salt}-r{a.replicate}"
        perturb_cfg = {"salt": _salt, "severity": a.severity,
                       "perturb_values": not a.no_perturb_values,
                       "perturb_ranges": not a.no_perturb_ranges,
                       "randomise_units": not a.no_randomise_units}
        # `arm` is the whole path segment and holds more than the train arm. arm_suffix only says
        # whether anything was perturbed. Without perturbation_tag, every severity, value-only or
        # range-only set and replicate would share the `__pert` directory, and the last build would
        # overwrite the rest. The naming matches 09/10/11. perturbation_tag also refuses a clean arm
        # with a severity or a replicate, because there is no perturbation for them to scale or
        # redraw.
        train_seg = arm_suffix(a.condition, perturb_cfg["perturb_values"],
                               perturb_cfg["perturb_ranges"])
        arm = train_seg + perturbation_tag(a.severity, perturb_cfg["perturb_values"],
                                           perturb_cfg["perturb_ranges"], a.replicate,
                                           "24_singleobs_dataset")
        print(f"  replicate: {a.replicate if a.replicate is not None else 'none (primary arm)'}")
        print(f"  v3 perturbation: {perturb_cfg}  -> path segment {arm!r}")
    # a --limit build is partial. As in 16, 18 and 23, it stays off the canonical path, which is
    # deleted before a build.
    if a.limit and a.stage == "build":
        arm += f"__smoke{a.limit}"
        print(f"  [PARTIAL BUILD] --limit set -> path segment {arm!r} (not the canonical path)")
    if a.match_dump == "AUTO":
        a.match_dump = default_match_dump(a.condition, train_seg)
        print(f"  --match-dump resolved to {a.match_dump}")
    if a.stage == "gate":
        stage_gate(a.limit or 40, a.condition, perturb_cfg)
    else:
        stage_build(a.condition, a.limit, a.jobs, a.match_dump, perturb_cfg, arm)


if __name__ == "__main__":
    main()
