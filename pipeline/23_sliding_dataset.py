#!/usr/bin/env python
"""Build the evaluation-time sliding-window test sets (thesis Chapters 5 and 6). Each patient's whole
record is split into windows, one dataset row per window. A model trained on the first 8192 tokens can
then be scored on the rest of the record.

The standard builders (09, 10, 11) call the tokenizer once per patient, and it truncates at MAX_LEN
tokens, possibly in the middle of a resource. Here `sliding_window_utils.make_windows` packs the
sorted resource list into non-overlapping windows of whole resources. The Patient and the most recent
Encounter are prepended to each window as a preamble. The per-patient logic of the builders
(paraphrase assignment, chronological sort, tokenizer construction) is imported from them. Only the
final tokenize step is replaced, and the builders are unchanged.

Join key: each row carries `obs_offset`, the number of labelled observations in all earlier windows
of that patient. 15_eval_runner.py emits obs_ordinal = range(offset, offset + n), so
(patient_uuid, obs_ordinal) stays the patient-level chronological index. Without the offset, window 2
would also start at ordinal 0, and its keys would collide with those of window 1.

Window 1 ends at a resource boundary. The standard build truncates at a token boundary, and its
truncated sequences are all 8192 tokens long. Window 1 therefore holds a prefix of the standard run's
observations, which may not be all of them.

Reads the inputs of 09, 10 and 11. --stage count counts the windows without writing. --stage build
writes a test split with a manifest to WORK_DIR/datasets/{model}__sliding__{condition}{arm}/. Prints
aggregate counts only.

  python pipeline/23_sliding_dataset.py --model mlm --condition para_xsite --test-arm pert --stage build
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "data" / "splits"
WINDOW_CAP = 128          # windows per patient at most. It binds for one test patient of 10,000, whose
                          # record needs 159 windows. The resources past the cap leave about 0.1% of
                          # positions unscored, and the build counts them (n_hit_cap, dropped_payload).

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, has_description, includes_reference_range,  # noqa: E402
                        representation, resolve_test_arm)
# This script implements v3. tests/test_conditions_refrange.py
# (test_every_condition_taking_script_handles_or_refuses_v3) recognises that from this import, so
# keep the import at module level.
import perturbation as pert  # noqa: E402



def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mlm_module():
    return _load("m9", HERE / "09_mlm_dataset.py")


def bcm_module():
    return _load("m10", HERE / "10_bcm_dataset.py")


def bcm_sep_cost(formatter, tokenizer) -> int:
    """BPE cost of the resource separator as it tokenizes between two blocks.

    ModernBERT tokenizes "\n\n" as one token in isolation but as two between blocks. Costing the
    separator in isolation would under-count by 1 token per resource boundary, roughly 100 tokens per
    100-resource window, and the windows would regularly overflow MAX_LEN. The cost is measured once
    by tokenizing "x<sep>x" and subtracting the two "x" tokens.
    """
    a = tokenizer("x", add_special_tokens=False)["input_ids"]
    ctx = tokenizer("x" + formatter.sep + "x", add_special_tokens=False)["input_ids"]
    return len(ctx) - 2 * len(a)


def bpe_resource_costs(formatter, tokenizer, sorted_resources, sep_cost) -> list:
    """Per-resource BPE cost for BCM/Llama, aligned 1:1 with `sorted_resources`.

    Resources emitting an empty block (Specimen, hasMember-inlined children, types with no
    relevant_keys) cost 0 and ride along free.
    """
    bundle_table = {r["id"]: r for r in sorted_resources if "id" in r}
    seen_ids = set()
    costs = [0] * len(sorted_resources)
    for i, r in enumerate(sorted_resources):
        rid = r.get("id")
        if rid in seen_ids:
            continue
        block, _ = formatter._format_resource(r, bundle_table=bundle_table, seen_ids=seen_ids,
                                              currently_inlining=set())
        if rid:
            seen_ids.add(rid)
        if not block:
            continue
        costs[i] = len(tokenizer(block, add_special_tokens=False)["input_ids"]) + sep_cost
    return costs


def mlm_resource_costs(fhir_tokenizer, resources):
    """Per-resource NdjsonTokenizer cost, aligned 1:1 with `resources`.

    Resources the tokenizer skips cost 0 and ride along free: Specimen (no relevant_keys entry),
    hasMember-inlined children (the parent Observation absorbs them), unknown types, and anything
    emitting no ids.
    """
    bundle = fhir_tokenizer._convert_to_bundle(resources)
    fhir_tokenizer.state = fhir_tokenizer._create_state(bundle)   # state is shared across calls
    id_to_idx = {r["id"]: i for i, r in enumerate(resources) if r.get("id")}
    costs = [0] * len(resources)
    for entry in bundle.entry:
        res = entry.resource
        rid = res.id
        if rid in fhir_tokenizer.state["seen_keys"]:
            continue                                              # already inlined by a parent
        ids = fhir_tokenizer.process_resource(res)
        if not ids:
            continue
        idx = id_to_idx.get(rid)
        if idx is not None:
            costs[idx] = len(ids)
    return costs


def process_patient_windows(M9, uuid, tok, pool, mimic_to_loinc, lab_min, lab_max, patch, stats,
                            mappable=None, paraphrase_salt=None, cap=WINDOW_CAP, perturb=None):
    """Returns a list of rows (one per window) instead of one truncated row.

    Everything before the tokenize step follows the builder. It uses the same per-patient paraphrase
    assignment and the same chronological sort. The assignment is identical across the windows of a
    patient, which keeps the models paraphrase-matched.
    """
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    from paraphrase_utils import (patient_rng, assign_paraphrases_per_patient,
                                  apply_paraphrases_to_resources)
    from sliding_window_utils import make_windows

    path = M9.NDJSON_DIR / f"{uuid}.ndjson"
    if not path.exists():
        stats["n_missing_file"] = stats.get("n_missing_file", 0) + 1
        return []
    try:
        with open(path) as f:
            resources = ndjson.load(f)
    except Exception:
        stats["n_other_error"] = stats.get("n_other_error", 0) + 1
        return []
    # v3: the second site, applied to the whole patient before windowing. It has to come before
    # `make_windows`, because windows are packed by per-resource token cost, which is read off the
    # rendered resource. Perturbing afterwards would window one version and score another.
    if perturb is not None:
        perturb(resources, uuid, stats)
    if patch:
        rng = patient_rng(uuid, salt=paraphrase_salt) if paraphrase_salt else patient_rng(uuid)
        mapping = assign_paraphrases_per_patient(resources, pool, mimic_to_loinc, rng)
        stats["total_obs_patched"] = stats.get("total_obs_patched", 0) + apply_paraphrases_to_resources(resources, mapping, mimic_to_loinc)
    enc = get_encounters(resources)
    resources = sorted(resources, key=lambda r: sort_key(r, enc))

    # Classify failures as 09 does. Building the tokenizer state raises KeyError('t0') for patients
    # with no usable t0, and these go to n_state_t0_error. fhirclient raises FHIRValidationError on
    # malformed bundles, and these go to n_other_error, as in 09, and are also counted in
    # n_validation_error. The standard build drops both kinds of patient.
    try:
        costs = mlm_resource_costs(tok, resources)
        wr = make_windows(resources, costs, max_len=M9.MAX_LEN, cap=cap)
    except KeyError as e:
        if str(e).strip("'\"") == "t0":
            stats["n_state_t0_error"] += 1
        else:
            stats["n_other_error"] = stats.get("n_other_error", 0) + 1
            print(f"  windowing KeyError: {e}")
        return []
    except Exception as e:
        stats["n_other_error"] = stats.get("n_other_error", 0) + 1
        stats["n_validation_error"] = stats.get("n_validation_error", 0) + (
            1 if type(e).__name__ == "FHIRValidationError" else 0)
        return []
    if wr.hit_cap:
        stats["n_hit_cap"] = stats.get("n_hit_cap", 0) + 1
        stats["dropped_payload"] = stats.get("dropped_payload", 0) + wr.dropped_payload_count

    rows, offset = [], 0
    for wi, widx in enumerate(wr.windows):
        sub = [resources[i] for i in widx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = tok(sub)
        except KeyError as e:
            if str(e).strip("'\"") == "t0":
                stats["n_state_t0_error"] += 1
            else:
                stats["n_other_error"] = stats.get("n_other_error", 0) + 1
            continue
        except Exception:
            stats["n_other_error"] = stats.get("n_other_error", 0) + 1
            continue
        # count only what the collator will label (see mappable_lab_tokens())
        n_lab = sum(1 for t, a in zip(out["input_ids"], out["attention_mask"])
                    if a == 1 and lab_min <= t <= lab_max
                    and (mappable is None or t in mappable))
        if n_lab == 0:
            continue                       # nothing scoreable, and it contributes no offset
        rows.append({"input_ids": out["input_ids"], "token_type_ids": out["token_type_ids"],
                     "attention_mask": out["attention_mask"],
                     "special_tokens_mask": out["special_tokens_mask"],
                     "uuid": uuid, "window_idx": wi, "obs_offset": offset,
                     "n_windows": len(wr.windows)})
        offset += n_lab
        stats["total_lab_tokens"] += n_lab
    if rows:
        stats["n_ok"] = stats.get("n_ok", 0) + 1
    else:
        stats["n_no_lab_tokens"] += 1
    return rows


def mappable_lab_tokens():
    """The same token set that the MLM collator labels, built as in 15_eval_runner.build_collator.

    The collator is constructed with `loinc_mappable_tokens=mappable` and assigns -100 to every other
    lab token, so an unmappable lab token is never scored. Counting raw lab tokens for `obs_offset`
    would overcount by about 2% and leave gaps in obs_ordinal. For example, window 2 would start at
    320 when window 1 scored only 314. The ordinals would then not line up with the chronological
    index of TF-IDF. This is not a collision, but it breaks the join all the same.
    """
    import pickle, pandas as pd
    with open(f"{locations.TOKENIZER}/codes_to_tokens.pkl", "rb") as f:
        lab_map = pickle.load(f)["http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"]
    seen = pd.read_csv(str(locations.UPSTREAM / "seen_codes.csv"))
    seen = seen[seen.observation_type == "lab"]
    return {lab_map[str(r.mimic_code)] + 30522                      # TEXT_VOCAB_SIZE
            for r in seen.itertuples() if str(r.mimic_code) in lab_map and str(r.loinc_code)}


def process_patient_windows_bpe(M, uuid, formatter, tokenizer, code2class, class2loinc,
                                mask_token_id, max_len, pool, mimic_to_loinc, patch, stats,
                                sep_cost, paraphrase_salt=None, cap=WINDOW_CAP, perturb=None):
    """BCM / Llama windowed processing. Returns a list of rows.

    It differs from the MLM path in two ways:
      * These builders format resources into a text prompt, then tokenize and align. Each window is
        therefore formatted separately.
      * They emit explicit `labels` (class id at MASK, -100 elsewhere). The scored count is then
        (labels != -100).sum(), and this path does not need the mappable-token reconstruction that
        the MLM path relies on.
    `_process_one_patient` does not sort, because format_resources sorts internally. Windowing has
    to follow chronological order, so this function sorts the resources first.
    """
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    from paraphrase_utils import (patient_rng, assign_paraphrases_per_patient,
                                  apply_paraphrases_to_resources)
    from fhir_data_utils import tokenize_and_align
    from sliding_window_utils import make_windows

    path = M.NDJSON_DIR / f"{uuid}.ndjson"
    if not path.exists():
        stats["n_missing_file"] = stats.get("n_missing_file", 0) + 1
        return []
    try:
        with open(path) as f:
            resources = ndjson.load(f)
    except Exception:
        stats["n_other_error"] = stats.get("n_other_error", 0) + 1
        return []
    # The same hook as in the MLM path above, for the same reason. The two bodies are near-identical,
    # so a change to one must be made to the other.
    if perturb is not None:
        perturb(resources, uuid, stats)
    if patch:
        rng = patient_rng(uuid, salt=paraphrase_salt) if paraphrase_salt else patient_rng(uuid)
        mapping = assign_paraphrases_per_patient(resources, pool, mimic_to_loinc, rng)
        stats["total_obs_patched"] = stats.get("total_obs_patched", 0) + apply_paraphrases_to_resources(resources, mapping, mimic_to_loinc)
    try:
        enc = get_encounters(resources)
        resources = sorted(resources, key=lambda r: sort_key(r, enc))
        costs = bpe_resource_costs(formatter, tokenizer, resources, sep_cost)
        wr = make_windows(resources, costs, max_len=max_len, cap=cap)
    except Exception as e:
        stats["n_other_error"] = stats.get("n_other_error", 0) + 1
        print(f"  windowing error: {type(e).__name__}")
        return []
    if wr.hit_cap:
        stats["n_hit_cap"] = stats.get("n_hit_cap", 0) + 1
        stats["dropped_payload"] = stats.get("dropped_payload", 0) + wr.dropped_payload_count

    rows, offset = [], 0
    for wi, widx in enumerate(wr.windows):
        sub = [resources[i] for i in widx]
        try:
            prompt, codes = formatter.format_resources(sub)
        except Exception:
            stats["n_format_error"] = stats.get("n_format_error", 0) + 1
            continue
        if not codes:
            continue
        try:
            ex = tokenize_and_align(prompt, codes, tokenizer, code2class, class2loinc,
                                    mask_token_id=mask_token_id, max_len=max_len)
        except Exception:
            stats["n_other_error"] = stats.get("n_other_error", 0) + 1
            continue
        if ex is None:
            continue
        n_lab = sum(1 for l in ex["labels"] if l != -100)     # the count the eval will score
        if n_lab == 0:
            continue
        stats["n_overflow"] = stats.get("n_overflow", 0) + (1 if len(ex["input_ids"]) >= max_len else 0)
        ex = dict(ex)
        ex.update({"uuid": uuid, "window_idx": wi, "obs_offset": offset,
                   "n_windows": len(wr.windows)})
        rows.append(ex)
        offset += n_lab
        stats["total_lab_tokens"] = stats.get("total_lab_tokens", 0) + n_lab
    if rows:
        stats["n_ok"] = stats.get("n_ok", 0) + 1
    else:
        stats["n_no_labels"] = stats.get("n_no_labels", 0) + 1
    return rows


def build_context(M9, condition="para"):
    """Tokenizer, paraphrase pool and label range, built the same way as in the builder.

    The two axes come from the slug through conditions.py. They are not read from
    `condition == "para"`, because that test is False for `para_norr`, which also has a description."""
    _wd, _wr = has_description(condition), includes_reference_range(condition)
    tok = M9._build_fhir_tokenizer(_wd, _wr)
    lab_min, lab_max = M9._lab_token_range(tok)
    pool = None
    if has_description(condition):
        _, _, te, st = M9.load_paraphrases_3way(
            HERE.parent / "data" / "paraphrases" / "clean", verbose=False)
        pool = te                                  # test-split paraphrases, as the test build uses
        print(f"  paraphrase split sha256={st.get('split_sha256', '')[:16]}...")
    m2l = M9._load_seen_codes_mimic_to_loinc(str(locations.UPSTREAM / "seen_codes.csv"))
    mappable = mappable_lab_tokens()
    print(f"  mappable lab tokens: {len(mappable):,}")
    return tok, pool, m2l, lab_min, lab_max, mappable


def test_uuids(limit=None):
    """Split files store 'uuid.ndjson'. The suffix is stripped and the IDs are sorted as in 09's
    read_uuid_list, so the patient order matches the standard build."""
    with open(SPLITS / "test_patients.txt") as f:
        ids = sorted({ln.strip().replace(".ndjson", "") for ln in f if ln.strip()})
    return ids[:limit] if limit else ids


def build_model_context(model, condition="para"):
    """Returns (module, kind, kwargs) for the requested model. `kind` selects the windowing path."""
    from fhir_data_utils import build_lab_code_mappings
    if model == "mlm":
        M = mlm_module()
        tok, pool, m2l, lo, hi, mappable = build_context(M, condition)
        return M, "mlm", dict(tok=tok, pool=pool, m2l=m2l, lo=lo, hi=hi, mappable=mappable)
    M = bcm_module() if model == "bcm" else _load("m11", HERE / "11_llama_dataset.py")
    _wd, _wr = has_description(condition), includes_reference_range(condition)
    # pass the representation. `rep` in the builders defaults to "v1", which would render raw units
    # and no 4 s.f. scientific values, and a v3 checkpoint would be scored on them. The MLM path needs
    # nothing like this, because its numbers are percentile tokens and the v2/v3 text rendering does
    # not apply to it.
    _rep = representation(condition)
    tokzr, fmt = (M._build_bcm_tokenizer_formatter(_wd, _wr, _rep) if model == "bcm"
                  else M._build_llama_tokenizer_formatter(_wd, _wr, _rep))
    c2c, c2l = build_lab_code_mappings(M.CODES_TO_TOKENS_PATH, M.SEEN_CODES_PATH)
    # Llama has no mask token. The id is a module constant and must not be read from the tokenizer.
    mask_id = tokzr.mask_token_id if model == "bcm" else M.LLAMA_MASK_TOKEN_ID
    pool = None
    if has_description(condition):
        M9 = mlm_module()
        _, _, pool, _ = M9.load_paraphrases_3way(
            HERE.parent / "data" / "paraphrases" / "clean", verbose=False)
    m2l = (mlm_module())._load_seen_codes_mimic_to_loinc(
        str(locations.UPSTREAM / "seen_codes.csv"))
    sep = bcm_sep_cost(fmt, tokzr)
    print(f"  {model}: in-context separator cost {sep}, mask_token_id {mask_id}")
    return M, "bpe", dict(fmt=fmt, tokzr=tokzr, c2c=c2c, c2l=c2l, mask_id=mask_id,
                          pool=pool, m2l=m2l, sep=sep)


def rows_for(model, M, kind, kw, uuid, patch, stats, perturb=None):
    if kind == "mlm":
        return process_patient_windows(M, uuid, kw["tok"], kw["pool"], kw["m2l"], kw["lo"], kw["hi"],
                                       patch=patch, stats=stats, mappable=kw["mappable"],
                                       perturb=perturb)
    return process_patient_windows_bpe(M, uuid, kw["fmt"], kw["tokzr"], kw["c2c"], kw["c2l"],
                                       kw["mask_id"], M.MAX_LEN, kw["pool"], kw["m2l"], patch,
                                       stats, kw["sep"], perturb=perturb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlm", choices=["mlm", "bcm", "llama"])
    ap.add_argument("--stage", choices=["count", "build"], default="count")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--condition", default="para", choices=MLM_CONDITIONS,
                    help="see pipeline/conditions.py for the slug grammar")
    ap.add_argument("--out", default=None, help="dataset dir; default datasets/{model}__sliding__{cond}{arm}")
    ap.add_argument("--test-arm", default=None,
                    help="v3 (xsite) only, and required there: which copy of the test split to "
                         "window. 23 builds an eval set for an already-trained checkpoint, so this "
                         "is its only arm. "
                         "15_eval_runner reads `{model}__sliding__{cond}{arm}`, so the "
                         "suffix must match what resolve_test_arm returns.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: perturbation severity multiplier")
    ap.add_argument("--perturb-salt", default=None, help="v3 only: override the base salt")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    a = ap.parse_args()

    _rep = representation(a.condition)
    if _rep == "v3" and a.test_arm is None:
        raise SystemExit("23_sliding_dataset: --test-arm is required for a v3 condition "
                         "('clean' or 'pert'). Without it, unperturbed resources would be windowed "
                         "under a perturbed name.")
    if _rep != "v3" and a.test_arm is not None:
        raise SystemExit(f"23_sliding_dataset: --test-arm is v3-only; {a.condition!r} is {_rep}.")

    # The suffix comes from `resolve_test_arm`, which 15 also calls, so the directory written here is
    # the one 15 looks up. The train-arm suffix is "" because this script builds data and no model.
    arm_suffix, _ = resolve_test_arm(a.condition, "", a.test_arm, "23_sliding_dataset") \
        if _rep == "v3" else ("", "")

    perturb = None
    if _rep == "v3":
        _m2l = (mlm_module())._load_seen_codes_mimic_to_loinc(
            str(locations.UPSTREAM / "seen_codes.csv"))
        _spec = pert.load_spec()
        _rand = None if a.no_randomise_units else pert.load_randomisation()
        _salt = a.perturb_salt or pert.DEFAULT_SALT
        _on = a.test_arm == "pert"
        _code_of = lambda r: pert.lab_loinc_of(r, _m2l)  # noqa: E731

        def perturb(resources, uuid, stats):    # noqa: F811
            """Perturb the whole patient in place, before windowing. Unit randomisation is outside
            the perturbation switches, because it is part of the v3 representation and on in every arm."""
            if _rand is not None:
                stats["total_units_randomised"] = stats.get("total_units_randomised", 0) + \
                    pert.randomise_units(resources, uuid, _rand, _code_of, salt=_salt)
            stats["total_fields_perturbed"] = stats.get("total_fields_perturbed", 0) + \
                pert.perturb_patient(resources, uuid, _spec, _code_of, severity=a.severity,
                                     perturb_values=_on, perturb_ranges=_on, salt=_salt)
        print(f"  representation v3 | test arm {a.test_arm!r}: values {'on' if _on else 'off'}, "
              f"ranges {'on' if _on else 'off'}, units randomised "
              f"{'on' if _rand is not None else 'off'}, severity {a.severity:g}, salt {_salt!r}")

    M, kind, kw = build_model_context(a.model, a.condition)
    stats = M._new_stats(0)
    uuids = test_uuids(a.limit)
    patch = has_description(a.condition)
    print(f"  model={a.model} condition={a.condition}{arm_suffix} patients={len(uuids):,}")

    rows, n_rows = [], 0
    for i, u in enumerate(uuids):
        r = rows_for(a.model, M, kind, kw, u, patch, stats, perturb=perturb)
        n_rows += len(r)
        if a.stage == "build":
            rows.extend(r)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(uuids)} patients, {n_rows:,} windows", flush=True)

    print(f"  windows: {n_rows:,}   patients kept: {stats.get('n_ok', 0):,}")
    print(f"  hit cap: {stats.get('n_hit_cap', 0)}   dropped payload: {stats.get('dropped_payload', 0)}")
    print(f"  scored observations: {stats.get('total_lab_tokens', 0):,}")

    if a.stage == "build":
        import datasets, json
        out = Path(a.out) if a.out else (
            locations.WORK / "datasets"
            / f"{a.model}__sliding__{a.condition}{arm_suffix}")
        # A --limit build is partial and gets a separate path. At the canonical path, a runner that
        # takes dataset_dict.json as the done marker would skip the real build, and 15 would score a
        # truncated split as if it were complete. 16 and 18 do the same.
        if a.limit and a.out is None:
            out = out.with_name(out.name + f"__smoke{a.limit}")
            print(f"  [PARTIAL BUILD] --limit set -> '{out.name}' (not the canonical path)")
        ds = datasets.Dataset.from_list(rows)
        datasets.DatasetDict({"test": ds}).save_to_disk(str(out))
        (out / "manifest.json").write_text(json.dumps({
            "model": a.model, "condition": a.condition, "representation": _rep,
            "test_arm": a.test_arm, "arm_suffix": arm_suffix, "window_cap": WINDOW_CAP,
            # True for Llama, whose text puts each Observation's Code line last (CODE_LAST in 11).
            # False for the others.
            "code_last": getattr(M, "CODE_LAST", False),
            "n_patients_in": len(uuids), "n_windows": len(ds), "build_stats": stats,
            "v3_perturbation": None if _rep != "v3" else {
                "salt": a.perturb_salt or pert.DEFAULT_SALT, "severity": a.severity,
                "perturb_values": a.test_arm == "pert", "perturb_ranges": a.test_arm == "pert",
                "randomise_units": not a.no_randomise_units},
        }, indent=2, default=str))
        print(f"  wrote {len(ds):,} rows -> {out}")


if __name__ == "__main__":
    main()
