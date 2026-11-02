"""Build the RELMA-substituted test split for one deep-model family (thesis Chapters 4 and 6).

Rebuilds only the test split of mlm, bcm or llama. Each observation gets a real local lab name from the
RELMA pool (pipeline/17_relma_pool.py) as its description in place of the MedGemma paraphrase. The
script runs the family builders (pipeline/09, 10 and 11) through importlib and reuses their
tokenisation and per-patient logic. Those builders take a `paraphrase_salt`, which is passed from
`_build_split_dataset` through `_process_chunk` to `_process_one_patient`. None keeps the unsalted
draw. The salt re-randomises the assignment of names to patients and is the RELMA assignment seed.

Only description conditions are eligible, because RELMA substitutes the description. A nodesc model
never saw one and cannot be evaluated here.

Assignment sidecar. Names are assigned per (patient, LOINC). A compact sidecar records which RELMA
name each test patient's observations of each LOINC received, and whether that name is clean, meaning
not a training paraphrase (see pipeline/17_relma_pool.py). Accuracy on clean names then comes from
filtering, with no second set of runs. The sidecar is an exact record of the assignment, because it
calls the same `assign_paraphrases_per_patient` with the same pool and the same
`patient_rng(uuid, salt)`. That function iterates `sorted(set(...))`, so it does not depend on
resource order. It also skips LOINCs absent from the pool without consuming an RNG draw. Recomputing
it outside the build therefore gives the same assignment as the build.

Codes without a RELMA name keep their MIMIC display text, because `apply_paraphrases_to_resources`
leaves them untouched. Their observations are dropped at scoring time but still give context to
neighbouring observations.

Writes WORK_DIR/datasets/<model>__relma_<strategy>_a<seed>.../ with the test split, the assignment
sidecar and a manifest, and a copy of the manifest to results/18_relma_dataset/. The sidecar is
patient-level and stays beside the dataset. Stdout is aggregate only.

  python pipeline/18_relma_dataset.py --model bcm --condition para_xsite --test-arm pert --strategy s1 --jobs 16 --write
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

_SCRIPTS_FOR_CONDITIONS = str(Path(__file__).resolve().parent)
if _SCRIPTS_FOR_CONDITIONS not in sys.path:
    sys.path.insert(0, _SCRIPTS_FOR_CONDITIONS)
from conditions import (CONDITIONS, has_description,  # noqa: E402
                        includes_reference_range, representation)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = locations.REPO
SCRIPTS = REPO / "pipeline"
POOL_DIR = REPO / "data" / "relma"
SPLITS_DIR = REPO / "data" / "splits"
NDJSON_DIR = locations.PATIENTS
OUTPUT_BASE = str(locations.WORK / "datasets")
RESULTS_DIR = REPO / "results" / "18_relma_dataset"

FAMILY_SCRIPT = {"mlm": "09_mlm_dataset.py", "bcm": "10_bcm_dataset.py", "llama": "11_llama_dataset.py"}


def load_family(model: str):
    spec = importlib.util.spec_from_file_location(f"fam_{model}", SCRIPTS / FAMILY_SCRIPT[model])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def salt_for(strategy: str, assignment_seed: int) -> str:
    """One namespace per (strategy, seed) so S1/S2 and each assignment seed are independent draws."""
    return f"relma-{strategy}-a{assignment_seed}"


def load_pool(strategy: str):
    """Return (pool {loinc:[names]}, clean lookup {(loinc,name): (clean_same, clean_global)})."""
    marked = json.loads((POOL_DIR / f"pool_{strategy}.json").read_text())
    pool, clean = {}, {}
    for loinc, v in marked.items():
        pool[loinc] = list(v["names"])
        for n, cs, cg in zip(v["names"], v["clean_same_loinc"], v["clean_global"]):
            clean[(loinc, n)] = (bool(cs), bool(cg))
    return pool, clean


def build_sidecar(uuids, pool, clean, mimic_to_loinc, salt, jobs=8):
    """Exact record of the realised assignment: (patient_uuid, loinc, relma_name, clean flags).
    Calls the same function the dataset build calls, with the same pool and salt."""
    from joblib import Parallel, delayed

    def one(uuid):
        import ndjson
        from paraphrase_utils import patient_rng, assign_paraphrases_per_patient
        p = NDJSON_DIR / f"{uuid}.ndjson"
        if not p.exists():
            return []
        try:
            with open(p) as f:
                resources = ndjson.load(f)
        except Exception:
            return []
        mapping = assign_paraphrases_per_patient(resources, pool, mimic_to_loinc,
                                                 patient_rng(uuid, salt=salt))
        return [(uuid, l, n) + clean.get((l, n), (False, False)) for l, n in mapping.items()]

    out = Parallel(n_jobs=jobs)(delayed(one)(u) for u in uuids)
    return [r for chunk in out for r in chunk]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(FAMILY_SCRIPT), required=True)
    ap.add_argument("--condition", default="para",
                    help="the condition whose representation this test set is built in. The default "
                         "'para' reproduces the v1 behaviour byte for byte, including the run slug. "
                         "A v3 slug (para_xsite) also needs --test-arm. nodesc slugs are refused, "
                         "because RELMA substitutes the description and a nodesc model was trained "
                         "without one.")
    ap.add_argument("--test-arm", choices=["clean", "pert"], default=None,
                    help="v3 only, and required there: which copy of the test split to build. "
                         "RELMA is a test-time substitution with no training arm, so this is the "
                         "script's only arm.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: perturbation severity multiplier")
    ap.add_argument("--perturb-salt", default=None, help="v3 only: override the base salt")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--refrange", choices=["full", "norr"], default=None,
                    help="deprecated, replaced by --condition. Kept so old commands still parse. "
                         "It must agree with --condition, or the run is refused. Use --condition "
                         "para_norr instead of --refrange norr.")
    ap.add_argument("--strategy", choices=["s1", "s2"], default="s1")
    ap.add_argument("--assignment-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N test patients (smoke)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.condition not in CONDITIONS:
        sys.exit(f"unknown --condition {args.condition!r}; expected one of {CONDITIONS}")
    if not has_description(args.condition):
        sys.exit(f"--condition {args.condition!r} has no description. RELMA substitutes "
                 "the description, and a nodesc model was trained without one.")
    _rep = representation(args.condition)
    if _rep == "v3" and args.test_arm is None:
        sys.exit("--test-arm is required for a v3 condition: 'pert' for the coherent second site, "
                 "or 'clean' for a names-only shift.")
    if _rep != "v3" and args.test_arm is not None:
        sys.exit(f"--test-arm is v3-only; --condition {args.condition!r} is {_rep}.")

    sys.path.insert(0, str(SCRIPTS))
    fam = load_family(args.model)
    for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    salt = salt_for(args.strategy, args.assignment_seed)
    # The condition slug decides the reference-range axis. --refrange is kept only so old commands
    # parse, and it has to agree with the slug. If the two flags disagreed, the build could produce a
    # para dataset with the range stripped.
    with_refrange = includes_reference_range(args.condition)
    if args.refrange is not None and (args.refrange == "full") != with_refrange:
        sys.exit(f"--refrange {args.refrange!r} contradicts --condition {args.condition!r}, which "
                 f"says the prompt should {'carry' if with_refrange else 'omit'} a reference range. "
                 "Drop --refrange and name the condition you want.")
    # the slug carries the reference-range axis so a norr RELMA set cannot overwrite the original
    tag = f"relma_{args.strategy}_a{args.assignment_seed}" + ("" if with_refrange else "_norr")
    # v1 and v2 keep the plain slug (`mlm__relma_s1_a0`). Only v3 appends the condition and arm,
    # because only v3 has an arm to name.
    _arm_suffix = f"__{args.condition}__{args.test_arm}" if _rep == "v3" else ""
    run = f"{args.model}__{tag}{_arm_suffix}"

    # -- v3: the second site, built the same way as in 06/09 --
    perturb_cfg = None
    if _rep == "v3":
        import perturbation as pert
        _salt = args.perturb_salt or pert.DEFAULT_SALT
        _on = args.test_arm == "pert"
        perturb_cfg = {"salt": _salt, "severity": args.severity,
                       "perturb_values": _on, "perturb_ranges": _on,
                       # outside the switches: part of the v3 representation, on in every arm
                       "randomise_units": not args.no_randomise_units}
        print(f"  representation v3 | test arm {args.test_arm!r}: values {'on' if _on else 'off'}, "
              f"ranges {'on' if _on else 'off'}, units randomised "
              f"{'on' if perturb_cfg['randomise_units'] else 'off'}, severity {args.severity:g}, "
              f"salt {_salt!r}, spec {pert.SPEC_PATH.name}")
    # A --limit build is partial and gets a separate path. At the canonical dataset path,
    # 15_eval_runner would evaluate the truncated test split as if it were complete.
    if args.limit:
        run += f"__smoke{args.limit}"
        print(f"  [PARTIAL BUILD] --limit set -> '{run}' (not the canonical path)")
    # the family builder puts its temporary shards beside CACHE_DIR, which only the builder's main()
    # sets.
    # Without this line they would land in the working folder and stay there.
    fam.CACHE_DIR = f"{OUTPUT_BASE}/{run}.cache"
    # RELMA is a with-description condition: it works by substituting the description.
    with_desc = True
    print(f"=== 18_relma_dataset | {run} | salt='{salt}' | "
          f"with_desc={with_desc} with_refrange={with_refrange} ===")

    pool, clean = load_pool(args.strategy)
    n_names = sum(len(v) for v in pool.values())
    print(f"RELMA pool [{args.strategy}]: {len(pool)} LOINC, {n_names:,} names")

    uuids = fam.read_uuid_list(SPLITS_DIR / "test_patients.txt") if hasattr(fam, "read_uuid_list") \
        else [l.strip().removesuffix(".ndjson") for l in open(SPLITS_DIR / "test_patients.txt") if l.strip()]
    if args.limit:
        uuids = uuids[: args.limit]
    print(f"test patients: {len(uuids):,}")

    # -- family-specific setup, reusing each script's own helpers --
    if args.model == "mlm":
        mimic_to_loinc = fam._load_seen_codes_mimic_to_loinc(fam.SEEN_CODES_PATH)
        tok = fam._build_fhir_tokenizer(with_desc, with_refrange)
        lab_min, lab_max = fam._lab_token_range(tok)
        # FHIR-MLM takes the perturbation but not the v2/v3 rendering. Its numbers are percentile
        # tokens, so standardising the text half would change nothing (see 09). BCM and Llama
        # take both, via `rep`. The difference lies in 09 and 10/11 themselves.
        build = lambda: fam._build_split_dataset(
            uuids, pool, mimic_to_loinc, tok, with_desc, lab_min, lab_max,
            f"test[{tag}]", True, args.jobs, paraphrase_salt=salt,
            with_refrange=with_refrange, perturb_cfg=perturb_cfg)
    else:
        from fhir_data_utils import build_lab_code_mappings
        mimic_code_to_class, class_to_loinc = build_lab_code_mappings(
            fam.CODES_TO_TOKENS_PATH, fam.SEEN_CODES_PATH)
        mimic_to_loinc = fam._load_seen_codes_mimic_to_loinc(fam.SEEN_CODES_PATH)
        builder = fam._build_bcm_tokenizer_formatter if args.model == "bcm" \
            else fam._build_llama_tokenizer_formatter
        # `rep` defaults to "v1". Without it, a v3 build would get the v1 rendering (raw units,
        # no 4 s.f., no standardised reference range) and a v3 checkpoint would be scored on it.
        # The formatter and the split builder both need it.
        tokenizer, formatter = builder(with_desc, with_refrange, _rep)
        build = lambda: fam._build_split_dataset(
            uuids, pool, mimic_code_to_class, class_to_loinc, mimic_to_loinc, formatter, tokenizer,
            tokenizer.mask_token_id, with_desc, fam.MAX_LEN, f"test[{tag}]", True, args.jobs,
            paraphrase_salt=salt, with_refrange=with_refrange, rep=_rep,
            perturb_cfg=perturb_cfg)

    # -- sidecar first: cheap, and it validates the pool/salt before the expensive build --
    t0 = time.time()
    side = build_sidecar(uuids, pool, clean, mimic_to_loinc, salt, jobs=min(args.jobs, 12))
    n_clean = sum(1 for r in side if r[3])
    print(f"assignment sidecar: {len(side):,} (patient,LOINC) pairs in {time.time()-t0:.0f}s | "
          f"clean_same_loinc {n_clean:,} ({100*n_clean/max(len(side),1):.1f}%)")

    t0 = time.time()
    ds, stats, shard_root = build()
    print(f"built test split: {len(ds):,} rows in {time.time()-t0:.0f}s | stats={stats}")

    if not args.write:
        print("\n[dry] --write not set; nothing saved")
        if shard_root:
            shutil.rmtree(shard_root, ignore_errors=True)
        return

    import datasets
    out = f"{OUTPUT_BASE}/{run}"
    if Path(out).exists() and not args.force:
        raise FileExistsError(f"{out} exists (pass --force to overwrite)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    datasets.DatasetDict({"test": ds}).save_to_disk(out)
    print(f"saved -> {out}")

    import pandas as pd
    sc = pd.DataFrame(side, columns=["patient_uuid", "loinc", "relma_name",
                                     "clean_same_loinc", "clean_global"])
    sc.to_parquet(f"{out}/assignment_sidecar.parquet", index=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    man = {
        "run": run, "model": args.model, "strategy": args.strategy,
        "assignment_seed": args.assignment_seed, "salt": salt,
        "condition": f"{args.condition} (RELMA-substituted)", "representation": _rep,
        "test_arm": args.test_arm, "with_refrange": with_refrange,
        # True for Llama, whose text puts each Observation's Code line last (CODE_LAST in 11).
        # False for BCM and MLM.
        "code_last": getattr(fam, "CODE_LAST", False),
        "v3_perturbation": perturb_cfg,
        "n_test_patients": len(uuids), "n_rows": len(ds), "build_stats": stats,
        "pool": {"n_loinc": len(pool), "n_names": n_names},
        "sidecar": {"n_pairs": len(side), "n_clean_same_loinc": n_clean,
                    "pct_clean_same_loinc": round(100 * n_clean / max(len(side), 1), 2)},
        "note": ("LOINCs absent from the RELMA pool keep their NATIVE MIMIC display and must be DROPPED "
                 "at scoring time; they remain as context for neighbouring observations."),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (Path(out) / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    (RESULTS_DIR / f"{run}.json").write_text(json.dumps(man, indent=2, default=str))
    print(f"sidecar -> {out}/assignment_sidecar.parquet | manifest -> {RESULTS_DIR/(run+'.json')}")
    if shard_root:                        # after the manifest: `ds` reads the shards until saved
        shutil.rmtree(shard_root, ignore_errors=True)


if __name__ == "__main__":
    main()
