"""Build the TF-IDF dataset: one text prompt per lab observation (Description, Value, ReferenceRange,
Priority), labelled with its LOINC code (thesis Chapters 4 and 5).

The prompt format is what `utils.extract_structured_features_single` parses at training time.
Descriptions are paraphrases from the 70/15/15 per-LOINC split of the deep models. The split comes
from the same function and seed (09_mlm_dataset.load_paraphrases_3way), so the pools are identical,
and the sha256 of the split shows this. Paraphrases are assigned per (patient, LOINC) as for FHIR-MLM,
BCM and Llama, so every model sees the same text for an observation. The split also gives TF-IDF a
val set. Observations are restricted to the 679-code label space (data/labelspace/labelspace.csv).

One dataset serves both description conditions. The nodesc condition is applied at training time by
extract_structured_features_single(exclude_fields={"description"}) on the same prompts. Every
label-space LOINC has paraphrases, so the observation set is identical.

--representation picks the text. v1 has native units, plain decimals and the reference range. v2 has
standardised units, values in 4 s.f. scientific notation and no reference range. v3 is v2 plus the
reference range, standardised the same way. In v3 the unit notation is randomised, and values and
ranges are perturbed per patient to simulate a second site. The --no-perturb-* flags turn the
perturbation off, for the clean arm and the value-only and range-only sets.

Reads data/splits/{train,val,test}_patients.txt, data/paraphrases/clean, the per-patient ndjson files
(PATIENT_NDJSON_DIR) and, from UPSTREAM_DIR, seen_codes.csv and the unit and qualitative-value maps.
Writes per-patient JSONL ({prompt, label}) and label files to {split}/{samples,labels}/ under
WORK_DIR/tfidf/dataset (v1), dataset_v2 or dataset_v3__{arm}{tag}, with a manifest. Patient-level
rows stay in WORK_DIR. Stdout shows only aggregates, except that --show-samples prints real prompts
for checking the format by eye.

  python pipeline/06_tfidf_dataset.py --representation v3
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys

from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import perturbation_tag  # noqa: E402

COMMON_DIR = str(locations.COMMON)
# paraphrase_utils (per-patient assignment) is in common/. The path is added here, so this script
# does not depend on the import of 09 to add it.
for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for unit_standardisation (v2)
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"

PATIENT_DIR = str(locations.PATIENTS)
SPLITS_DIR = locations.REPO / "data/splits"
PARAPHRASE_DIR = str(locations.REPO / "data/paraphrases/clean")
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
SEEN_CODES = f"{locations.UPSTREAM}/seen_codes.csv"
QUAL_NORMALIZE_PATH = f"{locations.UPSTREAM}/pcornet_qual_normalize.json"
QUAL_PASSTHROUGH_PATH = f"{locations.UPSTREAM}/pcornet_qual_passthrough.json"
UNIT_SYNONYM_PATH = f"{locations.UPSTREAM}/unit_synonym_map.json"
OUTPUT_DIR = str(locations.WORK / "tfidf/dataset")
#: v2 is written to dataset_v2, so a v2 build never overwrites the v1 dataset.
OUTPUT_DIR_V2 = str(locations.WORK / "tfidf/dataset_v2")
#: v3 is kept apart too (conditions.dataset_suffix). v3 is not an extension of v2, which has no
#: reference range, so a v3 build must not land where a v2 build is read from.
OUTPUT_DIR_V3 = str(locations.WORK / "tfidf/dataset_v3")
SEED = 42


# --- Pure, unit-testable logic ----------------------------------------------
def extract_loinc_code(obs: dict, mimic_to_loinc: dict) -> "str | None":
    """Map coding[0] to LOINC. Use the code itself under a LOINC system, else the MIMIC itemid map."""
    codings = obs.get("code", {}).get("coding", [])
    if not codings:
        return None
    code_value = codings[0].get("code")
    system = codings[0].get("system", "") or ""
    if not code_value:
        return None
    if "loinc" in system.lower():
        return code_value
    if code_value in mimic_to_loinc:
        return mimic_to_loinc[code_value]
    return None


def is_lab_obs(obs: dict) -> bool:
    """True for a lab observation under the code system of the model head (mimic-d-labitems)."""
    if obs.get("resourceType") != "Observation":
        return False
    codings = obs.get("code", {}).get("coding", [])
    return bool(codings) and codings[0].get("system") == LAB_CODE_SYSTEM and bool(codings[0].get("code"))


def format_observation_as_prompt(obs: dict, paraphrase: str, unit_norm, qual_norm,
                                 quantity_fmt=None, include_reference_range: bool = True,
                                 range_fmt=None) -> str:
    """Format one lab observation as prompt text. unit_norm(raw) -> str; qual_norm(raw) -> str or None
    (None omits the Value line). Emits, in order, Description / Value / ReferenceRange / Priority:
    the field labels utils.extract_structured_features_single parses.

    The optional parameters give the v2 and v3 representations. Left at their defaults, they give
    the v1 output byte for byte.

      quantity_fmt(value, raw_unit) -> (value_text, unit_text)
          Replaces the separate value and `unit_norm(unit)` handling of valueQuantity. It takes both
          together because the unit sets the scale factor, and the factor changes the value. If the
          two are formatted separately, a rescaled number can end up printed beside its old unit.
      include_reference_range
          False omits the ReferenceRange line entirely. v1 always writes it and drops it at training
          time by regex. v2 never writes it, so a build check can count zero occurrences.
      range_fmt(referenceRange_list, value_raw_unit) -> [(label, text, unit), ...]
          v3 only: formats the bounds through the same transform as the value. The v1 loop below
          formats each bound with `unit_norm(unit)` and the raw number, without the scale factor of
          the value. This is the mistake the note on `quantity_fmt` warns about. Under v3 that loop would
          print a haemoglobin as `1.400e+04 mg/dL` beside `Low: 12.0 g/dL`, a value a thousand times
          its range. That would happen for every observation whose unit is rescaled and which
          carries a range.

    16_tfidf_eval_dump.py calls this function too. A dump must pass the same arguments as the build
    of its representation. Otherwise the scored prompts differ from the training text."""
    parts = [f"Description: {paraphrase}"]  # the lab path always uses the paraphrase

    value_raw_unit = ""
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        comparator = vq.get("comparator", "")
        raw_unit = value_raw_unit = vq.get("unit") or vq.get("code") or ""
        if quantity_fmt is None:
            val = vq.get("value", "")
            unit = unit_norm(raw_unit)
        else:
            # the comparator is kept verbatim and still precedes the number. ">" and "<" carry
            # clinical meaning ("<0.01") and are part of what the Value regex captures
            val, unit = quantity_fmt(vq.get("value", ""), raw_unit)
        parts.append(f"Value: {comparator}{val} {unit}".rstrip())
    elif "valueString" in obs:
        emitted = qual_norm(obs.get("valueString") or "")
        if emitted:
            parts.append(f"Value: {emitted}")
    elif "valueCodeableConcept" in obs:
        vcc = obs["valueCodeableConcept"]
        raw = vcc.get("text") or (vcc.get("coding", [{}])[0].get("display", "") if vcc.get("coding") else "") or ""
        emitted = qual_norm(raw)
        if emitted:
            parts.append(f"Value: {emitted}")

    if include_reference_range and range_fmt is not None:
        # v3: the bounds go through the same transform as the value, so a rescaled number is not
        # printed beside its old unit. In a sample of 400 patients (103,220 quantitative
        # observations), a referenceRange list never has more than one entry, and every entry has
        # both bounds. Flattening the list therefore emits the same single line that v1 emits.
        rp = [f"{label}: {text} {unit}".rstrip()
              for label, text, unit in range_fmt(obs.get("referenceRange", []), value_raw_unit)]
        if rp:
            parts.append(f"ReferenceRange: {', '.join(rp)}")
    else:
        for ref in (obs.get("referenceRange", []) if include_reference_range else []):
            rp = []
            if "low" in ref:
                lu = unit_norm(ref["low"].get("unit") or ref["low"].get("code") or "")
                rp.append(f"Low: {ref['low'].get('value', '')} {lu}".rstrip())
            if "high" in ref:
                hu = unit_norm(ref["high"].get("unit") or ref["high"].get("code") or "")
                rp.append(f"High: {ref['high'].get('value', '')} {hu}".rstrip())
            if rp:
                parts.append(f"ReferenceRange: {', '.join(rp)}")

    if "extension" in obs and obs["extension"]:  # priority (lab only)
        priority = obs["extension"][0].get("valueString", "")
        if priority:
            parts.append(f"Priority: {priority}")

    return "\n".join(parts)


def tally_patient_samples(resources, mimic_to_loinc, label_set, paraphrases, rng, unit_norm, qual_norm,
                          uuid=None, quantity_fmt=None, include_reference_range=True,
                          range_fmt=None):
    """Return the per-patient list of (prompt, loinc) for lab observations whose LOINC is in
    label_set and has paraphrases.

    Paraphrases are assigned per (patient, LOINC) with `assign_paraphrases_per_patient` and
    `patient_rng(uuid)`, as in the deep-model dataset builders. All four models then see the same
    text for an observation, and paired tests compare like with like. When uuid is None (the --smoke
    preview), each observation draws from `rng` instead."""
    mapping = None
    if uuid is not None:
        from paraphrase_utils import patient_rng, assign_paraphrases_per_patient
        mapping = assign_paraphrases_per_patient(resources, paraphrases, mimic_to_loinc,
                                                 patient_rng(uuid))
    out = []
    for r in resources:
        if not is_lab_obs(r):
            continue
        loinc = extract_loinc_code(r, mimic_to_loinc)
        if not loinc or loinc not in label_set:
            continue
        if mapping is not None:
            chosen = mapping.get(loinc)
            if not chosen:
                continue
        else:
            paras = paraphrases.get(loinc)
            if not paras:
                continue
            chosen = rng.choice(paras)
        prompt = format_observation_as_prompt(r, chosen, unit_norm, qual_norm,
                                              quantity_fmt=quantity_fmt,
                                              include_reference_range=include_reference_range,
                                              range_fmt=range_fmt)
        out.append((prompt, loinc))
    return out


# --- IO (main only) ---------------------------------------------------------
def load_label_space(path: Path) -> set:
    return {r["loinc"] for r in csv.DictReader(open(path)) if r["in_primary_ge10occ"] == "1"}


def load_mimic_to_loinc(path: str) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        if r["observation_type"] == "lab" and r["mimic_code"] and r["loinc_code"]:
            out[r["mimic_code"]] = r["loinc_code"]
    return out


def load_paraphrases(pdir: str, test_fraction=0.2, seed=42):
    """Per-LOINC 80/20 split into train and test paraphrases. Not used by the build, which takes the
    70/15/15 split from 09 (load_paraphrases_3way)."""
    train, test = {}, {}
    rng = random.Random(seed)
    for fp in sorted(glob.glob(os.path.join(pdir, "*.txt"))):
        loinc = Path(fp).stem
        lines = [l.strip() for l in open(fp, encoding="utf-8") if l.strip()]
        rng.shuffle(lines)
        n_test = max(1, int(len(lines) * test_fraction))
        test[loinc], train[loinc] = lines[:n_test], lines[n_test:]
    return train, test


def _load_uuids(path: Path):
    return [l.strip()[:-7] if l.strip().endswith(".ndjson") else l.strip() for l in open(path) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="process only first N train patients, print samples, no write")
    ap.add_argument("--representation", choices=["v1", "v2", "v3"], default="v1",
                    help="v1 = native units, decimal, reference range. "
                         "v2 = standardised units + fixed-width 4 s.f. scientific + no reference "
                         "range. "
                         "v3 = v2 plus the reference range, standardised the same way, with the "
                         "value and the range perturbed per patient to simulate a second site.")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="v3 only: multiplier on every perturbation magnitude. The register's "
                         "magnitudes are uncertain by about 3x, so 0.5, 1 and 2 span a 4x band "
                         "that covers this uncertainty")
    ap.add_argument("--no-perturb-values", action="store_true",
                    help="v3 only: leave values alone. For the value-vs-range ablation.")
    ap.add_argument("--no-perturb-ranges", action="store_true",
                    help="v3 only: leave reference ranges alone. For the value-vs-range ablation.")
    ap.add_argument("--replicate", type=int, default=None,
                    help="v3 only: build perturbation replicate N, an independent site draw "
                         "with its own salt and its own directory.")
    ap.add_argument("--splits", default="train,val,test",
                    help="which splits to build. An eval-only set (replicate, severity or "
                         "ablation) needs only 'test', since its train and val are not read.")
    ap.add_argument("--no-randomise-units", action="store_true",
                    help="v3 only: turn off unit-notation randomisation (on by default in every v3 arm)")
    ap.add_argument("--perturb-salt", default=None,
                    help="v3 only: changes every site draw in lockstep. Use a different salt for "
                         "the test split than for the train split, so the two are independent sites.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="build and write only the first N patients of each split (a small "
                         "dataset for the build-time gate)")
    ap.add_argument("--show-samples", type=int, default=0,
                    help="print N fully decoded prompts (patient data: leave at 0 for any run whose output is shared)")
    args = ap.parse_args()
    if args.output_dir is None:
        # the train-arm segment. The v3 clean and perturbed builds share the condition and the
        # representation and differ only by the perturbation switches. Without this segment one
        # would overwrite the other.
        args.output_dir = {"v1": OUTPUT_DIR, "v2": OUTPUT_DIR_V2, "v3": OUTPUT_DIR_V3}[
            args.representation]
        if args.representation == "v3":
            # the naming the deep-model builders use (conditions.arm_suffix and perturbation_tag),
            # so a TF-IDF eval set and its FHIR-MLM counterpart carry the same tag. Without the
            # perturbation tag, every severity and every value-only or range-only set would land on
            # `__pert`, the dataset the perturbed arm trains on.
            _pv, _pr = not args.no_perturb_values, not args.no_perturb_ranges
            arm = "clean" if not (_pv or _pr) else "pert"
            args.output_dir += (f"__{arm}"
                                + perturbation_tag(args.severity, _pv, _pr, args.replicate,
                                                   "06_tfidf_dataset"))

    sys.path.insert(0, COMMON_DIR)
    import ndjson
    from tqdm import tqdm
    from input_preprocessing_utils import (load_qual_normalizer, load_qual_passthrough,
                                           load_unit_alias_map, normalize_qual_value, normalize_unit)

    qual_v2c = load_qual_normalizer(QUAL_NORMALIZE_PATH)
    qual_lit, qual_rx = load_qual_passthrough(QUAL_PASSTHROUGH_PATH)
    unit_map = load_unit_alias_map(UNIT_SYNONYM_PATH)
    ctr = Counter()

    def unit_norm(raw):
        if not raw:
            return ""
        canon, mapped = normalize_unit(raw, unit_map)
        ctr["unit_mapped" if mapped else "unit_unmapped"] += 1
        return canon

    def qual_norm(raw):
        outcome, val = normalize_qual_value(raw, qual_v2c, qual_lit, qual_rx)
        ctr[f"qual_{outcome}"] += 1
        return None if outcome == "blank" else val

    # -- v2 / v3 representation --
    # v3 uses the same value transform as v2. It adds the reference range, standardised through
    # that same transform, and the per-patient perturbation.
    quantity_fmt, include_rr, range_fmt, perturb = None, True, None, None
    if args.representation in ("v2", "v3"):
        import unit_standardisation as us
        include_rr = args.representation == "v3"
        # 06 takes no condition. One dataset serves both description conditions, and no-desc is
        # applied at training time. Whether the prompt carries a reference range follows --representation.

        def quantity_fmt(value, raw_unit):
            txt, unit = us.standardise_quantity(value, raw_unit, unit_map)
            ctr["v2_value_scientific" if us.VALUE_RE.match(txt or "")
                else "v2_value_passthrough"] += 1
            _, factor = us.standardise_unit(raw_unit, unit_map)
            ctr["v2_unit_rescaled" if factor != 1.0 else "v2_unit_kept"] += 1
            return txt, unit
        print(f"  representation {args.representation}: {len(us.STANDARD_UNIT)} units -> "
              f"{len(us.STANDARD_TARGETS)} targets, values at {us.SIG_FIGS} s.f. scientific, "
              f"{'ReferenceRange standardised the same way' if include_rr else 'no ReferenceRange line'}")

    if args.representation == "v3":
        import perturbation as pert

        def range_fmt(rrs, value_raw_unit):
            out = us.standardise_reference_range(rrs, value_raw_unit, unit_map)
            ctr["v3_rr_bounds"] += len(out)
            return out

        _spec = pert.load_spec()
        _rand = None if args.no_randomise_units else pert.load_randomisation()
        _salt = args.perturb_salt or pert.DEFAULT_SALT
        # a replicate is a different site, so it needs a different salt
        if args.replicate is not None:
            _salt = f"{_salt}-r{args.replicate}"
        _do_v = not args.no_perturb_values
        _do_r = not args.no_perturb_ranges

        def perturb(resources, uuid):
            """Applied to the patient's resources before formatting. It cannot be per observation,
            because MCHC is recomputed from Hb and Hct in the same panel (see perturbation.py)."""
            # pert.lab_loinc_of is the predicate 09/10/11 use. It gives the same result as
            # extract_loinc_code here, because the map in 06 is already lab-only.
            # Unit-notation randomisation comes first and sits outside the perturbation switches,
            # because it is part of the v3 representation and on in every arm. It rewrites unit
            # strings only and leaves values alone, so its order relative to the perturbation does
            # not matter. The order is fixed here anyway.
            if _rand is not None:
                ctr["v3_units_randomised"] += pert.randomise_units(
                    resources, uuid, _rand, lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
                    salt=_salt)
            n = pert.perturb_patient(resources, uuid, _spec,
                                     lambda r: pert.lab_loinc_of(r, mimic_to_loinc),
                                     severity=args.severity, perturb_values=_do_v,
                                     perturb_ranges=_do_r, salt=_salt)
            ctr["v3_fields_perturbed"] += n
        print("  unit randomisation: "
              + (f"{len(_rand)} codes" if _rand is not None else "off"))
        print(f"  perturbation: {len(_spec)} codes, severity {args.severity:g}, "
              f"values {'on' if _do_v else 'off'}, ranges {'on' if _do_r else 'off'}, "
              f"salt {_salt!r}")

    label_set = load_label_space(LABELSPACE_CSV)
    mimic_to_loinc = load_mimic_to_loinc(SEEN_CODES)
    # the 70/15/15 split of the deep models, from the function in 09 with the same seed, so the
    # pools are identical. It gives TF-IDF a val split and the same paraphrase pools as the other
    # models. The sha256 of the split is printed below.
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("d9", Path(__file__).resolve().parent / "09_mlm_dataset.py")
    _d9 = _ilu.module_from_spec(_s); _s.loader.exec_module(_d9)
    train_para, val_para, test_para, _pstats = _d9.load_paraphrases_3way(_d9.PARAPHRASE_DIR, verbose=False)
    print(f"  paraphrase split (from 09): train {_pstats['n_train']:,} / val {_pstats['n_val']:,} / "
          f"test {_pstats['n_test']:,} | sha256={_pstats['split_sha256'][:16]}…")
    print(f"label space (679): {len(label_set)} | mimic->loinc: {len(mimic_to_loinc)} | "
          f"paraphrase LOINC: {len(train_para)}")
    # every label-space LOINC must have paraphrases
    missing = [l for l in label_set if l not in train_para]
    assert not missing, f"{len(missing)} label-space LOINC lack paraphrases: {missing[:5]}"

    def run_split(name, uuids, paras, write):
        out_s = Path(args.output_dir) / name / "samples"
        out_l = Path(args.output_dir) / name / "labels"
        if write:
            out_s.mkdir(parents=True, exist_ok=True)
            out_l.mkdir(parents=True, exist_ok=True)
        rng = random.Random(SEED)
        n_pat = n_samp = n_with = 0
        label_freq = Counter()
        for u in tqdm(uuids, desc=name):
            p = f"{PATIENT_DIR}/{u}.ndjson"
            if not os.path.exists(p):
                continue
            with open(p) as f:
                resources = ndjson.load(f)
            if perturb is not None:
                perturb(resources, u)
            samples = tally_patient_samples(resources, mimic_to_loinc, label_set, paras, rng,
                                            unit_norm, qual_norm, uuid=u,
                                            quantity_fmt=quantity_fmt,
                                            include_reference_range=include_rr,
                                            range_fmt=range_fmt)
            n_pat += 1
            if not samples:
                continue
            n_with += 1
            n_samp += len(samples)
            for _, lc in samples:
                label_freq[lc] += 1
            if write:
                with open(out_s / f"{u}.jsonl", "w") as fs, open(out_l / f"{u}.txt", "w") as fl:
                    for prompt, label in samples:
                        fs.write(json.dumps({"prompt": prompt, "label": label}) + "\n")
                        fl.write(label + "\n")
        print(f"  {name}: patients={n_pat} with_samples={n_with} obs={n_samp:,} distinct_LOINC={len(label_freq)}")
        return n_samp, label_freq

    if args.smoke:
        uuids = _load_uuids(SPLITS_DIR / "train_patients.txt")[:args.smoke]
        rng = random.Random(SEED)
        shown = 0
        for u in uuids:
            p = f"{PATIENT_DIR}/{u}.ndjson"
            if not os.path.exists(p):
                continue
            with open(p) as f:
                resources = ndjson.load(f)
            if perturb is not None:
                perturb(resources, u)
            samples = tally_patient_samples(resources, mimic_to_loinc, label_set, train_para, rng,
                                            unit_norm, qual_norm, quantity_fmt=quantity_fmt,
                                            include_reference_range=include_rr,
                                            range_fmt=range_fmt)
            for prompt, label in samples[:2]:
                # print only the field names and the length, to check the prompt format without
                # printing the patient's Value, ReferenceRange or Priority
                fields = [ln.split(":", 1)[0] for ln in prompt.split("\n")]
                print(f"[smoke] label={label}  fields={fields}  chars={len(prompt)}")
                if shown < args.show_samples:
                    # --show-samples is opt-in and prints real patient values, for checking the
                    # representation by eye on a handful of observations
                    print("        " + prompt.replace("\n", "\n        "))
                shown += 1
            if shown >= 6:
                break
        run_split("train", uuids, train_para, write=False)
        print(f"\n[smoke] normalization: {dict(ctr)}")
        return

    def _uu(name):
        u = _load_uuids(SPLITS_DIR / f"{name}_patients.txt")
        return u[:args.limit] if args.limit else u

    want = tuple(x for x in ("train", "val", "test") if x in args.splits.split(","))
    if not want:
        sys.exit(f"--splits {args.splits!r} selected nothing")
    pool = {"train": train_para, "val": val_para, "test": test_para}
    counts = {}
    for _s in want:
        counts[_s], _ = run_split(_s, _uu(_s), pool[_s], write=True)
    train_n, test_n = counts.get("train"), counts.get("test")
    manifest = {"label_space": len(label_set), "train_obs": train_n, "test_obs": test_n,
                "representation": args.representation,
                "train_arm": ("clean" if (args.no_perturb_values and args.no_perturb_ranges)
                              else "pert") if args.representation == "v3" else None,
                "limited_to_n_patients": args.limit or None,
                "splits_built": list(want), "replicate": args.replicate,
                "patient_dir": PATIENT_DIR, "paraphrase_dir": PARAPHRASE_DIR, "normalization": dict(ctr)}
    if args.representation in ("v2", "v3"):
        import unit_standardisation as us
        manifest[args.representation] = {
            "sig_figs": us.SIG_FIGS, "max_log10_factor": us.MAX_LOG10_FACTOR,
            "n_units": len(us.STANDARD_UNIT), "targets": us.STANDARD_TARGETS,
            "include_reference_range": include_rr}
    if args.representation == "v3":
        import perturbation as pert
        # every setting that changes a number goes in the manifest. A v3 dataset is not
        # reproducible without the salt, and two arms that differ only by salt must be told apart.
        # The recorded salt is the effective one, with the replicate suffix, as in the `_salt` used
        # for the draw above and as 09/10/11 record it. The manifest alone then reproduces the draw.
        _eff_salt = args.perturb_salt or pert.DEFAULT_SALT
        if args.replicate is not None:
            _eff_salt = f"{_eff_salt}-r{args.replicate}"
        manifest["v3_perturbation"] = {
            "spec": str(pert.SPEC_PATH), "salt": _eff_salt,
            "randomise_units": not args.no_randomise_units,
            "severity": args.severity, "perturb_values": not args.no_perturb_values,
            "perturb_ranges": not args.no_perturb_ranges}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote dataset + manifest to {args.output_dir}/")


if __name__ == "__main__":
    main()
