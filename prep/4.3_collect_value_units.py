"""Collect the units and qualitative values each LOINC code is reported with (MIMIC-IV, LOINC, UCUM,
CDISC, CIEL), and build the unit synonym map that standardises unit strings (thesis Chapter 4 and
Appendix A).

Reads the per-patient files from prep/2_create_loinc_patients.py (WORK_DIR/loinc_patients/), but only
for the training patients listed in UPSTREAM_DIR/train_test_splits_mlm/train_patients.txt. Also reads
seen_codes.csv and mimic_loinc.csv from prep/4_seen_codes.py. From UPSTREAM_DIR it reads Loinc.csv,
the CDISC LOINC-to-LB map and SDTM terminology, the PCORnet CDM v1.14 UNIT value set, ucum.csv and
unit_manual_aliases.json. CIEL units come from the Open Concept Lab API. Writes
UPSTREAM_DIR/loinc_measurements_enriched.jsonl (one record per LOINC code) and
UPSTREAM_DIR/unit_synonym_map.json. The patient scan and the CIEL download are checkpointed in
UPSTREAM_DIR/checkpoints/. The unit map in resources/ is the output of this script. The
jsonl is not shipped, and step 5 of the README needs it (analysis/a20, analysis/lib/units.py).

  python prep/4.3_collect_value_units.py
"""

import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
import sys
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

PATIENT_DIR = locations.WORK / "loinc_patients"
SEEN_CODES_PATH = locations.UPSTREAM / "seen_codes.csv"
MIMIC_LOINC_PATH = locations.UPSTREAM / "mimic_loinc.csv"
LOINC_CSV_PATH = locations.UPSTREAM / "Loinc.csv"
CDISC_MAP_PATH = locations.UPSTREAM / "loinc_to_cdisc_lb_map.csv"
SPLIT_DIR = locations.UPSTREAM / "train_test_splits_mlm"
TRAIN_SPLIT_PATH = SPLIT_DIR / "train_patients.txt"
TEST_SPLIT_PATH = SPLIT_DIR / "test_patients.txt"

CHECKPOINT_DIR = locations.UPSTREAM / "checkpoints"
TRAIN_CHECKPOINT_PATH = CHECKPOINT_DIR / "4.3_value_units_train.json"
TEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "4.3_value_units_test.json"
CIEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "ciel_checkpoint.json"
OUTPUT_PATH = locations.UPSTREAM / "loinc_measurements_enriched.jsonl"

CHECKPOINT_INTERVAL = 2000  # save progress every N patients scanned

# OpenConceptLab (OCL) API for CIEL concepts. Anonymous reads are rate-limited
# to 400/min and 10k/day. Requests throttle on the X-LimitRemaining-Minute
# response header. Results are checkpointed to CIEL_CHECKPOINT_PATH, so later
# runs skip the network. `includeRetired=false` is the OCL default but is
# passed explicitly in case the default changes.
OCL_API_BASE = "https://api.openconceptlab.org"
OCL_CIEL_MAPPINGS_URL = (
    OCL_API_BASE + "/orgs/CIEL/sources/CIEL/mappings/"
    "?toConceptSource=LOINC&includeRetired=false&limit=100"
)
OCL_CIEL_CONCEPT_URL = OCL_API_BASE + "/orgs/CIEL/sources/CIEL/concepts/{cid}/"
OCL_REQUEST_DELAY = 0.2   # baseline per-request pause; 400/min ceiling = 0.15s
OCL_RETRIES = 3

# Unit synonym merging. UCUM is the source of the canonical keys. PCORnet and
# CDISC merge in by normalized-code match, and unit_manual_aliases.json covers
# the remaining bucket strings that match none of the three. The steps are in
# build_unit_synonym_map() below.
UCUM_CSV_PATH = locations.UPSTREAM / "ucum.csv"
PCORNET_UNIT_PATH = locations.UPSTREAM / "PCORnet_CDM_ValueSet_ReferenceFile_v1.14_UNIT.csv"
CDISC_TERM_PATH = locations.UPSTREAM / "CDISC_SDTM_Terminology.csv"
MANUAL_ALIASES_PATH = locations.UPSTREAM / "unit_manual_aliases.json"
UNIT_SYNONYM_MAP_PATH = locations.UPSTREAM / "unit_synonym_map.json"

CDISC_UNIT_CODELIST = "C71620"  # SDTM codelist for lab/vital units
# Case-sensitive CDISC submission values that collide with a UCUM code but
# carry a different physical meaning. Skipped at CDISC load so the CDISC
# synonyms don't pollute a UCUM canonical's synonym list.
#   C = CDISC "CELSIUS" vs UCUM "coulomb"
#   F = CDISC "FAHRENHEIT" vs UCUM "farad"
CDISC_CONFLICT_SKIP = {"C", "F"}
UNIT_COVERAGE_THRESHOLD = 0.95

# MIMIC-to-LOINC mapping (lab-only, from seen_codes.csv). Used only to route
# FHIR Observation codings into the right LOINC bucket during the scan.
mimic_to_loinc = {}
with open(SEEN_CODES_PATH) as f:
    for row in csv.DictReader(f):
        if row["observation_type"] != "lab":
            continue
        mc, lc = row["mimic_code"], row["loinc_code"]
        if mc and lc:
            mimic_to_loinc[mc] = lc

print(f"Loaded {len(mimic_to_loinc)} MIMIC → LOINC lab mappings covering "
      f"{len(set(mimic_to_loinc.values()))} LOINC codes")

# Target LOINC code set for external-source enrichment. It matches the target
# list of prep/4.1 (every code in mimic_loinc.csv), so the descriptions and
# measurements JSONLs cover the same codes. The MIMIC scan still uses the
# lab-only mapping above. The LOINC, UCUM and CDISC buckets are filled for
# every target code, whether or not MIMIC observed it.
mimic_loinc_df = pd.read_csv(MIMIC_LOINC_PATH).fillna("")
target_codes = set(mimic_loinc_df["LOINC_NUM"])
loinc_meta = {
    row["LOINC_NUM"]: {
        "loinc_long_name": row["LONG_COMMON_NAME"].strip(),
        "loinc_scale_type": row["SCALE_TYP"].strip(),
    }
    for _, row in mimic_loinc_df.iterrows()
}
print(f"Target LOINC codes (external enrichment scope): {len(target_codes)}")

def load_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

train_files = load_split(TRAIN_SPLIT_PATH)
test_files = load_split(TEST_SPLIT_PATH)
print(f"Train: {len(train_files)} patients")
print(f"Test:  {len(test_files)} patients")

def _empty_bucket():
    return {
        "value_units": Counter(),
        "qualitative_values": Counter(),
        "shape_counts": Counter(),  # keys: quantitative, qualitative, absent
    }


def _tally_observation(obs, bucket):
    """Update a LOINC-bucket with one Observation resource's value/unit info."""
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        unit = vq.get("unit") or vq.get("code") or ""
        bucket["value_units"][unit] += 1
        bucket["shape_counts"]["quantitative"] += 1
    elif "valueString" in obs:
        val = obs["valueString"] or ""
        bucket["qualitative_values"][val] += 1
        bucket["shape_counts"]["qualitative"] += 1
    elif "valueCodeableConcept" in obs:
        vcc = obs["valueCodeableConcept"]
        val = vcc.get("text") or (vcc.get("coding", [{}])[0].get("display", "")) or ""
        bucket["qualitative_values"][val] += 1
        bucket["shape_counts"]["qualitative"] += 1
    else:
        bucket["shape_counts"]["absent"] += 1


def _extract_mimic_code(obs):
    """Return the `mimic-d-labitems` code string from an Observation, or None."""
    for coding in obs.get("code", {}).get("coding", []):
        if coding.get("system") == "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems":
            return coding.get("code")
    return None


def _bucket_to_dict(bucket):
    return {k: dict(v) for k, v in bucket.items()}


def _dict_to_bucket(d):
    return {k: Counter(v) for k, v in d.items()}


def _save_checkpoint(path, processed, per_loinc):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_patients": list(processed),
        "per_loinc": {lc: _bucket_to_dict(b) for lc, b in per_loinc.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def _load_checkpoint(path):
    if not path.exists():
        return set(), defaultdict(_empty_bucket)
    with open(path) as f:
        payload = json.load(f)
    processed = set(payload.get("processed_patients", []))
    per_loinc = defaultdict(_empty_bucket)
    for lc, bd in payload.get("per_loinc", {}).items():
        per_loinc[lc] = _dict_to_bucket(bd)
    return processed, per_loinc


def scan_patients(patient_files, mimic_to_loinc, checkpoint_path, desc="Scanning"):
    """Scan a list of patient NDJSON filenames; return {loinc_code: bucket_dict}.

    Resumes from checkpoint_path if present. Writes checkpoint every
    CHECKPOINT_INTERVAL patients plus a final flush at the end.
    """
    processed, per_loinc = _load_checkpoint(checkpoint_path)
    if processed:
        print(f"  Resuming from checkpoint: {len(processed)} patients already processed")

    todo = [fn for fn in patient_files if fn not in processed]
    print(f"  {len(todo)} patients to scan (of {len(patient_files)} total)")

    for i, fn in enumerate(tqdm(todo, desc=desc)):
        fpath = PATIENT_DIR / fn
        try:
            with open(fpath) as f:
                for line in f:
                    if '"resourceType": "Observation"' not in line:
                        continue
                    obs = json.loads(line)
                    if obs.get("resourceType") != "Observation":
                        continue
                    mc = _extract_mimic_code(obs)
                    if mc is None:
                        continue
                    lc = mimic_to_loinc.get(mc)
                    if lc is None:
                        continue
                    _tally_observation(obs, per_loinc[lc])
        except FileNotFoundError:
            print(f"WARN: missing {fpath}")
            continue

        processed.add(fn)
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, processed, per_loinc)

    _save_checkpoint(checkpoint_path, processed, per_loinc)
    return {lc: _bucket_to_dict(b) for lc, b in per_loinc.items()}

train_results = scan_patients(train_files, mimic_to_loinc, TRAIN_CHECKPOINT_PATH, desc="Train")
print(f"Train scan complete: {len(train_results)} LOINC codes with at least one observation")

def _parse_semicolon_list(s):
    """Split a LOINC unit field on ';' into stripped non-empty unique units."""
    if not s:
        return []
    return sorted({p.strip() for p in s.split(";") if p.strip()})


# LOINC canonical (human-readable) and UCUM (formal) example units.
loinc_example_units = {}   # {loinc_code: [unit, ...]} from EXAMPLE_UNITS
loinc_ucum_units = {}      # {loinc_code: [unit, ...]} from EXAMPLE_UCUM_UNITS
with open(LOINC_CSV_PATH) as f:
    for row in csv.DictReader(f):
        lc = row["LOINC_NUM"]
        if lc not in target_codes:
            continue
        ex = _parse_semicolon_list(row.get("EXAMPLE_UNITS", ""))
        uc = _parse_semicolon_list(row.get("EXAMPLE_UCUM_UNITS", ""))
        if ex:
            loinc_example_units[lc] = ex
        if uc:
            loinc_ucum_units[lc] = uc
print(f"LOINC EXAMPLE_UNITS:      {len(loinc_example_units):>3} / {len(target_codes)} target codes")
print(f"LOINC EXAMPLE_UCUM_UNITS: {len(loinc_ucum_units):>3} / {len(target_codes)} target codes")

# CDISC LBORRESU: the standardized "original result unit" per LBTESTCD in the
# CDISC Lab Test Code controlled terminology. The map has multiple rows per
# LOINC code (one per CDISC specimen/method variant), so the units are
# deduplicated per LOINC code. "(Must be null)" marks tests that are explicitly
# qualitative in the CDISC model. They are skipped because they carry no unit.
cdisc_units_raw = defaultdict(set)
with open(CDISC_MAP_PATH) as f:
    for row in csv.DictReader(f):
        lc = row["LOINC Code"].strip()
        if lc not in target_codes:
            continue
        u = (row.get("EXAMPLE CDISC LBORRESU") or "").strip()
        if u and u != "(Must be null)":
            cdisc_units_raw[lc].add(u)
cdisc_units = {lc: sorted(us) for lc, us in cdisc_units_raw.items()}
print(f"CDISC LBORRESU:           {len(cdisc_units):>3} / {len(target_codes)} target codes")

# CIEL (OpenMRS terminology) quantitative units, fetched from the OCL API.
# The first run downloads all CIEL-to-LOINC mappings, then fetches each mapped
# CIEL concept to get `extras.units`. Later runs read the snapshot. Several
# CIEL ids can map to one LOINC (mostly Test+LabSet panel pairs), and the
# union of their `extras.units` values is taken per LOINC.
def _ocl_get(url):
    """GET a JSON URL from OCL with retries and a throttle based on the
    response headers.

    It calls resp.headers.get directly, which is case-insensitive on
    HTTPMessage. Casting to dict first would lose that, and OCL's mixed-case
    X-LimitRemaining-* headers would be missed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "fhir-loinc/4.3"})
    last_err = None
    for attempt in range(OCL_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
                rem_min = int(resp.headers.get("X-LimitRemaining-Minute", 400))
                rem_day = int(resp.headers.get("X-LimitRemaining-Day", 10000))
                if rem_day < 100:
                    raise RuntimeError(
                        f"OCL daily budget near-exhausted ({rem_day} left); abort."
                    )
                if rem_min < 20:
                    time.sleep(20)
                else:
                    time.sleep(OCL_REQUEST_DELAY)
                return body
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                # Respect Retry-After if present, else exponential backoff.
                wait = int(e.headers.get("Retry-After", 0)) or (2 ** attempt)
                time.sleep(wait)
            else:
                time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OCL GET failed after {OCL_RETRIES} attempts for {url}: {last_err}")


def _fetch_ciel_mappings_to_loinc():
    """Page through /orgs/CIEL/sources/CIEL/mappings/ for the LOINC target source."""
    rows = []
    page = 1
    while True:
        batch = _ocl_get(f"{OCL_CIEL_MAPPINGS_URL}&page={page}")
        rows.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return rows


def _fetch_ciel_units_from_api(target_codes):
    """Return {loinc_code: sorted unique unit strings} via OCL API."""
    print(f"  fetching CIEL→LOINC mappings from {OCL_API_BASE} ...")
    mappings = _fetch_ciel_mappings_to_loinc()
    # Build CIEL id -> list of target LOINCs it maps to.
    ciel_to_loincs = defaultdict(set)
    for m in mappings:
        lc = m.get("to_concept_code")
        cid = m.get("from_concept_code")
        if lc in target_codes and cid:
            ciel_to_loincs[cid].add(lc)
    print(f"  fetching {len(ciel_to_loincs)} CIEL concepts ...")
    loinc_to_units = defaultdict(set)
    for cid, lcs in tqdm(sorted(ciel_to_loincs.items()), desc="CIEL concepts"):
        concept = _ocl_get(OCL_CIEL_CONCEPT_URL.format(cid=cid))
        unit = ((concept.get("extras") or {}).get("units") or "").strip()
        if not unit:
            continue
        for lc in lcs:
            loinc_to_units[lc].add(unit)
    return {lc: sorted(us) for lc, us in loinc_to_units.items()}


def _atomic_write_json(path, obj):
    """Write JSON via tmp + rename so a mid-write abort can't corrupt path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


if CIEL_CHECKPOINT_PATH.exists():
    ciel_units = json.loads(CIEL_CHECKPOINT_PATH.read_text())
else:
    ciel_units = _fetch_ciel_units_from_api(target_codes)
    _atomic_write_json(CIEL_CHECKPOINT_PATH, ciel_units)
print(f"CIEL extras.units:        {len(ciel_units):>3} / {len(target_codes)} target codes")

def _sorted_unique(counter):
    """Return sorted unique keys. Drops empty strings and values containing '___'
    (MIMIC's de-identification placeholder for PHI like dates, locations, names)."""
    return sorted({k for k in counter.keys() if k and "___" not in k})


# Iterate the full target-code list so the description and measurement JSONLs
# cover the same codes. The `mimic` bucket only appears for codes observed in
# the train split. Qualitative values in particular can contain uniquely
# identifying free text, so nothing is taken from the test split. The `loinc`,
# `ucum`, `cdisc` and `ciel` buckets are filled for every code with a non-empty
# source.
records = []
for lc in sorted(target_codes):
    meta = loinc_meta.get(lc, {"loinc_long_name": "", "loinc_scale_type": ""})
    measurements = {}

    tr = train_results.get(lc)
    if tr is not None:
        mimic_sys = {}
        mimic_quant = _sorted_unique(tr["value_units"])
        mimic_qual = _sorted_unique(tr["qualitative_values"])
        if mimic_quant:
            mimic_sys["quantitative"] = mimic_quant
        if mimic_qual:
            mimic_sys["qualitative"] = mimic_qual
        if mimic_sys:
            measurements["mimic"] = mimic_sys

    if lc in loinc_example_units:
        measurements["loinc"] = {"quantitative": loinc_example_units[lc]}
    if lc in loinc_ucum_units:
        measurements["ucum"] = {"quantitative": loinc_ucum_units[lc]}
    if lc in cdisc_units:
        measurements["cdisc"] = {"quantitative": cdisc_units[lc]}
    if lc in ciel_units:
        measurements["ciel"] = {"quantitative": ciel_units[lc]}

    rec = {
        "loinc_code": lc,
        "loinc_long_name": meta["loinc_long_name"],
        "loinc_scale_type": meta["loinc_scale_type"],
        "measurements": measurements,
    }
    if tr is not None:
        rec["mimic_shape_counts"] = dict(tr["shape_counts"])
    records.append(rec)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(records)} records to {OUTPUT_PATH}")

import re

# Normalization, used to merge the sources and to build the alias_to_canonical keys.
# - Strip {annotations}, except when the whole string is an annotation
#   (e.g. "{INR}", "{score}"). Those strings would otherwise collapse to "",
#   and every distinct score, index or titer annotation would share one key.
# - Strip [brackets] but keep the content ("[IU]" becomes "IU"), so CDISC
#   "IU" and UCUM "[IU]" hit the same key.
# - Rewrite 10^n as 10*n, so CDISC "10^3/uL" and UCUM "10*3/uL" agree.
# - Lowercase, and strip whitespace and underscores.
_RE_ANNOT = re.compile(r"\{[^}]*\}")
_RE_BRACKET = re.compile(r"\[([^\]]*)\]")
_RE_POW = re.compile(r"(\d)\^(\d)")
_RE_SPACE_UNDER = re.compile(r"[\s_]+")

def normalize_unit_key(s):
    if not s:
        return ""
    stripped = _RE_ANNOT.sub("", s)
    if not stripped.strip():
        stripped = s  # pure-annotation preservation
    stripped = _RE_BRACKET.sub(r"\1", stripped)
    stripped = _RE_POW.sub(r"\1*\2", stripped)
    return _RE_SPACE_UNDER.sub("", stripped.lower())


def _load_ucum_rows():
    rows = []
    with open(UCUM_CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                "code":     r["case-sensitive code"].strip(),
                "display":  r["name (display)"].strip(),
                "synonyms": [s.strip() for s in r.get("synonyms", "").split(";") if s.strip()],
            })
    return rows


def _load_pcornet_rows():
    rows = []
    with open(PCORNET_UNIT_PATH) as f:
        for r in csv.DictReader(f):
            code = (r.get("CODE") or r.get("\ufeffCODE") or "").strip()
            if not code:
                continue
            rows.append({
                "code":    code,
                "display": (r.get("DESCRIPTIVE_TEXT") or "").strip(),
            })
    return rows


def _load_cdisc_unit_rows():
    rows = []
    with open(CDISC_TERM_PATH) as f:
        for r in csv.DictReader(f):
            if r.get("Codelist Code", "").strip() != CDISC_UNIT_CODELIST:
                continue
            sub = (r.get("CDISC Submission Value") or "").strip()
            if not sub or sub in CDISC_CONFLICT_SKIP:
                continue
            rows.append({
                "code":     sub,
                "pref":     (r.get("NCI Preferred Term") or "").strip(),
                "synonyms": [s.strip() for s in (r.get("CDISC Synonym(s)") or "").split(";") if s.strip()],
            })
    return rows


def build_unit_synonym_map():
    ucum_rows = _load_ucum_rows()
    pcn_rows  = _load_pcornet_rows()
    cds_rows  = _load_cdisc_unit_rows()
    aliases_payload = json.loads(MANUAL_ALIASES_PATH.read_text())
    aliases = aliases_payload["aliases"]
    excluded = aliases_payload["exclude"]

    canonical_units = {}       # canonical_code -> {display, synonyms, sources, is_ucum}
    norm_to_canonical = {}     # normalized_code -> canonical_code (first-wins)

    def _register(code, display, syns, source, is_ucum):
        if code in canonical_units:
            entry = canonical_units[code]
            entry["synonyms"] = sorted(set(entry["synonyms"]) | set(syns))
            if source not in entry["sources"]:
                entry["sources"].append(source)
            return
        canonical_units[code] = {
            "display":  display,
            "synonyms": sorted(set([code, display] + list(syns)) - {""}),
            "sources":  [source],
            "is_ucum":  is_ucum,
        }

    # 1. UCUM seed.
    for r in ucum_rows:
        _register(r["code"], r["display"], r["synonyms"], "UCUM", is_ucum=True)
        norm_to_canonical.setdefault(normalize_unit_key(r["code"]), r["code"])

    # 2. PCORnet: merge into UCUM by normalized code, else create.
    for r in pcn_rows:
        key = normalize_unit_key(r["code"])
        tgt = norm_to_canonical.get(key)
        if tgt is not None:
            entry = canonical_units[tgt]
            entry["synonyms"] = sorted((set(entry["synonyms"]) | {r["code"], r["display"]}) - {""})
            if "PCORnet" not in entry["sources"]:
                entry["sources"].append("PCORnet")
        else:
            _register(r["code"], r["display"], [], "PCORnet", is_ucum=False)
            norm_to_canonical[key] = r["code"]

    # 3. CDISC: merge by normalized code (C/F pre-filtered). For new canonicals
    #    from CDISC, we keep the CDISC submission value as the key but it's not
    #    a UCUM code (is_ucum=False).
    for r in cds_rows:
        key = normalize_unit_key(r["code"])
        tgt = norm_to_canonical.get(key)
        if tgt is not None:
            entry = canonical_units[tgt]
            added = {r["code"], r["pref"]} | set(r["synonyms"])
            entry["synonyms"] = sorted((set(entry["synonyms"]) | added) - {""})
            if "CDISC" not in entry["sources"]:
                entry["sources"].append("CDISC")
        else:
            _register(r["code"], r["pref"], r["synonyms"], "CDISC", is_ucum=False)
            norm_to_canonical[key] = r["code"]

    # 4. Manual aliases. The target may be an existing canonical code. It may
    #    also be an annotation-decorated form whose normalized key matches an
    #    existing canonical, e.g. "{cells}/uL" and UCUM "/uL". It is then
    #    merged into that canonical so that no duplicate is created. Otherwise
    #    it is a new non-UCUM code (e.g. "[beth'U]", "{INR}") that needs a new
    #    canonical entry.
    for alias_surface, target_code in aliases.items():
        if target_code in canonical_units:
            tgt_canonical = target_code
        else:
            tgt_norm = normalize_unit_key(target_code)
            existing = norm_to_canonical.get(tgt_norm)
            if existing is not None:
                tgt_canonical = existing
                entry = canonical_units[tgt_canonical]
                entry["synonyms"] = sorted((set(entry["synonyms"]) | {target_code}) - {""})
            else:
                canonical_units[target_code] = {
                    "display":  target_code,
                    "synonyms": [target_code],
                    "sources":  ["manual"],
                    "is_ucum":  False,
                }
                norm_to_canonical[tgt_norm] = target_code
                tgt_canonical = target_code

        entry = canonical_units[tgt_canonical]
        entry["synonyms"] = sorted((set(entry["synonyms"]) | {alias_surface}) - {""})
        if "manual" not in entry["sources"]:
            entry["sources"].append("manual")
        entry.setdefault("manual_aliases", [])
        if alias_surface not in entry["manual_aliases"]:
            entry["manual_aliases"].append(alias_surface)

    # 5. Reverse lookup alias_to_canonical. Resolve collisions by:
    #    (a) code-normalized match, (b) display-normalized match,
    #    (c) UCUM-native + shortest code + alphabetical tiebreaker.
    claims = defaultdict(list)
    for canon, entry in canonical_units.items():
        for syn in entry["synonyms"]:
            k = normalize_unit_key(syn)
            if k:
                claims[k].append(canon)

    alias_to_canonical = {}
    ambiguities = []

    def _resolve(norm_key, candidates_set):
        code_matches = [c for c in candidates_set if normalize_unit_key(c) == norm_key]
        if len(code_matches) == 1:
            return code_matches[0], "code-match"
        winners = code_matches if code_matches else list(candidates_set)
        disp_matches = [c for c in winners
                        if normalize_unit_key(canonical_units[c]["display"]) == norm_key]
        if len(disp_matches) == 1:
            return disp_matches[0], "display-match"
        if disp_matches:
            winners = disp_matches
        winners.sort(key=lambda c: (not canonical_units[c]["is_ucum"], len(c), c))
        return winners[0], "tiebreaker"

    for k, seen in claims.items():
        cand_set = set(seen)
        if len(cand_set) == 1:
            alias_to_canonical[k] = next(iter(cand_set))
            continue
        winner, rule = _resolve(k, cand_set)
        alias_to_canonical[k] = winner
        if rule == "tiebreaker":
            ambiguities.append({
                "surface_norm": k,
                "winner": winner,
                "losers": sorted(cand_set - {winner}),
            })

    return {
        "canonical_units":           canonical_units,
        "alias_to_canonical":        alias_to_canonical,
        "excluded_non_units":        excluded,
        "ambiguities_auto_resolved": ambiguities,
        "build_notes": {
            "ucum_rows":              len(ucum_rows),
            "pcornet_rows":           len(pcn_rows),
            "cdisc_rows":             len(cds_rows),
            "canonical_entries":      len(canonical_units),
            "alias_lookup_keys":      len(alias_to_canonical),
            "cdisc_conflict_skipped": sorted(CDISC_CONFLICT_SKIP),
            "manual_alias_count":     len(aliases),
        },
    }


unit_synonym_map = build_unit_synonym_map()
_atomic_write_json(UNIT_SYNONYM_MAP_PATH, unit_synonym_map)

bn = unit_synonym_map["build_notes"]
print(f"Wrote {UNIT_SYNONYM_MAP_PATH}")
print(f"  source rows:       ucum={bn['ucum_rows']}  pcornet={bn['pcornet_rows']}  cdisc={bn['cdisc_rows']}")
print(f"  canonical entries: {bn['canonical_entries']}")
print(f"  alias lookup keys: {bn['alias_lookup_keys']}")
print(f"  manual aliases:    {bn['manual_alias_count']}")
print(f"  ambiguities (tiebreaker-resolved): {len(unit_synonym_map['ambiguities_auto_resolved'])}")
