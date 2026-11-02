"""Collect the names of each LOINC code from the LOINC table, MIMIC-IV, UMLS, OHDSI Athena and CDISC
(thesis Chapter 4 and Appendix C). These are the source descriptions that prep/4.2 paraphrases.

Reads UPSTREAM_DIR/mimic_loinc.csv and seen_codes.csv (from prep/4_seen_codes.py), the CDISC
LOINC-to-LB map (UPSTREAM_DIR/loinc_to_cdisc_lb_map.csv) and the Athena vocabulary export
(ATHENA_DIR). Queries the UMLS API with UMLS_API_KEY from the environment. Writes
UPSTREAM_DIR/loinc_descriptions_enriched.jsonl, one record per LOINC code: loinc_code, umls_cui,
metadata (the six LOINC axes) and descriptions ({vocabulary: [names]}). The UMLS answers and the
Athena extraction are cached in UPSTREAM_DIR/umls_checkpoint.json and athena_checkpoint.json, so an
interrupted run resumes where it stopped.

  UMLS_API_KEY=<key> python prep/4.1_collect_external_descriptions.py
"""

import os
import csv
import json
import time
import requests
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

UMLS_API_KEY = os.environ["UMLS_API_KEY"]  # set in the environment

# UMLS API
UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
REQUEST_DELAY = 0.2  # seconds between API calls
BACKOFF_DELAY = 30   # seconds to wait on 429 rate limit
MAX_RETRIES = 3      # retries per request on transient errors
CHECKPOINT_INTERVAL = 50  # save progress every N codes

# Athena OMOP vocabulary export (tab-separated despite .csv extension)
ATHENA_DIR = str(locations.ATHENA)
ATHENA_CONCEPT_PATH = f"{ATHENA_DIR}/CONCEPT.csv"
ATHENA_RELATIONSHIP_PATH = f"{ATHENA_DIR}/CONCEPT_RELATIONSHIP.csv"
ATHENA_SYNONYM_PATH = f"{ATHENA_DIR}/CONCEPT_SYNONYM.csv"
ATHENA_ENGLISH_LANG_ID = "4180186"  # English language concept ID in OMOP
ATHENA_SYNONYM_RELATIONSHIPS = {"Mapped from", "Is a"}  # 1-hop traversal
# Skip synthetic LOINC composite classes when following relationships
ATHENA_SKIP_LOINC_CLASSES = {"LOINC Hierarchy", "LOINC Group", "LOINC Class"}

# Input files
MIMIC_LOINC_PATH = str(locations.UPSTREAM / "mimic_loinc.csv")
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
CDISC_MAP_PATH = str(locations.UPSTREAM / "loinc_to_cdisc_lb_map.csv")

# Output files
CHECKPOINT_PATH = str(locations.UPSTREAM / "umls_checkpoint.json")
ATHENA_CHECKPOINT_PATH = str(locations.UPSTREAM / "athena_checkpoint.json")
OUTPUT_PATH = str(locations.UPSTREAM / "loinc_descriptions_enriched.jsonl")

def umls_get(url, params, retries=MAX_RETRIES):
    """Make a GET request to the UMLS API with retry and backoff logic."""
    params["apiKey"] = UMLS_API_KEY
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            print(f"  Rate limited, waiting {BACKOFF_DELAY}s (attempt {attempt + 1}/{retries})")
            time.sleep(BACKOFF_DELAY)
            continue
        if resp.status_code in (500, 502, 503):
            print(f"  Server error {resp.status_code}, retrying in 5s (attempt {attempt + 1}/{retries})")
            time.sleep(5)
            continue
        # Non-retryable error
        resp.raise_for_status()
    print(f"  Failed after {retries} retries (last status: {resp.status_code})")
    return None


def get_cui_for_loinc(loinc_code):
    """Retrieve the UMLS CUI for a LOINC code via exact source-code search."""
    data = umls_get(
        f"{UMLS_BASE_URL}/search/current",
        {"string": loinc_code, "inputType": "sourceUi", "searchType": "exact", "sabs": "LNC"},
    )
    if data is None:
        return None
    results = data.get("result", {}).get("results", [])
    if results and results[0].get("ui") != "NONE":
        return results[0]["ui"]
    return None


def get_atoms_for_cui(cui):
    """Retrieve all English-language atoms for a CUI, handling pagination."""
    atoms = []
    page = 1
    while True:
        data = umls_get(
            f"{UMLS_BASE_URL}/content/current/CUI/{cui}/atoms",
            {"language": "ENG", "pageSize": 25, "pageNumber": page},
        )
        if data is None:
            break
        page_results = data.get("result", [])
        if not page_results:
            break
        for atom in page_results:
            name = atom.get("name", "").strip()
            source = atom.get("rootSource", "").strip()
            if name and source:
                atoms.append({"name": name, "source": source})
        # Stop if fewer results than page size (last page)
        if len(page_results) < 25:
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    return atoms


def load_checkpoint(path=CHECKPOINT_PATH):
    """Load existing checkpoint if available."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(data, path=CHECKPOINT_PATH):
    """Save checkpoint to disk."""
    with open(path, "w") as f:
        json.dump(data, f)


def dedupe_list(values):
    """Deduplicate a list of strings case-insensitively, keeping first casing. Return sorted list."""
    seen = set()
    result = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            result.append(v)
    return sorted(result, key=str.lower)


# ----------------------------------------------------------------------------
# Athena OMOP vocabulary helpers
#
# Athena exports are tab-separated despite the .csv extension. Each file is
# streamed with csv.DictReader inside a `with` block, so the file handle is
# released promptly. This matters when several 200–500 MB files are scanned
# in sequence in one process.
# ----------------------------------------------------------------------------

def load_athena_loinc_concepts(target_codes):
    """Stream CONCEPT.csv once to find concept_ids for our target LOINC codes.

    Returns (loinc_to_cid, cid_to_loinc).
    """
    target_set = set(target_codes)
    loinc_to_cid = {}
    with open(ATHENA_CONCEPT_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Scan CONCEPT for LOINC"):
            if row["vocabulary_id"] == "LOINC" and row["concept_code"] in target_set:
                loinc_to_cid[row["concept_code"]] = row["concept_id"]
    cid_to_loinc = {v: k for k, v in loinc_to_cid.items()}
    return loinc_to_cid, cid_to_loinc


def load_athena_relationships(loinc_cid_set, rel_types):
    """Stream CONCEPT_RELATIONSHIP.csv once to collect 1-hop non-self neighbors.

    Returns {loinc_concept_id: set(related_concept_ids)}.
    """
    neighbors = defaultdict(set)
    with open(ATHENA_RELATIONSHIP_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Scan CONCEPT_RELATIONSHIP"):
            if row.get("invalid_reason"):
                continue
            cid1 = row["concept_id_1"]
            cid2 = row["concept_id_2"]
            if cid1 in loinc_cid_set and row["relationship_id"] in rel_types and cid1 != cid2:
                neighbors[cid1].add(cid2)
    return neighbors


def load_athena_concept_info(concept_ids):
    """Stream CONCEPT.csv once to look up (vocab, name, class) for the given concept_ids.

    Returns {concept_id: {"vocabulary_id": ..., "concept_name": ..., "concept_class_id": ...}}.
    """
    want = set(concept_ids)
    info = {}
    with open(ATHENA_CONCEPT_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Scan CONCEPT for neighbors"):
            if row["concept_id"] in want:
                info[row["concept_id"]] = {
                    "vocabulary_id": row["vocabulary_id"],
                    "concept_name": row["concept_name"],
                    "concept_class_id": row["concept_class_id"],
                }
    return info


def load_athena_synonyms(concept_ids):
    """Stream CONCEPT_SYNONYM.csv once to collect English synonyms for the given concept_ids.

    Returns {concept_id: [synonym_strings]}.
    """
    want = set(concept_ids)
    synonyms = defaultdict(list)
    with open(ATHENA_SYNONYM_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="Scan CONCEPT_SYNONYM"):
            if row["concept_id"] in want and row["language_concept_id"] == ATHENA_ENGLISH_LANG_ID:
                synonyms[row["concept_id"]].append(row["concept_synonym_name"])
    return synonyms


def build_athena_descriptions_for_code(
    loinc_code, loinc_cid, neighbors, concept_info, synonyms_by_cid
):
    """Assemble {vocab_lowercase: [description_strings]} for one LOINC code.

    Rules:
    - The LOINC concept's primary name and synonyms go under "lnc".
    - Each neighbor concept goes under its vocabulary_id, lowercased.
    - Skip neighbors whose class is in ATHENA_SKIP_LOINC_CLASSES (synthetic composites).
    - Append the neighbor's primary name and its English synonyms.
    """
    result = defaultdict(list)

    # Direct LOINC concept info
    info = concept_info.get(loinc_cid)
    if info:
        if info["concept_name"]:
            result["lnc"].append(info["concept_name"])
        result["lnc"].extend(synonyms_by_cid.get(loinc_cid, []))

    # 1-hop neighbors
    for nbr_cid in neighbors.get(loinc_cid, []):
        nbr_info = concept_info.get(nbr_cid)
        if not nbr_info:
            continue
        vocab = nbr_info["vocabulary_id"]
        # Skip synthetic LOINC composite classes
        if vocab == "LOINC" and nbr_info["concept_class_id"] in ATHENA_SKIP_LOINC_CLASSES:
            continue
        key = vocab.lower()
        if nbr_info["concept_name"]:
            result[key].append(nbr_info["concept_name"])
        result[key].extend(synonyms_by_cid.get(nbr_cid, []))

    return dict(result)

# Load LOINC metadata from mimic_loinc.csv (one row per code)
mimic_loinc_df = pd.read_csv(MIMIC_LOINC_PATH).fillna("")
loinc_codes = mimic_loinc_df["LOINC_NUM"].tolist()
print(f"Loaded {len(loinc_codes)} LOINC codes from {MIMIC_LOINC_PATH}")

# Load MIMIC descriptions from seen_codes.csv (may have multiple descs per code)
mimic_descs = defaultdict(set)
with open(SEEN_CODES_PATH, newline="") as f:
    for row in csv.DictReader(f):
        desc = row["mimic_desc"].strip()
        if desc:
            mimic_descs[row["loinc_code"]].add(desc)
print(f"Loaded MIMIC descriptions for {len(mimic_descs)} codes from {SEEN_CODES_PATH}")

# Load CDISC LB mapping (LBTEST = full test name, LBTESTCD = short code)
cdisc_df = pd.read_csv(CDISC_MAP_PATH).fillna("")
cdisc_descs = defaultdict(set)
for _, row in cdisc_df.iterrows():
    code = str(row["LOINC Code"]).strip()
    lbtest = row["CDISC LBTEST"].strip()
    lbtestcd = row["CDISC LBTESTCD"].strip()
    if lbtest:
        cdisc_descs[code].add(lbtest)
    if lbtestcd:
        cdisc_descs[code].add(lbtestcd)

cdisc_overlap = set(loinc_codes) & set(cdisc_descs.keys())
print(f"Loaded CDISC descriptions for {len(cdisc_descs)} LOINC codes from {CDISC_MAP_PATH} ({len(cdisc_overlap)} overlap with our codes)")

assert UMLS_API_KEY, "Please set UMLS_API_KEY before running this cell."

checkpoint = load_checkpoint()
print(f"Checkpoint has {len(checkpoint)} codes already queried.")

codes_to_query = [c for c in loinc_codes if c not in checkpoint]
print(f"Remaining codes to query: {len(codes_to_query)}")

for i, code in enumerate(tqdm(codes_to_query, desc="Querying UMLS")):
    # Step 1: Get CUI
    cui = get_cui_for_loinc(code)
    time.sleep(REQUEST_DELAY)

    # Step 2: Get atoms if CUI found
    atoms = []
    if cui:
        atoms = get_atoms_for_cui(cui)
        time.sleep(REQUEST_DELAY)

    checkpoint[code] = {"cui": cui, "atoms": atoms}

    # Periodic checkpoint save
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(checkpoint)
        print(f"  Checkpoint saved at {i + 1}/{len(codes_to_query)}")

# Final save
save_checkpoint(checkpoint)
print(f"Done. Queried {len(codes_to_query)} codes. Total in checkpoint: {len(checkpoint)}")

if os.path.exists(ATHENA_CHECKPOINT_PATH):
    print(f"Loading Athena descriptions from checkpoint: {ATHENA_CHECKPOINT_PATH}")
    athena_by_loinc = load_checkpoint(ATHENA_CHECKPOINT_PATH)
    print(f"Loaded {len(athena_by_loinc)} codes from checkpoint.")
else:
    assert os.path.exists(ATHENA_DIR), f"Athena directory not found: {ATHENA_DIR}"

    # Step 1: find concept_ids for our target LOINC codes
    loinc_to_cid, cid_to_loinc = load_athena_loinc_concepts(loinc_codes)
    print(f"  Found {len(loinc_to_cid)}/{len(loinc_codes)} LOINC codes in Athena CONCEPT table")
    loinc_cid_set = set(loinc_to_cid.values())

    # Step 2: collect 1-hop neighbors via Mapped from / Is a
    neighbors = load_athena_relationships(loinc_cid_set, ATHENA_SYNONYM_RELATIONSHIPS)
    n_with_neighbors = sum(1 for s in neighbors.values() if s)
    total_edges = sum(len(s) for s in neighbors.values())
    print(f"  {n_with_neighbors} codes have neighbors; {total_edges} total edges")

    # Step 3: look up vocab/name/class for LOINC concepts + all neighbors
    neighbor_ids = set()
    for s in neighbors.values():
        neighbor_ids.update(s)
    all_cids_to_look_up = loinc_cid_set | neighbor_ids
    concept_info = load_athena_concept_info(all_cids_to_look_up)
    print(f"  Looked up info for {len(concept_info)}/{len(all_cids_to_look_up)} concepts")

    # Step 4: collect English synonyms for LOINC concepts + neighbors
    synonyms_by_cid = load_athena_synonyms(all_cids_to_look_up)
    n_with_synonyms = len(synonyms_by_cid)
    total_synonyms = sum(len(v) for v in synonyms_by_cid.values())
    print(f"  {n_with_synonyms} concepts have English synonyms; {total_synonyms} total synonyms")

    # Step 5: build per-LOINC-code description dict
    athena_by_loinc = {}
    for loinc_code, loinc_cid in loinc_to_cid.items():
        descs = build_athena_descriptions_for_code(
            loinc_code, loinc_cid, neighbors, concept_info, synonyms_by_cid
        )
        if descs:
            athena_by_loinc[loinc_code] = descs

    save_checkpoint(athena_by_loinc, ATHENA_CHECKPOINT_PATH)
    print(f"Saved Athena checkpoint to {ATHENA_CHECKPOINT_PATH}")

# Summary of Athena extraction
vocab_counter = Counter()
for code, descs in athena_by_loinc.items():
    for vocab in descs:
        vocab_counter[vocab] += 1
print("\nAthena extraction summary:")
print(f"  Codes with any Athena data: {len(athena_by_loinc)}/{len(loinc_codes)}")
print("  Vocabularies discovered:")
for vocab, n_codes in vocab_counter.most_common():
    print(f"    {vocab}: {n_codes} codes")

umls_checkpoint = load_checkpoint(CHECKPOINT_PATH)
athena_checkpoint = load_checkpoint(ATHENA_CHECKPOINT_PATH)


def _is_joined_relatednames_blob(s):
    """Athena's CONCEPT_SYNONYM stores the LOINC RELATEDNAMES2 field as a single
    semicolon-joined synonym on almost every LOINC concept. Detect and drop it.
    Real descriptions don't contain 5+ semicolons."""
    return s.count(";") >= 5


# Phrases that mark a derived or computed test that is distinct from a direct
# measurement. Athena's "Mapped from" relationship sometimes pulls in SNOMED
# procedures that share an analyte with a LOINC code but describe a different
# test. For example, SNOMED "Fractional excretion of magnesium" maps to LOINC
# 19123-9 (Magnesium SerPl-mCnc) "Magnesium [Mass/vol] in Ser/Plas". The fractional
# excretion formula uses a serum magnesium measurement, but the clinical
# procedure is different. Any non-LOINC description that contains these phrases
# is dropped. None of the target LOINC codes describes fractional excretion.
CROSS_TEST_NOISE_PATTERNS = (
    "fractional excretion",
)


def _is_cross_test_noise(desc):
    d_lower = desc.lower()
    return any(p in d_lower for p in CROSS_TEST_NOISE_PATTERNS)


# Normalize UMLS source names to the canonical vocab key used elsewhere.
# LNC is UMLS's abbreviation for LOINC, so it folds into the same bucket.
UMLS_SOURCE_ALIASES = {"lnc": "loinc"}


# Build one JSON object per LOINC code
records = []
for _, loinc_row in mimic_loinc_df.iterrows():
    code = loinc_row["LOINC_NUM"]
    umls_entry = umls_checkpoint.get(code, {"cui": None, "atoms": []})
    athena_entry = athena_checkpoint.get(code, {})

    # --- Metadata: the 6 LOINC elements ---
    metadata = {
        "loinc_component": (loinc_row.get("COMPONENT") or "").strip(),
        "loinc_property": (loinc_row.get("PROPERTY") or "").strip(),
        "loinc_time_aspect": (loinc_row.get("TIME_ASPCT") or "").strip(),
        "loinc_system": (loinc_row.get("SYSTEM") or "").strip(),
        "loinc_scale_type": (loinc_row.get("SCALE_TYP") or "").strip(),
        "loinc_method_type": (loinc_row.get("METHOD_TYP") or "").strip(),
    }

    # --- Collect raw (un-deduped) descriptions by vocab from all sources ---
    raw_by_vocab = defaultdict(list)

    # MIMIC
    raw_by_vocab["mimic"].extend(mimic_descs.get(code, []))

    # LOINC table fields: primary names only. RELATEDNAMES2 is skipped. It supplies
    # about 79% of the LOINC strings, and 89% of its fragments are 15 characters or
    # fewer. These are abbreviations such as "CT", "PCR" and "QL", or broad
    # categories such as "Microbiology". They add noise and give little to paraphrase.
    for field in ("LONG_COMMON_NAME", "SHORTNAME", "DisplayName"):
        val = (loinc_row.get(field) or "").strip()
        if val:
            raw_by_vocab["loinc"].append(val)

    # UMLS atoms, grouped by source vocabulary (LNC is aliased to loinc)
    for atom in umls_entry.get("atoms", []):
        src = atom["source"].lower()
        src = UMLS_SOURCE_ALIASES.get(src, src)
        if src != "loinc" and _is_cross_test_noise(atom["name"]):
            continue
        raw_by_vocab[src].append(atom["name"])

    # CDISC mapping
    raw_by_vocab["cdisc"].extend(cdisc_descs.get(code, []))

    # Athena extraction. The extraction stores direct LOINC concept info under the
    # hard-coded "lnc" key and neighbor LOINC codes under "loinc". Both are the
    # same vocabulary, so they are merged. The joined RELATEDNAMES2 synonym and
    # cross-test noise are also dropped (e.g., SNOMED "Fractional excretion of X"
    # pulled in via "Mapped from").
    for vocab, descs in athena_entry.items():
        key = "loinc" if vocab in ("lnc", "loinc") else vocab
        for d in descs:
            if _is_joined_relatednames_blob(d):
                continue
            if key != "loinc" and _is_cross_test_noise(d):
                continue
            raw_by_vocab[key].append(d)

    # --- Dedupe each vocab's list and drop empty keys ---
    descriptions = {}
    for vocab, vals in raw_by_vocab.items():
        deduped = dedupe_list(vals)
        if deduped:
            descriptions[vocab] = deduped

    record = {
        "loinc_code": code,
        "umls_cui": umls_entry.get("cui") or "",
        "metadata": metadata,
        "descriptions": descriptions,
    }
    records.append(record)

print(f"Built {len(records)} records")

# Preview first record
print(json.dumps(records[0], indent=2))

with open(OUTPUT_PATH, "w") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved {len(records)} records to {OUTPUT_PATH}")
