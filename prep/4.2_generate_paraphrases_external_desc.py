"""Generate paraphrases of each LOINC code with MedGemma-4B, clean them with rules, verify them with
Llama-3.1-8B-Instruct, and finalise the pool (thesis Chapter 4 and Appendix C).

Reads UPSTREAM_DIR/loinc_descriptions_enriched.jsonl (each code's LOINC axes and its known names from
six vocabularies, written by prep/4.1). Writes, in UPSTREAM_DIR, paraphrases_ext/ (raw generations),
paraphrases_ext_clean/, paraphrases_ext_verified/ (the final pool, one txt per code) and
paraphrases_ext_rejected/. A checkpoint file, UPSTREAM_DIR/4.2_checkpoint.json, lets an interrupted run
resume. Needs a GPU, and HF_TOKEN (from the environment or a .env file) for the gated models.

The script was converted from a notebook. The `# %% [cell N]` markers keep the notebook's cell numbers,
which some comments refer to.

  python prep/4.2_generate_paraphrases_external_desc.py
"""

# %% [cell 2]
import os
import re
import glob
import json
import gc
import random
import sys
from pathlib import Path

import torch
import torch._dynamo
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

torch.set_float32_matmul_precision("high")

# Raise the dynamo recompile limit as a safety net. Cells 17 and 21 pin the
# model to a single GPU (see _pick_best_gpu in cell 6) to avoid the
# multi-device forward-dispatch hook in accelerate, which gets wrapped by
# torch.compile. Each model.generate() call allocates a fresh DynamicCache
# whose obj_id fails the dynamo guards and triggers a recompile. At the
# default cache_size_limit=8 the run fails partway with
# FailOnRecompileLimitHit. 64 leaves room in case some other compile path
# fires.
torch._dynamo.config.cache_size_limit = 64

# %% [cell 4]
# Load secrets from .env if present. Both strict KEY=value and loose
# KEY = "value" formats are accepted, so typos in .env don't break the
# script. HF_TOKEN is required, and the script stops with an error if it is
# missing.
_ENV_PATH = ".env"
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

HF_TOKEN = os.environ["HF_TOKEN"]
GEN_MODEL_NAME    = "google/medgemma-4b-it"
VERIFY_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ENRICHED_DESC_PATH      = str(locations.UPSTREAM / "loinc_descriptions_enriched.jsonl")
PARAPHRASE_DIR          = str(locations.UPSTREAM / "paraphrases_ext")
PARAPHRASE_CLEAN_DIR    = str(locations.UPSTREAM / "paraphrases_ext_clean")
PARAPHRASE_VERIFIED_DIR = str(locations.UPSTREAM / "paraphrases_ext_verified")
PARAPHRASE_REJECTED_DIR = str(locations.UPSTREAM / "paraphrases_ext_rejected")

# Vocab order controls two things. The first is cross-vocab dedup in prompt
# rendering. LOINC comes first and wins, so later vocabs only add new
# descriptions. The second is the order in which non-LOINC vocab
# descriptions are merged during finalize.
VOCAB_ORDER = ["loinc", "mimic", "snomed", "cdisc", "ciel", "mth"]
NON_LOINC_VOCABS = [v for v in VOCAB_ORDER if v != "loinc"]

VOCAB_LABELS = {
    "loinc":  "Official LOINC naming",
    "mimic":  "MIMIC hospital naming",
    "snomed": "SNOMED CT",
    "cdisc":  "CDISC LB",
    "ciel":   "CIEL",
    "mth":    "UMLS Metathesaurus",
}

GEN_BATCH_SIZE        = 8
GEN_SEEDS             = [123, 456, 789, 42, 43]
GEN_MAX_INPUT_LEN     = 1024   # room for the prompt's names from all six vocabularies
GEN_MAX_NEW_TOKENS    = 150

VERIFY_BATCH_SIZE     = 4
VERIFY_MAX_INPUT_LEN  = 1536   # the verifier prompt includes few-shot examples
VERIFY_MAX_NEW_TOKENS = 5

MAX_PARAPHRASE_LEN    = 150   # p99 of enriched descs is 134 chars; 120 would clip 2.5% of real LOINC long-common-names

# Lightweight checkpoint file. Tracks which (seed, strategy) pairs of
# generation have finished and which files verify has finished. Lets
# a rerun after an interruption resume where the previous run
# stopped. Safe to hand-edit between runs: flip `done` to false to
# force a stage to run, or clear the `completed_*` lists to redo specific
# units. If the file is missing, a default structure is used in memory
# and it gets created on the first successful unit of work.
CHECKPOINT_PATH       = str(locations.UPSTREAM / "4.2_checkpoint.json")


# Prompting strategies. Each entry drives one pass over all LOINC codes.
# Design notes:
# - Every prompt carries the names from all 6 vocabularies, so the LLM only
#   has to vary the style. It does not have to generate new content. There
#   is no billing-justification strategy. A "test for diagnosing X" framing
#   invites the model to name whatever test is used for X in place of the
#   given test (cross-test contamination).
# - `ehr_dropdown` is emphasized because the postfix-comma qualifier
#   style (e.g. "Glucose, CSF") is the dominant multi-token pattern in
#   the real MIMIC `code.display` distribution (~17% of displays).
# - `method_specimen_explicit` targets the method + specimen phrasing
#   that the vocabularies cover only thinly.
# - `department_specific` is limited to an urgency/context prefix, with
#   an explicit guardrail against naming different tests.
# - `legacy_systems` focuses on truncated uppercase. It leaves out
#   underscores because only 1 of 980 MIMIC displays has one.
PROMPTING_STRATEGIES = [
    {
        "name": "abbreviation_heavy",
        "system": "You are documenting lab tests in a busy emergency department. Use common medical abbreviations and shorthand that real clinicians scribble on rushed lab slips.",
        "instruction": "List 5 ultra-brief test names using standard medical abbreviations. Each should be under 30 characters, unique, and recognizable to any lab technician.",
        "temperature": 0.9,
        "examples": "Examples: 'CBC w/ diff', 'Lytes', 'UA c&s', 'PT/INR', 'BUN/Cr'"
    },
    {
        "name": "verbose_formal",
        "system": "You are writing formal laboratory requisitions for a reference laboratory manual.",
        "instruction": "List 5 complete test descriptions with medical terms mostly spelled out. Include the specimen type and what is being measured. Keep each under 100 characters.",
        "temperature": 0.7,
        "examples": "Examples: 'Hemoglobin mass per volume in whole blood', 'Glucose concentration in serum or plasma', 'Total calcium in serum'"
    },
    {
        "name": "method_specimen_explicit",
        "system": "You are writing entries for a reference lab catalog where each entry names the analytical method and the specimen.",
        "instruction": "List 5 test descriptions that explicitly name the analytical method (e.g. HPLC, immunoassay, enzymatic, BCG, hexokinase, automated counter) AND the specimen (e.g. serum, plasma, whole blood, urine, CSF). Only use plausible methods for the given test — do not invent methodology.",
        "temperature": 0.7,
        "examples": "Examples: 'Hemoglobin A1c by HPLC, whole blood', 'Albumin, serum, bromocresol green method', 'Glucose, plasma, hexokinase', 'TSH by chemiluminescent immunoassay, serum'"
    },
    {
        "name": "alternate_terminology",
        "system": "You are a medical coder familiar with regional and international terminology variations.",
        "instruction": "Write 5 descriptions using British / European spellings, older nomenclature still in use, or regional variants. Stay in English.",
        "temperature": 0.7,
        "examples": "Examples: 'Leucocyte count' (for WBC), 'Glycated haemoglobin' (for HbA1c), 'Haematocrit' (for HCT), 'Serum urea' (for BUN)"
    },
    {
        "name": "ehr_dropdown",
        "system": "You are designing dropdown entries for an EMR order-entry screen. Real examples from production hospital systems: 'Glucose, CSF', 'Cholesterol, LDL, Calculated', 'Testosterone, Free', 'Amylase, Joint Fluid', 'Iron Binding Capacity, Total'.",
        "instruction": "List 5 test names in the comma-separated postfix-qualifier style shown above: base analyte first, then qualifiers (specimen, calculation, fraction, method) after commas. Keep each under 50 characters.",
        "temperature": 0.7,
        "examples": "Examples: 'Glucose, CSF', 'Cholesterol, LDL, Calculated', 'Protein, Urine, Random', 'Hemoglobin, Whole Blood'"
    },
    {
        "name": "mixed_format",
        "system": "You are compiling lab descriptions from various sources and systems.",
        "instruction": "Write 5 descriptions with varied formats: one with an abbreviation in parentheses, one with a postfix specimen qualifier, one with a measurement unit, and free choice for the others.",
        "temperature": 0.7,
        "examples": "Examples: 'Alanine Aminotransferase (ALT)', 'Estimated GFR (MDRD equation)', 'Hemoglobin, whole blood', 'Glucose mg/dL'"
    },
    {
        "name": "common_variations",
        "system": "You are reviewing actual clinical orders from various healthcare providers.",
        "instruction": "List 5 test names the way busy clinicians actually write them in real orders — including panel nicknames, shortcut names, and common informal shorthand.",
        "temperature": 0.7,
        "examples": "Examples: 'CMP14', 'Chem 7', 'H&H', 'BUN/Cr', 'Troponin I', 'TSH w/ reflex'"
    },
    {
        "name": "department_specific",
        "system": "You are labeling tests with the urgency and context prefix conventions used on EHR order screens.",
        "instruction": "List 5 variants of the SAME test, each prefixed with a real-world urgency or context marker (e.g. STAT, Routine, POC, AM Draw, Add-on). Do NOT name a different test — vary only the prefix while keeping the underlying test identical to what the prompt describes.",
        "temperature": 0.7,
        "examples": "Examples: 'STAT Glucose', 'Routine CBC', 'POC Lactate', 'AM Draw TSH', 'Add-on Magnesium'"
    },
    {
        "name": "legacy_systems",
        "system": "You are working with test names exported from older laboratory information systems.",
        "instruction": "List 5 test names using legacy LIS conventions: truncated uppercase forms, short fixed-width codes, and older abbreviation styles. Keep each under 20 characters.",
        "temperature": 0.7,
        "examples": "Examples: 'GLUC', 'HGB A1C', 'CBC DIFF', 'NA SERUM', 'TSH QN'"
    },
]

# %% [cell 6]
def load_enriched_records(path):
    """Load the enriched descriptions JSONL as a dict keyed by loinc_code.

    Raises ValueError on duplicate codes, so a repeated row in the JSONL
    gets noticed.
    """
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            code = record["loinc_code"]
            if code in records:
                raise ValueError(
                    f"Duplicate loinc_code {code} in {path} at line {line_num}"
                )
            records[code] = record
    return records


def _normalize_text(s):
    """Case-insensitive, whitespace-normalized key for dedup / strip.

    Python 3's str.split() with no sep collapses all Unicode whitespace
    (including NBSP), so this also handles non-ASCII whitespace.
    """
    return " ".join(s.split()).lower()


def format_metadata_block(metadata):
    """Render the 6 LOINC axes as a bulleted block for the LLM prompt.

    The axis labels are in title case. This reads more naturally than ALL
    CAPS and matches the style of the rest of the prompt. MedGemma
    recognises the LOINC-canonical abbreviations of axis values (MCnc,
    Ser/Plas, Qn, ...), so they are kept verbatim from the JSONL.
    """
    fields = [
        ("Component",   "loinc_component"),
        ("Property",    "loinc_property"),
        ("Time aspect", "loinc_time_aspect"),
        ("System",      "loinc_system"),
        ("Scale type",  "loinc_scale_type"),
        ("Method type", "loinc_method_type"),
    ]
    lines = []
    for label, key in fields:
        val = (metadata.get(key) or "").strip()
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "Not specified"


def format_descriptions_block(descriptions, vocab_order=VOCAB_ORDER,
                              vocab_labels=VOCAB_LABELS):
    """Render per-vocab descriptions for the LLM prompt, grouped by vocab.

    Emits one header per vocab (e.g. 'Official LOINC naming:') followed
    by bulleted entries. Cross-vocab duplicates are dropped. The vocabs
    are walked in `vocab_order`, and any description that normalizes to
    the same key as one in an earlier vocab is skipped. With LOINC first,
    duplicates in MTH / SNOMED / CDISC / CIEL / MIMIC are filtered out. A
    vocab whose entries are all duplicates gets no header.

    A header per vocab, in place of inline '[Label] text' prefixes, stops
    MedGemma from echoing vocab labels in its generated paraphrases.
    """
    seen = set()
    groups = []
    for vocab in vocab_order:
        descs = descriptions.get(vocab) or []
        label = vocab_labels.get(vocab, vocab.upper())
        lines = []
        for d in descs:
            key = _normalize_text(d)
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(f"- {d}")
        if lines:
            groups.append(f"{label}:\n" + "\n".join(lines))
    return "\n\n".join(groups) if groups else "Not specified"


def _default_checkpoint():
    """Fresh checkpoint structure used when UPSTREAM_DIR/4.2_checkpoint.json
    doesn't exist yet. Sub-progress lists are only tracked for the
    long-running stages (generation, verify). Clean and finalize are fast,
    so a single `done` flag is enough for them.
    """
    return {
        "generation": {
            "done": False,
            "completed_pairs": [],   # list of "{seed}__{strategy_name}" strings
        },
        "clean":    {"done": False},
        "verify": {
            "done": False,
            "completed_files": [],   # list of "{loinc_code}.txt" filenames
        },
        "finalize": {"done": False},
    }


def load_checkpoint(path=CHECKPOINT_PATH):
    """Load the per-stage checkpoint.

    Returns a dict with the default structure if the file doesn't exist
    (without creating the file). Missing keys in an existing checkpoint
    are filled in from the default, so adding new stages to this file
    later won't break old checkpoints. The file is only written when
    save_checkpoint() is called with real progress.
    """
    default = _default_checkpoint()
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    for stage, stage_default in default.items():
        if stage not in ckpt:
            ckpt[stage] = stage_default
        elif isinstance(stage_default, dict):
            for k, v in stage_default.items():
                if k not in ckpt[stage]:
                    ckpt[stage][k] = v
    return ckpt


def save_checkpoint(ckpt, path=CHECKPOINT_PATH):
    """Atomic checkpoint write: json.dump to a temp file then os.replace.
    A crash mid-write can't leave a truncated checkpoint on disk.
    Creates the parent directory if it doesn't exist.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _pick_best_gpu():
    """Return the CUDA device index with the most free memory, or None
    if CUDA is unavailable.

    The MedGemma and Llama loaders use it to pin the model to a single
    GPU with device_map={"": f"cuda:{idx}"}. On multi-GPU machines this
    works better than device_map="auto". Both models fit on one 24GB card
    (MedGemma-4B bf16 ~8GB, Llama-3.1-8B 4-bit ~6GB), so sharding only
    adds overhead. device_map="auto" also installs an accelerate
    forward-dispatch hook that gets wrapped by torch.compile. Each
    generate() call creates a fresh DynamicCache whose obj_id fails the
    dynamo guards and forces a recompile. At the default cache_size_limit
    of 8 this crashes partway through generation.

    The GPU is picked when the model is loaded. If another user has taken
    GPU 0, the card with more free memory at that moment is used.
    """
    if not torch.cuda.is_available():
        return None
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i
    print(f"Selected cuda:{best_idx} ({best_free / (1024**3):.1f} GB free)")
    return best_idx

# %% [cell 7]
def save_paraphrases(paraphrases, output_dir, check_existing=True):
    """Save paraphrases to individual files, one file per LOINC code.

    Args:
        paraphrases: Dictionary with LOINC codes as keys and list of
            paraphrases as values.
        output_dir: Directory to save the files (e.g. UPSTREAM_DIR/paraphrases_ext).
        check_existing: If True, append unique paraphrases to existing
            files. If False, overwrite.

    Returns:
        Dictionary with statistics about files created/updated.
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "files_created": 0,
        "files_updated": 0,
        "total_paraphrases_written": 0,
        "duplicates_skipped": 0,
    }

    for loinc_code, paraphrase_list in paraphrases.items():
        safe_filename = loinc_code.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(output_dir, f"{safe_filename}.txt")

        # Normalized comparison set: case/whitespace-insensitive so
        # "Glucose, serum" and "glucose,  serum" collapse to one entry.
        # The file itself is still written with original casing.
        existing_normalized = set()
        if check_existing and os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_normalized.add(_normalize_text(line))
            except Exception as e:
                print(f"Warning: could not read existing file {filepath}: {e}")

        if check_existing:
            new_paraphrases = []
            for paraphrase in paraphrase_list:
                paraphrase = paraphrase.strip()
                if not paraphrase:
                    continue
                norm = _normalize_text(paraphrase)
                if norm not in existing_normalized:
                    new_paraphrases.append(paraphrase)
                    existing_normalized.add(norm)
                else:
                    stats["duplicates_skipped"] += 1

            if new_paraphrases:
                mode = "a" if os.path.exists(filepath) else "w"
                with open(filepath, mode, encoding="utf-8") as f:
                    for paraphrase in new_paraphrases:
                        f.write(paraphrase + "\n")

                stats["total_paraphrases_written"] += len(new_paraphrases)
                if mode == "a":
                    stats["files_updated"] += 1
                else:
                    stats["files_created"] += 1
        else:
            unique_paraphrases = []
            seen_normalized = set()
            for paraphrase in paraphrase_list:
                paraphrase = paraphrase.strip()
                if not paraphrase:
                    continue
                norm = _normalize_text(paraphrase)
                if norm not in seen_normalized:
                    unique_paraphrases.append(paraphrase)
                    seen_normalized.add(norm)

            with open(filepath, "w", encoding="utf-8") as f:
                for paraphrase in unique_paraphrases:
                    f.write(paraphrase + "\n")

            stats["files_created"] += 1
            stats["total_paraphrases_written"] += len(unique_paraphrases)

    print("=" * 60)
    print("SAVE SUMMARY:")
    print(f"Files created: {stats['files_created']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Total paraphrases written: {stats['total_paraphrases_written']}")
    if check_existing:
        print(f"Duplicates skipped: {stats['duplicates_skipped']}")
    print("=" * 60)

    return stats

# %% [cell 8]
def build_generation_user_content(record, strategy):
    """Build the user-turn content for one (record, strategy) pair.

    This is the text that goes inside a single Gemma user turn. It
    contains (in order):
      1. the strategy persona (its "system" text)
      2. the LOINC axes block
      3. the known names and descriptions block
      4. the strategy's instruction and examples
      5. a single consolidated output-format directive

    MedGemma's canonical chat template folds system messages into the
    first user turn anyway, so no separate system turn is built.
    Everything goes into one user message, and
    tokenizer.apply_chat_template wraps it with the correct turn tokens,
    BOS and generation prompt.
    """
    metadata_text = format_metadata_block(record["metadata"])
    descriptions_text = format_descriptions_block(record["descriptions"])

    return (
        f"{strategy['system']}\n\n"
        f"LOINC axes:\n{metadata_text}\n\n"
        f"Known names and descriptions:\n{descriptions_text}\n\n"
        f"{strategy['instruction']}\n"
        f"{strategy['examples']}\n\n"
        "Output format: one test description per line, plain text only. "
        "Do NOT output code, field names (like 'description:'), markdown, "
        "numbering, bullets, explanations, or introductions. Nothing but "
        "the test descriptions."
    )


def generate_diverse_paraphrases(model, tokenizer, enriched_records,
                                 strategies=PROMPTING_STRATEGIES,
                                 batch_size=GEN_BATCH_SIZE,
                                 random_seed=None,
                                 output_dir=PARAPHRASE_DIR):
    """Generate diverse paraphrases using multiple prompting strategies.

    Saves after each strategy to avoid memory issues.

    Args:
        enriched_records: Dict keyed by LOINC code, values are records
            from UPSTREAM_DIR/loinc_descriptions_enriched.jsonl.
        strategies: List of prompting strategies to use.
        batch_size: Number of LOINC codes to process at once.
        random_seed: Random seed for reproducible generation (optional).
        output_dir: Directory to save paraphrases (saves after each
            strategy).

    Returns:
        Dictionary with statistics about generation and saving.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")

    total_stats = {
        "files_created": 0,
        "files_updated": 0,
        "total_paraphrases_written": 0,
        "duplicates_skipped": 0,
    }

    loinc_codes = list(enriched_records.keys())

    # Per-(seed, strategy) checkpoint. It is loaded once at the start of
    # this function. The on-disk copy is updated after each successful
    # strategy save, so an interrupted run can resume without redoing work.
    # random_seed==None means the caller opted out of seeding, and then
    # checkpointing is off too.
    _ckpt = load_checkpoint()
    _completed_pairs = set(_ckpt["generation"]["completed_pairs"])

    for strategy in strategies:
        _pair_key = f"{random_seed}__{strategy['name']}"
        if random_seed is not None and _pair_key in _completed_pairs:
            print(f"Checkpoint: skipping strategy '{strategy['name']}' "
                  f"(seed {random_seed} already complete)")
            continue

        print(f"Applying strategy: {strategy['name']}")

        strategy_results = {code: [] for code in loinc_codes}

        for i in tqdm(range(0, len(loinc_codes), batch_size)):
            batch_codes = loinc_codes[i:i + batch_size]

            # Build prompts with apply_chat_template so they match MedGemma's
            # canonical format (BOS, turn tokens, generation prompt). A
            # hand-built f-string with a <start_of_turn>system block would
            # not be canonical.
            prompts = []
            for loinc_code in batch_codes:
                record = enriched_records[loinc_code]
                content = build_generation_user_content(record, strategy)
                messages = [{"role": "user", "content": content}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            # add_special_tokens=False because apply_chat_template already
            # emitted the literal <bos> token. If the tokenizer added
            # another one, the input would start with two BOS tokens.
            # padding=True pads to the longest sequence in the current
            # batch. The average prompt is ~300 tokens and
            # GEN_MAX_INPUT_LEN=1024, so padding to max_length would
            # waste ~70% of compute on PAD tokens. truncation=True with
            # max_length still caps stray long prompts.
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=GEN_MAX_INPUT_LEN,
                add_special_tokens=False,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    temperature=strategy["temperature"],
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Repeating the test name on each of the 5 list items
                    # is correct, but a repetition_penalty discourages it
                    # and can push the model to name other tests
                    # (cross-test contamination).
                    # no_repeat_ngram_size=4 prevents verbatim 4-gram
                    # loops and does not penalise single-word reuse
                    # across lines.
                    no_repeat_ngram_size=4,
                )

            for j, output in enumerate(outputs):
                loinc_code = batch_codes[j]
                prompt_length = inputs["input_ids"][j].shape[0]
                generated = output[prompt_length:]
                response = tokenizer.decode(generated, skip_special_tokens=True)

                response = response.replace("<end_of_turn>", "").strip()
                lines = response.split("\n")

                for line in lines:
                    # Strip any stray '[Label] ' prefix that the model still
                    # emits despite the chat template and the grouped
                    # headers. This runs before the general strip of
                    # leading characters.
                    line = re.sub(r"^\s*\[[^\]]*\]\s*", "", line)
                    # Strip leading list markers / numbering / brackets.
                    line = re.sub(r"^[\d\.\-\*\•\)\]\}\>\s]+", "", line).strip()
                    # len(line) >= 2 because the abbreviation_heavy and
                    # legacy_systems strategies are meant to produce very
                    # short outputs ('GLUC', 'H&H', 'Lytes'). A higher minimum
                    # such as >10 would reject them. The cleaner (cell 10) and
                    # the verifier (cell 11) filter real garbage. This filter
                    # only drops empty and intro lines.
                    if line and len(line) >= 2 and not any(
                        skip in line.lower()
                        for skip in ["here are", "below are", "i'll generate"]
                    ):
                        strategy_results[loinc_code].append(line)

        # Dedup within this strategy.
        for loinc_code in strategy_results:
            seen = set()
            unique_paraphrases = []
            for p in strategy_results[loinc_code]:
                normalized = _normalize_text(p)
                if normalized not in seen:
                    seen.add(normalized)
                    unique_paraphrases.append(p)
            strategy_results[loinc_code] = unique_paraphrases

        print(f"Saving paraphrases for strategy: {strategy['name']}")
        strategy_stats = save_paraphrases(strategy_results, output_dir, check_existing=True)

        total_stats["files_created"] += strategy_stats["files_created"]
        total_stats["files_updated"] += strategy_stats["files_updated"]
        total_stats["total_paraphrases_written"] += strategy_stats["total_paraphrases_written"]
        total_stats["duplicates_skipped"] += strategy_stats["duplicates_skipped"]

        # Mark this (seed, strategy) pair as complete. The on-disk file is
        # re-read before appending so that hand-edits made between
        # strategies are kept.
        if random_seed is not None:
            _ckpt = load_checkpoint()
            if _pair_key not in _ckpt["generation"]["completed_pairs"]:
                _ckpt["generation"]["completed_pairs"].append(_pair_key)
                save_checkpoint(_ckpt)
            _completed_pairs.add(_pair_key)

        # Clean up between (seed, strategy) pairs to prevent a slow leak
        # in the CUDA private pools and static addresses. Cell 17 explains
        # the cause. The order matters:
        #   1. Drop any HybridCache still attached to the model. Forcing
        #      "dynamic" in cell 17 should stop it from being created, so
        #      this is a fallback.
        #   2. torch._dynamo.reset() flushes the compiled-graph cache,
        #      which would otherwise pin static-address tensors.
        #   3. Drop local refs, then run gc.collect() before
        #      empty_cache() so that unreferenced tensors return their
        #      memory to the CUDA allocator.
        if hasattr(model, "_cache"):
            model._cache = None
        torch._dynamo.reset()
        del strategy_results
        gc.collect()
        torch.cuda.empty_cache()

        # Log free GPU memory after each strategy. In an unattended run, a
        # leak then shows up as free memory that keeps shrinking.
        if torch.cuda.is_available():
            _dev = model.device
            _idx = _dev.index if _dev.index is not None else 0
            _free, _total = torch.cuda.mem_get_info(_idx)
            print(f"  GPU{_idx} after '{strategy['name']}': "
                  f"{_free / (1024**3):.2f} / {_total / (1024**3):.2f} GB free")

    return total_stats

# %% [cell 9]
def generate_paraphrases_multiple_seeds(model, tokenizer, enriched_records,
                                        seeds, strategies=PROMPTING_STRATEGIES,
                                        batch_size=GEN_BATCH_SIZE,
                                        output_dir=PARAPHRASE_DIR):
    """Generate diverse paraphrases using multiple random seeds."""
    combined_stats = {
        "files_created": 0,
        "files_updated": 0,
        "total_paraphrases_written": 0,
        "duplicates_skipped": 0,
    }

    print(f"Generating paraphrases with {len(seeds)} different random seeds...")

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 60}")
        print(f"SEED {seed_idx}/{len(seeds)}: {seed}")
        print(f"{'=' * 60}")

        seed_stats = generate_diverse_paraphrases(
            model, tokenizer, enriched_records,
            strategies=strategies,
            batch_size=batch_size,
            random_seed=seed,
            output_dir=output_dir,
        )

        for k in combined_stats:
            combined_stats[k] += seed_stats[k]

        print(f"Completed seed {seed}. Total paraphrases written so far: "
              f"{combined_stats['total_paraphrases_written']}")

    print(f"\n{'=' * 60}")
    print("ALL SEEDS COMPLETED")
    print(f"Final statistics: {combined_stats}")
    print(f"{'=' * 60}")

    return combined_stats

# %% [cell 10]
# Precompiled once at cell execution. clean_paraphrases below applies the
# patterns and rejects a line that matches any of them.
#
# Design notes:
# - The cleaner favours precision over recall. An LLM verification layer
#   runs after it, so only lines that are clearly not descriptions are
#   rejected (code, markdown, field-name echoes, model preambles). A valid
#   LOINC test name rejected here is lost, while noise that gets through can
#   still be caught by the verifier.
# - `:` is the main field-name catcher. 0 of the 980 MIMIC Observation.display
#   strings contain a colon, so rejecting any line with one is safe. The
#   filter covers Test:, Description:, Component:, Property:, SYSTEM:,
#   METHOD:, Label:, Value:, Rules:, Here are:, Below are:, etc.
# - `**` catches markdown bold the model sometimes emits around field names.
# - Uppercase-only schema patterns (e.g. COMPONENT) are a backup for rare
#   cases where the model drops the trailing `:`. They are case-sensitive so
#   English prose ("component of the test", "complete blood count") passes.
_REMOVAL_PATTERNS = [
    # --- Structural / code / formatting anywhere in the line ---
    re.compile(r":"),                               # any colon -> field-name echo
    re.compile(r"\*\*"),                            # markdown bold
    re.compile(r"""=\s*[\[\{"']"""),                # x = [...] / {...} / "..."
    re.compile(r"\{"),                              # any { -> dict/f-string/template
    re.compile(r"\[\]"),                            # empty list literal
    re.compile(r"\($"),                             # trailing (
    re.compile(r"\[$"),                             # trailing [
    re.compile(r"</?\w+\b[^>]*>"),                  # HTML/XML tag
    re.compile(r"--"),                              # double dash
    re.compile("\u0142"),                           # stray Polish L (model artifact)

    # --- Line-start markers: Python/pseudocode, markdown, lists ---
    re.compile(r"^\s*def\s", re.I),
    re.compile(r"^\s*if\s", re.I),
    re.compile(r"^\s*import\s+\w+", re.I),
    re.compile(r"^\s*from\s+\w+\s+import\b", re.I),
    re.compile(r"^\s*return\b", re.I),
    re.compile(r"^\s*print\s*\(", re.I),
    re.compile(r"^\s*#"),                           # python comment / md header
    re.compile(r'^\s*"""'),                         # docstring open
    re.compile(r"^\s*```"),                         # markdown code fence
    re.compile(r"^\s*\{\{"),                        # template placeholder
    re.compile(r"^\s*\("),                          # starts with (
    re.compile(r"^\s*[-*]\s+"),                     # bullet list
    re.compile(r"^\s*\d+[.)]\s+"),                  # numbered list
    re.compile(r"^\s*%(pip|matplotlib|load|run|config|env|magic)\b", re.I),
    # A lone field word on its own line (a fallback, since `:` catches most of these)
    re.compile(
        r'^\s*"?(test|example|attribute|tool|title|name|description|core|label|value|text|rules)\s*$',
        re.I,
    ),

    # --- LOINC schema field names in UPPERCASE (no re.I to avoid English prose) ---
    re.compile(r"\bCOMPONENTS?\b"),
    re.compile(r"\bPROPERTY\b"),
    re.compile(r"\bPROPERTIES\b"),
    re.compile(r"\bTIME[ _]?ASPC?T?\b"),
    re.compile(r"\bSCALE[ _]?TYPE?\b"),
    re.compile(r"\bLONG[ _]?COMMON[ _]?NAME\b"),
    re.compile(r"\bSHORT[ _]?NAMES?\b"),
    re.compile(r"\bRELATED[ _]?NAMES?\b"),
    re.compile(r"\bCURRENT[ _]?NAMES?\b"),
    re.compile(r"\bDISPLAY[ _]?NAMES?\b"),
    re.compile(r"\bCORE[ _]?ATTRIBUTES?\b"),
    re.compile(r"\bDispaly\b", re.I),               # historical typo of "Display"

    # --- Model preambles and disclaimers ---
    re.compile(r"^\s*I am\b"),                      # conversational line start
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi('m| am) (sorry|unable|an? (ai|language))\b", re.I),
    re.compile(r"\bi cannot\b", re.I),
    re.compile(r"\bi'?ll generate\b", re.I),
    re.compile(r"\bthank you\b", re.I),
    re.compile(r"\byou'?re welcome\b", re.I),

    # --- Model meta-commentary / self-talk ---
    # MedGemma's chat fine-tuning sometimes leaks through. The model then
    # evaluates its own output or echoes the task instructions in place of
    # test descriptions. These patterns gave 0 real false positives on the
    # 115K lines of an earlier cleaned corpus and 0 on LOINC/MIMIC ground
    # truth.
    re.compile(
        r"^The (descriptions|output|test|provided|generated|list|responses|"
        r"measurement|specimen|component|time aspect|LOINC|MIMIC|CDISC|"
        r"fifth|fourth|third|second|first|above)", re.I
    ),
    re.compile(
        r"(plausible|well-formed|well-written|requested format|"
        r"meet all|all the specified|is complete and|follow the prompt|"
        r"following the|the constraints)", re.I
    ),
    # Exact-match meta words that appear in 100+ LOINC files as
    # cross-cutting noise (e.g. "annotation" in 238 files, "output"
    # in 237). Case-insensitive + anchored to avoid substring hits.
    re.compile(
        r"^(annotation|output|model|Display|enumerate|code|"
        r"test names|test descriptions|Test Name|Test Description|"
        r"descriptions provided|descriptions are provided|"
        r"example output|example_output)$", re.I
    ),
    re.compile(r"^Note (that|also)\b", re.I),

    # --- LOINC field-name prefixes without colons ---
    # MedGemma sometimes uses LOINC field names as line prefixes
    # (e.g. "Display Albumin", "Description Cholesterol", "Name CD15").
    # These patterns have 0 hits in the earlier cleaned corpus, the LOINC
    # ground truth or the MIMIC ground truth.
    re.compile(r"^Display\s", re.I),
    re.compile(r"^Descriptions?\s", re.I),
    re.compile(r"^Name\s", re.I),

    # --- Filenames / paths ---
    # 0 LOINC/MIMIC ground-truth names end in a file extension.
    re.compile(r"\.(txt|py|csv|json|md|html|xml|ipynb)$", re.I),

    # --- Formatting artifacts ---
    re.compile(r"^[\\/_\-\]\[}{|`\s*]+$"),   # lines of pure special chars
    re.compile(r"^(g\s*)+$"),                      # gibberish g / g g / g g g
    re.compile(r"^[\]\[)(}{]+$"),                 # lone brackets
    re.compile(r"^\\[\\`_\-\]\[n]"),         # escaped-backslash lead
    re.compile(r"^title\s", re.I),                 # "title Sodium ..." prefix

    # --- Non-Latin script (multilingual pretraining bleed) ---
    # All hits in the earlier cleaned corpus (10 Chinese, 3 Cyrillic) are
    # model self-talk or Unicode confusable homoglyphs, so rejecting them
    # loses no test name.
    re.compile(r"[\u4e00-\u9fff]"),               # CJK Unified Ideographs
    re.compile(r"[\u0400-\u04ff]"),               # Cyrillic

    # --- Lone ordering prefixes without a test name ---
    # "AM Draw", "STAT", etc. are valid prefixes in real names like
    # "STAT Glucose" but appear alone in 60–105 files as truncated
    # generator output. Anchored to exact-match only.
    re.compile(r"^(AM Draw|Routine|POC|STAT|Add-on|Add-On|CDISC LB)$", re.I),
]


def clean_paraphrases(input_dir, output_dir=PARAPHRASE_CLEAN_DIR):
    """Clean generated paraphrases by removing noise and applying
    formatting rules.

    An LLM verification layer runs afterwards and does the finer filtering.
    This layer therefore prefers precision over recall. It only rejects
    lines that are clearly not natural-language LOINC test descriptions.

    Returns a dictionary with cleaning statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "files_processed": 0,
        "total_lines_input": 0,
        "total_lines_output": 0,
        "lines_removed": 0,
    }

    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Processing {len(input_files)} files from {input_dir}")

    for input_filepath in tqdm(input_files):
        filename = os.path.basename(input_filepath)
        output_filepath = os.path.join(output_dir, filename)

        try:
            with open(input_filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            original_count = len(lines)
            stats["total_lines_input"] += original_count

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Strip outer quotes and punctuation before pattern matching,
                # so that e.g. 'annotation.' becomes 'annotation' and the
                # exact-match meta-word pattern catches it.
                line = line.strip("\"'`,.!?")

                if not line:
                    continue

                if any(rx.search(line) for rx in _REMOVAL_PATTERNS):
                    continue

                if len(line) > MAX_PARAPHRASE_LEN:
                    continue

                cleaned_lines.append(line.strip())

            unique_lines = list(set(cleaned_lines))
            unique_lines.sort(key=str.lower)

            with open(output_filepath, "w", encoding="utf-8") as f:
                for line in unique_lines:
                    f.write(line + "\n")

            final_count = len(unique_lines)
            removed_count = original_count - final_count

            stats["files_processed"] += 1
            stats["total_lines_output"] += final_count
            stats["lines_removed"] += removed_count

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total input lines: {stats['total_lines_input']}")
    print(f"Total output lines: {stats['total_lines_output']}")
    print(f"Lines removed: {stats['lines_removed']}")
    if stats["total_lines_input"] > 0:
        print(f"Removal percentage: {(stats['lines_removed'] / stats['total_lines_input']) * 100:.1f}%")
    print("=" * 60)

    return stats

# %% [cell 11]
VERIFY_SYSTEM_PROMPT = """\
You are a quality filter for lab-test descriptions. For each candidate you
receive:
- The ground-truth LOINC axes of a specific lab test
- Several known correct names for that test (the reference)
- A candidate description to evaluate

Your job is to decide whether the candidate is consistent with the reference
using SURFACE-LEVEL comparison. You do NOT need deep medical knowledge. Only
reject cases where the mismatch is obvious from the text alone.

REJECT only if ONE of the following is clearly true:

1. GIBBERISH: candidate is random characters or non-words
   (e.g. "gvgvgbhddjd", "test123abc")

2. NOT A LAB TEST: candidate names something that is clearly not a laboratory
   measurement (e.g. "Chest X-Ray", "patient intake form",
   "gastroenteritis symptoms", "ophthalmology visit")

3. SPECIMEN CONTRADICTION: candidate names a specimen that clearly conflicts
   with the reference names. E.g. candidate says "in urine" but every
   reference name says serum / plasma; candidate says "whole blood" but
   reference says CSF.

4. COMPONENT CONTRADICTION: candidate names a completely different substance
   from the reference. E.g. candidate says "Potassium" but every reference
   name is about "Chloride"; candidate says "Glucose" but reference is about
   "Hemoglobin A1c".

5. TOO VAGUE: candidate is too generic to identify any specific test
   (e.g. "lab result", "blood test", "a chemistry panel", "analysis")

6. META-COMMENTARY: candidate is the model describing itself, the task,
   the output format, or the quality of the output rather than naming a
   specific lab test.
   (e.g. "The descriptions are well-formed", "example output",
    "Display 5 results", "The test descriptions are plausible",
    "Note that PrThr is a Property", "annotation", "test names")

ACCEPT everything else. In particular, ALWAYS ACCEPT:

- Standard medical abbreviations (CBC, CMP, CMP14, Chem 7, H&H, HbA1c,
  PT/INR, BUN/Cr, TSH, LFT, BMP, UA, Lytes)
- Legacy LIS codes with underscores or all-caps (IMMAT_LEUKO_WB,
  KTONE_MASS_VOL, HGB_A1C, NA_SERUM, TSH_QN, GLUC)
- Postfix qualifier style ("Glucose, CSF", "Cholesterol, LDL, Calculated",
  "Testosterone, Free")
- British / European spellings (leucocyte, haemoglobin, haematocrit)
- Informal shorthand ("lytes", "chem panel", "trop", "CBC w/ diff")
- Verbose formal descriptions with spelled-out medical terms
- Anything you are uncertain about - when in doubt, ACCEPT

You are a first-pass filter. Another process handles deeper review. Your
value is rejecting obviously wrong cases without rejecting anything you
cannot verify from the text alone.

EXAMPLES:

Reference names: "Chloride [Moles/volume] in Serum or Plasma",
                 "Chloride SerPl-sCnc", "Chloride, Plasma"
Candidate: "Cl- level, plasma"
Answer: true
(Cl- is the standard abbreviation for Chloride; specimen matches.)

Reference names: "Chloride [Moles/volume] in Serum or Plasma"
Candidate: "Potassium level, serum"
Answer: false
(Candidate names Potassium; all reference names say Chloride. Clear
component mismatch.)

Reference names: "Chloride [Moles/volume] in Serum or Plasma"
Candidate: "Chloride in urine, 24-hour collection"
Answer: false
(Candidate says urine; all reference names say serum/plasma. Clear
specimen mismatch.)

Reference names: "Glucose [Mass/volume] in Serum or Plasma",
                 "Glucose SerPl-mCnc"
Candidate: "GLUC_SER_MASS_VOL"
Answer: true
(Legacy LIS shorthand; parses to Glucose, Serum, Mass per Volume,
consistent with reference.)

Reference names: any lab test
Candidate: "gvgvgbhddjdkdjd"
Answer: false
(Gibberish, not real words.)

Reference names: any lab test
Candidate: "Chest X-Ray"
Answer: false
(Imaging study, not a lab test.)

Answer with ONLY the word "true" or "false". No punctuation, no explanation.\
"""


def build_verification_user_content(metadata_text, descriptions_text, paraphrase):
    """Build the user-turn content for one verification call.

    The LOINC code string is left out because Llama has no memorized
    LOINC mapping, and the code would only add noise to the prompt. The
    metadata block and the reference names are the ground truth.
    """
    return (
        f"Ground truth LOINC axes for the target test:\n{metadata_text}\n\n"
        f"Reference names for this test:\n{descriptions_text}\n\n"
        f"Candidate description:\n{paraphrase}\n\n"
        "Is the candidate consistent with the reference above by the "
        "surface-level criteria in your instructions?\n\n"
        "Answer (true/false only):"
    )


def _parse_verification_response(response):
    """Parse a verifier model response into (accepted, first_word).

    accepted is True (accept) or False (reject). first_word is the first
    alphabetic word, "empty" for an empty response, or the first 20
    characters when there is no alphabetic word. Checking the first word
    handles responses like 'true', 'True.', 'yes' and 'Yes, this is valid'.
    Anything unrecognised gives False, so rejection stays the conservative
    default.
    """
    stripped = response.strip().lower()
    if not stripped:
        return False, "empty"
    # Pull the first alphabetic run.
    m = re.match(r"[a-z]+", stripped)
    if not m:
        return False, stripped[:20]
    first = m.group(0)
    if first in ("true", "yes", "y", "accept", "valid"):
        return True, first
    if first in ("false", "no", "n", "reject", "invalid"):
        return False, first
    return False, first


def verify_paraphrases(model, tokenizer, enriched_records,
                       input_dir=PARAPHRASE_CLEAN_DIR,
                       output_dir=PARAPHRASE_VERIFIED_DIR,
                       rejected_dir=PARAPHRASE_REJECTED_DIR,
                       batch_size=VERIFY_BATCH_SIZE):
    """Verify that cleaned paraphrases keep the meaning of the original
    LOINC code.

    Uses Llama's apply_chat_template with a system message and a user
    message, because Llama 3 supports a system role natively. Generation
    is greedy (do_sample=False) for reproducibility and speed. Response
    parsing reads the first alphabetic word. Substring matching on
    "true" / "false" would give false negatives on responses like "yes".

    The system prompt limits verification to surface-level checks
    (gibberish, off-topic, specimen/component contradiction, extreme
    vagueness) because Llama 3.1 8B lacks deep medical knowledge. Subtle
    wrong-axis errors (wrong cell subtype, invalid method, scale mismatch)
    are not checked here. They rely on the generator's accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    stats = {
        "files_processed": 0,
        "total_paraphrases_input": 0,
        "total_paraphrases_verified": 0,
        "total_paraphrases_rejected": 0,
        "verification_rate": 0.0,
    }

    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Verifying paraphrases for {len(input_files)} LOINC codes...")

    # Per-file checkpoint. It is loaded once at the start, and the on-disk
    # copy is updated after each successful file write. The set is kept in
    # sync with the in-memory ckpt, so re-entering the loop in the same cell
    # execution still skips completed files. That does not happen in
    # practice, but it is cheap to support.
    _ckpt = load_checkpoint()
    _completed_files = set(_ckpt["verify"]["completed_files"])
    if _completed_files:
        print(f"Checkpoint: {len(_completed_files)} file(s) already verified, resuming")

    for input_filepath in tqdm(input_files):
        filename = os.path.basename(input_filepath)

        if filename in _completed_files:
            continue
        # LOINC codes have no "/" in practice, so this reverse mapping
        # is a no-op, kept symmetrical with save_paraphrases.
        loinc_code = filename.replace(".txt", "").replace("_", "/")

        if loinc_code not in enriched_records:
            print(f"Warning: LOINC code {loinc_code} not in enriched records, skipping")
            continue

        try:
            with open(input_filepath, "r", encoding="utf-8") as f:
                paraphrases = [line.strip() for line in f if line.strip()]

            if not paraphrases:
                continue

            stats["total_paraphrases_input"] += len(paraphrases)

            record = enriched_records[loinc_code]
            metadata_text = format_metadata_block(record["metadata"])
            descriptions_text = format_descriptions_block(record["descriptions"])

            verified_paraphrases = []
            rejected_paraphrases = []

            for i in range(0, len(paraphrases), batch_size):
                batch_paraphrases = paraphrases[i:i + batch_size]

                prompts = []
                for paraphrase in batch_paraphrases:
                    user_content = build_verification_user_content(
                        metadata_text, descriptions_text, paraphrase,
                    )
                    messages = [
                        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(prompt)

                # add_special_tokens=False because apply_chat_template
                # already emitted the literal BOS token.
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=VERIFY_MAX_INPUT_LEN,
                    add_special_tokens=False,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=VERIFY_MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for j, output in enumerate(outputs):
                    prompt_length = inputs["input_ids"][j].shape[0]
                    generated = output[prompt_length:]
                    response = tokenizer.decode(generated, skip_special_tokens=True)

                    is_valid, first_word = _parse_verification_response(response)
                    if is_valid:
                        verified_paraphrases.append(batch_paraphrases[j])
                        stats["total_paraphrases_verified"] += 1
                    else:
                        rejected_paraphrases.append(batch_paraphrases[j])
                        stats["total_paraphrases_rejected"] += 1
                        if first_word not in ("false", "no", "n", "reject", "invalid"):
                            print(
                                "Unexpected verifier response "
                                f"(first_word={first_word!r}): {response!r}"
                            )

                torch.cuda.empty_cache()

            if verified_paraphrases:
                output_filepath = os.path.join(output_dir, filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    for paraphrase in verified_paraphrases:
                        f.write(paraphrase + "\n")

            if rejected_paraphrases:
                rejected_filepath = os.path.join(rejected_dir, filename)
                with open(rejected_filepath, "w", encoding="utf-8") as f:
                    for paraphrase in rejected_paraphrases:
                        f.write(paraphrase + "\n")

            stats["files_processed"] += 1

            # Mark the file done in the checkpoint so an interrupted run can
            # skip it on resume. The checkpoint is reloaded before appending
            # so that hand-edits are kept.
            _ckpt = load_checkpoint()
            if filename not in _ckpt["verify"]["completed_files"]:
                _ckpt["verify"]["completed_files"].append(filename)
                save_checkpoint(_ckpt)
            _completed_files.add(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if stats["total_paraphrases_input"] > 0:
        stats["verification_rate"] = (
            stats["total_paraphrases_verified"] / stats["total_paraphrases_input"]
        ) * 100

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total paraphrases input: {stats['total_paraphrases_input']}")
    print(f"Total paraphrases verified: {stats['total_paraphrases_verified']}")
    print(f"Total paraphrases rejected: {stats['total_paraphrases_rejected']}")
    print(f"Verification rate: {stats['verification_rate']:.1f}%")
    print("=" * 60)

    return stats

# %% [cell 12]
def finalize_paraphrases(enriched_records,
                         verified_dir=PARAPHRASE_VERIFIED_DIR,
                         non_loinc_vocabs=NON_LOINC_VOCABS):
    """Merge non-LOINC vocab descriptions into each code's verified file
    and strip any line that matches the code's LOINC bucket.

    For each code in `enriched_records`:
      1. Read verified_dir/<code>.txt if it exists (may not exist if
         verification wrote nothing for the code).
      2. Extend with descriptions from each vocab in `non_loinc_vocabs`
         in order.
      3. Build loinc_strip_set = { _normalize_text(d)
                                   for d in descriptions["loinc"] }.
      4. Walk the merged list; keep entries whose _normalize_text is not
         in loinc_strip_set and not in a running seen set. Preserve
         order (LLM paraphrases first, then vocab descriptions in
         non_loinc_vocabs order).
      5. Overwrite verified_dir/<code>.txt with the result.

    It is idempotent. Re-running it produces identical output.

    Returns stats dict with counts of what was kept / stripped.
    """
    os.makedirs(verified_dir, exist_ok=True)

    stats = {
        "files_written": 0,
        "files_skipped_empty": 0,
        "llm_kept": 0,
        "llm_stripped_as_loinc": 0,
        "vocab_added": 0,
        "vocab_stripped_as_loinc": 0,
        "vocab_stripped_as_duplicate": 0,
    }

    for loinc_code, record in tqdm(enriched_records.items(), desc="Finalize"):
        safe_filename = loinc_code.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(verified_dir, f"{safe_filename}.txt")

        # 1. LLM paraphrases (if any).
        llm_lines = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                llm_lines = [line.strip() for line in f if line.strip()]

        # 2. Non-LOINC vocab descriptions in order.
        vocab_lines = []
        for vocab in non_loinc_vocabs:
            for d in record["descriptions"].get(vocab) or []:
                d = d.strip()
                if d:
                    vocab_lines.append(d)

        # 3. LOINC strip set (per-code, from the JSONL loinc bucket only).
        loinc_strip_set = {
            _normalize_text(d)
            for d in (record["descriptions"].get("loinc") or [])
            if d.strip()
        }

        # 4. Merge with LLM first, then vocab; dedup and strip.
        seen = set()
        merged = []
        for line in llm_lines:
            key = _normalize_text(line)
            if not key:
                continue
            if key in loinc_strip_set:
                stats["llm_stripped_as_loinc"] += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append(line)
            stats["llm_kept"] += 1

        for line in vocab_lines:
            key = _normalize_text(line)
            if not key:
                continue
            if key in loinc_strip_set:
                stats["vocab_stripped_as_loinc"] += 1
                continue
            if key in seen:
                stats["vocab_stripped_as_duplicate"] += 1
                continue
            seen.add(key)
            merged.append(line)
            stats["vocab_added"] += 1

        # 5. Write back (in place). Empty files are still written so
        # downstream tools can iterate the full code set, but only if
        # there was ever any content to consider. If both llm_lines and
        # vocab_lines were empty before stripping, skip writing.
        if not llm_lines and not vocab_lines:
            stats["files_skipped_empty"] += 1
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            for line in merged:
                f.write(line + "\n")
        stats["files_written"] += 1

    print("\n" + "=" * 60)
    print("FINALIZE SUMMARY:")
    print(f"Files written:                   {stats['files_written']}")
    print(f"Files skipped (nothing to write): {stats['files_skipped_empty']}")
    print(f"LLM paraphrases kept:            {stats['llm_kept']}")
    print(f"LLM paraphrases stripped (LOINC collision): {stats['llm_stripped_as_loinc']}")
    print(f"Vocab descriptions added:        {stats['vocab_added']}")
    print(f"Vocab descriptions stripped (LOINC collision): {stats['vocab_stripped_as_loinc']}")
    print(f"Vocab descriptions stripped (duplicate of LLM or earlier vocab): {stats['vocab_stripped_as_duplicate']}")
    print("=" * 60)

    return stats

# %% [cell 15]
enriched_records = load_enriched_records(ENRICHED_DESC_PATH)
print(f"Loaded {len(enriched_records)} enriched records")

# Peek at a rich example so the prompt rendering is visible at a glance.
example_code = "11572-5"  # Rheumatoid fact SerPl-aCnc
r = enriched_records[example_code]
print(f"\nExample prompt blocks for {example_code}:")
print("--- metadata ---")
print(format_metadata_block(r["metadata"]))
print("--- descriptions ---")
print(format_descriptions_block(r["descriptions"]))

# %% [cell 17]
_ckpt = load_checkpoint()
if _ckpt["generation"]["done"]:
    print("Checkpoint: generation already done, skipping MedGemma load and generation.")
    print("(Flip generation.done to false in UPSTREAM_DIR/4.2_checkpoint.json to force rerun.)")
else:
    gen_tokenizer = AutoTokenizer.from_pretrained(
        GEN_MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    # Decoder-only generation needs left padding. With right padding the
    # model sees [prompt, PAD, PAD, ...], and attention over the trailing
    # PAD positions degrades output quality. Llama's loader (cell 21) sets
    # it too.
    gen_tokenizer.padding_side = "left"

    # Try flash_attention_2 first. If that load raises ImportError, ValueError
    # or RuntimeError (not installed, older GPU, model unsupported), reload
    # with the transformers default attention. flash-attn is missing
    # from the data-analysis environment but present in the deep-models
    # one, so the fallback lets the script run from either.
    # Pin the model to a single GPU. When the model is sharded across two
    # GPUs, the accelerate dispatch hooks get wrapped by torch.compile,
    # which causes FailOnRecompileLimitHit.
    _gpu_idx = _pick_best_gpu()

    try:
        gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{_gpu_idx}"},
            token=HF_TOKEN,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("MedGemma loaded with flash_attention_2")
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to default attention")
        gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{_gpu_idx}"},
            token=HF_TOKEN,
            trust_remote_code=True,
        )

    # Gemma3 defaults to HybridCache, which allocates KV tensors with
    # torch._dynamo.mark_static_address and captures CUDA Graphs into
    # private memory pools. torch.cuda.empty_cache() does not release
    # those pools. Over many generate() calls the GPU fills up and runs
    # out of memory (after about 2.5 strategies on a 24 GB 3090). Forcing
    # the dynamic path makes generation_utils._prepare_cache_for_generation
    # take the `DynamicCache()` branch (line ~110 in transformers 4.52.4).
    # That branch allocates a fresh cache per call, which is
    # garbage-collected normally. Setting cache_implementation to None
    # would not work, because _prepare_cache_for_generation then falls back to
    # self.config.get_text_config().cache_implementation, which Gemma3
    # hardcodes to "hybrid". Expected slowdown: 10–25% per batch.
    gen_model.generation_config.cache_implementation = "dynamic"

    print(f"Generating paraphrases for {len(enriched_records)} LOINC codes...")
    gen_stats = generate_paraphrases_multiple_seeds(
        gen_model, gen_tokenizer, enriched_records,
        seeds=GEN_SEEDS,
        batch_size=GEN_BATCH_SIZE,
        output_dir=PARAPHRASE_DIR,
    )
    print(f"Generation stats: {gen_stats}")

    # Mark generation done in checkpoint.
    _ckpt = load_checkpoint()
    _ckpt["generation"]["done"] = True
    save_checkpoint(_ckpt)

    # Free MedGemma GPU memory before loading Llama.
    del gen_model
    del gen_tokenizer
    torch.cuda.empty_cache()

# %% [cell 19]
_ckpt = load_checkpoint()
if _ckpt["clean"]["done"]:
    print("Checkpoint: clean already done, skipping.")
    print("(Flip clean.done to false in UPSTREAM_DIR/4.2_checkpoint.json to force rerun.)")
else:
    clean_stats = clean_paraphrases(PARAPHRASE_DIR, PARAPHRASE_CLEAN_DIR)
    print(f"Clean stats: {clean_stats}")

    _ckpt = load_checkpoint()
    _ckpt["clean"]["done"] = True
    save_checkpoint(_ckpt)

# %% [cell 21]
_ckpt = load_checkpoint()
if _ckpt["verify"]["done"]:
    print("Checkpoint: verify already done, skipping Llama load.")
else:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    llama_tokenizer = AutoTokenizer.from_pretrained(
        VERIFY_MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"

    # Flash attention 2 with a fallback, and the model pinned to a single
    # GPU. Cell 17 explains both.
    _gpu_idx = _pick_best_gpu()

    try:
        llama_model = AutoModelForCausalLM.from_pretrained(
            VERIFY_MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": f"cuda:{_gpu_idx}"},
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("Llama-3.1-8B-Instruct (4-bit) loaded with flash_attention_2")
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to default attention")
        llama_model = AutoModelForCausalLM.from_pretrained(
            VERIFY_MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": f"cuda:{_gpu_idx}"},
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("Llama-3.1-8B-Instruct (4-bit) loaded.")

    # Llama 3.1's generation_config.json ships with temperature=0.6
    # and top_p=0.9 for its default do_sample=True mode. The verifier
    # uses greedy decoding (do_sample=False), so these settings have no
    # effect. transformers still warns on every generate() call because
    # they differ from the greedy-mode defaults (1.0). Resetting them
    # stops the warning and does not change the output.
    llama_model.generation_config.temperature = 1.0
    llama_model.generation_config.top_p = 1.0

# %% [cell 22]
_ckpt = load_checkpoint()
if _ckpt["verify"]["done"]:
    print("Checkpoint: verify already done, skipping.")
    print("(Flip verify.done to false in UPSTREAM_DIR/4.2_checkpoint.json to force rerun.)")
else:
    verify_stats = verify_paraphrases(
        llama_model, llama_tokenizer, enriched_records,
        input_dir=PARAPHRASE_CLEAN_DIR,
        output_dir=PARAPHRASE_VERIFIED_DIR,
        rejected_dir=PARAPHRASE_REJECTED_DIR,
        batch_size=VERIFY_BATCH_SIZE,
    )
    print(f"Verify stats: {verify_stats}")

    _ckpt = load_checkpoint()
    _ckpt["verify"]["done"] = True
    save_checkpoint(_ckpt)

# %% [cell 24]
_ckpt = load_checkpoint()
if _ckpt["finalize"]["done"]:
    print("Checkpoint: finalize already done, skipping.")
    print("(Flip finalize.done to false in UPSTREAM_DIR/4.2_checkpoint.json to force rerun.)")
else:
    finalize_stats = finalize_paraphrases(
        enriched_records,
        verified_dir=PARAPHRASE_VERIFIED_DIR,
        non_loinc_vocabs=NON_LOINC_VOCABS,
    )
    print(f"Finalize stats: {finalize_stats}")

    _ckpt = load_checkpoint()
    _ckpt["finalize"]["done"] = True
    save_checkpoint(_ckpt)
