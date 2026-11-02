"""
Helpers for turning FHIR patient records into tokenized lab-code prediction examples, and the
lab-code class maps that the pipeline and the analyses share (thesis Chapter 5). Tokenizer-agnostic:
the caller passes whatever tokenizer and mask token id the model uses.

Public API:
- build_lab_code_mappings: build {mimic_code -> class_idx} and {class_idx -> loinc}.
- enumerate_arrow_patients: collect patient UUIDs from an FHIR-MLM Arrow dataset.
- load_val_patients / derive_split: MLM-aligned train/val split.
- tokenize_and_align: tokenize a formatter-output prompt and build aligned labels.
"""

from __future__ import annotations
from pathlib import Path
import csv
import pickle

import datasets


# Code-system constants. The FHIR-MLM training script (pipeline/12_mlm_train.py) uses the same
# lab-token range, offset by the size of the text vocabulary.
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
LAB_TOKEN_MIN_RAW = 4429
LAB_TOKEN_MAX_RAW = 6051
N_LAB_CLASSES = LAB_TOKEN_MAX_RAW - LAB_TOKEN_MIN_RAW + 1  # 1623


def build_lab_code_mappings(
    codes_to_tokens_path: str | Path,
    seen_codes_path: str | Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return:
      - mimic_code_to_class: {mimic_code_str -> int in [0, 1623)}; class index =
        raw_token_id - LAB_TOKEN_MIN_RAW. Covers all 1623 MIMIC lab codes.
      - class_to_loinc: {class_idx -> loinc_str}, for the LOINC-mappable classes only.
        Used to (a) decide which positions get a label vs -100 at train time
        and (b) post-map predictions to LOINC at eval time.
    """
    with open(codes_to_tokens_path, "rb") as f:
        codes_to_tokens = pickle.load(f)
    lab_map = codes_to_tokens[LAB_CODE_SYSTEM]  # {mimic_code: raw_token_id}

    mimic_code_to_class: dict[str, int] = {
        str(code): int(raw_id) - LAB_TOKEN_MIN_RAW
        for code, raw_id in lab_map.items()
    }

    class_to_loinc: dict[int, str] = {}
    with open(seen_codes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("observation_type") != "lab":
                continue
            mimic_code = row.get("mimic_code") or ""
            loinc_code = row.get("loinc_code") or ""
            if not (mimic_code and loinc_code):
                continue
            cls = mimic_code_to_class.get(mimic_code)
            if cls is not None:
                class_to_loinc[cls] = loinc_code

    return mimic_code_to_class, class_to_loinc


def enumerate_arrow_patients(arrow_path: str | Path) -> set[str]:
    """All unique patient UUIDs in the MLM Arrow dataset's `files` column,
    with .ndjson suffix stripped."""
    ds = datasets.load_from_disk(str(arrow_path))
    uuids: set[str] = set()
    # Iterating the column directly is faster than per-row dict access.
    for files in ds["files"]:
        for f in files:
            uuids.add(f.replace(".ndjson", ""))
    return uuids


def load_val_patients(val_patients_path: str | Path) -> set[str]:
    """Read mlm_val_patients.txt, which has one UUID per line and no .ndjson suffix."""
    with open(val_patients_path) as f:
        return {line.strip() for line in f if line.strip()}


def derive_split(
    arrow_patients: set[str],
    val_patients: set[str],
) -> tuple[set[str], set[str]]:
    """train = arrow_patients - val_patients; val = arrow_patients ∩ val_patients."""
    val = arrow_patients & val_patients
    train = arrow_patients - val_patients
    return train, val


def tokenize_and_align(
    prompt_text: str,
    codes: list[str],
    tokenizer,
    mimic_code_to_class: dict[str, int],
    class_to_loinc: dict[int, str],
    mask_token_id: int,
    max_len: int = 8192,
) -> dict | None:
    """Tokenize `prompt_text` and build aligned labels at mask positions.

    Labels: -100 everywhere except at mask positions whose corresponding MIMIC
    code is LOINC-mappable, where the label is the class index in [0, 1623).
    Non-LOINC-mappable masks stay -100, as in the FHIR-MLM loss mask.

    Truncation keeps the first max_len tokens, as the FHIR-MLM tokenizer's
    `_truncate` does (third_party/fhir_mlm/fhir_tokenizer.py). Mask positions
    past max_len are dropped, which matches FHIR-MLM, where those code tokens
    are truncated too.

    Returns {input_ids, attention_mask, labels} (Python lists), or None if the
    post-truncation sequence has zero non-(-100) labels.

    Raises ValueError if the mask-token count in the tokenized prompt disagrees
    with len(codes). That points to a bug in the formatter/tokenizer contract.
    A data problem is an unlikely cause.
    """
    enc = tokenizer(prompt_text, add_special_tokens=True, return_tensors=None)
    input_ids: list[int] = enc["input_ids"]
    attention_mask: list[int] = enc["attention_mask"]

    mask_positions = [i for i, t in enumerate(input_ids) if t == mask_token_id]
    if len(mask_positions) != len(codes):
        raise ValueError(
            f"Mask alignment broken: {len(mask_positions)} mask tokens in tokenized "
            f"prompt vs {len(codes)} codes from formatter."
        )

    # Truncate from the back. Mask positions increase monotonically, so once
    # one is past max_len, all later ones are dropped too.
    input_ids = input_ids[:max_len]
    attention_mask = attention_mask[:max_len]

    labels = [-100] * len(input_ids)
    for pos, code in zip(mask_positions, codes):
        if pos >= len(input_ids):
            break
        cls = mimic_code_to_class.get(code)
        if cls is None:
            # The formatter only emits masks for LAB_CODE_SYSTEM codes, which
            # are all in mimic_code_to_class. Reaching this line means the
            # formatter and codes_to_tokens are out of sync.
            continue
        if cls in class_to_loinc:
            labels[pos] = cls

    if not any(l != -100 for l in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
