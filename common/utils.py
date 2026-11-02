"""Shared helpers for TF-IDF and data preparation (thesis Chapters 4 and 5).

`extract_structured_features_single` turns a verbose observation prompt into the compact text that
TF-IDF reads. It is used by pipeline/06, 07, 08, 08b and 16. `load_mimic_iv_map` reads the MIMIC-IV
LOINC map for prep/4_seen_codes.py. The pipeline does not call the dataset loaders with
class-balanced sampling. pipeline/07_tfidf_train_eval.py reimplements the threshold sampler with a
seeded RNG.
"""
import os
import re
import csv
import math
import json
import glob
import gzip
import ndjson
import random
import pickle
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from collections import defaultdict

from typing import List, Tuple

# Get a list of all patient files
def load_patient_files(data_dir):
    return glob.glob(os.path.join(data_dir, "*.ndjson"))

# Load a single patient file
def load_resources(patient_file):
    with open(patient_file, "r") as f:
        return ndjson.load(f)
    
# Check if the FHIR resource is a lab observation
def is_lab_observation(resource):
    return resource.get("resourceType") == "Observation" and "laboratory" in resource['category'][0]['coding'][0]['code']
    
# Check if the FHIR resource is a non-lab observation
def is_nonlab_observation(resource):
    return resource.get("resourceType") == "Observation" and "laboratory" not in resource['category'][0]['coding'][0]['code']

# Load MIMIC to LOINC mapping from CSV and extract LOINC vocabulary
def load_mimic_iv_map(file_path, code_key=True):
    mimic_to_loinc = {}
    loinc_codes_set = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mimic_code = row["itemid (omop_source_code)"]
            mimic_desc = row["label"]
            loinc_code = row["omop_concept_code"]
            target_code_system = row["omop_vocabulary_id"]
            if code_key and mimic_code and loinc_code:
                mimic_to_loinc[mimic_code] = {"mimic_code": mimic_code,
                                              "mimic_desc": mimic_desc,
                                              "loinc_code": loinc_code,
                                              "target_code_system": target_code_system}
                loinc_codes_set.add(loinc_code)
            elif not code_key and mimic_desc and loinc_code:
                mimic_to_loinc[mimic_desc] = {"mimic_code": mimic_code,
                                              "mimic_desc": mimic_desc,
                                              "loinc_code": loinc_code,
                                              "target_code_system": target_code_system}
    return mimic_to_loinc, sorted(list(loinc_codes_set))

_RR_BOUND_RE = re.compile(r'\b(Low|High):\s*(-?\d*\.?\d+)', re.IGNORECASE)


def _fmt_num(v: float) -> str:
    """Render a coarsened bound without scientific notation. This matches the source formatting,
    where values arrive as `f"{json_number}"`, i.e. '4' or '3.5', never '4.0' or '1e+05'."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10f}".rstrip("0").rstrip(".")


def _coarsen_bound(v: float, how: str) -> float:
    """Coarsen one reference-range bound.

    '1sf'  -> one significant figure. It rounds half to even, like numpy.round, so the result is
              bit-identical to the a15 identifiability measurement (analysis/a15_field_identifiability.py,
              `_sigfig_vec`). Keep the two in sync. Otherwise the trained arm is not the measured one.
    'mag'  -> floor(log10(|v|)), the integer decade index. 0 means "in [1, 10)". Zero maps to the
              -99 sentinel, same as a15.
    """
    if how == "1sf":
        if v == 0:
            return 0.0
        f = 10.0 ** (0 - math.floor(math.log10(abs(v))))
        return round(v * f) / f
    if how == "mag":
        return -99.0 if v == 0 else float(math.floor(math.log10(abs(v))))
    raise ValueError(f"unknown coarsen_reference_range: {how!r}")


def _coarsen_reference_range(ref_range: str, how: str) -> str:
    """Rewrite only the numeric bounds inside a 'Low: x unit, High: y unit' string.

    The match is anchored on the Low:/High: labels because units in this corpus contain digits
    (10*3/uL, 10*12/L, log10 IU/mL, log10 copies/mL). Substituting bare numbers would corrupt them.
    Units may also be missing ('Low: 1.005, High: 1.035'). That case works too, because the match
    stops at the number.
    """
    def _sub(m):
        try:
            v = _coarsen_bound(float(m.group(2)), how)
        except (TypeError, ValueError):
            return m.group(0)
        return f"{m.group(1)}: {_fmt_num(v)}"
    return _RR_BOUND_RE.sub(_sub, ref_range)


def extract_structured_features_single(prompt: str, exclude_fields: set = None,
                                       coarsen_reference_range: str = None) -> str:
    """
    Extract the data values from the verbose prompt format of the single-resource dataset.

    Removes the instructional text and keeps, in this order:
    - Test description (paraphrased LOINC code)
    - Measured value with comparators
    - Reference range, as "reference <range>"
    - Units, when the prompt has a Units line
    - Status and priority, unless they are the common defaults
    - Data-absent reason
    - Notes/interpretations shorter than 200 characters
    The category and the datetime are not kept.

    Example: Description, Value and Reference Range lines become
    "<description> <value> reference <range>".

    Args:
        prompt: Verbose prompt string from dataset
        exclude_fields: Optional set of field names to exclude from the extracted features.
            Valid field names: "description", "value", "reference_range", "units",
            "status", "priority", "data_absent_reason", "note".
            If None (default), all fields are included. This is useful for ablation
            studies to measure each field's contribution to prediction accuracy.
        coarsen_reference_range: Optional None | "1sf" | "mag". Reduces the precision of the
            reference-range bounds instead of removing the field, for a graded reference-range
            ablation. None (default) leaves the bounds unchanged. Ignored when
            "reference_range" is in exclude_fields, since the field is dropped anyway.

    Returns:
        Compact string with only the data values ("unknown" if there are none)
    """
    if exclude_fields is None:
        exclude_fields = set()
    
    features = []
    
    # Extract description (the paraphrased LOINC code and the most important field)
    if "description" not in exclude_fields:
        desc_match = re.search(r'Description:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
            # Remove [MASKED] if present (from whole history format)
            description = re.sub(r'\[MASKED\]\s*', '', description)
            features.append(description)
    
    # Extract value with comparator (e.g., ">100", "<5", "7.2")
    if "value" not in exclude_fields:
        value_match = re.search(r'Value:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if value_match:
            value = value_match.group(1).strip()
            # Keep comparators and numeric values
            if value and value.lower() not in ['none', 'n/a', 'null', '']:
                features.append(value)
    
    # Extract reference range
    if "reference_range" not in exclude_fields:
        ref_match = re.search(r'Reference\s*Range:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if ref_match:
            ref_range = ref_match.group(1).strip()
            if coarsen_reference_range:
                ref_range = _coarsen_reference_range(ref_range, coarsen_reference_range)
            if ref_range and ref_range.lower() not in ['none', 'n/a', 'null', '']:
                features.append(f"reference {ref_range}")
    
    # Extract units (helps distinguish same test with different units)
    # Only an explicit Units line is read. Units inside the Value or Reference Range line are not extracted.
    if "units" not in exclude_fields:
        units_match = re.search(r'Units?:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if units_match:
            units = units_match.group(1).strip()
            if units and units.lower() not in ['none', 'n/a', 'null', '', 'unitless']:
                features.append(units)
    
    # Extract status (final, preliminary, etc.)
    if "status" not in exclude_fields:
        status_match = re.search(r'Status:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).strip()
            if status and status.lower() not in ['final', 'registered']:  # Skip common defaults
                features.append(f"status-{status}")
    
    # Extract priority (routine, urgent, stat, etc.)
    if "priority" not in exclude_fields:
        priority_match = re.search(r'Priority:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if priority_match:
            priority = priority_match.group(1).strip()
            if priority and priority.lower() not in ['routine', 'none', 'n/a']:
                features.append(f"priority-{priority}")
    
    # Extract data absent reason (if value is missing)
    if "data_absent_reason" not in exclude_fields:
        absent_match = re.search(r'DataAbsentReason:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if absent_match:
            reason = absent_match.group(1).strip()
            features.append(f"absent-{reason}")
    
    # Extract notes (clinical interpretations)
    if "note" not in exclude_fields:
        note_match = re.search(r'Note:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
        if note_match:
            note = note_match.group(1).strip()
            if note and len(note) < 200:  # Only include short notes
                features.append(note)
    
    # Join all features with spaces
    compact = ' '.join(features)
    
    # Clean up extra whitespace
    compact = re.sub(r'\s+', ' ', compact).strip()
    
    return compact if compact else "unknown"

def load_single_resource_dataset(data_dir: str, split: str = "train") -> Tuple[List[str], List[str]]:
    """
    Load the single resource dataset from JSONL files.
    
    Args:
        data_dir: Root directory containing train/test splits
        split: Either "train" or "test"
    
    Returns:
        Tuple of (texts, labels) where texts are prompts and labels are LOINC codes
    """
    samples_dir = Path(data_dir) / split / "samples"
    
    texts = []
    labels = []
    
    # Get all patient JSONL files
    sample_files = sorted(list(samples_dir.glob("*.jsonl")))
    
    print(f"Loading {split} dataset from {samples_dir}")
    print(f"Found {len(sample_files)} patient files")
    
    for sample_file in tqdm(sample_files, desc=f"Loading {split} data"):
        with open(sample_file, 'r') as f:
            for line in f:
                # Each line is a JSON object with 'prompt' and 'label'
                sample = json.loads(line)
                texts.append(sample['prompt'])
                labels.append(sample['label'])
    
    print(f"Loaded {len(texts)} samples from {split} set")
    
    return texts, labels


def load_from_cache(cache_path: str) -> Tuple[List[str], List[str]]:
    """
    Load data from consolidated cache file.
    """
    print(f"Loading from cache: {cache_path}")
    
    with gzip.open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    print(f"Loaded {cache_data['n_samples']:,} samples")
    
    return cache_data['texts'], cache_data['labels']


def create_consolidated_cache(data_dir: str, split: str, cache_path: str) -> Tuple[List[str], List[str]]:
    """
    Load data and create a consolidated cache file.
    """
    print(f"Creating consolidated cache for {split} set...")
    texts, labels = load_single_resource_dataset(data_dir, split)
    
    # Save to compressed cache
    cache_data = {
        'texts': texts,
        'labels': labels,
        'n_samples': len(texts),
        'n_classes': len(set(labels))
    }
    
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cache created ({file_size_mb:.1f} MB)")
    
    return texts, labels


def load_dataset_with_cache(data_dir: str, split: str, cache_dir: str = "cache/") -> Tuple[List[str], List[str]]:
    """
    Load dataset using cache if available, otherwise create cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename based on data directory and split
    cache_filename = f"{split}_data.pkl.gz"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"Found existing cache for {split} set")
        try:
            return load_from_cache(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Falling back to original loading method...")
    
    print(f"No cache found for {split} set. Creating cache...")
    return create_consolidated_cache(data_dir, split, cache_path)

def load_and_prepare_data_single(
    data_dir: str,
    cache_dir: str = "cache/",
    use_subset: bool = True,
    subset_size: int = 100000,
    min_samples_per_class: int = 10,
    exclude_fields: set = None,
    visualize: bool = False
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Complete data loading and preparation pipeline for single resource dataset.
    
    Steps:
    1. Check for cached processed data
    2. Load from cache (or create cache)
    3. Pre-filter singleton classes
    4. Subsample if requested
    5. Post-filter by minimum samples per class
    6. Apply feature extraction
    7. Cache the final processed data
    
    Args:
        data_dir: Root directory containing train/test splits
        cache_dir: Directory for cache files
        use_subset: Whether to subsample
        subset_size: Target number of training samples
        min_samples_per_class: Minimum samples required per class
        visualize: If True, plot class distribution as 2x1 subplot (train on top, test on bottom)
    
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    print("="*80)
    print("DATA LOADING AND PREPARATION PIPELINE")
    print("="*80)
    
    # Step 1. Check for cached processed data
    exclude_str = "_".join(sorted(exclude_fields)) if exclude_fields else "none"
    params_str = f"{use_subset}_{subset_size}_{min_samples_per_class}_{exclude_str}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    processed_cache_path = os.path.join(cache_dir, f"processed_data_{params_hash}.pkl.gz")
    
    if os.path.exists(processed_cache_path):
        print("\n✓ Found cached processed data with matching parameters")
        print(f"  Parameters: use_subset={use_subset}, subset_size={subset_size}, min_samples_per_class={min_samples_per_class}, exclude_fields={exclude_fields}")
        print(f"  Loading from: {processed_cache_path}")
        
        with gzip.open(processed_cache_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        print("\nLoaded processed data:")
        print(f"  Training: {processed_data['n_train_samples']:,} samples, {processed_data['n_train_classes']} classes")
        print(f"  Test: {processed_data['n_test_samples']:,} samples, {processed_data['n_test_classes']} classes")
        print(f"\n{'='*80}")
        print("DATA PREPARATION COMPLETE (from cache)")
        print(f"{'='*80}")
        
        return processed_data['train_texts'], processed_data['train_labels'], processed_data['test_texts'], processed_data['test_labels']
    
    print("\nNo cached processed data found. Processing from scratch...")
    
    # Step 2: Load from cache
    train_texts, train_labels = load_dataset_with_cache(data_dir, "train", cache_dir)
    test_texts, test_labels = load_dataset_with_cache(data_dir, "test", cache_dir)
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nInitial dataset size:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 3: Pre-filter singleton classes (required for stratified sampling)
    train_non_singleton = {code for code, count in train_label_counts.items() if count >= 2}
    test_non_singleton = {code for code, count in test_label_counts.items() if count >= 2}
    non_singleton_classes = train_non_singleton & test_non_singleton
    
    train_texts = [t for t, l in zip(train_texts, train_labels) if l in non_singleton_classes]
    train_labels = [l for l in train_labels if l in non_singleton_classes]
    test_texts = [t for t, l in zip(test_texts, test_labels) if l in non_singleton_classes]
    test_labels = [l for l in test_labels if l in non_singleton_classes]
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nAfter removing singletons:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 4: Subsample if requested
    if use_subset and len(train_texts) > subset_size:
        from sklearn.model_selection import train_test_split
        
        subset_fraction = subset_size / len(train_texts)
        print(f"\nSubsampling to {subset_size:,} samples (fraction: {subset_fraction:.4f})")
        
        train_texts, _, train_labels, _ = train_test_split(
            train_texts, train_labels,
            train_size=subset_fraction,
            stratify=train_labels,
            random_state=42
        )
        
        test_texts, _, test_labels, _ = train_test_split(
            test_texts, test_labels,
            train_size=subset_fraction,
            stratify=test_labels,
            random_state=43
        )
        
        train_label_counts = Counter(train_labels)
        test_label_counts = Counter(test_labels)
        
        print("After subsampling:")
        print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
        print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 5: Post-filter by minimum samples per class
    valid_classes = {code for code, count in train_label_counts.items() if count >= min_samples_per_class}
    
    train_texts = [t for t, l in zip(train_texts, train_labels) if l in valid_classes]
    train_labels = [l for l in train_labels if l in valid_classes]
    test_texts = [t for t, l in zip(test_texts, test_labels) if l in valid_classes]
    test_labels = [l for l in test_labels if l in valid_classes]
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print(f"\nAfter filtering classes with < {min_samples_per_class} samples:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 6: Feature extraction
    print(f"\nApplying feature extraction (exclude_fields={exclude_fields})...")
    train_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(train_texts, desc="Train")]
    test_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(test_texts, desc="Test")]
    
    # Step 7: Cache the final processed data
    print("\nCaching processed data for future use...")
    processed_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'n_train_samples': len(train_texts),
        'n_test_samples': len(test_texts),
        'n_train_classes': len(set(train_labels)),
        'n_test_classes': len(set(test_labels)),
        'params': {
            'use_subset': use_subset,
            'subset_size': subset_size,
            'min_samples_per_class': min_samples_per_class,
            'exclude_fields': sorted(exclude_fields) if exclude_fields else None
        }
    }
    
    with gzip.open(processed_cache_path, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(processed_cache_path) / (1024 * 1024)
    print(f"✓ Saved processed data to cache: {processed_cache_path}")
    print(f"  Cache size: {file_size_mb:.1f} MB")
    print("  A later run with the same parameters will load from this cache.")
    
    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    
    # Visualization
    if visualize:
        print("\nGenerating class distribution plots...")
        
        # Get sorted counts
        train_sorted_counts = sorted(train_label_counts.values(), reverse=True)
        test_sorted_counts = sorted(test_label_counts.values(), reverse=True)
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Train plot
        ax1.bar(range(len(train_sorted_counts)), train_sorted_counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class Rank', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title(f'Training Set: Class Distribution ({len(train_sorted_counts)} classes, {len(train_texts):,} samples)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Test plot
        ax2.bar(range(len(test_sorted_counts)), test_sorted_counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Class Rank', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title(f'Test Set: Class Distribution ({len(test_sorted_counts)} classes, {len(test_texts):,} samples)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return train_texts, train_labels, test_texts, test_labels


def load_and_prepare_data_single_sqrt(
    data_dir: str,
    frequency_counter_path: str = "data/code_frequency/le_code_counter_loinc.pkl",
    cache_dir: str = "cache/",
    subset_size: int = 2000000,
    downsample_min: int = 500,
    min_samples_per_class: int = 10,
    exclude_fields: set = None,
    visualize: bool = False
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Complete data loading and preparation pipeline with square root downsampling.
    
    This function implements a class-balanced sampling strategy:
    - Classes with <= downsample_min samples: take all available samples
    - Classes with > downsample_min samples: apply sqrt scaling then scale to target
    
    Steps:
    1. Check for cached processed data
    2. Load from cache (or create cache)
    3. Load frequency counter to determine per-class targets
    4. Calculate sqrt-based sampling targets
    5. Stratified sampling based on targets
    6. Post-filter by minimum samples per class
    7. Apply feature extraction
    8. Cache the final processed data
    
    Args:
        data_dir: Root directory containing train/test splits
        frequency_counter_path: Path to the frequency counter pickle file
        cache_dir: Directory for cache files
        subset_size: Target total number of training samples (~2M)
        downsample_min: Threshold - classes below this always take all samples
        min_samples_per_class: Minimum samples required per class after sampling
        visualize: If True, plot class distribution as 2x1 subplot (train on top, test on bottom)
    
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    print("="*80)
    print("DATA LOADING AND PREPARATION PIPELINE (SQRT SAMPLING)")
    print("="*80)
    
    # Step 1. Check for cached processed data
    exclude_str = "_".join(sorted(exclude_fields)) if exclude_fields else "none"
    params_str = f"sqrt_{subset_size}_{downsample_min}_{min_samples_per_class}_{exclude_str}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    processed_cache_path = os.path.join(cache_dir, f"processed_data_{params_hash}.pkl.gz")
    
    if os.path.exists(processed_cache_path):
        print("\n✓ Found cached processed data with matching parameters")
        print(f"  Parameters: sqrt sampling, target={subset_size}, downsample_min={downsample_min}, min_samples={min_samples_per_class}, exclude_fields={exclude_fields}")
        print(f"  Loading from: {processed_cache_path}")
        
        with gzip.open(processed_cache_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        print("\nLoaded processed data:")
        print(f"  Training: {processed_data['n_train_samples']:,} samples, {processed_data['n_train_classes']} classes")
        print(f"  Test: {processed_data['n_test_samples']:,} samples, {processed_data['n_test_classes']} classes")
        print(f"\n{'='*80}")
        print("DATA PREPARATION COMPLETE (from cache)")
        print(f"{'='*80}")
        
        return processed_data['train_texts'], processed_data['train_labels'], processed_data['test_texts'], processed_data['test_labels']
    
    print("\nNo cached processed data found. Processing from scratch...")
    
    # Step 2: Load from cache
    train_texts, train_labels = load_dataset_with_cache(data_dir, "train", cache_dir)
    test_texts, test_labels = load_dataset_with_cache(data_dir, "test", cache_dir)
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nInitial dataset size:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 3: Load frequency counter
    print(f"\nLoading frequency counter from: {frequency_counter_path}")
    with open(frequency_counter_path, 'rb') as f:
        frequency_counter = pickle.load(f)
    
    print(f"  Loaded {len(frequency_counter)} codes from frequency counter")
    print(f"  Total samples in counter: {sum(frequency_counter.values()):,}")
    
    # Step 4: Calculate sqrt-based sampling targets
    print("\nCalculating sqrt-based sampling targets...")
    print(f"  Downsample threshold: {downsample_min} (classes <= this: take all samples)")
    print(f"  Target total samples: {subset_size:,}")
    
    # Separate classes into two groups
    take_all_classes = {}  # Classes below threshold - take all
    downsample_classes = {}  # Classes above threshold - apply sqrt
    
    for code, count in train_label_counts.items():
        if count <= downsample_min:
            take_all_classes[code] = count
        else:
            downsample_classes[code] = count
    
    # Calculate budget allocation
    take_all_budget = sum(take_all_classes.values())
    remaining_budget = subset_size - take_all_budget
    
    print(f"\n  Classes <= {downsample_min}: {len(take_all_classes)} classes, {take_all_budget:,} samples (take all)")
    print(f"  Classes > {downsample_min}: {len(downsample_classes)} classes")
    print(f"  Remaining budget for sqrt sampling: {remaining_budget:,} samples")
    
    # Apply sqrt scaling to classes above threshold
    sqrt_counts = {code: np.sqrt(count) for code, count in downsample_classes.items()}
    total_sqrt = sum(sqrt_counts.values())
    
    # Calculate per-class targets
    target_counts = {}
    
    # Classes below threshold: take all
    for code, count in take_all_classes.items():
        target_counts[code] = count
    
    # Classes above threshold: sqrt-scaled allocation
    for code, sqrt_count in sqrt_counts.items():
        if total_sqrt > 0:
            allocated = int((sqrt_count / total_sqrt) * remaining_budget)
            # Apply floor of downsample_min and don't exceed available samples
            # This keeps classes just above threshold from being scaled below it
            target_counts[code] = min(max(allocated, downsample_min), downsample_classes[code])
        else:
            target_counts[code] = downsample_classes[code]
    
    # Show statistics
    total_target = sum(target_counts.values())
    print(f"\n  Total target samples after sqrt allocation: {total_target:,}")
    
    # Show some examples of the scaling
    if len(downsample_classes) > 0:
        print("\n  Sample sqrt scaling (top 5 frequent classes):")
        for code, original_count in sorted(downsample_classes.items(), key=lambda x: x[1], reverse=True)[:5]:
            target = target_counts[code]
            reduction = (1 - target/original_count) * 100
            print(f"    {code}: {original_count:,} -> {target:,} ({reduction:.1f}% reduction)")
    
    # Step 5: Stratified sampling based on targets
    print("\nPerforming stratified sampling...")
    
    # Group samples by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(train_labels):
        label_to_indices[label].append(idx)
    
    # Sample from each class according to targets
    sampled_indices = []
    sampling_stats = {'took_all': 0, 'downsampled': 0}
    
    for label, indices in tqdm(label_to_indices.items(), desc="Sampling classes"):
        target = target_counts.get(label, len(indices))
        
        if len(indices) <= target:
            # Take all samples
            sampled_indices.extend(indices)
            sampling_stats['took_all'] += 1
        else:
            # Randomly sample
            sampled = random.sample(indices, target)
            sampled_indices.extend(sampled)
            sampling_stats['downsampled'] += 1
    
    # Shuffle to mix classes
    random.shuffle(sampled_indices)
    
    # Apply sampling
    train_texts = [train_texts[i] for i in sampled_indices]
    train_labels = [train_labels[i] for i in sampled_indices]
    
    train_label_counts = Counter(train_labels)
    
    print("\nSampling statistics:")
    print(f"  Classes where all samples taken: {sampling_stats['took_all']}")
    print(f"  Classes that were downsampled: {sampling_stats['downsampled']}")
    print(f"  Training samples after sampling: {len(train_texts):,}")
    print(f"  Training classes after sampling: {len(train_label_counts)}")
    
    # Apply proportional sampling to test set
    print("\nApplying proportional sampling to test set...")
    test_label_to_indices = defaultdict(list)
    for idx, label in enumerate(test_labels):
        test_label_to_indices[label].append(idx)
    
    test_sampled_indices = []
    for label, indices in test_label_to_indices.items():
        if label in train_label_counts:
            # Calculate proportional test size
            train_count = train_label_counts[label]
            original_train_count = label_to_indices[label]
            if len(original_train_count) > 0:
                fraction = train_count / len(original_train_count)
                test_target = max(1, int(len(indices) * fraction))
                
                if len(indices) <= test_target:
                    test_sampled_indices.extend(indices)
                else:
                    sampled = random.sample(indices, test_target)
                    test_sampled_indices.extend(sampled)
    
    test_texts = [test_texts[i] for i in test_sampled_indices]
    test_labels = [test_labels[i] for i in test_sampled_indices]
    
    test_label_counts = Counter(test_labels)
    
    print(f"  Test samples after proportional sampling: {len(test_texts):,}")
    print(f"  Test classes after proportional sampling: {len(test_label_counts)}")
    
    # Step 6: Post-filter by minimum samples per class
    print(f"\nFiltering classes with < {min_samples_per_class} training samples...")
    valid_classes = {code for code, count in train_label_counts.items() if count >= min_samples_per_class}
    
    # Also keep only classes that exist in both train and test
    valid_classes = valid_classes & set(test_labels)
    
    train_texts = [t for t, l in zip(train_texts, train_labels) if l in valid_classes]
    train_labels = [l for l in train_labels if l in valid_classes]
    test_texts = [t for t, l in zip(test_texts, test_labels) if l in valid_classes]
    test_labels = [l for l in test_labels if l in valid_classes]
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nAfter filtering:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Show class distribution statistics
    train_counts_list = list(train_label_counts.values())
    print("\n  Training class distribution:")
    print(f"    Min samples per class: {min(train_counts_list)}")
    print(f"    Max samples per class: {max(train_counts_list)}")
    print(f"    Mean samples per class: {np.mean(train_counts_list):.1f}")
    print(f"    Median samples per class: {np.median(train_counts_list):.1f}")
    
    # Step 7: Feature extraction
    print(f"\nApplying feature extraction (exclude_fields={exclude_fields})...")
    train_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(train_texts, desc="Train")]
    test_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(test_texts, desc="Test")]
    
    # Step 8: Cache the final processed data
    print("\nCaching processed data for future use...")
    processed_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'n_train_samples': len(train_texts),
        'n_test_samples': len(test_texts),
        'n_train_classes': len(set(train_labels)),
        'n_test_classes': len(set(test_labels)),
        'params': {
            'method': 'sqrt',
            'subset_size': subset_size,
            'downsample_min': downsample_min,
            'min_samples_per_class': min_samples_per_class,
            'exclude_fields': sorted(exclude_fields) if exclude_fields else None
        }
    }
    
    with gzip.open(processed_cache_path, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(processed_cache_path) / (1024 * 1024)
    print(f"✓ Saved processed data to cache: {processed_cache_path}")
    print(f"  Cache size: {file_size_mb:.1f} MB")
    print("  A later run with the same parameters will load from this cache.")
    
    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE (SQRT SAMPLING)")
    print(f"{'='*80}")
    
    # Visualization
    if visualize:
        print("\nGenerating class distribution plots...")
        
        # Get sorted counts
        train_sorted_counts = sorted(train_label_counts.values(), reverse=True)
        test_sorted_counts = sorted(test_label_counts.values(), reverse=True)
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Train plot
        ax1.bar(range(len(train_sorted_counts)), train_sorted_counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class Rank', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title(f'Training Set: Class Distribution ({len(train_sorted_counts)} classes, {len(train_texts):,} samples)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Test plot
        ax2.bar(range(len(test_sorted_counts)), test_sorted_counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Class Rank', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title(f'Test Set: Class Distribution ({len(test_sorted_counts)} classes, {len(test_texts):,} samples)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return train_texts, train_labels, test_texts, test_labels


def load_natural_test_set(
    data_dir: str,
    valid_classes: set,
    cache_dir: str = "cache/",
    target_size: int = None,
    exclude_fields: set = None,
    random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load a uniformly-downsampled test set that preserves the natural class distribution.

    It applies the same sampling fraction to every class, so the class frequencies in
    the test set reflect the true population. The proportional test sampling in
    load_and_prepare_data_single_threshold() instead mirrors the boosted rare-class
    distribution of the train set.

    Args:
        data_dir: Root directory containing train/test splits
        valid_classes: Set of class labels to keep (e.g. the classes the threshold
                       pipeline kept). Only these classes appear in the output.
        cache_dir: Directory for cache files (reuses the raw test cache)
        target_size: Desired total number of test samples. If None, defaults to
                     the number of raw test samples belonging to valid_classes
                     (i.e. no downsampling, only class filtering).
        exclude_fields: Fields to exclude during feature extraction (same semantics
                        as extract_structured_features_single).
        random_state: Random seed for reproducible sampling.

    Returns:
        Tuple of (test_texts, test_labels) after uniform downsampling and feature
        extraction.
    """
    print("=" * 80)
    print("LOADING NATURAL (UNIFORM) TEST SET")
    print("=" * 80)

    # Step 1: Load raw test data from cache
    test_texts, test_labels = load_dataset_with_cache(data_dir, "test", cache_dir)
    print(f"\nRaw test set: {len(test_texts):,} samples, {len(set(test_labels))} classes")

    # Step 2: Filter to valid classes
    filtered = [
        (t, l) for t, l in zip(test_texts, test_labels) if l in valid_classes
    ]
    test_texts = [t for t, _ in filtered]
    test_labels = [l for _, l in filtered]

    label_counts = Counter(test_labels)
    print(f"After filtering to {len(valid_classes)} valid classes: "
          f"{len(test_texts):,} samples, {len(label_counts)} classes")

    # Step 3: Uniform downsampling (same fraction for every class)
    if target_size is not None and target_size < len(test_texts):
        fraction = target_size / len(test_texts)
        print(f"\nUniform downsampling to ~{target_size:,} samples (fraction={fraction:.4f})")

        rng = random.Random(random_state)
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(test_labels):
            label_to_indices[label].append(idx)

        sampled_indices = []
        for label, indices in label_to_indices.items():
            k = max(1, int(len(indices) * fraction))
            k = min(k, len(indices))
            sampled_indices.extend(rng.sample(indices, k))

        rng.shuffle(sampled_indices)
        test_texts = [test_texts[i] for i in sampled_indices]
        test_labels = [test_labels[i] for i in sampled_indices]

        label_counts = Counter(test_labels)
        print(f"After uniform sampling: {len(test_texts):,} samples, {len(label_counts)} classes")
    else:
        print("\nNo downsampling applied (target_size >= available samples or not set)")

    # Step 4: Feature extraction
    print(f"\nApplying feature extraction (exclude_fields={exclude_fields})...")
    test_texts = [
        extract_structured_features_single(text, exclude_fields=exclude_fields)
        for text in tqdm(test_texts, desc="Natural test")
    ]

    # Summary statistics
    counts = list(label_counts.values())
    print("\nNatural test set ready:")
    print(f"  Samples: {len(test_texts):,}")
    print(f"  Classes: {len(label_counts)}")
    print(f"  Min samples/class: {min(counts)}")
    print(f"  Max samples/class: {max(counts)}")
    print(f"  Mean samples/class: {np.mean(counts):.1f}")
    print(f"  Median samples/class: {np.median(counts):.1f}")
    print("=" * 80)

    return test_texts, test_labels


def load_and_prepare_data_single_threshold(
    data_dir: str,
    frequency_counter_path: str = "data/code_frequency/le_code_counter_loinc.pkl",
    cache_dir: str = "cache/",
    subset_size: int = 2000000,
    downsample_min: int = 500,
    min_samples_per_class: int = 10,
    exclude_fields: set = None,
    visualize: bool = False
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Complete data loading and preparation pipeline with threshold sampling. It works like
    load_and_prepare_data_single_sqrt, except that classes above downsample_min share the
    remaining budget in proportion to their size. The sqrt version uses the square root of the size.

    Args:
        data_dir: Root directory containing train/test splits
        frequency_counter_path: Path to the frequency counter pickle file
        cache_dir: Directory for cache files
        subset_size: Target total number of training samples (~2M)
        downsample_min: Threshold - classes below this always take all samples
        min_samples_per_class: Minimum samples required per class after sampling
        visualize: If True, plot class distribution as 2x1 subplot (train on top, test on bottom)
    
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    print("="*80)
    print("DATA LOADING AND PREPARATION PIPELINE (UNIFORM SAMPLING WITH THRESHOLD)")
    print("="*80)
    
    # Step 1. Check for cached processed data
    exclude_str = "_".join(sorted(exclude_fields)) if exclude_fields else "none"
    params_str = f"threshold_{subset_size}_{downsample_min}_{min_samples_per_class}_{exclude_str}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    processed_cache_path = os.path.join(cache_dir, f"processed_data_{params_hash}.pkl.gz")
    
    if os.path.exists(processed_cache_path):
        print("\n✓ Found cached processed data with matching parameters")
        print(f"  Parameters: threshold sampling, target={subset_size}, downsample_min={downsample_min}, min_samples={min_samples_per_class}, exclude_fields={exclude_fields}")
        print(f"  Loading from: {processed_cache_path}")
        
        with gzip.open(processed_cache_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        print("\nLoaded processed data:")
        print(f"  Training: {processed_data['n_train_samples']:,} samples, {processed_data['n_train_classes']} classes")
        print(f"  Test: {processed_data['n_test_samples']:,} samples, {processed_data['n_test_classes']} classes")
        print(f"\n{'='*80}")
        print("DATA PREPARATION COMPLETE (from cache)")
        print(f"{'='*80}")
        
        return processed_data['train_texts'], processed_data['train_labels'], processed_data['test_texts'], processed_data['test_labels']
    
    print("\nNo cached processed data found. Processing from scratch...")
    
    # Step 2: Load from cache
    train_texts, train_labels = load_dataset_with_cache(data_dir, "train", cache_dir)
    test_texts, test_labels = load_dataset_with_cache(data_dir, "test", cache_dir)
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nInitial dataset size:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Step 3: Load frequency counter
    print(f"\nLoading frequency counter from: {frequency_counter_path}")
    with open(frequency_counter_path, 'rb') as f:
        frequency_counter = pickle.load(f)
    
    print(f"  Loaded {len(frequency_counter)} codes from frequency counter")
    print(f"  Total samples in counter: {sum(frequency_counter.values()):,}")
    
    # Step 4: Calculate threshold-based sampling targets
    print("\nCalculating threshold-based sampling targets...")
    print(f"  Downsample threshold: {downsample_min} (classes <= this: take all samples)")
    print(f"  Target total samples: {subset_size:,}")
    
    # Separate classes into two groups
    take_all_classes = {}  # Classes below threshold - take all
    downsample_classes = {}  # Classes above threshold - apply uniform scaling
    
    for code, count in train_label_counts.items():
        if count <= downsample_min:
            take_all_classes[code] = count
        else:
            downsample_classes[code] = count
    
    # Calculate budget allocation
    take_all_budget = sum(take_all_classes.values())
    remaining_budget = subset_size - take_all_budget
    
    print(f"\n  Classes <= {downsample_min}: {len(take_all_classes)} classes, {take_all_budget:,} samples (take all)")
    print(f"  Classes > {downsample_min}: {len(downsample_classes)} classes")
    print(f"  Remaining budget for uniform sampling: {remaining_budget:,} samples")
    
    # Apply uniform scaling to classes above threshold
    total_uniform = sum(downsample_classes.values())
    
    # Calculate per-class targets
    target_counts = {}
    
    # Classes below threshold: take all
    for code, count in take_all_classes.items():
        target_counts[code] = count
    
    # Classes above threshold: uniform-scaled allocation
    for code, uniform_count in downsample_classes.items():
        if total_uniform > 0:
            allocated = int((uniform_count / total_uniform) * remaining_budget)
            # Apply floor of downsample_min and don't exceed available samples
            # This keeps classes just above threshold from being scaled below it
            target_counts[code] = min(max(allocated, downsample_min), downsample_classes[code])
        else:
            target_counts[code] = downsample_classes[code]
    
    # Show statistics
    total_target = sum(target_counts.values())
    print(f"\n  Total target samples after uniform allocation: {total_target:,}")
    
    # Step 5: Stratified sampling based on targets
    print("\nPerforming stratified sampling...")
    
    # Group samples by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(train_labels):
        label_to_indices[label].append(idx)
    
    # Sample from each class according to targets
    sampled_indices = []
    sampling_stats = {'took_all': 0, 'downsampled': 0}
    
    for label, indices in tqdm(label_to_indices.items(), desc="Sampling classes"):
        target = target_counts.get(label, len(indices))
        
        if len(indices) <= target:
            # Take all samples
            sampled_indices.extend(indices)
            sampling_stats['took_all'] += 1
        else:
            # Randomly sample
            sampled = random.sample(indices, target)
            sampled_indices.extend(sampled)
            sampling_stats['downsampled'] += 1
    
    # Shuffle to mix classes
    random.shuffle(sampled_indices)
    
    # Apply sampling
    train_texts = [train_texts[i] for i in sampled_indices]
    train_labels = [train_labels[i] for i in sampled_indices]
    
    train_label_counts = Counter(train_labels)
    
    print("\nSampling statistics:")
    print(f"  Classes where all samples taken: {sampling_stats['took_all']}")
    print(f"  Classes that were downsampled: {sampling_stats['downsampled']}")
    print(f"  Training samples after sampling: {len(train_texts):,}")
    print(f"  Training classes after sampling: {len(train_label_counts)}")
    
    # Apply proportional sampling to test set
    print("\nApplying proportional sampling to test set...")
    test_label_to_indices = defaultdict(list)
    for idx, label in enumerate(test_labels):
        test_label_to_indices[label].append(idx)
    
    test_sampled_indices = []
    for label, indices in test_label_to_indices.items():
        if label in train_label_counts:
            # Calculate proportional test size
            train_count = train_label_counts[label]
            original_train_count = label_to_indices[label]
            if len(original_train_count) > 0:
                fraction = train_count / len(original_train_count)
                test_target = max(1, int(len(indices) * fraction))
                
                if len(indices) <= test_target:
                    test_sampled_indices.extend(indices)
                else:
                    sampled = random.sample(indices, test_target)
                    test_sampled_indices.extend(sampled)
    
    test_texts = [test_texts[i] for i in test_sampled_indices]
    test_labels = [test_labels[i] for i in test_sampled_indices]
    
    test_label_counts = Counter(test_labels)
    
    print(f"  Test samples after proportional sampling: {len(test_texts):,}")
    print(f"  Test classes after proportional sampling: {len(test_label_counts)}")
    
    # Step 6: Post-filter by minimum samples per class
    print(f"\nFiltering classes with < {min_samples_per_class} training samples...")
    valid_classes = {code for code, count in train_label_counts.items() if count >= min_samples_per_class}
    
    # Also keep only classes that exist in both train and test
    valid_classes = valid_classes & set(test_labels)
    
    train_texts = [t for t, l in zip(train_texts, train_labels) if l in valid_classes]
    train_labels = [l for l in train_labels if l in valid_classes]
    test_texts = [t for t, l in zip(test_texts, test_labels) if l in valid_classes]
    test_labels = [l for l in test_labels if l in valid_classes]
    
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    print("\nAfter filtering:")
    print(f"  Training: {len(train_texts):,} samples, {len(train_label_counts)} classes")
    print(f"  Test: {len(test_texts):,} samples, {len(test_label_counts)} classes")
    
    # Show class distribution statistics
    train_counts_list = list(train_label_counts.values())
    print("\n  Training class distribution:")
    print(f"    Min samples per class: {min(train_counts_list)}")
    print(f"    Max samples per class: {max(train_counts_list)}")
    print(f"    Mean samples per class: {np.mean(train_counts_list):.1f}")
    print(f"    Median samples per class: {np.median(train_counts_list):.1f}")
    
    # Step 7: Feature extraction
    print(f"\nApplying feature extraction (exclude_fields={exclude_fields})...")
    train_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(train_texts, desc="Train")]
    test_texts = [extract_structured_features_single(text, exclude_fields=exclude_fields) for text in tqdm(test_texts, desc="Test")]
    
    # Step 8: Cache the final processed data
    print("\nCaching processed data for future use...")
    processed_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'n_train_samples': len(train_texts),
        'n_test_samples': len(test_texts),
        'n_train_classes': len(set(train_labels)),
        'n_test_classes': len(set(test_labels)),
        'params': {
            'method': 'threshold',
            'subset_size': subset_size,
            'downsample_min': downsample_min,
            'min_samples_per_class': min_samples_per_class,
            'exclude_fields': sorted(exclude_fields) if exclude_fields else None
        }
    }
    
    with gzip.open(processed_cache_path, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(processed_cache_path) / (1024 * 1024)
    print(f"✓ Saved processed data to cache: {processed_cache_path}")
    print(f"  Cache size: {file_size_mb:.1f} MB")
    print("  A later run with the same parameters will load from this cache.")
    
    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE (UNIFORM SAMPLING WITH THRESHOLD)")
    print(f"{'='*80}")
    
    # Visualization
    if visualize:
        print("\nGenerating class distribution plots...")
        
        # Get sorted counts
        train_sorted_counts = sorted(train_label_counts.values(), reverse=True)
        test_sorted_counts = sorted(test_label_counts.values(), reverse=True)
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Train plot
        ax1.bar(range(len(train_sorted_counts)), train_sorted_counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class Rank', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title(f'Training Set: Class Distribution ({len(train_sorted_counts)} classes, {len(train_texts):,} samples)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Test plot
        ax2.bar(range(len(test_sorted_counts)), test_sorted_counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Class Rank', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title(f'Test Set: Class Distribution ({len(test_sorted_counts)} classes, {len(test_texts):,} samples)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return train_texts, train_labels, test_texts, test_labels
