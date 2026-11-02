"""Count the lab and chart event codes in MIMIC-IV on FHIR, overall and restricted to the codes with a
LOINC mapping (thesis Chapter 4).

Reads MimicObservationLabevents.ndjson and MimicObservationChartevents.ndjson from MIMIC_FHIR_DIR, the
mimic-code LOINC mappings (MIMIC_CODE_DIR/mimic-iv/mapping/d_labitems_to_loinc.csv and
meas_chartevents_main.csv) and UPSTREAM_DIR/Loinc.csv. Writes Counters of code counts as pickles to
UPSTREAM_DIR/code_frequency/. le_code_counter.pkl and ce_code_counter.pkl cover every code.
le_code_counter_loinc.pkl and ce_code_counter_loinc.pkl cover the codes with a LOINC mapping, and
prep/2 and prep/4 read these two.

  python prep/1_code_frequency_analysis.py
"""

import os
import time
import pickle
from tqdm import tqdm
from collections import Counter
import logging
from typing import Generator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

logger = logging.getLogger(__name__)

# MIMIC-IV on FHIR NDJSON files
data_dir = str(locations.MIMIC_FHIR)

# Create output directory for frequency counters
output_dir = str(locations.UPSTREAM / "code_frequency")
os.makedirs(output_dir, exist_ok=True)

def generator_from_file(read_file: str) -> Generator:
    """Yield the resources of an NDJSON file one line at a time, skipping malformed lines and
    summarising them at the end."""
    import json
    
    error_count = 0
    total_count = 0
    error_lines = []
    
    with open(read_file, "r") as fr:
        for line_num, line in enumerate(fr, 1):
            total_count += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                # Use json.loads instead of ndjson.reader to avoid stderr messages
                resource = json.loads(line)
                yield resource
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count == 1:
                    print("Warning: Found malformed JSON records, skipping them...")
                # Store first few error lines for summary (but don't log each one)
                if error_count <= 5:
                    error_lines.append(f"Line {line_num}: {str(e)}")
                continue
    
    if error_count > 0:
        print(f"Skipped {error_count:,} malformed JSON records out of {total_count:,} total lines ({error_count/total_count*100:.2f}%)")
        if error_lines:
            print("Sample errors:")
            for error_line in error_lines:
                print(f"  {error_line}")
            if error_count > 5:
                print(f"  ... and {error_count - 5} more similar errors")

def extract_code_from_resource(resource):
    """Extract code from resource, safely handling nested structure"""
    try:
        return resource["code"]["coding"][0]["code"]
    except (KeyError, IndexError, TypeError):
        return None

# Analyze lab events

print("Processing Lab Events...")
le_file = os.path.join(data_dir, "MimicObservationLabevents.ndjson")
le_code_counter = Counter()

# Get file size for progress tracking
file_size = os.path.getsize(le_file)
print(f"File size: {file_size / (1024**3):.2f} GB")

# Sample first 1000 lines to estimate total records
print("Sampling file to estimate total records...")
sample_count = 0
with open(le_file, "r") as f:
    while sample_count < 1000:
        line = f.readline()
        if not line:  # End of file
            break
        sample_count += 1
    sample_position = f.tell()

# Estimate total records based on sample
if sample_count > 0:
    avg_bytes_per_record = sample_position / sample_count
    estimated_total_records = int(file_size / avg_bytes_per_record)
    print(f"Estimated total records: {estimated_total_records:,} (based on {sample_count} sample records)")
else:
    estimated_total_records = None
    print("Could not sample file for estimation")

start_time = time.time()
le_gen = generator_from_file(le_file)

for resource in tqdm(le_gen, total=estimated_total_records, desc="Lab events", unit="records"):
    code = extract_code_from_resource(resource)
    if code:
        le_code_counter[code] += 1

elapsed_time = time.time() - start_time
print(f"\nCompleted. Total unique lab event codes: {len(le_code_counter):,}")
print(f"Total lab events processed: {sum(le_code_counter.values()):,}")
print(f"Processing time: {elapsed_time:.1f} seconds")
print("\nTop 10 most frequent lab event codes:")
for code, count in le_code_counter.most_common(10):
    print(f"  {code}: {count}")

# Save lab events counter
le_output_file = os.path.join(output_dir, "le_code_counter.pkl")
with open(le_output_file, "wb") as f:
    pickle.dump(le_code_counter, f)
print(f"Lab events code counter saved to {le_output_file}")

# Analyze chart events
print("Processing Chart Events...")
ce_file = os.path.join(data_dir, "MimicObservationChartevents.ndjson")
ce_code_counter = Counter()

# Get file size for progress tracking
file_size = os.path.getsize(ce_file)
print(f"File size: {file_size / (1024**3):.2f} GB")

# Sample first 1000 lines to estimate total records
print("Sampling file to estimate total records...")
sample_count = 0
with open(ce_file, "r") as f:
    while sample_count < 1000:
        line = f.readline()
        if not line:  # End of file
            break
        sample_count += 1
    sample_position = f.tell()

# Estimate total records based on sample
if sample_count > 0:
    avg_bytes_per_record = sample_position / sample_count
    estimated_total_records = int(file_size / avg_bytes_per_record)
    print(f"Estimated total records: {estimated_total_records:,} (based on {sample_count} sample records)")
else:
    estimated_total_records = None
    print("Could not sample file for estimation")

start_time = time.time()
ce_gen = generator_from_file(ce_file)

for resource in tqdm(ce_gen, total=estimated_total_records, desc="Chart events", unit="records"):
    code = extract_code_from_resource(resource)
    if code:
        ce_code_counter[code] += 1

elapsed_time = time.time() - start_time
print(f"\nCompleted. Total unique chart event codes: {len(ce_code_counter):,}")
print(f"Total chart events processed: {sum(ce_code_counter.values()):,}")
print(f"Processing time: {elapsed_time:.1f} seconds")
print("\nTop 10 most frequent chart event codes:")
for code, count in ce_code_counter.most_common(10):
    print(f"  {code}: {count}")

# Save chart events counter
ce_output_file = os.path.join(output_dir, "ce_code_counter.pkl")
with open(ce_output_file, "wb") as f:
    pickle.dump(ce_code_counter, f)
print(f"Chart events code counter saved to {ce_output_file}")

# Load mapping files and LOINC master list
import pandas as pd
import csv

# Define file paths for mappings
LAB_OBS_MAP_FILEPATH = str(locations.MIMIC_CODE / "mimic-iv/mapping/d_labitems_to_loinc.csv")
OTHER_OBS_MAP_FILEPATH = str(locations.MIMIC_CODE / "mimic-iv/mapping/meas_chartevents_main.csv")
LOINC_MASTER_PATH = str(locations.UPSTREAM / "Loinc.csv")

print("Loading mapping files and LOINC master list...")

# Load LOINC master list to validate LOINC codes
try:
    loinc_master_df = pd.read_csv(LOINC_MASTER_PATH, low_memory=False)
    valid_loinc_codes = set(loinc_master_df['LOINC_NUM'].dropna().unique())
    print(f"Loaded {len(valid_loinc_codes):,} valid LOINC codes from master list")
except FileNotFoundError:
    print(f"Warning: {LOINC_MASTER_PATH} not found, will use mapping file validation only")
    valid_loinc_codes = None

# Load lab events mapping (MIMIC code -> LOINC)
lab_mapping = {}
try:
    with open(LAB_OBS_MAP_FILEPATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mimic_code = row["itemid (omop_source_code)"]
            loinc_code = row["omop_concept_code"]
            target_system = row["omop_vocabulary_id"]
            
            # Only include if we have valid codes and target system is LOINC
            if mimic_code and loinc_code and target_system == "LOINC":
                # Additional validation against master list if available
                if valid_loinc_codes is None or loinc_code in valid_loinc_codes:
                    lab_mapping[mimic_code] = loinc_code
    
    print(f"Loaded {len(lab_mapping):,} MIMIC lab codes with valid LOINC mappings")
except FileNotFoundError:
    print(f"Warning: {LAB_OBS_MAP_FILEPATH} not found")
    lab_mapping = {}

# Load chart events mapping (MIMIC code -> LOINC)
chart_mapping = {}
try:
    with open(OTHER_OBS_MAP_FILEPATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mimic_code = row["itemid (omop_source_code)"]
            loinc_code = row["omop_concept_code"]
            target_system = row["omop_vocabulary_id"]
            
            # Only include if we have valid codes and target system is LOINC
            if mimic_code and loinc_code and target_system == "LOINC":
                # Additional validation against master list if available
                if valid_loinc_codes is None or loinc_code in valid_loinc_codes:
                    chart_mapping[mimic_code] = loinc_code
    
    print(f"Loaded {len(chart_mapping):,} MIMIC chart codes with valid LOINC mappings")
except FileNotFoundError:
    print(f"Warning: {OTHER_OBS_MAP_FILEPATH} not found")
    chart_mapping = {}

# Create LOINC-filtered lab events counter
print("Creating LOINC-filtered lab events counter...")
le_code_counter_loinc = Counter()

# Filter lab events counter to only include codes that map to LOINC
for mimic_code, count in le_code_counter.items():
    if mimic_code in lab_mapping:
        le_code_counter_loinc[mimic_code] = count

# Calculate statistics
original_le_codes = len(le_code_counter)
original_le_events = sum(le_code_counter.values())
filtered_le_codes = len(le_code_counter_loinc)
filtered_le_events = sum(le_code_counter_loinc.values())

print("Lab Events LOINC Filtering Results:")
print(f"  Original: {original_le_codes:,} unique codes, {original_le_events:,} total events")
print(f"  LOINC-filtered: {filtered_le_codes:,} unique codes, {filtered_le_events:,} total events")
print(f"  Retention: {filtered_le_codes/original_le_codes*100:.1f}% of codes, {filtered_le_events/original_le_events*100:.1f}% of events")

print("\nTop 10 most frequent LOINC-mappable lab event codes:")
for code, count in le_code_counter_loinc.most_common(10):
    loinc_code = lab_mapping.get(code, "Unknown")
    print(f"  {code} -> {loinc_code}: {count:,}")

# Save LOINC-filtered lab events counter
le_loinc_output_file = os.path.join(output_dir, "le_code_counter_loinc.pkl")
with open(le_loinc_output_file, "wb") as f:
    pickle.dump(le_code_counter_loinc, f)
print(f"\nLOINC-filtered lab events counter saved to {le_loinc_output_file}")

# Create LOINC-filtered chart events counter
print("Creating LOINC-filtered chart events counter...")
ce_code_counter_loinc = Counter()

# Filter chart events counter to only include codes that map to LOINC
for mimic_code, count in ce_code_counter.items():
    if mimic_code in chart_mapping:
        ce_code_counter_loinc[mimic_code] = count

# Calculate statistics
original_ce_codes = len(ce_code_counter)
original_ce_events = sum(ce_code_counter.values())
filtered_ce_codes = len(ce_code_counter_loinc)
filtered_ce_events = sum(ce_code_counter_loinc.values())

print("Chart Events LOINC Filtering Results:")
print(f"  Original: {original_ce_codes:,} unique codes, {original_ce_events:,} total events")
print(f"  LOINC-filtered: {filtered_ce_codes:,} unique codes, {filtered_ce_events:,} total events")
print(f"  Retention: {filtered_ce_codes/original_ce_codes*100:.1f}% of codes, {filtered_ce_events/original_ce_events*100:.1f}% of events")

print("\nTop 10 most frequent LOINC-mappable chart event codes:")
for code, count in ce_code_counter_loinc.most_common(10):
    loinc_code = chart_mapping.get(code, "Unknown")
    print(f"  {code} -> {loinc_code}: {count:,}")

# Save LOINC-filtered chart events counter
ce_loinc_output_file = os.path.join(output_dir, "ce_code_counter_loinc.pkl")
with open(ce_loinc_output_file, "wb") as f:
    pickle.dump(ce_code_counter_loinc, f)
print(f"\nLOINC-filtered chart events counter saved to {ce_loinc_output_file}")
