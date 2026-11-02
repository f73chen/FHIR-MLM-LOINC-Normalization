"""Filter MIMIC-IV on FHIR to the observations whose code has a LOINC mapping, write one file per
patient, and count each patient's resources (thesis Chapter 4).

Reads the MIMIC-IV on FHIR NDJSON files in MIMIC_FHIR_DIR and the LOINC-mapped code counts from
prep/1_code_frequency_analysis.py (UPSTREAM_DIR/code_frequency/). Writes the filtered lab and chart
observations next to the source files (LOINCFilteredObservationLabEvents.ndjson and
LOINCFilteredObservationChartEvents.ndjson). Also writes one NDJSON file per patient to
WORK_DIR/loinc_patients/ and each patient's resource counts to UPSTREAM_DIR/loinc_patients_summary.csv.
pipeline/01_build_split.py reads the summary.

  python prep/2_create_loinc_patients.py
"""

import os
import json
import time
import ndjson
import pickle
import logging
import polars as pl
from tqdm import tqdm
from typing import Generator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

logger = logging.getLogger(__name__)

# Load frequency counters
with open(str(locations.UPSTREAM / "code_frequency/le_code_counter_loinc.pkl"), 'rb') as f:
    le_counter = pickle.load(f)

with open(str(locations.UPSTREAM / "code_frequency/ce_code_counter_loinc.pkl"), 'rb') as f:
    ce_counter = pickle.load(f)

# Adapted from third_party/fhir_mlm/04_write_patient_dataset.py. Malformed lines are skipped and
# summarised at the end
def generator_from_file(read_file: str) -> Generator:
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

# MIMIC-IV on FHIR NDJSON files. The filtered files are written next to them
data_dir = str(locations.MIMIC_FHIR)

# Filter Lab Events with progress estimation
print("Processing Lab Events...")
le_file = os.path.join(data_dir, "MimicObservationLabevents.ndjson")
le_write_file = os.path.join(data_dir, "LOINCFilteredObservationLabEvents.ndjson")

# The loop below appends, so start from an empty file
if os.path.exists(le_write_file):
    os.remove(le_write_file)

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

num_le_filtered = 0
start_time = time.time()

# le_counter holds only the codes with a LOINC mapping (prep/1)
le_counter_keys = set(le_counter.keys())

# Use generator with tqdm and estimated total
le_gen = generator_from_file(le_file)
for resource in tqdm(le_gen, total=estimated_total_records, desc="Filtering Lab Events", unit="records"):
    code = resource.get("code", {}).get("coding", [{}])[0].get("code", "")
    if code in le_counter_keys:
        num_le_filtered += 1
        with open(le_write_file, "a+") as fw:
            writer = ndjson.writer(fw)
            writer.writerow(resource)

elapsed_time = time.time() - start_time
print(f"\nCompleted. {num_le_filtered:,} lab events filtered and saved to {le_write_file}")
print(f"Total processing time: {elapsed_time:.1f} seconds")

# Filter Chart Events with progress estimation
print("Processing Chart Events...")
ce_file = os.path.join(data_dir, "MimicObservationChartevents.ndjson")
ce_write_file = os.path.join(data_dir, "LOINCFilteredObservationChartEvents.ndjson")

# The loop below appends, so start from an empty file
if os.path.exists(ce_write_file):
    os.remove(ce_write_file)

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

num_ce_filtered = 0
start_time = time.time()

# ce_counter holds only the codes with a LOINC mapping (prep/1)
ce_counter_keys = set(ce_counter.keys())

# Use generator with tqdm and estimated total
ce_gen = generator_from_file(ce_file)
for resource in tqdm(ce_gen, total=estimated_total_records, desc="Filtering Chart Events", unit="records"):
    code = resource.get("code", {}).get("coding", [{}])[0].get("code", "")
    if code in ce_counter_keys:
        num_ce_filtered += 1
        with open(ce_write_file, "a+") as fw:
            writer = ndjson.writer(fw)
            writer.writerow(resource)

elapsed_time = time.time() - start_time
print(f"\nCompleted. {num_ce_filtered:,} chart events filtered and saved to {ce_write_file}")
print(f"Total processing time: {elapsed_time:.1f} seconds")

# Combine the filtered resources into one history per patient, as
# third_party/fhir_mlm/04_write_patient_dataset.py does
print("Creating patient histories from filtered resources...")

# Set up directories
read_dir = data_dir
write_dir = str(locations.WORK / "loinc_patients")
os.makedirs(write_dir, exist_ok=True)

# Initialize patient statistics tracking
patient_stats = {}  # Will store stats for each patient

print("Writing initial patient files...")
# Write initial patient files
patient_file = os.path.join(read_dir, "MimicPatient.ndjson")

# Sample first 1000 lines to estimate total patients
print("Sampling patient file to estimate total...")
sample_count = 0
file_size = os.path.getsize(patient_file)
with open(patient_file, "r") as f:
    while sample_count < 1000:
        line = f.readline()
        if not line:  # End of file
            break
        sample_count += 1
    sample_position = f.tell()

# Estimate total patients based on sample
if sample_count > 0:
    avg_bytes_per_patient = sample_position / sample_count
    estimated_total_patients = int(file_size / avg_bytes_per_patient)
    print(f"Estimated total patients: {estimated_total_patients:,} (based on {sample_count} sample records)")
else:
    estimated_total_patients = None
    print("Could not sample file for estimation")

patient_gen = generator_from_file(patient_file)
patients_created = 0
for i, resource in tqdm(enumerate(patient_gen), total=estimated_total_patients, desc="Creating patient files", unit="patients"):
    patient_id = resource["id"]
    write_file = os.path.join(write_dir, patient_id + ".ndjson")
    with open(write_file, "w") as fw:
        writer = ndjson.writer(fw)
        writer.writerow(resource)
    patients_created += 1
    
    # Initialize patient stats
    patient_stats[patient_id] = {
        'filename': patient_id + '.ndjson',
        'total_resources': 1,  # Start with 1 for the Patient resource
        'Encounter': 0,
        'Procedure': 0,
        'LabObservation': 0,
        'ChartObservation': 0,
        'Specimen': 0
    }

print(f"Done writing patient files: {patients_created:,} created")

# Process other resource types and append to patient files
print("Processing other resource types...")

# Define resource files to process (using filtered versions where available)
resource_files = [
    "MimicEncounter.ndjson",
    "MimicEncounterICU.ndjson", 
    "LOINCFilteredObservationLabEvents.ndjson",  # filtered above
    "LOINCFilteredObservationChartEvents.ndjson",  # filtered above
    "MimicSpecimen.ndjson",
    "MimicSpecimenLab.ndjson",
    "MimicProcedureICU.ndjson",
]

for file in resource_files:
    print(f"Starting to process {file}")
    read_file = os.path.join(read_dir, file)
    
    # Check if file exists
    if not os.path.exists(read_file):
        print(f"Warning: {file} not found, skipping...")
        continue
    
    # Sample first 1000 lines to estimate total records
    sample_count = 0
    file_size = os.path.getsize(read_file)
    with open(read_file, "r") as f:
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
        print(f"  Estimated records: {estimated_total_records:,}")
    else:
        estimated_total_records = None
        print("  Empty file, skipping...")
        continue
    
    file_gen = generator_from_file(read_file)
    processed_count = 0
    
    for resource in tqdm(file_gen, total=estimated_total_records, desc=f"Processing {file}", unit="records"):
        # Get patient ID from subject reference
        ref = resource.get("subject", {}).get("reference", "")
        patient_id = ref.split("/")[-1] if ref else ""
        
        if not patient_id:
            continue
            
        # Append to patient file
        write_file = os.path.join(write_dir, patient_id + ".ndjson")
        if os.path.exists(write_file):  # resources of patients without a Patient resource are skipped
            with open(write_file, "a") as fw:
                writer = ndjson.writer(fw)
                writer.writerow(resource)
            processed_count += 1
            
            # Update patient statistics
            if patient_id in patient_stats:
                patient_stats[patient_id]['total_resources'] += 1
                
                # Categorize resource type
                resource_type = resource.get('resourceType', '')
                if resource_type == 'Encounter':
                    patient_stats[patient_id]['Encounter'] += 1
                elif resource_type == 'Procedure':
                    patient_stats[patient_id]['Procedure'] += 1
                elif resource_type == 'Observation':
                    # Distinguish between Lab and Chart observations based on filename
                    if 'LabEvents' in file:
                        patient_stats[patient_id]['LabObservation'] += 1
                    elif 'ChartEvents' in file:
                        patient_stats[patient_id]['ChartObservation'] += 1
                elif resource_type == 'Specimen':
                    patient_stats[patient_id]['Specimen'] += 1
    
    print(f"Done processing {file}: appended {processed_count:,} resources")

print("Finished creating patient histories.")

# Write patient statistics to CSV
print("Writing patient statistics to CSV...")
summary_file = str(locations.UPSTREAM / "loinc_patients_summary.csv")
os.makedirs(os.path.dirname(summary_file), exist_ok=True)

# Convert patient_stats to list of dictionaries for easier CSV writing
stats_list = list(patient_stats.values())

# Write to CSV with polars
if stats_list:
    df_stats = pl.DataFrame(stats_list)
    df_stats.write_csv(summary_file)
    print(f"Patient statistics written to {summary_file}")
    print(f"Total patients with statistics: {len(stats_list):,}")
    
    # Show summary statistics
    print("\nSummary statistics:")
    print(f"Total resources across all patients: {df_stats['total_resources'].sum():,}")
    print(f"Total Encounters: {df_stats['Encounter'].sum():,}")
    print(f"Total Procedures: {df_stats['Procedure'].sum():,}")
    print(f"Total Lab Observations: {df_stats['LabObservation'].sum():,}")
    print(f"Total Chart Observations: {df_stats['ChartObservation'].sum():,}")
    print(f"Total Specimens: {df_stats['Specimen'].sum():,}")
else:
    print("No patient statistics to write")
