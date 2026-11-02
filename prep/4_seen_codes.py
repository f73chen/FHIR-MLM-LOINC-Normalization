"""Write seen_codes.csv and mimic_loinc.csv (thesis Chapter 4).

seen_codes.csv is the code list. It holds every MIMIC-IV lab and chart code with a LOINC mapping,
and its LOINC target. mimic_loinc.csv is the LOINC table restricted to those targets.

Reads the LOINC-mapped code counts from prep/1_code_frequency_analysis.py
(UPSTREAM_DIR/code_frequency/), the mimic-code LOINC mappings (MIMIC_CODE_DIR/mimic-iv/mapping/) and
UPSTREAM_DIR/Loinc.csv. Writes UPSTREAM_DIR/seen_codes.csv with the columns mimic_code, mimic_desc,
loinc_code, target_code_system and observation_type ('lab' or 'nonlab'). Most later steps read it as
the MIMIC-to-LOINC map. Also writes UPSTREAM_DIR/mimic_loinc.csv (selected LOINC columns), which is
the input to prep/4.1 and prep/4.3.

  python prep/4_seen_codes.py
"""

import os
import copy
import pickle
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
import utils

# File paths
LAB_OBS_MAP_FILEPATH = str(locations.MIMIC_CODE / "mimic-iv/mapping/d_labitems_to_loinc.csv")
OTHER_OBS_MAP_FILEPATH = str(locations.MIMIC_CODE / "mimic-iv/mapping/meas_chartevents_main.csv")

def record_seen_codes(output_path, le_counter, ce_counter):
    """
    Write seen_codes.csv from the precomputed code counters: each MIMIC code with its LOINC mapping
    and observation type ('lab' or 'nonlab').

    Args:
        output_path: Path to save the CSV file
        le_counter: Lab events counter with MIMIC codes as keys
        ce_counter: Chart events counter with MIMIC codes as keys
    """
    # Load mapping files
    lab_obs_map, _ = utils.load_mimic_iv_map(LAB_OBS_MAP_FILEPATH)
    other_obs_map, _ = utils.load_mimic_iv_map(OTHER_OBS_MAP_FILEPATH)
    
    # Dictionary to store unique codes and their mappings
    unique_codes = {}
    
    # Process lab events codes from le_counter
    print(f"Processing {len(le_counter)} lab event codes...")
    for mimic_code in le_counter.keys():
        if mimic_code in lab_obs_map:
            mapping = copy.deepcopy(lab_obs_map[mimic_code])
            mapping["observation_type"] = "lab"
            unique_codes[mimic_code] = mapping
        else:
            print(f"Warning: Lab code {mimic_code} not found in mapping file")
    
    # Process chart events codes from ce_counter
    print(f"Processing {len(ce_counter)} chart event codes...")
    for mimic_code in ce_counter.keys():
        if mimic_code in other_obs_map:
            mapping = copy.deepcopy(other_obs_map[mimic_code])
            mapping["observation_type"] = "nonlab"
            unique_codes[mimic_code] = mapping
        else:
            print(f"Warning: Chart code {mimic_code} not found in mapping file")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(list(unique_codes.values()))
    
    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Calculate unique counts
    unique_mimic_codes = len(df)
    unique_loinc_codes = df['loinc_code'].nunique()
    
    # Count MIMIC codes with null LOINC codes
    null_loinc_count = df['loinc_code'].isnull().sum()
    
    print(f"Saved {len(df)} unique codes to {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique MIMIC codes: {unique_mimic_codes}")
    print(f"Unique LOINC codes: {unique_loinc_codes}")
    print(f"MIMIC codes with null LOINC codes: {null_loinc_count}")
    
    return df

def filter_loinc_database(seen_codes_path, loinc_path, output_path):
    """
    Filter the LOINC CSV to only include rows where LOINC_NUM is in the seen codes
    and keep only the specified columns.
    """
    # Read the seen codes to get the list of LOINC codes we need
    seen_codes_df = pd.read_csv(seen_codes_path)
    seen_loinc_codes = set(seen_codes_df['loinc_code'].dropna().unique())
    
    print(f"Found {len(seen_loinc_codes)} unique LOINC codes in seen codes")
    
    # Read the full LOINC dataset
    loinc_df = pd.read_csv(loinc_path, low_memory=False)
    print(f"Original LOINC dataset shape: {loinc_df.shape}")
    
    # Define the columns we want to keep
    desired_columns = [
        'LOINC_NUM', 'COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 
        'SCALE_TYP', 'METHOD_TYP', 'LONG_COMMON_NAME', 'SHORTNAME', 
        'RELATEDNAMES2', 'DisplayName'
    ]
    
    # Check which columns exist in the LOINC dataset
    available_columns = [col for col in desired_columns if col in loinc_df.columns]
    missing_columns = [col for col in desired_columns if col not in loinc_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in LOINC dataset: {missing_columns}")
    
    print(f"Available columns: {available_columns}")
    
    # Filter rows where LOINC_NUM is in our seen codes
    filtered_df = loinc_df[loinc_df['LOINC_NUM'].isin(seen_loinc_codes)]
    
    # Select only the available desired columns
    filtered_df = filtered_df[available_columns]
    
    print(f"Filtered LOINC dataset shape: {filtered_df.shape}")
    print(f"Number of matching LOINC codes found: {len(filtered_df)}")
    
    # Save the filtered dataset
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered LOINC data to {output_path}")
    
    return filtered_df

loinc_path = str(locations.UPSTREAM / "Loinc.csv")
seen_codes_path = str(locations.UPSTREAM / "seen_codes.csv")
mimic_loinc_path = str(locations.UPSTREAM / "mimic_loinc.csv")

# Load frequency counters
with open(str(locations.UPSTREAM / "code_frequency/le_code_counter_loinc.pkl"), 'rb') as f:
    le_counter = pickle.load(f)
    le_counter_keys = set(le_counter.keys())

with open(str(locations.UPSTREAM / "code_frequency/ce_code_counter_loinc.pkl"), 'rb') as f:
    ce_counter = pickle.load(f)
    ce_counter_keys = set(ce_counter.keys())

# Create seen codes CSV using the pre-computed frequency counters
seen_codes_df = record_seen_codes(output_path=seen_codes_path, le_counter=le_counter, ce_counter=ce_counter)

# Filter LOINC database based on the codes we identified
filtered_df = filter_loinc_database(seen_codes_path=seen_codes_path, loinc_path=loinc_path, output_path=mimic_loinc_path)
