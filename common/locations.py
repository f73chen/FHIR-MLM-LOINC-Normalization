"""Where the code reads its inputs and writes its outputs.

Every location outside the repository can be set with an environment variable. The defaults keep
everything inside the repository, in folders that git ignores (data/, work/, results/).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "common"
PIPELINE = REPO / "pipeline"
ANALYSIS = REPO / "analysis"
THIRD_PARTY = REPO / "third_party" / "fhir_mlm"
RESOURCES = REPO / "resources"


def _location(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default))


# Inputs that come from outside this repository's pipeline.
MIMIC_FHIR = _location("MIMIC_FHIR_DIR", REPO / "data" / "mimic-iv-fhir")            # MIMIC-IV on FHIR v1.0 ndjson (PhysioNet)
PATIENTS = _location("PATIENT_NDJSON_DIR", REPO / "data" / "hosp_patients")         # one ndjson file per patient
UPSTREAM = _location("UPSTREAM_DIR", REPO / "data" / "upstream")                    # prep/ outputs, LOINC tables, the RELMA export
TOKENIZER = _location("FHIR_MLM_TOKENIZER_DIR", REPO / "data" / "fhir_mlm_tokenizer")  # FHIR-MLM tokenizer files
FHIR_MLM_BASE = _location("FHIR_MLM_BASE_CHECKPOINT", REPO / "data" / "fhir_mlm_base")  # pretrained FHIR-MLM checkpoint
PRETRAIN_DATASETS = _location("PRETRAIN_DATASETS_DIR", REPO / "data" / "pretrain_datasets")  # FHIR-MLM pretraining sets
SOURCE_DOCS = _location("SOURCE_DOCS_DIR", REPO / "data" / "source_documents")      # CLIA and RCPAQAP documents (PDF)
MIMIC_CODE = _location("MIMIC_CODE_DIR", REPO / "data" / "mimic-code")              # github.com/MIT-LCP/mimic-code
ATHENA = _location("ATHENA_DIR", REPO / "data" / "athena")                         # OHDSI Athena vocabulary export (prep/4.1)
CMU_FONTS = _location("CMU_FONT_DIR", REPO / "data" / "cm-unicode")               # Computer Modern Unicode OpenType fonts (TeX Live), for the figures

# Large outputs: datasets, models, evaluation outputs, caches.
WORK = _location("WORK_DIR", REPO / "work")
