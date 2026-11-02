# FHIR-MLM-LOINC-Normalization

This is the code for the MASc thesis *Mapping Local Laboratory Codes to LOINC Using FHIR Patient Records* by
Fenglin (Grace) Chen (University of Waterloo, 2026).

The code starts from the MIMIC-IV on FHIR files and ends with the results in the thesis. It prepares the data,
builds the label space and the paraphrase pools, and simulates a second site by perturbing values, reference
ranges, and units. It trains four models: TF-IDF with logistic regression, FHIR-MLM (a FHIR-pretrained encoder),
BioClinical ModernBERT (BCM), and Llama-3.1-8B-Instruct. It also prompts off-the-shelf language models as
baselines and runs the analyses.

This is research code. It is not intended for clinical use.

## Layout

| Folder | Contents |
|---|---|
| `prep/` | data preparation: code counts, per-patient summaries, the code list, external descriptions, paraphrase generation, and the unit and qualitative-value maps |
| `pipeline/` | the split, label space, paraphrase cleaning, perturbation, dataset builders, training, and evaluation, numbered in run order |
| `analysis/` | scripts for the numbers, tables, and figures in the thesis. `analysis/appendix/` writes the tables and numbers of Appendices A–F, and `analysis/lib/` holds modules the analysis scripts share |
| `common/` | modules used across the repository. `common/locations.py` says where every input and output lives |
| `third_party/fhir_mlm/` | where Trevor Yu's FHIR-MLM code goes (not included, see below) |
| `resources/` | a few small files (see below) |
| `environment/` | pinned requirements for the two Python environments |
| `tests/` | unit tests |

## Data

The repository has no data. It does not include MIMIC-IV records, splits, patient lists, model weights,
paraphrase pools, or LOINC and RELMA files. Running the code needs credentialed access to MIMIC-IV on PhysioNet
and the downloads in the table below.

`resources/` holds a few small files. These are the unit synonym map, which `prep/4.3` builds from MIMIC-IV
unit spellings and public vocabularies, three hand-edited alias lists, and the perturbation-evidence registers
from public sources (CLIA 2026, RCPAQAP, and a literature register).

Trevor Yu's FHIR-MLM code is not included either. It consists of the tokenizer, data collator, and model
code, and the two scripts that write the per-patient files. FHIR-MLM's tokenizer files, pretrained checkpoint, and
pretraining datasets are not included. His thesis describes them (Yu, *Encoding FHIR Medical Data for
Transformers*, University of Waterloo, 2025). The code expects his scripts in `third_party/fhir_mlm/`. Without
his work, step 1, the split in step 3, and every step that builds, trains, or evaluates FHIR-MLM cannot run.

| Data | Source | Used by |
|---|---|---|
| MIMIC-IV on FHIR v1.0 | PhysioNet (credentialed access) | everything |
| `mimic-iv/mapping/*.csv` | github.com/MIT-LCP/mimic-code | `prep/1`, `prep/4` |
| LOINC 2.82 (`Loinc.csv`, `LoincTableCore.csv`) | loinc.org | label space, paraphrase cleaning, perturbation |
| local test names from RELMA's Community Mappings | RELMA (Regenstrief Institute), exported by hand one LOINC code at a time | `pipeline/17` |
| UMLS Metathesaurus API | NLM UTS (licence and API key) | `prep/4.1` |
| OHDSI Athena vocabulary export | athena.ohdsi.org | `prep/4.1` |
| CDISC SDTM terminology and the LOINC-to-LB map | CDISC | `prep/4.1`, `prep/4.3` |
| PCORnet CDM v1.14 value sets (QUAL, UNIT) | PCORnet | `prep/4.3`, `prep/4.4` |
| CIEL concepts | Open Concept Lab API | `prep/4.3` |
| `ucum.csv` | NLM ucum-lhc | `prep/4.3`, `analysis/lib/units.py` |
| CLIA 2026 proficiency-testing limits (PDF) | CMS | `pipeline/parse_clia_limits.py`, `parse_clia_subspecialty.py` |
| RCPAQAP allowable limits (PDF) | RCPAQAP | `pipeline/parse_rcpaqap_categories.py` |
| `parsed_units_df` from the lab2clean R package, v2.0.0 (GPL-3) | CRAN | `analysis/a16`, `a19`, `a20` |
| table of LOINC-code-changing unit conversions | Hauser et al., JAMIA 2018 | `analysis/a16`, `a19`, `a20` |
| `loinc_conversion_map.csv` | PrecisionHealthIntelligence | `analysis/a16`, `a19`, `a20` |
| FHIR-MLM tokenizer files, pretrained checkpoint, and pretraining datasets | Trevor Yu's FHIR-MLM work (not included) | `pipeline/01`, FHIR-MLM datasets, training, and evaluation |
| `google/medgemma-4b-it`, `meta-llama/Llama-3.1-8B-Instruct`, `thomas-sounack/BioClinical-ModernBERT-large`, `google/gemma-3-12b-it`, `aaditya/Llama3-OpenBioLLM-8B`, `epfl-llm/meditron-7b` | Hugging Face (some gated) | paraphrases, models, and baselines |

Most downloads go in `UPSTREAM_DIR`, under the file names that each script's docstring gives.

`prep/4` builds `mimic_loinc.csv`, the input to `prep/4.1` and `4.3`. For the thesis it was built from a LOINC
release older than 2.82, the version the rest of the code uses. Rebuilding it with 2.82 gives the same 862
codes, but 125 of them get different names.

## Setup

The deep models were trained with different library versions from the rest of the code, so there are two
Python environments. The deep-models environment runs the deep models' datasets, training, and evaluation
(`pipeline/09`–`15`, `18`, and `23`–`25`), the three analysis scripts in step 7 that read their datasets,
and the check `analysis/verify_obs_ordinal_join.py`. The data-analysis environment runs everything else.

Data-analysis environment (Python 3.10):

```
pip install -r environment/requirements-data-analysis.txt
```

Deep-models environment (Python 3.11, CUDA 12.4):

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/requirements-deep-models.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

The last line installs a prebuilt flash-attn wheel for CUDA 12, torch 2.6, and Python 3.11. Building flash-attn
from source needs the CUDA toolkit. The two `parse_clia_*` scripts and `parse_rcpaqap_categories.py` also need
`pdftotext` (poppler-utils).

Set these environment variables to point the code at your copies. Each one defaults to a folder inside the
repository that git ignores (see `common/locations.py`). Some outputs always go to `data/` and `results/` inside
the repository, which git also ignores.

| Variable | Holds |
|---|---|
| `MIMIC_FHIR_DIR` | the MIMIC-IV on FHIR ndjson files |
| `PATIENT_NDJSON_DIR` | the per-patient files from step 1 |
| `UPSTREAM_DIR` | the `prep/` outputs and most downloads |
| `FHIR_MLM_TOKENIZER_DIR`, `FHIR_MLM_BASE_CHECKPOINT` | FHIR-MLM's tokenizer files and pretrained checkpoint |
| `PRETRAIN_DATASETS_DIR` | FHIR-MLM's pretraining datasets, whose patients the split keeps out |
| `SOURCE_DOCS_DIR` | the CLIA and RCPAQAP PDFs |
| `MIMIC_CODE_DIR`, `ATHENA_DIR` | the mimic-code checkout and the Athena export |
| `WORK_DIR` | large outputs: datasets, models, evaluation outputs, and caches |
| `CMU_FONT_DIR` | the Computer Modern Unicode fonts used in the figures (the `cm-unicode` OpenType files from TeX Live) |

Two more variables hold keys. `prep/4.1` reads your UMLS API key from `UMLS_API_KEY`. `HF_TOKEN` holds a
Hugging Face token for the gated models, and `prep/4.2` also reads it from a `.env` file in the folder you run it
from. Accept each gated model's terms on Hugging Face before you download it.

The thesis ran on one machine with two NVIDIA GeForce RTX 3090 GPUs (24 GB each).

## Running the code

Run the steps in this order. Each script's docstring says what it reads and writes, and gives its command line.

### 1. Per-patient files

Trevor Yu's `04_write_patient_dataset.py` and then `04_just_add_medrequest.py` (not included) write one ndjson
file per patient, `<patient id>.ndjson`, to `PATIENT_NDJSON_DIR`. Each file starts with the Patient resource,
followed by the resources whose `subject` is that patient, file by file in this order: Encounter, Condition,
Procedure, ObservationLabevents, ObservationMicroTest, ObservationMicroOrg, ObservationMicroSusc, Specimen,
SpecimenLab, MedicationDispense, MedicationAdministration, and MedicationRequest.

### 2. Preparation

```
python prep/0_install_resources.py
python prep/1_code_frequency_analysis.py
python prep/2_create_loinc_patients.py
python prep/4_seen_codes.py                              # the code list
UMLS_API_KEY=<key> python prep/4.1_collect_external_descriptions.py
python prep/4.2_generate_paraphrases_external_desc.py    # GPU
python prep/shared_paraphrase_garbage_check.py
python prep/4.4_pcornet_qual_normalize.py
python prep/4.3_collect_value_units.py
```

`prep/4.3` writes `loinc_measurements_enriched.jsonl`, which step 5 needs, and rebuilds the unit synonym map
(the copy in `resources/` is its output). It reads the patient lists of an earlier split
(`UPSTREAM_DIR/train_test_splits_mlm/`, not included). Without them, run it after step 3, with step 3's
`data/splits/train_patients.txt` and `test_patients.txt` copied there. The unit lists may then differ from
those in the thesis.

`prep/4.2` writes paraphrases with MedGemma-4B and checks them with Llama-3.1-8B-Instruct. Two more LLM passes
over the paraphrases ran outside this code, and their outputs are not included. The first ran after `prep/4.2`
and removed garbage lines that the Llama check had kept. `analysis/appendix/appendix_c.py` counts its rejected
lines from `UPSTREAM_DIR/paraphrases_ext_rejected_second/`, which holds one file per LOINC code. Those lines
were also removed from `paraphrases_ext_verified/`. The second pass is the review in step 4.

### 3. Split and label space

```
python pipeline/01_build_split.py --contamination-set legacy8192
python pipeline/02_derive_labelspace.py
```

The split in the thesis kept out only the patients from FHIR-MLM's second pretraining stage, and
`--contamination-set legacy8192` reproduces it. For a new split, leave the flag out. The default (`union`) keeps
out the patients from both stages.

### 4. Paraphrase pools

```
python pipeline/03_paraphrase_hygiene.py
python pipeline/03b_build_concepts.py
python pipeline/04_finalize_paraphrases.py --removals-json <review result>
python pipeline/05_supplemental_garbage.py
```

An LLM review of the pools from `03` checked that MedGemma followed its instructions and removed obvious
garbage lines that a reviewer without medical training could spot. It did not judge clinical validity. Its
removal lists are not included. With an empty folder as `--removals-json`, `04` removes no lines, and the pools
differ from those in the thesis.

### 5. Perturbation evidence

The `parse_*` scripts rebuild the CLIA and RCPAQAP registers from the PDFs. Without the PDFs, use the copies that
`prep/0_install_resources.py` installed in step 2. Then run:

```
python analysis/a15_field_identifiability.py --stage all --jobs 16
python pipeline/build_numeric_obs_counts.py
python analysis/a30_measurement_scales.py
python analysis/a37_stated_intervals.py
python analysis/a20_unit_code_concentration.py
python analysis/a31_unit_disagreement_resolution.py
python analysis/a27_register_verification.py
python pipeline/build_unit_randomisation.py
python pipeline/build_perturbation_spec.py
```

### 6. Models

TF-IDF uses `06`, `07`, `08`, `08b`, and `16`, and `17` builds the RELMA pools. The deep models' datasets come
from `09`–`11`, `18`, and `23`–`25`, training from `12`–`14`, and evaluation from `15`.

Use the flags in each docstring's example. Without flags, `06` builds an earlier text format
(`--representation v1`), and `18`, `23`–`25`, and `analysis/b4_oneshot_baselines.py` use the earlier condition
`para`. The thesis used `--representation v3` and the conditions `para_xsite` and `nodesc_xsite`. `para_xsite`
gives each observation a paraphrased description, and `nodesc_xsite` gives none. The `clean` arm keeps the
original numbers, and the `pert` arm uses the perturbed ones. `analysis/lib/registry.py` lists the runs the
analyses read. Run `15` and `16` with `--dump-full-logits`. The calibration analyses need the full logits, and
the analyses that compare models run only when every model's main evaluation has them.

Training runs on two GPUs, with `torchrun --nproc_per_node=2` for `12` and `13` and
`accelerate launch --num_processes=2` for `14`. GPU training is not bit-for-bit reproducible, so repeated runs
of one command differ slightly. In the BCM runs for the thesis, the new classification heads were initialized
before the seed was set, so those checkpoints cannot be rebuilt from their seeds. `13_bcm_train.py` sets the seed
before it builds the model, so the head of a new run follows `--seed`.

### 7. Analysis

```
python analysis/refrange_diagnostic.py --n-train 100000 --write
python analysis/a1_train_counts.py
python analysis/b4_oneshot_baselines.py --model <model> --condition para_xsite --test-arm <clean|pert> --write
python analysis/b4_oneshot_baselines.py --stage score --write
python analysis/v3_cost.py --stage tfidf
python analysis/v3_all.py
python analysis/appendix/tables.py
```

`refrange_diagnostic.py` records the local code of every scored observation, which the Chapter 6 analyses read.
The baselines need a GPU, and `b4_oneshot_baselines.py` runs once per model and test arm. `<model>` is
`medgemma-4b`, `llama31-8b`, `gemma3-12b`, `openbio-8b`, or `meditron-7b`. `v3_cost.py --stage tfidf` times
TF-IDF on the CPU, so run it once on a quiet machine.

`v3_all.py` runs every Chapter 6 analysis in order, from the evaluation outputs of `15` and `16`. Its last steps
write the tables, the figures, and the numbers book (`results/analysis/v3/numbers.md`). The numbers book lists
the numbers that Chapters 6–8 quote, with the observations each one is computed on. `appendix/tables.py` writes
the tables and numbers of the appendices.

The Chapter 2, 4, and 5 scripts (`ch2_*`, `ch4_*`, and `ch5_*`) run on their own. So do the perturbation
analyses that step 5 does not need: `a16`, `a17` then `a22`, `a19`, `a21`, `a25`, `a26`, and
`a32_bin_change_final`. `ch2_encoding_density_v3_codelast.py`, `ch2_two_encodings_tokens.py`, and
`ch4_relma_coverage_v3.py` run in the deep-models environment, because they read the deep models' datasets or
tokenizer. Results go to `results/`, which git ignores.

## Tables and figures

The scripts name tables and figures by ID (T-41, F-26), and `v3_tables.py` and `v3_figures.py` take these IDs in
`--only`. The thesis numbers them by chapter:

| Thesis | ID | Script |
|---|---|---|
| Table 2.1 | T-40 | `analysis/ch2_loinc_axes_table.py` |
| Figure 2.1 | F-16 | `analysis/ch2_fhir_observation_example.py` |
| Figure 2.2 | F-24 | `analysis/ch2_two_encodings_figure.py` |
| Table 4.1 | T-01 | `analysis/ch4_split_observations_table.py` |
| Table 4.2 | T-04 | `analysis/ch4_frequency_bins_table.py` |
| Table 4.3 | T-38 | `analysis/ch4_magnitude_sources_table.py` |
| Figure 4.1 | F-15 | `analysis/ch4_imbalance_figure.py` |
| Figure 4.2 | F-27 | `analysis/ch4_paraphrase_similarity_figure.py` |
| Table 5.1 | T-08 | `analysis/ch5_model_sizes_table.py` |
| Table 5.2 | T-39 | `analysis/ch5_train_test_matrix_table.py` |
| Figure 5.1 | F-19 | `analysis/ch5_two_units_bridge_figure.py` |

Chapter 6 tables (`analysis/v3_tables.py`):

| Table | ID | Table | ID | Table | ID | Table | ID |
|---|---|---|---|---|---|---|---|
| 6.1 | T-10 | 6.8 | T-15 | 6.15 | T-42 | 6.22 | T-30 |
| 6.2 | T-11 | 6.9 | T-16 | 6.16 | T-43 | 6.23 | T-19 |
| 6.3 | T-12 | 6.10 | T-17 | 6.17 | T-47 | 6.24 | T-20 |
| 6.4 | T-14 | 6.11 | T-49 | 6.18 | T-48 | 6.25 | T-25 |
| 6.5 | T-32 | 6.12 | T-41 | 6.19 | T-24 | 6.26 | T-28 |
| 6.6 | T-34 | 6.13 | T-46 | 6.20 | T-21 | 6.27 | T-26 |
| 6.7 | T-35 | 6.14 | T-45 | 6.21 | T-22 | 6.28 | T-50 |

Chapter 6 figures (`analysis/v3_figures.py`):

| Figure | ID | Figure | ID | Figure | ID | Figure | ID |
|---|---|---|---|---|---|---|---|
| 6.1 | F-12 | 6.3 | F-26 | 6.5 | F-06 | 6.7 | F-10a |
| 6.2 | F-02 | 6.4 | F-05 | 6.6 | F-11 | 6.8 | F-10b |

The appendix tables (A.1–F.3) have the same numbers as in the thesis. `analysis/appendix/tables.py` writes them.

## Tests

```
pytest tests -m "not slow"
```

Most tests run on small synthetic inputs. Tests that need the data or the deep-models environment fail or are
skipped without them. The slow tests rebuild a FHIR tokenizer or scan real evaluation outputs.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{chen2026mapping,
  author  = {Chen, Fenglin},
  title   = {Mapping Local Laboratory Codes to {LOINC} Using {FHIR} Patient Records},
  school  = {University of Waterloo},
  address = {Waterloo, ON, Canada},
  year    = {2026}
}
```

`CITATION.cff` has the same reference.

## Credits and licence

The FHIR-MLM tokenizer, data collator, and model code are Trevor Yu's (University of Waterloo) and are not
included. This repository is released under the Apache License 2.0 (see `LICENSE`), and `NOTICE` lists the
attributions.

PhysioNet distributes MIMIC-IV and MIMIC-IV on FHIR under the PhysioNet Credentialed Health Data License. If you
use them, cite the datasets and PhysioNet, as PhysioNet asks.

This material contains content from LOINC (http://loinc.org). LOINC is copyright © Regenstrief
Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is
available at no cost under the license at http://loinc.org/license. LOINC® is a registered United
States trademark of Regenstrief Institute, Inc.
