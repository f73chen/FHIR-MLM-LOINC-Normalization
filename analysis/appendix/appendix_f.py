"""The values and tables of thesis Appendix F (reproducibility: data, models, software, compute, code).

Sources: the data files the pipeline read (their dates), the Hugging Face cache (the model revisions),
the environment files of the release, the machine (GPUs, CPU, memory) and the LICENSE of the release.
Dates are file modification dates, i.e. when each copy was made. The asserts stop the render if a fact
the appendix states no longer holds. analysis/appendix/tables.py renders the output:

  python analysis/appendix/tables.py
"""
import datetime
import os
import re
import subprocess
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

RELEASE = locations.REPO
HF = Path.home() / ".cache" / "huggingface" / "hub"
UP = locations.UPSTREAM
MONTH = "%-d %B %Y"


#: the models in Table F.2 (name, role, Hugging Face ID). D.3 and E.1 refer to it for the IDs and revisions.
MODELS = [
    ("BioClinical ModernBERT [Sounack 2025]", "Fine-tuned (BCM)", "thomas-sounack/BioClinical-ModernBERT-large"),
    ("Llama-3.1-8B-Instruct [Grattafiori 2024]", "Fine-tuned; off-the-shelf baseline; paraphrase cleaner", "meta-llama/Llama-3.1-8B-Instruct"),
    ("MedGemma-4B [Sellergren 2025]", "Paraphrase generation; off-the-shelf baseline", "google/medgemma-4b-it"),
    ("Gemma-3-12B [Gemma3 2025]", "Off-the-shelf baseline", "google/gemma-3-12b-it"),
    ("OpenBioLLM-8B [OpenBioLLM-8B 2024]", "Off-the-shelf baseline", "aaditya/Llama3-OpenBioLLM-8B"),
    ("Meditron-7B [Meditron 2023]", "Off-the-shelf baseline", "epfl-llm/meditron-7b"),
]
#: each table as the appendix prints it: key -> (header, rows, align). tables.py draws the PNG and .tex from here.
TABLES = {}


def _table(key, header, rows, align=""):
    TABLES[key] = (list(header), [[str(c) for c in r] for r in rows], align)


def _date(p):
    return datetime.date.fromtimestamp(os.path.getmtime(p)).strftime(MONTH)


def _snapshot(repo):
    d = HF / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(os.listdir(d))
    assert len(snaps) == 1, (repo, snaps)   # one cached snapshot: the revision every run loaded
    return snaps[0]


def _pins(path, names):
    pins = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m and m.group(1).lower() in names:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def values():
    # -- data --
    assert (UP / "Loinc_2.82").is_dir()
    athena = locations.ATHENA
    assert not (athena / "VOCABULARY.csv").exists()
    cdisc_days = sorted([datetime.date.fromtimestamp(os.path.getmtime(UP / f))
                         for f in ("loinc_to_cdisc_lb_map.csv", "CDISC_SDTM_Terminology.csv")])
    assert cdisc_days[0].strftime("%B %Y") == cdisc_days[1].strftime("%B %Y")
    # a few words per cell, one line per row. Some facts are true of the copies but left out of the table.
    # The MIMIC-IV PhysioNet Credentialed Health Data Use Agreement is version 1.5.0. The map is
    # d_labitems_to_loinc.csv in the MIT-LCP mimic-code repository. The RELMA export holds LOINC 2.81, the release
    # then current. UMLS 2025AB was the release then current, queried through the UTS API on 11 April 2026. The
    # Athena export holds SNOMED CT, CIEL and LOINC and does not record their versions. The PCORnet value set is
    # the CDM's value set for qualitative results.
    data_rows = [
        ["MIMIC-IV [MIMIC-IV v2.2] [Johnson 2023]", "2.2", "PhysioNet, credentialed"],
        ["MIMIC-IV on FHIR [MIMIC-IV on FHIR v1.0] [Bennett 2023]", "1.0", "PhysioNet, credentialed"],
        ["MIMIC-IV LOINC map", _date(UP / "mimic_loinc.csv"), "GitHub, public"],
        ["LOINC [LOINC Users' Guide, 2026]", "2.82", "Regenstrief, LOINC licence"],
        ["RELMA Community Mappings [Vreeman 2015]", f"RELMA 7.14, {_date(UP / 'relma_custom_codes.csv')}", "Regenstrief, no redistribution"],
        ["UMLS Metathesaurus [Bodenreider 2004]", "2025AB", "UMLS licence"],
        ["OHDSI Athena vocabularies", _date(athena / "CONCEPT.csv"), "OHDSI"],
        ["CDISC terminology and LOINC map", f"{cdisc_days[0].day} and {cdisc_days[1].strftime(MONTH)}", "CDISC"],
        ["PCORnet value set", "1.14", "PCORnet"],
    ]
    _table("F.1", ["Source", "Version or date", "Access"], data_rows)

    # -- models --
    models = MODELS
    assert _snapshot("meta-llama/Llama-3.1-8B-Instruct") == _snapshot("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model_rows = [["FHIR-MLM [Yu 2025 (MASc thesis)]", "Fine-tuned", "Private", "—"]]
    model_rows += [[m, role, repo, _snapshot(repo)] for m, role, repo in models]
    assert all(len(r[3]) == 40 for r in model_rows[1:])            # full commit hashes
    _table("F.2", ["Model", "Role", "Hugging Face ID", "Revision"], model_rows)

    # -- software --
    keys = {"torch", "transformers", "peft", "bitsandbytes", "accelerate", "scikit-learn", "optuna"}
    da = _pins(RELEASE / "environment" / "requirements-data-analysis.txt", keys)
    dm = _pins(RELEASE / "environment" / "requirements-deep-models.txt", keys)
    head_da = open(RELEASE / "environment" / "requirements-data-analysis.txt").read()
    head_dm = open(RELEASE / "environment" / "requirements-deep-models.txt").read()
    assert "Python 3.10" in head_da and "Python 3.11, CUDA 12.4" in head_dm and "flash-attn==2.7.4.post1" in head_dm
    order = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "scikit-learn", "optuna"]
    sw_rows = [["Python", "3.10", "3.11 (CUDA 12.4)"]]
    sw_rows += [[k, da.get(k, "—"), dm.get(k, "—")] for k in order]
    sw_rows += [["flash-attn", "—", "2.7.4.post1"]]
    _table("F.3", ["Package", "Data and analysis", "Deep models"], sw_rows)

    # -- hardware --
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()
    assert len(gpus) == 2 and all("RTX 3090" in g for g in gpus)
    gpu_mem = round(int(re.search(r"(\d+) MiB", gpus[0]).group(1)) / 1024)
    cpu = re.search(r"Model name:\s+(.*)", subprocess.run(["lscpu"], capture_output=True, text=True).stdout).group(1).strip()
    mem_gib = round(int(re.search(r"MemTotal:\s+(\d+) kB", open("/proc/meminfo").read()).group(1)) / 1024 ** 2)

    lic = open(RELEASE / "LICENSE").read()
    assert "Apache License" in lic and "Version 2.0" in lic

    v = {
        "gpu": f"two NVIDIA GeForce RTX 3090 GPUs with {gpu_mem} GB of memory each",
        "cpu": cpu, "mem": f"{mem_gib} GiB",
        "repo": "https://github.com/f73chen/FHIR-MLM-LOINC-Normalization",
    }
    return v
