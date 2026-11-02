#!/usr/bin/env python
"""Write the data for F-24: what the input pipeline of each model makes of one lab Observation (thesis
Chapter 2).

Runs the three pipelines as the v3 dataset builders call them, on a synthetic patient (one Patient,
one Encounter, one sodium Observation), and writes the tokens to
results/analysis/ch2/two_encodings_tokens.json for ch2_two_encodings_figure.py to draw:
  - FHIR-MLM: 09_mlm_dataset._build_fhir_tokenizer(with_desc=True) on the sorted resources. The
    Observation tokens are the ones with the Observation token type.
  - BCM / Llama: 10_bcm_dataset / 11_llama_dataset ._build_*_tokenizer_formatter(with_desc=True,
    rep="v3"). The text block of the Observation is tokenised on its own, and every piece is assigned
    to the line (field) that its first character sits on.

The Observation is the running example of the thesis, which F-16 and F-17 also show. It is sodium
(MIMIC item 50983, mapped to LOINC 2951-2, Sodium SerPl-sCnc), 139 mEq/L, with reference range
133-145 mEq/L. There is no patient data. Every resource is made up, and only the shape follows
MIMIC-IV on FHIR. Its timestamp carries a +-hh:mm offset because every real one does. Sodium is one
of the 25 codes whose unit notation v3 draws per patient (mEq/L or mmol/L). This is the mEq/L draw,
which the text formatter writes as "meq/L".

Needs the FHIR-MLM tokenizer code (third_party/fhir_mlm) and the Hugging Face tokenizers, so it runs
in the deep-models environment:

  python analysis/ch2_two_encodings_tokens.py
"""
from __future__ import annotations
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "pipeline"
OUT = HERE.parent / "results" / "analysis" / "ch2" / "two_encodings_tokens.json"

U = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-units"
EDT = "2154-03-02T09:41:00-05:00"


def q(v, unit="mEq/L"):
    return {"code": unit, "unit": unit, "value": v, "system": U}


PATIENT = {"resourceType": "Patient", "id": "pat-synth", "gender": "female", "birthDate": "2094-01-01"}
ENCOUNTER = {"resourceType": "Encounter", "id": "enc-synth", "status": "finished",
             "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP"},
             "subject": {"reference": "Patient/pat-synth"},
             "period": {"start": "2154-02-28T08:00:00-05:00", "end": "2154-03-05T14:00:00-05:00"}}
OBSERVATION = {
    "resourceType": "Observation", "id": "obs-synth", "status": "final",
    "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-observation-labevents"]},
    "code": {"coding": [{"code": "50983", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems",
                         "display": "Sodium"}]},
    "subject": {"reference": "Patient/pat-synth"}, "encounter": {"reference": "Encounter/enc-synth"},
    "category": [{"coding": [{"code": "laboratory", "system": "http://terminology.hl7.org/CodeSystem/observation-category"}]}],
    "extension": [{"url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-lab-priority", "valueString": "ROUTINE"}],
    "effectiveDateTime": EDT, "valueQuantity": q(139), "referenceRange": [{"low": q(133), "high": q(145)}],
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _sorted_resources():
    from fhir_formatter import get_encounters, sort_key
    rs = [copy.deepcopy(r) for r in (PATIENT, ENCOUNTER, OBSERVATION)]
    encs = get_encounters(rs)
    return sorted(rs, key=lambda r: sort_key(r, encs))


def fhir_mlm_tokens(mlm):
    tok = mlm._build_fhir_tokenizer(with_desc=True)
    names = {}
    for mapping in tok.code_mapping.values():
        for code, i in mapping.items():
            names.setdefault(i, code)
    out = tok(_sorted_resources())
    obs_type = tok.token_type_id_mapping["Observation"]
    ids = [i for i, t, a in zip(out["input_ids"], out["token_type_ids"], out["attention_mask"]) if a == 1 and t == obs_type]
    return [names[i] if i in names else tok.text_tokenizer.convert_ids_to_tokens(i) for i in ids]


def text_pieces(builder):
    tokenizer, formatter = builder(True, True, "v3")
    prompt, _ = formatter.format_resources(_sorted_resources())
    block = prompt[prompt.find("Resource: Observation"):].split("\n\n")[0].rstrip("\n")
    enc = tokenizer(block, add_special_tokens=False, return_offsets_mapping=True)
    starts = [0] + [m.end() for m in re.finditer("\n", block)]
    lines = block.split("\n")
    rows = [[] for _ in lines]
    for piece, (a, _) in zip(tokenizer.convert_ids_to_tokens(enc["input_ids"]), enc["offset_mapping"]):
        rows[max(i for i, s in enumerate(starts) if s <= a)].append(piece)
    return {"block": block, "lines": lines, "pieces_by_line": rows, "n": len(enc["input_ids"])}


def main():
    sys.argv = sys.argv[:1]                       # the dataset modules parse argv at import
    mlm = _load(SCRIPTS / "09_mlm_dataset.py", "mlm09")
    bcm = _load(SCRIPTS / "10_bcm_dataset.py", "bcm10")
    llama = _load(SCRIPTS / "11_llama_dataset.py", "llama11")
    result = {"observation": OBSERVATION,
              "fhir_mlm": fhir_mlm_tokens(mlm),
              "bcm": text_pieces(bcm._build_bcm_tokenizer_formatter),
              "llama": text_pieces(llama._build_llama_tokenizer_formatter)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("FHIR-MLM:", len(result["fhir_mlm"]), result["fhir_mlm"])
    for k in ("bcm", "llama"):
        print(f"{k}: {result[k]['n']} pieces |", [len(r) for r in result[k]["pieces_by_line"]])
        for line, row in zip(result[k]["lines"], result[k]["pieces_by_line"]):
            print(f"   {line!r}: {row}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
