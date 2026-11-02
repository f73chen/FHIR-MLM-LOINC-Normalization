#!/usr/bin/env python
"""Count how many lab observations fit in one 8,192-token window, per model, on the v3 test splits
(thesis Chapter 2).

The FHIR-native encoding fits more lab observations per window than the text serialisation BCM and Llama
read. v3 writes every number in fixed-width scientific notation, reference-range bounds included, and
standardises unit strings. That affects BCM and Llama only, because FHIR-MLM turns each number into one
percentile token. The Llama datasets are the code-last builds, with the Code line last in each
Observation.

Method:
    * per family (mlm / bcm / llama), the test split of `{fam}__para_xsite__{clean,pert}`. One row is the
      first window of one patient
    * patients present in all three families (in effect the FHIR-MLM patients, i.e. the test patients
      with an Encounter)
    * per patient, the attended FHIR-MLM lab tokens restricted to LOINC-mappable classes, and the labelled
      BCM and Llama positions (they label only mappable positions)
    * reported on a fixed sample (the first 500 uuids in sorted order) and on every shared patient
    * a "full window" is a row whose attended length reaches the window (8,192 tokens). These are the
      patients whose record was cut. For them the per-window count measures the capacity of the
      encoding

Reads the datasets in WORK_DIR/datasets. Writes aggregate counts only, with no patient records or uuids,
to results/analysis/ch2/encoding_density_v3_codelast.json (--no-write prints them only).

The datasets were written with the `datasets` library of the deep-models environment. The older version
in the data-analysis environment cannot read that format, so the script runs in the deep-models
environment:

  python analysis/ch2_encoding_density_v3_codelast.py
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))     # fhir_data_utils

HERE = Path(__file__).resolve().parent
_J = importlib.util.spec_from_file_location("vooj", HERE / "verify_obs_ordinal_join.py")
J = importlib.util.module_from_spec(_J); _J.loader.exec_module(J)

OUT = HERE.parent / "results" / "analysis" / "ch2" / "encoding_density_v3_codelast.json"
REPS = {"v3_clean": "para_xsite__clean", "v3_pert": "para_xsite__pert"}
FAMS = ("mlm", "bcm", "llama")
N_SAMPLE = 500                                  # the fixed sample: the first 500 uuids in sorted order


def per_patient_counts(ds, uuids, mappable, fam):
    """{uuid: (n_mappable_lab_obs_in_window, attended_len)} for the given uuids."""
    idx = {u: i for i, u in enumerate(ds["uuid"])}
    out = {}
    for u in uuids:
        r = ds[idx[u]]
        am = np.asarray(r["attention_mask"])
        if fam == "mlm":
            n = sum(1 for c in J.mlm_class_seq(r["input_ids"], am) if c in mappable)
        else:
            n = len(J.labelled_class_seq(r["labels"]))
        out[u] = (n, int(am.sum()))
    return out


def summarise(counts, uuids, max_len):
    res = {}
    for fam in FAMS:
        n = np.array([counts[fam][u][0] for u in uuids], dtype=float)
        res[f"{fam}_mean"] = round(float(n.mean()), 2)
    res["ratio_mlm_bcm"] = round(res["mlm_mean"] / res["bcm_mean"], 3)
    res["ratio_mlm_llama"] = round(res["mlm_mean"] / res["llama_mean"], 3)
    # full windows: the rows of all three families reach the window, so each count is the capacity
    # of that encoding
    full = [u for u in uuids if all(counts[f][u][1] >= max_len for f in FAMS)]
    res["n_full_all_three"] = len(full)
    if full:
        for fam in FAMS:
            res[f"{fam}_mean_full"] = round(float(np.mean([counts[fam][u][0] for u in full])), 2)
        res["ratio_mlm_bcm_full"] = round(res["mlm_mean_full"] / res["bcm_mean_full"], 3)
        res["ratio_mlm_llama_full"] = round(res["mlm_mean_full"] / res["llama_mean_full"], 3)
    return res


def main(write=True):
    import datasets
    from fhir_data_utils import build_lab_code_mappings
    _, class_to_loinc = build_lab_code_mappings(J.CODES_TO_TOKENS_PATH, J.SEEN_CODES_PATH)
    mappable = set(class_to_loinc)
    result = {"method": "verify_obs_ordinal_join.py's sequences on the v3 test splits; Llama code-last",
              "mappable_classes": len(mappable), "reps": {}}
    for rep, slug in REPS.items():
        ds, max_len = {}, None
        for fam in FAMS:
            path = f"{J.DATASET_BASE}/{fam}__{slug}"
            ds[fam] = datasets.load_from_disk(path)["test"]
            m = json.load(open(f"{path}/manifest.json"))
            assert max_len in (None, m["max_len"]), (rep, fam, m["max_len"])
            max_len = m["max_len"]
            if fam == "llama":
                assert m.get("code_last") is True, f"{path} is not a code-last build"
        common = sorted(set(ds["mlm"]["uuid"]) & set(ds["bcm"]["uuid"]) & set(ds["llama"]["uuid"]))
        counts = {fam: per_patient_counts(ds[fam], common, mappable, fam) for fam in FAMS}
        s500, alls = summarise(counts, common[:N_SAMPLE], max_len), summarise(counts, common, max_len)
        result["reps"][rep] = {"datasets": f"{{mlm,bcm,llama}}__{slug}", "max_len": max_len,
                               "n_common_patients": len(common),
                               "e_d_07_sample_500": s500, "all_common_patients": alls}
        print(f"{rep:9s} common {len(common):,} | all: mlm {alls['mlm_mean']:.1f} bcm {alls['bcm_mean']:.1f} "
              f"llama {alls['llama_mean']:.1f} ratio {alls['ratio_mlm_bcm']:.2f}/{alls['ratio_mlm_llama']:.2f} | "
              f"full-window patients {alls['n_full_all_three']:,}")
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2))
        print("wrote", OUT)


if __name__ == "__main__":
    main(write="--no-write" not in sys.argv)
