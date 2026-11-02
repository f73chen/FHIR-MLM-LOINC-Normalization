"""The values and tables of thesis Appendix E (off-the-shelf baselines).

Sources: analysis/b4_oneshot_baselines.py (models, sample, exemplar choice, decoding), lib/oneshot.py
(prompt, extraction, scoring), the b4 summary (sample sizes), the v3 TF-IDF training prompts (the shape
shares, counted here) and the formatter in pipeline/06 (the made-up examples). The real exemplars are
training observations and are not printed. Asserts compare the code with what the appendix says and
stop the render on a mismatch. analysis/appendix/tables.py renders the output:

  python analysis/appendix/tables.py
"""
import importlib.util
import json
from collections import Counter

import common as C
import appendix_f

#: each table as the appendix prints it: key -> (header, rows, align). tables.py draws the PNG and .tex from here.
TABLES = {}


def _table(key, header, rows, align=""):
    TABLES[key] = (list(header), [[str(c) for c in r] for r in rows], align)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def values():
    import sys
    sys.path.insert(0, str(C.ANALYSIS))
    b4 = _load("b4", C.ANALYSIS / "b4_oneshot_baselines.py")
    O = b4.O
    src = open(C.ANALYSIS / "b4_oneshot_baselines.py", encoding="utf-8").read()
    for frag in ("max_new_tokens=16, do_sample=False", "max_length=2048", "batch_size=1"):
        assert frag in src, frag
    assert b4.EXEMPLAR_SHAPES == ("numeric+range", "qualitative", "numeric")
    assert (b4.N_PATIENTS, b4.N_OBS, b4.SEED) == (500, 5000, 0)
    assert O.LOINC_RE.pattern == r"\b(\d{1,5}-\d)\b"

    label = {"llama31-8b": "Llama-3.1-8B-Instruct", "gemma3-12b": "Gemma-3-12B", "openbio-8b": "OpenBioLLM-8B",
             "meditron-7b": "Meditron-7B", "medgemma-4b": "MedGemma-4B"}
    cite = {"llama31-8b": "[Grattafiori 2024]", "gemma3-12b": "[Gemma3 2025]", "openbio-8b": "[OpenBioLLM-8B 2024]",
            "meditron-7b": "[Meditron 2023]", "medgemma-4b": "[Sellergren 2025]"}
    assert set(b4.MODELS) == set(label)
    rows = []
    for key in ("llama31-8b", "gemma3-12b", "medgemma-4b", "openbio-8b", "meditron-7b"):
        repo, style, four_bit = b4.MODELS[key]
        assert repo in {r for *_, r in appendix_f.MODELS}, repo     # the caption sends readers to Table F.2
        rows.append([f"{label[key]} {cite[key]}", "Chat template" if style == "chat" else "Plain text",
                     "4-bit NF4" if four_bit else "bf16"])
    _table("E.1", ["Model", "Prompt form", "Precision"], rows)
    assert b4.MODELS["openbio-8b"][1] == "plain" and b4.MODELS["meditron-7b"][1] == "plain"

    # sample sizes actually scored (both arms)
    summ = {r["model"]: int(r["n"]) for r in C.read_csv(C.RESULTS / "analysis" / "b4" / "b4_summary.csv")}
    fs = {k: v for k, v in summ.items() if "__v3" in k and k.endswith("__fs3")}
    assert len(fs) == 10 and len(set(fs.values())) == 1
    n_sample = next(iter(fs.values()))

    # the shapes of the training prompts (what the exemplars cover), counted over the same 400 files
    _, train_dir = b4.dataset_dirs("para_xsite", test_arm="pert")
    shapes = Counter()
    for f in sorted(train_dir.glob("*.jsonl"))[:400]:
        for line in open(f):
            if line.strip():
                shapes[b4.prompt_shape(json.loads(line)["prompt"])] += 1
    n_sh = sum(shapes.values())
    cover = sum(shapes[s] for s in b4.EXEMPLAR_SHAPES)

    # made-up exemplars and query, through the formatter in 06 and the real prompt builders
    m06 = _load("m06", C.SCRIPTS / "06_tfidf_dataset.py")
    import unit_standardisation as us
    from input_preprocessing_utils import (load_qual_normalizer, load_qual_passthrough, load_unit_alias_map,
                                           normalize_qual_value, normalize_unit)
    amap = load_unit_alias_map(C.UNIT_ALIAS_JSON)
    v2c = load_qual_normalizer(m06.QUAL_NORMALIZE_PATH)
    lit, rx = load_qual_passthrough(m06.QUAL_PASSTHROUGH_PATH)
    unit_norm = lambda raw: normalize_unit(raw, amap)[0] if raw else ""        # noqa: E731
    qual_norm = lambda raw: (lambda o, v: None if o == "blank" else v)(*normalize_qual_value(raw, v2c, lit, rx))  # noqa: E731
    fmt = lambda obs, desc: m06.format_observation_as_prompt(                   # noqa: E731
        dict(obs, resourceType="Observation"), desc, unit_norm, qual_norm,
        quantity_fmt=lambda v, u: us.standardise_quantity(v, u, amap), include_reference_range=True,
        range_fmt=lambda r, u: us.standardise_reference_range(r, u, amap))
    rr = lambda lo, hi, u: [{"low": {"value": lo, "unit": u}, "high": {"value": hi, "unit": u}}]  # noqa: E731
    ex = [
        (fmt({"valueQuantity": {"value": 139, "unit": "mEq/L"}, "referenceRange": rr(135, 145, "mEq/L"),
              "extension": [{"valueString": "ROUTINE"}]}, "Na, serum"), "2951-2"),                  # Sodium SerPl-sCnc
        (fmt({"valueString": "Negative"}, "Urine protein, dipstick"), "5804-0"),                    # Prot Ur Strip-mCnc
        (fmt({"valueQuantity": {"value": 6.5, "unit": "units"}}, "Urine pH, dipstick"), "5803-2"),  # pH Ur Strip
    ]
    assert [b4.prompt_shape(p) for p, _ in ex] == list(b4.EXEMPLAR_SHAPES)
    query = fmt({"valueQuantity": {"value": 4.2, "unit": "mEq/L"}, "referenceRange": rr(3.3, 5.1, "mEq/L"),
                 "extension": [{"valueString": "STAT"}]}, "K, serum")
    lab = {r["loinc"] for r in C.read_csv(C.RERUN / "data" / "labelspace" / "labelspace.csv") if r["in_primary_ge10occ"] == "1"}
    assert all(code in lab for _, code in ex)
    msgs = O.build_fewshot_messages(query, ex)
    chat = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)
    plain = O.build_plain_fewshot(query, ex)
    ex_names = "; ".join(f"{code}, {C.long_name(code)}" for _, code in ex)

    v = {
        "n_patients": f"{b4.N_PATIENTS:,}", "n_per": f"{b4.N_OBS // b4.N_PATIENTS}", "n_sample": f"{n_sample:,}",
        "seed": str(b4.SEED),
        "sh_nr": C.pct(shapes["numeric+range"], n_sh), "sh_q": C.pct(shapes["qualitative"], n_sh),
        "sh_n": C.pct(shapes["numeric"], n_sh), "sh_cover": C.pct(cover, n_sh), "sh_nv": C.pct(shapes["no-value"], n_sh),
        "n_sh": f"{n_sh:,}",
        "chat_example": chat, "plain_example": plain, "ex_names": ex_names,
        "loinc_re": O.LOINC_RE.pattern,
    }
    return v
