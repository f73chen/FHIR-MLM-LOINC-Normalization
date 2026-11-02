"""The values and tables of thesis Appendix C (descriptions: paraphrases and RELMA names).

Sources: the paraphrase script `prep/4.2` and the stage directories it wrote, the output of `prep/4.1`
(`loinc_descriptions_enriched.jsonl`), the cleaning steps 03–05 and their manifests,
`09_mlm_dataset.load_paraphrases_3way` (the split), and the RELMA export with the manifest of
`17_relma_pool.py`. `prep/4.2` is read with `ast` and is not imported, because importing it would load
the Hugging Face token. An assert stops the render when a file no longer matches the appendix text.
analysis/appendix/tables.py renders the output:

  python analysis/appendix/tables.py
"""
import ast
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

PREP42 = str(locations.REPO / "prep/4.2_generate_paraphrases_external_desc.py")
UP = locations.UPSTREAM
ENRICHED = UP / "loinc_descriptions_enriched.jsonl"
PARA = C.RERUN / "data" / "paraphrases"
LABELSPACE = C.RERUN / "data" / "labelspace" / "labelspace.csv"
RELMA_CSV = UP / "relma_custom_codes.csv"
RELMA_MANIFEST = C.RERUN / "data" / "relma" / "manifest_s1.json"
VOCABS = ["mimic", "snomed", "cdisc", "ciel", "mth"]
#: readable labels for the nine strategies (the prompts themselves are quoted verbatim)
STRATEGY_LABEL = {
    "abbreviation_heavy": "Heavy abbreviation", "verbose_formal": "Verbose and formal",
    "method_specimen_explicit": "Method and specimen named", "alternate_terminology": "Alternate terminology",
    "ehr_dropdown": "EHR drop-down entry", "mixed_format": "Mixed format",
    "common_variations": "Clinicians' shorthand", "department_specific": "Urgency or context prefix",
    "legacy_systems": "Legacy system codes",
}
norm = lambda s: " ".join(s.split()).lower()   # noqa: E731 (the script's _normalize_text)

#: each table as the appendix prints it: key -> (header, rows, align). tables.py draws the PNG and .tex from here.
TABLES = {}


def _table(key, header, rows, align=""):
    TABLES[key] = (list(header), [[str(c) for c in r] for r in rows], align)


def _constants():
    """Module-level constants of prep/4.2, by literal evaluation, and the two prompt builders."""
    tree = ast.parse(open(PREP42, encoding="utf-8").read())
    out, funcs = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(node, ast.FunctionDef) and node.name in (
                "build_generation_user_content", "build_verification_user_content"):
            funcs[node.name] = node
    return out, funcs


def _template(funcs):
    """The generation user turn with placeholders, built by the function in the script."""
    ns = {"format_metadata_block": lambda m: "<the code's six LOINC axes, one per line>",
          "format_descriptions_block": lambda d: "<the code's known names, grouped by vocabulary>"}
    mod = ast.Module(body=[funcs["build_generation_user_content"], funcs["build_verification_user_content"]],
                     type_ignores=[])
    exec(compile(mod, PREP42, "exec"), ns)
    gen = ns["build_generation_user_content"](
        {"metadata": {}, "descriptions": {}},
        {"system": "<persona>", "instruction": "<instruction>", "examples": "<examples>"})
    ver = ns["build_verification_user_content"]("<the code's six LOINC axes>",
                                                "<the code's known names>", "<candidate line>")
    return gen, ver


def _lines(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def values():
    k, funcs = _constants()
    strategies = k["PROMPTING_STRATEGIES"]
    assert [s["name"] for s in strategies] == list(STRATEGY_LABEL)
    assert k["GEN_MODEL_NAME"] == "google/medgemma-4b-it"
    assert k["VERIFY_MODEL_NAME"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert k["VOCAB_ORDER"] == ["loinc"] + VOCABS
    src = open(PREP42, encoding="utf-8").read()
    for frag in ("do_sample=True", "top_p=0.9", "no_repeat_ngram_size=4", "do_sample=False",
                 'if first in ("true", "yes", "y", "accept", "valid")'):
        assert frag in src, frag
    gen_tmpl, ver_tmpl = _template(funcs)

    # -- the source names (Table C.1) --
    lab = sorted(r["loinc"] for r in C.read_csv(LABELSPACE) if r["in_primary_ge10occ"] == "1")
    enr = {}
    for line in open(ENRICHED, encoding="utf-8"):
        r = json.loads(line)
        enr[r["loinc_code"]] = r["descriptions"]
    assert all(c in enr for c in lab)
    six = ["loinc"] + VOCABS
    cov = {v: sum(1 for c in lab if enr[c].get(v)) for v in six}
    nstr = {v: sum(len(enr[c].get(v, [])) for c in lab) for v in six}
    assert {key for c in lab for key in enr[c]} <= set(six)
    mean_src = sum(nstr.values()) / len(lab)
    # a few words per cell. In full, LOINC gives its long common name, short name and display name, plus the
    # LOINC synonyms that UMLS and Athena list. SNOMED CT and CIEL concepts are one "Mapped from" or "Is a" step
    # from the LOINC concept in OHDSI Athena. CDISC comes from a LOINC-to-CDISC laboratory mapping file. The
    # Metathesaurus concept is found by searching the code (UTS API).
    how = {
        "loinc": ("Official names, synonyms", "Same code"),
        "mimic": ("Lab item labels", "MIMIC-IV's LOINC map"),
        "snomed": ("Concept names, synonyms", "OHDSI Athena"),
        "cdisc": ("Lab test names", "LOINC–CDISC map"),
        "ciel": ("Concept names, synonyms", "OHDSI Athena"),
        "mth": ("Concept name", "UTS search"),
    }
    label = {"loinc": "LOINC", "mimic": "MIMIC-IV", "snomed": "SNOMED CT", "cdisc": "CDISC",
             "ciel": "CIEL", "mth": "UMLS Metathesaurus"}
    rows1 = [[label[v], how[v][0], how[v][1], C.num(cov[v]), C.num(nstr[v])] for v in six]
    rows1.append(["Total", "", "", "", C.num(sum(nstr.values()))])    # names, counted per vocabulary
    _table("C.1", ["Vocabulary", "Kind of name", "Linked by", "Codes", "Names"],
                           rows1, "lllrr")

    # -- the strategies (Table C.2) --
    temps = sorted({s["temperature"] for s in strategies})
    # the caption says: all at the lower temperature except heavy abbreviation
    assert [s["name"] for s in strategies if s["temperature"] == temps[-1]] == ["abbreviation_heavy"]
    srows = [[STRATEGY_LABEL[s["name"]], f"{s['system']} {s['instruction']} {s['examples']}"] for s in strategies]
    _table("C.2", ["Strategy", "Persona, instruction and examples (verbatim)"], srows)
    max_per_prompt = {int(w) for s in strategies for w in s["instruction"].split()[1:2] if w.isdigit()}
    assert max_per_prompt == {5}

    # -- stage counts over the codes that have a pool --
    codes = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(PARA / "clean" / "*.txt")))
    cnt = lambda d: sum(len(_lines(os.path.join(d, c + ".txt"))) for c in codes)   # noqa: E731
    n_gen, n_rule = cnt(UP / "paraphrases_ext"), cnt(UP / "paraphrases_ext_clean")
    # the second LLM check was run outside this repository. Its rejected lines, one file per code, are
    # read from paraphrases_ext_rejected_second/. They were removed from paraphrases_ext_verified/.
    n_llama_rej, n_llm1_rej = cnt(UP / "paraphrases_ext_rejected"), cnt(UP / "paraphrases_ext_rejected_second")
    n_ver = cnt(UP / "paraphrases_ext_verified")
    # the merge, recomputed: lines the check kept, plus the other vocabularies, minus the official LOINC names
    acc = strip_llm = vocab_add = 0
    merged_total = 0
    for c in codes:
        rej = set(_lines(UP / "paraphrases_ext_rejected" / f"{c}.txt"))
        kept = [l for l in _lines(UP / "paraphrases_ext_clean" / f"{c}.txt") if l not in rej]
        acc += len(kept)
        d = enr.get(c, {})
        strip = {norm(x) for x in d.get("loinc", []) if x.strip()}
        seen = set()
        for l in kept:
            key = norm(l)
            if key in strip:
                strip_llm += 1
            elif key not in seen:
                seen.add(key)
        for v in VOCABS:
            for x in d.get(v, []):
                key = norm(x)
                if x.strip() and key not in strip and key not in seen:
                    seen.add(key)
                    vocab_add += 1
        merged_total += len(seen)
    assert acc == n_rule - n_llama_rej
    assert merged_total - n_llm1_rej == n_ver, (merged_total, n_llm1_rej, n_ver)
    n_partA = cnt(PARA / "partA_pool")
    man = json.load(open(PARA / "clean" / "manifest.json"))
    assert man["lines_in"] == n_partA and man["lines_out"] == n_partA - man["removed_matched"]
    n_final = cnt(PARA / "clean")
    assert man["lines_out_final"] == n_final == man["lines_out"] - man["supplemental_safe_garbage_removed"]
    # the merge removes the LOINC-name matches and the duplicates, and adds the names from the other vocabularies
    n_dups = acc - strip_llm - (merged_total - vocab_add)
    assert acc - (strip_llm + n_dups) + vocab_add == merged_total
    stage = [
        ["Generated by MedGemma", C.num(n_gen), "", ""],
        ["Rule-based cleaning", C.num(n_rule), C.num(n_gen - n_rule), ""],
        ["Check by Llama", C.num(acc), C.num(n_llama_rej), ""],
        ["Merging", C.num(merged_total), C.num(strip_llm + n_dups), C.num(vocab_add)],
        ["Second LLM check", C.num(n_ver), C.num(n_llm1_rej), ""],
        ["Structural clean-up", C.num(n_partA), C.num(n_ver - n_partA), ""],
        ["LLM review", C.num(man["lines_out"]), C.num(man["removed_matched"]), ""],
        ["Exact-match removals", C.num(n_final), C.num(man["supplemental_safe_garbage_removed"]), ""],
    ]
    _table("C.3", ["Step", "Lines", "Removed", "Added"], stage, "lrrr")

    # -- the final pool on the label space, and its split --
    spec = importlib.util.spec_from_file_location("m09", C.SCRIPTS / "09_mlm_dataset.py")
    m09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m09)
    assert m09.SEED == 42 and m09.FRACTIONS == (0.70, 0.15, 0.15)
    tr, va, te, st = m09.load_paraphrases_3way(str(PARA / "clean"), verbose=False)
    sizes = {c: len(tr.get(c, [])) + len(va.get(c, [])) + len(te.get(c, [])) for c in lab}
    n_lab = sum(sizes.values())
    gen_n = vocab_n = mimic_n = 0
    for c in lab:
        d = enr[c]
        vs = {norm(x) for v in VOCABS for x in d.get(v, [])}
        ms = {norm(x) for x in d.get("mimic", [])}
        for l in tr.get(c, []) + va.get(c, []) + te.get(c, []):
            key = norm(l)
            vocab_n += key in vs
            mimic_n += key in ms
    gen_n = n_lab - vocab_n
    n_mimic_labels = sum(len({norm(x) for x in enr[c].get("mimic", [])}) for c in lab)
    # expected share of scored test observations whose drawn description is a vocabulary name
    occ = {r["loinc"]: int(r["test_occ"]) for r in C.read_csv(LABELSPACE) if r["in_primary_ge10occ"] == "1"}
    W = sum(occ.values())
    e_v = e_m = 0.0
    for c in lab:
        pool = te.get(c, [])
        if not pool:
            continue
        vs = {norm(x) for v in VOCABS for x in enr[c].get(v, [])}
        ms = {norm(x) for x in enr[c].get("mimic", [])}
        e_v += occ[c] * sum(norm(l) in vs for l in pool) / len(pool)
        e_m += occ[c] * sum(norm(l) in ms for l in pool) / len(pool)
    n_tr = sum(len(tr.get(c, [])) for c in lab)
    n_va = sum(len(va.get(c, [])) for c in lab)
    n_te = sum(len(te.get(c, [])) for c in lab)

    # -- RELMA --
    rows = C.read_csv(RELMA_CSV)
    eng = [r for r in rows if "english" in (r.get("Language") or "").lower()]
    labset = set(lab)
    eng_lab = [r for r in eng if (r.get("LOINC Code") or "").strip() in labset and (r.get("Local Test Name") or "").strip()]
    rm = json.load(open(RELMA_MANIFEST))["stats"]
    insts = {(r.get("Institution") or "").strip() for r in eng_lab}
    assert rm["n_loinc_in_pool"] == len({r["LOINC Code"].strip() for r in eng_lab})
    assert list(rows[0].keys()) == ["LOINC Code", "Battery Code", "Battery Name", "Local Code", "Local Test Name",
                                    "Units", "Language", "Institution"]

    v = {
        "n_lab": C.num(len(lab)),
        "mean_src": f"{mean_src:.1f}",
        "gen_model": k["GEN_MODEL_NAME"], "ver_model": k["VERIFY_MODEL_NAME"],
        "n_strategies": C.num(len(strategies)), "seeds": ", ".join(str(s) for s in k["GEN_SEEDS"]),
        "n_seeds": C.num(len(k["GEN_SEEDS"])),
        "temp_lo": f"{temps[0]:.1f}", "temp_hi": f"{temps[-1]:.1f}",
        "max_new": C.num(k["GEN_MAX_NEW_TOKENS"]),
        "gen_template": gen_tmpl,
        "max_len": C.num(k["MAX_PARAPHRASE_LEN"]),
        "verify_system": k["VERIFY_SYSTEM_PROMPT"], "verify_user": ver_tmpl,
        "ver_max_new": C.num(k["VERIFY_MAX_NEW_TOKENS"]),
        "n_codes_pool": C.num(len(codes)),
        "n_strip": C.num(strip_llm), "n_dups": C.num(n_dups),
        "pct_llama_rej": C.pct(n_llama_rej, n_rule), "n_llm1_rej": C.num(n_llm1_rej),
        "n_partB": C.num(man["removed_matched"]), "n_partB_skip": C.num(man["removed_unmatched_skipped"]),
        "n_tail": C.num(man["supplemental_safe_garbage_removed"]),
        "n_final_lab": C.num(n_lab), "mean_lab": f"{n_lab / len(lab):.1f}", "min_lab": C.num(min(sizes.values())),
        "mean_gen": f"{gen_n / len(lab):.1f}", "mean_vsrc": f"{vocab_n / len(lab):.1f}",
        "n_vocab": C.num(vocab_n),
        "n_mimic_in": C.num(mimic_n), "n_mimic_labels": C.num(n_mimic_labels),
        "e_vocab": f"{100 * e_v / W:.1f}%", "e_mimic": f"{100 * e_m / W:.1f}%",
        "n_tr": C.num(n_tr), "n_va": C.num(n_va), "n_te": C.num(n_te),
        "min_te": C.num(min(len(te[c]) for c in lab)),
        "relma_rows": C.num(len(rows)), "relma_eng": C.num(len(eng)),
        "relma_insts": C.num(len(insts)), "relma_codes": C.num(rm["n_loinc_in_pool"]),
        "relma_names": C.num(rm["n_names_total"]),
    }
    return v
