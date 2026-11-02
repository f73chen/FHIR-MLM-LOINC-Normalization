"""Appendices A–F: their tables as booktabs LaTeX and print-size PNG/PDF, and their in-text numbers as JSON.

The rows and numbers come from the code, constants and outputs of the pipeline, through appendix_<x>.py.
The tables are drawn with analysis/lib/{figstyle,booktabs}.py, like the tables of Chapters 4–6. A table
too long for one page is drawn in parts (<name>_part1.png, ...) and written as a longtable.

    python analysis/appendix/tables.py          write results/appendix/tables/<name>.{tex,png,pdf} and
                                                results/appendix/values_<A-F>.json
    python analysis/appendix/tables.py --check  fail if a .tex differs from a fresh build
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import common as C
import appendix_a as A
import appendix_b as B
import appendix_c as CC
import appendix_d as D
import appendix_e as E
import appendix_f as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = locations.REPO / "results" / "appendix" / "tables"
LIB = C.ANALYSIS / "lib"
SPAN = 6.30          # inches: the widest a table is drawn, inside the 6.375 in text width of the thesis


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LIB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FS, BT = _load("figstyle"), _load("booktabs")
FS.FONT = "stix"     # the LaTeX column widths here are measured in STIX. The fitted copies in the thesis build on them.
_UNSUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
_MATH = re.compile(r"(\$[^$]*\$)")
_TEX = [("{", r"\{"), ("}", r"\}"), ("#", r"\#"), ("_", r"\_"), ("&", r"\&"),       # reserved first,
        ("~", r"$\sim$"),                      # "~2,000,000" means about, but LaTeX reads a bare ~ as a space
        ("†", r"$^{\dagger}$"), ("‡", r"$^{\ddagger}$"), ("÷", r"$\div$")]          # then the symbols
_OPEN1 = re.compile(r"(?<![^\s(\[])'")      # a single quote that opens: at the start, or after a space or (
_POW = re.compile(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_MANT = re.compile(r"(\d+(?:\.\d+)?) × 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")     # 2 × 10⁻⁴ as one span, never split
_GREEK = {"α": r"$\alpha$", "β": r"$\beta$"}


def _math(text):
    """What the appendix modules write in Unicode and print sets as mathematics: 10⁻⁵ as $10^{-5}$, β as
    $\\beta$."""
    text = _MANT.sub(lambda g: "$" + g.group(1) + r" \times 10^{" + g.group(2).translate(_UNSUP) + "}$", text)
    text = _POW.sub(lambda g: "$10^{" + g.group(1).translate(_UNSUP) + "}$", text)
    for a, b in _GREEK.items():
        text = text.replace(a, b)
    return text
_OPEN2 = re.compile(r'(?<![^\s(\[])"')


def _outside_math(text, fn):
    """`fn` applied to the parts of `text` outside $...$ spans."""
    return "".join(p if len(p) > 1 and p[0] == p[-1] == "$" else fn(p) for p in _MATH.split(text))


def tex_text(text):
    """UCUM codes and LOINC names carry { } # _ &, which LaTeX reserves. † ‡ ÷ become math. The
    translation that booktabs does itself (%, <, dashes, ×) follows in BT.tex. $...$ spans are left
    alone."""
    def esc(t):
        for a, b in _TEX:
            t = t.replace(a, b)
        t = _OPEN2.sub("``", _OPEN1.sub("`", t))       # LaTeX opening quotes. A closing " is ''
        return t.replace('"', "''").replace("\n", r"\newline ")   # a line break inside a p{} cell
    return _outside_math(_math(text), esc)


def draw_text(text):
    """Typographic quotes, as LaTeX sets them: ‘ “ where a quote opens, ’ ” elsewhere (an apostrophe is ’)."""
    def curl(t):
        t = _OPEN2.sub("“", _OPEN1.sub("‘", t))
        return t.replace("'", "’").replace('"', "”")
    return _outside_math(_math(text), curl)


TITLES = {
    "A.1": "Unit strings seen in the training set, grouped into kinds, with each kind's target unit.",
    "A.2": "LOINC codes whose MIMIC units disagreed with the preferred unit in other terminologies.",
    "A.3": "LOINC codes whose units are drawn at random between mEq/L and mmol/L.",
    "B.1": "Literature sources.",
    "B.2": "Published magnitudes for perturbing each named analyte.",
    "B.3": "LOINC Property groups.",
    "B.4": "Perturbation groups, with each group's most frequent code.",
    "B.5": "Group magnitude defaults for groups containing named analytes.",
    "B.6": "Range variations relative to each analyte's original half-width.",
    "B.7": "Measurement scales and the rules for applying perturbation magnitudes to each.",
    "B.8": "The numerical guards.",
    "C.1": "Vocabularies providing canonical names for paraphrase generation.",
    "C.2": "Prompting strategies.",
    "C.3": "Lines in the paraphrase pool after each cleaning step.",
    "D.1": "TF-IDF data and model settings.",
    "D.2": "TF-IDF Optuna hyperparameter optimization search space.",
    "D.3": "Fine-tuning settings for the three deep models.",
    "E.1": "The off-the-shelf models.",
    "F.1": "Data sources.",
    "F.2": "All models used and the Hugging Face revision for each.",
    "F.3": "Software versions in the two Python environments.",
}


def caption(table_id):
    return TITLES[table_id]


def math_factor(f):
    """A factor for LaTeX and the drawing: powers of ten as $10^{3}$, other numbers as C.factor prints them."""
    e = round(math.log10(f))
    if e not in (0, 1) and abs(f - 10.0 ** e) <= 1e-9 * 10.0 ** e:
        return f"$10^{{{e}}}$"
    return C.factor(f)


def fill(header, body, shares):
    """Widths (points) for the wrapped columns, with `shares` = {column: weight}. The other columns keep
    their natural width, and the wrapped ones split what is left of SPAN by weight. If the wrapped
    columns fit at their natural width, they keep it and nothing wraps."""
    FS.apply_paper(BT.SIZE)
    width = BT._widthfn()[0]
    nat = [max(width(ln) for r in list(body) + [header] for ln in draw_text(r[k]).split("\n"))
           for k in range(len(header))]
    free = SPAN * 72 - 2 * BT.PAD * len(header) - sum(w for k, w in enumerate(nat) if k not in shares)
    if sum(nat[k] for k in shares) <= free:
        return {k: nat[k] for k in shares}
    total = sum(shares.values())
    return {k: free * s / total for k, s in shares.items()}


def balance(header, body, widths, n=2):
    """Split points for a long table drawn in `n` parts of about equal height."""
    FS.apply_paper(BT.SIZE)
    measure = BT._widthfn()[1]
    tall = [max(sum(len(BT._wrap(par, w, measure)) for par in draw_text(r[k]).split("\n"))
                for k, w in widths.items()) for r in body]
    cum, parts, start = 0, [], 0
    for i, t in enumerate(tall):
        cum += t
        if len(parts) < n - 1 and cum >= sum(tall) * (len(parts) + 1) / n:
            parts.append((start, i + 1))
            start = i + 1
    return parts + [(start, len(body))]


def _from(mod, key):
    """A table's header, rows and right-aligned columns, from the appendix module's TABLES."""
    header, rows, align = mod.TABLES[key]
    return header, rows, {k for k, a in enumerate(align) if a == "r"}


def _meta(table_id, path):
    return f"Table {table_id} -- generated by analysis/appendix/tables.py from {path}"


# -- Appendix A --
def table_a1(ctx):
    return {
        "name": "A1_unit_conversion", "header": A.A1_HEADER,
        "body": A.conversion_rows(A.conversion_groups(), math_factor), "right": {3},
        "caption": caption("A.1"), "label": "tab:unit-conversion",
        "comment": _meta("A.1", "unit_standardisation.STANDARD_UNIT"),
    }


def table_a2(ctx):
    # the two unit columns sit under one "Example unit" header. There is no Action column, because the
    # caption says what was done with each case.
    assert A.A2_HEADER[1:3] == ["MIMIC-IV unit", "Terminology unit"] and A.A2_HEADER[5] == "Action"
    return {
        "name": "A2_unit_disagreements",
        "header": [A.A2_HEADER[0], "MIMIC-IV", "Terminologies", *A.A2_HEADER[3:5]],
        "groups": [("Example unit", 1, 2)], "body": [r[:5] for r in A.resolution_rows(A.disagreements())],
        "right": {3, 4}, "total": True,
        "caption": caption("A.2"), "label": "tab:unit-disagreements",
        "comment": _meta("A.2", "results/analysis/a31/disagreement_resolution.csv"),
    }


def table_a3(ctx):
    rows, starts = A.randomized_rows(A.disagreements().frozen)
    return {
        "name": "A3_randomized_codes", "header": A.A3_HEADER, "body": rows, "right": set(),
        "gaps": starts,                       # a little space between analytes, as booktabs' \addlinespace
        "caption": caption("A.3"), "label": "tab:randomized-codes",
        "comment": _meta("A.3", "data/variability/unit_randomisation.csv; names from LOINC 2.82"),
    }


# -- Appendix B --
def table_b1(ctx):
    header, rows, right = _from(B, "B.1")
    return {
        "name": "B1_sources", "header": header, "body": rows, "right": right,
        "groups": [("Analytes", 3, 4)],                 # the last two columns count the analytes each paper supplies
        "caption": caption("B.1"), "label": "tab:magnitude-sources-literature",
        "comment": _meta("B.1", "variability_register.SOURCES and RECORDS"),
    }


def table_b2(ctx):
    header, rows, right = _from(B, "B.2")
    assert header == ["Analyte", "Value CV (%)", "Source", "Range variation (%)", "Source"]
    head = ["Analyte", "%", "Source", "%", "Source"]     # the quantities move into the group titles
    return {
        "name": "B2_analyte_magnitudes", "header": head, "body": rows, "right": right,
        "groups": [("Value CV", 1, 2), ("Range variation", 3, 4)], "widths": fill(head, rows, {0: 1}),
        "caption": caption("B.2"), "label": "tab:analyte-magnitudes",
        "comment": _meta("B.2", "variability_register.RECORDS via resolve()"),
    }


def table_b3(ctx):
    header, rows, right = _from(B, "B.3")
    return {
        "name": "B3_property_groups", "header": header, "body": rows, "right": right,
        "caption": caption("B.3"), "label": "tab:property-groups",
        "comment": _meta("B.3", "a25_partition_comparison.PROP_GROUP"),
    }


def table_b4(ctx):
    header, rows, right = _from(B, "B.4")
    widths = fill(header, rows, {3: 1})
    return {
        "name": "B4_code_groups", "header": header, "body": rows, "right": right, "widths": widths,
        "groups": [("Most frequent code", 2, 3)],
        "parts": balance(header, rows, widths), "long": True,     # too long for one page: two parts
        "caption": caption("B.4"), "label": "tab:code-groups",
        "comment": _meta("B.4", "data/variability/perturbation_spec.csv"),
    }


def table_b5(ctx):
    header, rows, right = _from(B, "B.5")
    assert header[1:] == ["Value CV (%)", "From", "Codes", "Range variation (%)", "From", "Codes", "Capped"]
    head = ["Group", "%", "From", "Codes", "%", "From", "Codes", "Capped"]
    return {
        "name": "B5_group_defaults", "header": head, "body": rows, "right": right,
        "groups": [("Value CV default", 1, 3), ("Range variation default", 4, 7)],
        "widths": fill(head, rows, {2: 1}),      # "Urine leukocytes" wraps, so the table fits the text width
        "caption": caption("B.5"), "label": "tab:group-defaults",
        "comment": _meta("B.5", "data/variability/perturbation_spec.csv"),
    }


def table_b6(ctx):
    header, rows, right = _from(B, "B.6")
    return {
        "name": "B6_range_cap", "header": header, "body": rows, "right": right,
        "caption": caption("B.6"), "label": "tab:range-cap",
        "comment": _meta("B.6", "perturbation_spec.csv and stated_intervals.csv (a37's calibration)"),
    }


#: the formulas of Table B.7, set as mathematics in the drawing and the .tex (appendix_b keeps plain text)
B7_MATH = {"1 + z × CV": r"$1 + z \times \mathrm{CV}$", "z × √value": r"$z \times \sqrt{\mathrm{value}}$"}


def table_b7(ctx):
    header, rows, right = _from(B, "B.7")
    body = []
    for r in rows:
        rule = r[3]
        for plain, tex in B7_MATH.items():
            rule = rule.replace(plain, tex)
        body.append(r[:3] + [rule])
    assert sum("$" in r[3] for r in body) == 2, [r[3] for r in body]     # ratio, count
    return {
        "name": "B7_measurement_scales", "header": header, "body": body, "right": right,      # one line per row
        "caption": caption("B.7"),
        "label": "tab:measurement-scales", "comment": _meta("B.7", "perturbation_spec.csv; rules from perturbation.py"),
    }


def table_b8(ctx):
    header, rows, right = _from(B, "B.8")
    return {
        "name": "B8_guards", "header": header, "body": rows, "right": right,
        "caption": caption("B.8"), "label": "tab:perturbation-guards",
        "comment": _meta("B.8", "perturbation_spec.csv; perturbation.perturb_number and perturb_patient"),
    }


# -- Appendix C --
def table_c1(ctx):
    header, rows, right = _from(CC, "C.1")
    return {
        "name": "C1_source_vocabularies", "header": header, "body": rows, "right": right, "total": True,   # one line per row
        "caption": caption("C.1"), "label": "tab:source-vocabularies",
        "comment": _meta("C.1", "fhir/data/loinc_descriptions_enriched.jsonl (written by prep/4.1)"),
    }


def table_c2(ctx):
    header, rows, right = _from(CC, "C.2")
    widths = fill(header, rows, {0: 0.22, 1: 0.78})
    return {
        "name": "C2_prompting_strategies", "header": header, "body": rows, "right": right,
        "widths": widths, "gaps": set(range(1, len(rows))),
        "parts": balance(header, rows, widths), "long": True,     # too tall for one page: two parts
        "caption": caption("C.2"), "label": "tab:prompting-strategies",
        "comment": _meta("C.2", "prep/4.2 PROMPTING_STRATEGIES (verbatim)"),
    }


def table_c3(ctx):
    header, rows, right = _from(CC, "C.3")
    return {
        "name": "C3_pool_steps", "header": header, "body": rows, "right": right,
        "caption": caption("C.3"), "label": "tab:pool-steps",
        "comment": _meta("C.3", "the pool's stage folders and the clean-up manifest"),
    }


# -- Appendix D --
def table_d1(ctx):
    header, rows, right = _from(D, "D.1")
    body = [[r[0].replace("strength C", "strength $C$"), r[1]] for r in rows]   # as in D.2
    assert sum("$C$" in r[0] for r in body) == 1
    groups = {3: "Stop words", 10: "Classifier"}          # training data | text features | classifier
    assert all(body[k][0] == name for k, name in groups.items())
    return {
        "name": "D1_tfidf_settings", "header": header, "body": body, "right": right, "gaps": set(groups),
        "caption": caption("D.1"), "label": "tab:tfidf-settings",
        "comment": _meta("D.1", "tfidf_recipe.PINNED and 07_tfidf_train_eval.py"),
    }


def table_d2(ctx):
    header, rows, right = _from(D, "D.2")
    body = [[r[0].replace("strength C", "strength $C$"), r[1]] for r in rows]
    assert sum("$C$" in r[0] for r in body) == 1
    return {
        "name": "D2_tfidf_search_space", "header": header, "body": body, "right": right,
        "caption": caption("D.2"), "label": "tab:tfidf-search-space",
        "comment": _meta("D.2", "08_tfidf_optuna_b0.SPACE, C_RANGE, MAX_ITER"),
    }


def table_d3(ctx):
    header, rows, right = _from(D, "D.3")
    return {
        "name": "D3_deep_model_settings", "header": header, "body": rows, "right": right,
        "widths": fill(header, rows, {1: 1, 2: 1, 3: 1}), "gaps": set(range(1, len(rows))),
        "caption": caption("D.3"), "label": "tab:deep-model-settings",
        "comment": _meta("D.3", "12_mlm_train.py, 13_bcm_train.py and 14_llama_train.py"),
    }


# -- Appendix E --
def table_e1(ctx):
    header, rows, right = _from(E, "E.1")
    return {
        "name": "E1_baseline_models", "header": header, "body": rows, "right": right,
        "caption": caption("E.1"), "label": "tab:baseline-models",
        "comment": _meta("E.1", "b4_oneshot_baselines.MODELS"),
    }


# -- Appendix F --
def table_f1(ctx):
    header, rows, right = _from(F, "F.1")
    return {
        "name": "F1_data_sources", "header": header, "body": rows, "right": right,          # one line per row
        "draw_widths": fill(header, rows, {0: 1}),     # the citation names are long: only the PNG wraps them
        "caption": caption("F.1"), "label": "tab:data-sources",
        "comment": _meta("F.1", "the data files' dates and the release's records"),
    }


def table_f2(ctx):
    header, rows, right = _from(F, "F.2")
    assert header == ["Model", "Role", "Hugging Face ID", "Revision"]
    # the ID and the full 40-character revision, one above the other, so neither has to wrap
    head = ["Model", "Role", "Hugging Face ID and revision"]
    body = [[m, role, repo if rev == "—" else f"{repo}\n{rev}"] for m, role, repo, rev in rows]
    widths = fill(head, body, {0: 1.15, 1: 1.0})
    widths[2] = max(BT._widthfn()[0](ln) for r in body for ln in r[2].split("\n"))
    assert sum(widths.values()) + 2 * BT.PAD * len(head) <= SPAN * 72 + 1e-6
    return {
        "name": "F2_models", "header": head, "body": body, "right": right, "widths": widths,
        "gaps": set(range(1, len(body))),
        "caption": caption("F.2"), "label": "tab:models",
        "comment": _meta("F.2", "the Hugging Face cache's snapshots"),
    }


def table_f3(ctx):
    header, rows, right = _from(F, "F.3")
    return {
        "name": "F3_software", "header": header, "body": rows, "right": right,
        "groups": [("Environment", 1, 2)],
        "caption": caption("F.3"), "label": "tab:software",
        "comment": _meta("F.3", "the release's environment/requirements-*.txt"),
    }


TABLES = [table_a1, table_a2, table_a3,
          table_b1, table_b2, table_b3, table_b4, table_b5, table_b6, table_b7, table_b8,
          table_c1, table_c2, table_c3,
          table_d1, table_d2, table_d3,
          table_e1,
          table_f1, table_f2, table_f3]


def main(check=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = {"A": A.values(), "B": B.values(), "C": CC.values(), "D": D.values(), "E": E.values(), "F": F.values()}
    stale = []
    for build in TABLES:
        t = build(ctx)
        layout = {"groups": t.get("groups", ()), "right": t["right"], "total": t.get("total", False),
                  "gaps": t.get("gaps", ())}
        if t.get("widths"):
            layout["widths"] = t["widths"]
        rows = [[tex_text(c) for c in r] for r in t["body"]]
        for r in rows:          # a row opening with "[" would be read as \\[..], \midrule[..] or \addlinespace[..]
            if r[0].startswith("["):
                r[0] = "{" + r[0] + "}"
        tex = BT.tex(t["header"], rows, caption=tex_text(t["caption"]),
                     label=t["label"], comment=t["comment"], long=t.get("long", False), **layout)
        path = OUT / f"{t['name']}.tex"
        if check:
            same = path.exists() and path.read_text(encoding="utf-8") == tex
            print(f"{t['name']}: {'in step' if same else 'DIFFERS'}")
            stale += [] if same else [t["name"]]
            continue
        path.write_text(tex, encoding="utf-8")
        head, body = [draw_text(h) for h in t["header"]], [[draw_text(c) for c in r] for r in t["body"]]
        if t.get("draw_widths"):
            layout["widths"] = t["draw_widths"]
        if t.get("parts"):
            pngs, w = BT.draw_parts(FS, head, body, str(OUT / t["name"]), t["parts"], **layout)
        else:
            png, w = BT.draw(FS, head, body, str(OUT / t["name"]), **layout)
            pngs = [png]
        assert w <= FS.TEXTWIDTH, (t["name"], w)
        print(f"{t['name']}: wrote {path.name} and {', '.join(Path(p).name for p in pngs)} "
              f"(width {w:.2f} in; text width {FS.TEXTWIDTH} in)")
    if stale:
        sys.exit(f"stale: {', '.join(stale)} (run analysis/appendix/tables.py)")
    if not check:                    # the in-text numbers, as the appendices print them
        for key, values in ctx.items():
            (OUT.parent / f"values_{key}.json").write_text(json.dumps(
                {k: v for k, v in values.items() if isinstance(v, (str, int, float))}, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
