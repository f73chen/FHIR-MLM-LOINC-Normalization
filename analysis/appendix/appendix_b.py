"""Values for Appendix B (perturbation magnitudes): Tables B.1–B.8 and the numbers its text quotes.

Sources: pipeline/variability_register.py (the literature records), pipeline/build_perturbation_spec.py
and its output data/variability/perturbation_spec.csv (one row per perturbed code), the property groups
of a25, the stated intervals of a37, the CLIA divisors of a22, numeric_obs_counts.csv, the RCPAQAP
programme lists, and the raw values in the training split (for the negative-value counts). If the code
and the appendix text no longer agree, an assert stops the build.

Built by the appendix table script:
  python analysis/appendix/tables.py
"""
import importlib.util
import re
import statistics
from collections import Counter, defaultdict

import common as C
import variability_register as vr

VAR = C.RERUN / "data" / "variability"
SPEC = VAR / "perturbation_spec.csv"
STATED = VAR / "stated_intervals.csv"
NUMERIC = VAR / "numeric_obs_counts.csv"
RCPAQAP = VAR / "rcpaqap_categories.csv"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bps = _load("bps", C.SCRIPTS / "build_perturbation_spec.py")
a25 = _load("a25", C.ANALYSIS / "a25_partition_comparison.py")

#: the short names in the register, as the reader sees them
ANALYTE = {
    "Sodium": "Sodium", "Calcium": "Calcium, total and ionized", "Creatinine": "Creatinine",
    "ALT": "Alanine aminotransferase (ALT)", "ALP": "Alkaline phosphatase (ALP)",
    "FT4": "Thyroxine, free (FT4)", "HbA1c": "Hemoglobin A1c", "PSA": "Prostate-specific antigen (PSA)",
    "Haemoglobin": "Hemoglobin", "Haematocrit": "Hematocrit", "Erythrocytes": "Erythrocytes in blood",
    "MCV": "Mean corpuscular volume (MCV)", "MCH": "Mean corpuscular hemoglobin (MCH)",
    "MCHC": "Mean corpuscular hemoglobin concentration (MCHC)",
    "RDW": "Erythrocyte distribution width (RDW)", "Leukocytes": "Leukocytes in blood",
    "Platelets": "Platelets", "Urine WBC": "Leukocytes in urine sediment",
    "Urine RBC": "Erythrocytes in urine sediment", "INR": "INR", "Fibrinogen": "Fibrinogen",
    "Potassium": "Potassium", "Chloride": "Chloride", "Magnesium": "Magnesium", "Glucose": "Glucose",
    "Urate": "Urate", "Total protein": "Protein, total (serum)", "Bilirubin": "Bilirubin (all fractions)",
    "AST": "Aspartate aminotransferase (AST)", "LDH": "Lactate dehydrogenase (LDH)",
    "GGT": "Gamma-glutamyl transferase (GGT)", "Amylase": "Amylase", "CK": "Creatine kinase (CK)",
    "TSH": "Thyrotropin (TSH)", "aPTT": "aPTT",
}
#: short names for prose and the smaller tables
SHORT = {"Sodium": "sodium", "Calcium": "calcium", "Creatinine": "creatinine", "ALT": "ALT", "ALP": "ALP",
         "FT4": "FT4", "HbA1c": "HbA1c", "PSA": "PSA", "Haemoglobin": "hemoglobin",
         "Haematocrit": "hematocrit", "Erythrocytes": "erythrocytes", "MCV": "MCV", "MCH": "MCH",
         "MCHC": "MCHC", "RDW": "RDW", "Leukocytes": "leukocytes", "Platelets": "platelets",
         "Urine WBC": "urine leukocytes", "Urine RBC": "urine erythrocytes", "INR": "INR",
         "Fibrinogen": "fibrinogen", "Potassium": "potassium", "Chloride": "chloride",
         "Magnesium": "magnesium", "Glucose": "glucose", "Urate": "urate", "Total protein": "total protein",
         "Bilirubin": "bilirubin", "AST": "AST", "LDH": "LDH", "GGT": "GGT", "Amylase": "amylase",
         "CK": "CK", "TSH": "TSH", "aPTT": "aPTT"}
CITE = {"adeli2017_t2": "Adeli 2017", "friedberg2007_t5": "Friedberg 2007", "ricos2019": "Ricós 2019",
        "paal2021": "Paal 2021", "vu2022_t5": "Vu 2022", "cwiklinska2020": "Ćwiklińska 2020",
        "meijer2018_t2": "Meijer 2018", "cunningham2012": "Cunningham 2012"}
#: Table B.1 in a few words per cell, to keep prose out of tables. The longer descriptions, and how each figure
#: was taken, are in pipeline/variability_register.py.
MEASURE = {
    "adeli2017_t2": "Between-lab CV", "friedberg2007_t5": "Reference limits", "ricos2019": "Specifications",
    "vu2022_t5": "Proficiency-test CV", "paal2021": "Proficiency-test CV", "meijer2018_t2": "Between-lab CV",
    "cwiklinska2020": "Between-lab CV", "cunningham2012": "Proficiency-test CV",
}
#: where the laboratories are, from the descriptions in the register (Canadian; 163 US; German EQA; 10 European
#: countries; 183 Polish; the College of American Pathologists), the title of Ricós 2019 (EQA programs in Spain)
#: and the abstract of Vu 2022 (EQA organisers in Vietnam)
SETTING = {
    "adeli2017_t2": "Canada", "friedberg2007_t5": "US", "ricos2019": "Spain", "vu2022_t5": "Vietnam",
    "paal2021": "Germany", "meijer2018_t2": "Europe", "cwiklinska2020": "Poland", "cunningham2012": "US",
}
CLASS_NAME = {
    "CHEM": "chemistry", "HEM/BC": "hematology and blood count", "UA": "urinalysis", "COAG": "coagulation",
    "DRUG/TOX": "drugs and toxicology", "MICRO": "microbiology", "CELLMARK": "cell markers",
    "SERO": "serology", "PULM": "pulmonary", "SPEC": "specimen characteristics", "CLIN": "clinical",
    "CYTO": "cytology", "HEMODYN.MOLEC": "hemodynamics", "BDYTMP.ATOM": "body temperature",
}


def _pct1(x):
    return f"{x:.1f}"


#: each table as key -> (header, rows, align). tables.py draws the PNG and .tex from here.
TABLES = {}


def _table(key, header, rows, align=""):
    TABLES[key] = (list(header), [[str(c) for c in r] for r in rows], align)


def _cap(name):
    """First letter up for a table cell, unless the name is an abbreviation with capitals (aPTT, ALT)."""
    return name if any(ch.isupper() for ch in name) else name[0].upper() + name[1:]


def _negatives():
    """Negative raw values per LOINC code in the training split (aggregate counts only)."""
    a15 = _load("a15", C.ANALYSIS / "a15_field_identifiability.py")
    from fhir_data_utils import build_lab_code_mappings
    _, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    df = a15.load_cache("train")[["class_idx", "val"]]
    df = df[df["val"].notna()]
    neg = df[df["val"] < 0]
    counts = Counter(c2l.get(int(ci), "?") for ci in neg["class_idx"])
    return len(df), dict(counts)


def values():
    spec = C.read_csv(SPEC)
    byc = {r["loinc"]: r for r in spec}
    numeric = {r["loinc"]: r for r in C.read_csv(NUMERIC)}
    n_num = sum(int(numeric[c]["train_obs_numeric"]) for c in byc)
    n_rr = sum(int(numeric[c]["train_obs_numeric_with_refrange"]) for c in byc)
    obs = lambda codes: sum(int(byc[c]["n_obs"]) for c in codes)   # noqa: E731 (numeric observations)
    assert all(int(byc[c]["n_obs"]) == int(numeric[c]["train_obs_numeric"]) for c in byc)

    # -- the register: Tables B.1 and B.2 --
    recs = vr.RECORDS
    assert set(CITE) == set(vr.SOURCES) == set(MEASURE) == set(SETTING)
    assert {r[0] for r in recs} == set(ANALYTE)
    val_an = sorted({r[0] for r in recs if r[2] == "value_cv"})
    rng_an = sorted({r[0] for r in recs if r[2] == "range_pctV"})
    resolved_v = {a: vr.resolve(a, "value_cv") for a in val_an}
    resolved_r = {a: vr.resolve(a, "range_pctV") for a in rng_an}
    n_meas = sum(t == "observed" for _, t, _ in resolved_v.values())
    n_specn = sum(t == "specification" for _, t, _ in resolved_v.values())
    assert n_meas + n_specn == len(val_an)
    assert all(t == "observed" for _, t, _ in resolved_r.values())
    assert statistics.median(m for m, _, _ in resolved_v.values()) == bps.VALUE_FALLBACK
    assert round(statistics.median(m for m, _, _ in resolved_r.values()), 1) == bps.RANGE_FALLBACK

    def src_cell(srcs, tier):            # † a specification, ‡ derived by us (the caption says so)
        s = "; ".join(f"[{CITE[x]}]" for x in srcs)
        if tier == "specification":
            s += "†"
        if srcs == ["friedberg2007_t5"]:
            s += "‡"
        return s
    rows = []
    for a in sorted(ANALYTE, key=lambda a: ANALYTE[a].lower()):
        v, g = resolved_v.get(a), resolved_r.get(a)
        rows.append([ANALYTE[a], _pct1(v[0]) if v else "—", src_cell(v[2], v[1]) if v else "",
                     _pct1(g[0]) if g else "—", src_cell(g[2], g[1]) if g else ""])
    assert not any("†" in r[4] or "‡" in r[2] for r in rows)   # specifications are values, derivations ranges
    _table("B.2", ["Analyte", "Value CV (%)", "Source", "Range variation (%)", "Source"],
                            rows, "lrlrl")
    # the published figures the rules set aside
    unused = []
    for a in val_an:
        mag, tier, srcs = resolved_v[a]
        for r in recs:
            if r[0] == a and r[2] == "value_cv" and (r[3], r[5]) != (mag, srcs[0]) and r[5] not in srcs:
                unused.append((SHORT[a], r[3], CITE[r[5]], r[4]))
    assert all(t == "specification" for *_, t in unused) and len(unused) == 5
    unused_spec = ", ".join(f"{a} {_pct1(m)}%" for a, m, _, _ in sorted(unused, key=lambda u: u[0].lower()))
    ca = [r for r in recs if r[0] == "Calcium" and r[2] == "range_pctV"]
    assert len(ca) == 2 and resolved_r["Calcium"][0] == max(r[3] for r in ca)

    use = defaultdict(lambda: Counter())
    for a in val_an:
        for s in resolved_v[a][2]:
            use[s]["value " + resolved_v[a][1]] += 1
    for a in rng_an:
        for s in resolved_r[a][2]:
            use[s]["range"] += 1
    srows = []
    order = ["adeli2017_t2", "friedberg2007_t5", "ricos2019", "vu2022_t5", "paal2021", "meijer2018_t2",
             "cwiklinska2020", "cunningham2012"]
    assert sorted(order) == sorted(vr.SOURCES)
    for s in order:
        u = use[s]
        assert not (u["value observed"] and u["value specification"]), s    # one kind of value figure per paper
        n_value = u["value observed"] + u["value specification"]
        srows.append([f"[{CITE[s]}]", MEASURE[s], SETTING[s], n_value or "—", u["range"] or "—"])
    _table("B.1", ["Source", "Measure", "Setting", "Value CV", "Range variation"], srows, "lllrr")

    # -- analyte to LOINC: values for the text --
    named_v = [r for r in spec if r["value_tier"] in ("observed", "specification")]
    named_r = [r for r in spec if r["range_tier"] == "observed"]
    sodium = sorted((r for r in spec if r["value_source"] == "Sodium"), key=lambda r: r["name"])
    assert vr.regex_for("Sodium") == r"^Sodium \["
    multi = [r["loinc"] for r in spec
             if sum(bool(re.search(vr.regex_for(a), r["name"], re.I)) for a in ANALYTE) > 1]
    list_sodium = "; ".join(r["name"] for r in sodium)
    rx_wbc, rx_uwbc = vr.regex_for("Leukocytes"), vr.regex_for("Urine WBC")

    # -- code groups: Tables B.3 and B.4 --
    groups = defaultdict(list)
    for r in spec:
        groups[r["class_group"]].append(r)
    def split(g):
        cls, rest = g.split(" / ", 1)
        m = re.match(r"(.*?)( \[(general chem|endocrine)\])?$", rest)
        prop = m.group(1) + ({"general chem": ", general", "endocrine": ", endocrine"}[m.group(3)]
                             if m.group(3) else "")
        return cls, prop
    assert {split(g)[0] for g in groups} <= set(CLASS_NAME)
    grows = []
    for g, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cls, prop = split(g)
        top = sorted(rs, key=lambda r: (-int(r["n_obs"]), r["name"]))[0]
        grows.append([f"{cls}, {prop}", len(rs), top["loinc"], top["name"]])   # code with its name (LOINC licence §10.3)
    _table("B.4", ["Group", "Codes", "LOINC", "Long common name"], grows, "lrll")
    pg = defaultdict(list)
    for p, grp in a25.PROP_GROUP.items():
        pg[grp].append(p)
    _table("B.3", ["Property group", "LOINC properties"],
                              [[_cap(grp), ", ".join(sorted(ps))] for grp, ps in sorted(pg.items())])
    chem_conc = {k: len(v) for k, v in groups.items() if k.startswith("CHEM / concentration")}
    # the endocrine groups against RCPAQAP's Endocrine and Tumour Markers lists
    endo = sorted((r for r in spec if "[endocrine]" in r["class_group"]), key=lambda r: r["name"])
    rc = C.read_csv(RCPAQAP)
    rc_et = {r["analyte"] for r in rc if r["category"] in ("Endocrine", "Tumour Markers")}
    rc_bnp = {r["analyte"] for r in rc if r["category"] == "BNP"}
    assert "NT-Pro-BNP" in rc_bnp and "NT-Pro-BNP" not in rc_et and not any("27-29" in a for a in rc_et)
    extra = [r for r in endo if r["name"].startswith("Natriuretic peptide") or "27-29" in r["name"]]
    assert len(extra) == 3, [r["name"] for r in extra]

    # -- inherited magnitudes, the group defaults: Table B.5 --
    drows = []
    for g, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        av = [(float(r["value_cv_pct"]), r["value_source"]) for r in rs
              if r["value_tier"] in ("observed", "specification")]
        ar = [(float(r["range_pctV"]), r["range_source"]) for r in rs if r["range_tier"] == "observed"]
        cv = [r for r in rs if r["value_tier"] == "class"]
        cr = [r for r in rs if r["range_tier"] == "class"]
        if not (cv or cr):
            continue
        vd, rd = (max(av) if av else None), (max(ar) if ar else None)
        assert all(float(r["value_cv_pct"]) == vd[0] for r in cv)
        # as the caption says, a dash means no named magnitude of that kind, and the group's codes take the median
        assert av or all(r["value_tier"] == "fallback" for r in rs if r["value_tier"] not in ("", "derived"))
        assert ar or all(r["range_tier"] == "fallback" for r in rs if r["range_tier"] not in ("", "derived"))
        assert all(float(r["range_pctV_uncapped"]) == rd[0] for r in cr)
        capped = sum(r["range_cap_applied"] == "1" for r in cr)
        cls, prop = split(g)
        drows.append([f"{cls}, {prop}",
                      *([_pct1(vd[0]), _cap(SHORT[vd[1]]), len(cv)] if cv else ["—", "", ""]),
                      *([_pct1(rd[0]), _cap(SHORT[rd[1]]), len(cr), capped] if cr else ["—", "", "", ""])])
    _table("B.5", ["Group", "Value CV (%)", "From", "Codes", "Range variation (%)", "From",
                                    "Codes", "Capped"], drows, "lrlrrlrr")
    n_fb_v = sum(r["value_tier"] == "fallback" for r in spec)
    n_fb_r = sum(r["range_tier"] == "fallback" for r in spec)

    # -- the range cap, a37's calibration reproduced: Table B.6 --
    stated = {r["loinc"]: r for r in C.read_csv(STATED)}
    cal = {}
    for r in spec:
        hw = stated.get(r["loinc"], {}).get("halfwidth_pct")
        if r["range_tier"] != "observed" or r["magnitude_kind"] != "relative_pct" or not r["range_pctV"] or not hw:
            continue
        n = int(stated[r["loinc"]]["n_with_range"])
        if r["range_source"] not in cal or n > cal[r["range_source"]][0]:
            cal[r["range_source"]] = (n, r["loinc"], float(r["range_pctV"]), float(hw))
    assert len(cal) == len(rng_an)
    ratios = {a: pv / hw for a, (_, _, pv, hw) in cal.items()}
    amax = max(ratios, key=ratios.get)
    assert round(ratios[amax], 2) == bps.MAX_MEASURED_RATIO
    crow = [[_cap(SHORT[a]), _pct1(cal[a][2]), f"{cal[a][3]:.1f}", f"{ratios[a]:.2f}"]
            for a in sorted(ratios, key=ratios.get)]
    _table("B.6", ["Analyte", "Range variation (%)", "Stated half-width (%)", "Ratio"], crow, "lrrr")
    capped_codes = [r["loinc"] for r in spec if r["range_cap_applied"] == "1"]
    n_capped_rr = sum(int(numeric[c]["train_obs_numeric_with_refrange"]) for c in capped_codes)

    # -- perturbation shapes, the scales: Table B.7 --
    sc = defaultdict(list)
    for r in spec:
        sc[r["scale_class"]].append(r)
    mk = lambda cls: Counter(r["magnitude_kind"] for r in sc[cls])   # noqa: E731
    assert mk("log") == Counter({"rung": 5, "absolute": 2, "relative_pct": 1})
    assert mk("ladder") == Counter({"rung": 10, "": 1})
    assert set(mk("count_poisson")) == {"poisson"} and set(mk("ratio_near_1")) == {"excess_over_1"}
    # one line per rule. The caption defines z and CV. The only unperturbed ladder code is among the "Not
    # perturbed" codes of Table B.8, and the log codes moved by an absolute offset are the pH codes.
    assert all("pH" in r["name"] for r in sc["log"] if r["magnitude_kind"] == "absolute")
    rule = {
        "ratio": "Multiplied by 1 + z × CV",
        "integer_grid": "As ratio, then rounded to the code's precision",
        "bounded_frac": "As ratio, then clamped to [0, 100]",
        "count_poisson": "z × √value added, then rounded to an integer",
        "log": "pH offset (2), one step (5), relative (1)",
        "ratio_near_1": "Distance from 1 perturbed",
        "ladder": "One step up or down",
    }
    label = {"ratio": "Ratio", "integer_grid": "Integer grid", "bounded_frac": "Bounded fraction",
             "count_poisson": "Count", "log": "Log", "ratio_near_1": "Near 1", "ladder": "Ladder"}
    assert set(sc) == set(rule)
    srows = [[label[k], len(sc[k]), C.pct(obs([r["loinc"] for r in sc[k]]), n_num).rstrip("%"), rule[k]]
             for k in sorted(sc, key=lambda k: -len(sc[k]))]
    _table("B.7", ["Scale", "Codes", "Obs. (%)", "Rule"], srows, "lrrl")
    special = [r["loinc"] for k in ("log", "ratio_near_1", "ladder") for r in sc[k]]
    n_step = sum(r["magnitude_kind"] == "rung" for k in ("log", "ratio_near_1", "ladder") for r in sc[k])

    # a relative factor 1 + z*m/100 can fall below zero, because z is an untruncated standard normal
    # draw (site_z). The non-negative guard then sets the value, or both range bounds, to zero. These
    # are the expected shares.
    import math
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))            # noqa: E731
    def negf(sev):
        ev = er = 0.0
        for r in spec:
            if r["magnitude_kind"] != "relative_pct" or r["tier0_no_perturb"] == "1" or r["allow_negative"] == "1":
                continue
            mv, mr = float(r["value_cv_pct"] or 0) * sev, float(r["range_pctV"] or 0) * sev
            ev += int(r["n_obs"]) * (phi(-100 / mv) if mv else 0)
            er += int(numeric[r["loinc"]]["train_obs_numeric_with_refrange"]) * (phi(-100 / mr) if mr else 0)
        return ev / n_num, er / n_rr
    negf1, negf2 = negf(1.0), negf(2.0)

    # the pH offset: the CLIA blood-pH limit over the a22 median ratio of CLIA limits to measured CVs
    a22 = C.read_csv(C.RESULTS / "analysis" / "a22" / "clia_divisor.csv")
    ks = [float(r["k"]) for r in a22]
    assert round(statistics.median(ks), 2) == 2.82 and abs(0.04 / statistics.median(ks) - bps.PH_ABSOLUTE_OFFSET) < 5e-4

    # -- numerical guards: Table B.8 --
    tier0 = [r for r in spec if r["tier0_no_perturb"] == "1"]
    # the only laboratory measurement that the specimen-class rule catches (a known misfile)
    visc = [r for r in tier0 if C.long_name(r["loinc"]) == "Viscosity of Serum"]
    assert len(visc) == 1 and C.loinc_table()[visc[0]["loinc"]]["CLASS"] == "SPEC"
    exempt = sorted({C.loinc_table()[r["loinc"]]["COMPONENT"] for r in spec if r["allow_negative"] == "1"})
    assert exempt == ["Anion gap", "Base excess"], exempt
    assert bps.NEGATIVE_OK == {"Base excess", "Anion gap", "Bilirubin.indirect", "Cholesterol in LDL"}
    miss = {"Bilirubin.indirect": "Bilirubin.non-glucuronidated", "Cholesterol in LDL": "Cholesterol.in LDL"}
    defect = [r for r in spec if C.loinc_table()[r["loinc"]]["COMPONENT"] in miss.values()]
    tot_train, neg = _negatives()
    assert tot_train == n_num
    defect_neg = [r for r in defect if neg.get(r["loinc"])]
    clamp100 = [r for r in spec if r["clamp_0_100"] == "1"]
    derived = [r for r in spec if r["derived_role"]]
    assert sorted({r["derived_role"] for r in derived}) == ["MCH", "MCHC", "MCV"]
    panel = [r for r in spec if r["fraction_panel"] == "1"]
    # the non-negative clamp (G1) and the re-rounding (G7) act on numbers perturbed by a numeric rule. A ladder code
    # steps to a neighbouring rung and returns before them, and a code that is never perturbed is left as it is.
    numeric_kinds = {"relative_pct", "absolute", "excess_over_1", "poisson"}
    assert {r["magnitude_kind"] for r in spec if r["tier0_no_perturb"] != "1"} == numeric_kinds | {"rung"}
    src_p = open(C.SCRIPTS / "perturbation.py", encoding="utf-8").read()
    for frag in ('elif kind == "rung":', "return value if stepped is None else stepped",
                 'if not spec["allow_negative"] and out < 0:', "return round(out, nd)"):
        assert frag in src_p, frag
    by_rule = [r for r in spec if r["tier0_no_perturb"] != "1" and r["magnitude_kind"] in numeric_kinds]
    nonneg = [r for r in by_rule if r["allow_negative"] != "1"]
    rungs = [r for r in spec if r["tier0_no_perturb"] != "1" and r["magnitude_kind"] == "rung"]
    n_ = lambda rs: C.num(len(rs))                                              # noqa: E731
    sh = lambda codes: C.pct(obs(codes), n_num).rstrip("%")                     # noqa: E731
    scale = [r["loinc"] for r in sc["count_poisson"]] + special
    grows2 = [                      # labels as the thesis prints them
        ["Not perturbed", n_(tier0), sh([r["loinc"] for r in tier0])],
        ["Non-negative values", n_(nonneg), sh([r["loinc"] for r in nonneg])],
        ["Percentages within 0–100%", n_(clamp100), sh([r["loinc"] for r in clamp100])],
        ["Re-derive from perturbed inputs", n_(derived), sh([r["loinc"] for r in derived])],
        ["White-cell differential keeps its total", n_(panel), sh([r["loinc"] for r in panel])],
        ["Scale-specific rules (Table B.7)", C.num(len(scale)), sh(scale)],
        ["Keep original precision", n_(by_rule), sh([r["loinc"] for r in by_rule])],
    ]
    _table("B.8", ["Guard", "Codes", "Obs. (%)"], grows2, "lrr")

    v = {
        "n_records": C.num(len(recs)),
        "n_sources_lit": C.num(len(vr.SOURCES)),
        "n_val_an": C.num(len(val_an)), "n_meas": C.num(n_meas), "n_specn": C.num(n_specn),
        "n_rng_an": C.num(len(rng_an)),
        "unused_spec": unused_spec,
        "ca_adeli": _pct1(min(r[3] for r in ca)), "ca_fried": _pct1(max(r[3] for r in ca)),
        "rx_sodium": vr.regex_for("Sodium"), "n_sodium": C.num(len(sodium)), "list_sodium": list_sodium,
        "rx_wbc": rx_wbc, "rx_uwbc": rx_uwbc,
        "n_named_v": C.num(len(named_v)), "n_named_r": C.num(len(named_r)),
        "multi_sentence": ("No code's name matches more than one analyte's pattern." if not multi else
                           f"{len(multi)} codes match more than one pattern; the analyte first in alphabetical order is used."),
        "n_groups": C.num(len(groups)),
        "n_chem_gen": C.num(chem_conc["CHEM / concentration [general chem]"]),
        "n_chem_endo": C.num(chem_conc["CHEM / concentration [endocrine]"]),
        "n_endo": C.num(len(endo)),
        "n_fb_v": C.num(n_fb_v), "n_fb_r": C.num(n_fb_r),
        "cap": f"{bps.MAX_MEASURED_RATIO:.2f}",
        "ratio_min": f"{min(ratios.values()):.2f}",
        "amin": SHORT[min(ratios, key=ratios.get)], "amax": SHORT[amax],
        "n_capped": C.num(len(capped_codes)), "pct_capped_rr": C.pct(n_capped_rr, n_rr),
        "ph_offset": f"{bps.PH_ABSOLUTE_OFFSET:g}",
        "n_special": C.num(len(special)), "pct_special": C.pct(obs(special), n_num), "n_step": C.num(n_step),
        "n_rung": C.num(len(rungs)),
        "defect_list": "; ".join(f"{r['loinc']}, {C.long_name(r['loinc'])}" for r in defect),
        "n_defect": C.num(len(defect)),
        "defect_neg_list": " and ".join(f"{neg[r['loinc']]} for {r['loinc']}" for r in defect_neg),
        "n_defect_obs": C.num(obs([r["loinc"] for r in defect])),
        "n_defect_neg": C.num(sum(neg.get(r["loinc"], 0) for r in defect)),
        "n_num": C.num(n_num),
        "sodium_cv": _pct1(resolved_v["Sodium"][0]),
        "class_names": ", ".join(f"{k} ({CLASS_NAME[k]})" for k in sorted({split(g)[0] for g in groups},
                                 key=lambda k: (-sum(len(rs) for g, rs in groups.items() if split(g)[0] == k), k))),
        "n_k": C.num(len(ks)), "k_min": f"{min(ks):.2f}", "k_max": f"{max(ks):.2f}", "k_med": f"{statistics.median(ks):.2f}",
        "tier0_list": "; ".join(sorted(C.long_name(r["loinc"]) for r in tier0)),
        "n_visc": C.num(obs([visc[0]["loinc"]])),
        "negf_r1": f"{100 * negf1[1]:.2f}%", "negf_r2": f"{100 * negf2[1]:.1f}%",
    }
    assert negf1[0] < 1e-5      # "almost no values" at the published magnitudes
    assert v["ph_offset"] == "0.014", v["ph_offset"]   # a small constant: keep it out of _pct1's rounding
    assert len(defect) == 3 and len(defect_neg) == 2
    return v
