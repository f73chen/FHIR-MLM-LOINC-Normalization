"""The values and tables of thesis Appendix A (unit standardization).

Sources: pipeline/unit_standardisation.py (the frozen table and the runtime rules), the alias map,
the attestation table from a20, the verdicts from a31 and the frozen randomisation table
(pipeline/build_unit_randomisation.py). The asserts stop the render if the code no longer says what
the appendix text claims. analysis/appendix/tables.py renders the output:

  python analysis/appendix/tables.py
"""
from collections import defaultdict
from types import SimpleNamespace

import common as C
import unit_standardisation as us
from input_preprocessing_utils import load_unit_alias_map

A20 = C.RESULTS / "analysis" / "a20" / "a20_external_attestation.csv"
A20_UNITS = C.RESULTS / "analysis" / "a20" / "a20_codes_per_unit.csv"
A31 = C.RESULTS / "analysis" / "a31" / "disagreement_resolution.csv"
FROZEN = C.RERUN / "data" / "variability" / "unit_randomisation.csv"

#: each target with the kind of quantity it measures, in the order Table A.1 lists them. The names are
#: standard where one exists (Schadow 1999, LOINC properties), and "molar concentration" follows
#: Chapter 4. The U/mL kind is "units", because its U/mL codes are tumour markers (CA 125, CA 27-29,
#: CA 19-9) and heparin assays (LOINC [Units/volume]). In that kind, only the 103 U/L observations
#: (amylase, lipase) are enzyme activities, by the code-unit counts in a16.
KINDS = [
    ("mg/dL", "Mass concentration"), ("mmol/L", "Molar concentration"),
    ("meq/L", "Equivalents per volume"), ("10*3/uL", "Number concentration"),
    ("[IU]/L", "International units per volume"), ("U/mL", "Units per volume"),
    ("U/g", "Units per mass"), ("[arb'U]", "Arbitrary units"),
    ("[GPL'U]", "IgG phospholipid units"), ("[MPL'U]", "IgM phospholipid units"),
    ("%", "Fraction or mass ratio"), ("fL", "Volume"), ("pg", "Mass"), ("s", "Time"),
    ("L/min", "Volume flow rate"), ("mm/h", "Velocity"), ("mg/(24.h)", "Mass rate"),
    ("mm[Hg]", "Pressure"), ("mosm/kg", "Osmolality"),
    ("{#}/[HPF]", "Number per high-power field"), ("{#}/[LPF]", "Number per low-power field"),
    ("{score}", "Score"),
]

#: a31's rule -> (how Table A.2 names the difference, with the MIMIC-IV unit first, and a check that the
#: name fits every code under the rule), in display order. The caption explains "interchangeable" and
#: "mapping issue".
RULES = {
    "eq vs mol: the factor depends on valence":
        ("Equivalents vs moles", lambda m, e: (m, e) == ("meq/L", "mmol/L")),
    "IU == U == arb'U, an assay-defined activity unit":
        ("IU or U vs arbitrary units",
         lambda m, e: m in ("[IU]/L", "[IU]/mL", "U/L", "U/mL") and e.startswith("[arb'U]/")),
    'MIMIC\'s "units" is ambiguous across properties':
        ("Arbitrary units vs pH", lambda m, e: (m, e) == ("[arb'U]", "[pH]")),
    "different QUANTITY, not different units":
        ("Fraction vs count", lambda m, e: m == "%" and set(e.split(",")) <= {"10*3/uL", "10*9/L"}),
    "a UCUM annotation is not meaning-bearing":
        ("Arbitrary units vs annotation", lambda m, e: m == "[arb'U]" and e[:1] == "{" and e[-1:] == "}"),
    "mass vs molar is a different LOINC code by definition":
        ("Mass vs molar concentration", lambda m, e: m.endswith("g/mL") and e.endswith("mol/L")),
    "MIMIC has no volume denominator":
        ("Arbitrary units vs concentration", lambda m, e: (m, e) == ("[arb'U]", "[arb'U]/mL")),
}
#: what was done, by a31's resolution (the pH rule is overridden in resolution_rows)
DONE = {"RANDOMISE": "Randomized (A.5)",
        "MERGE": "Kept (interchangeable)",
        "NOT_A_UNIT_ISSUE": "Kept (mapping issue)"}
A2_HEADER = ["Difference", "MIMIC-IV unit", "Terminology unit", "Codes", "Obs. (%)", "Action"]
A3_HEADER = ["LOINC code", "Long common name"]


A1_HEADER = ["Kind of quantity", "Target unit", "Source unit", "Factor"]


def conversion_groups():
    """The content of Table A.1 from the frozen table, as [(kind, target, [(source unit, factor), ...])]
    in KINDS order, with sources by falling factor. tables.py draws it (.tex, PNG)."""
    by_target = defaultdict(list)
    for u, (t, f) in us.STANDARD_UNIT.items():
        if u != t:
            by_target[t].append((u, f))
    targets = {t for t, _ in us.STANDARD_UNIT.values()}
    assert {t for t, _ in KINDS} == targets and len(KINDS) == len(targets)
    return [(label, t, sorted(by_target[t], key=lambda uf: -uf[1])) for t, label in KINDS]


def conversion_rows(groups, fmt):
    """One row per source unit, with the kind and target only on the first row of a group. `fmt`
    prints a factor."""
    rows = []
    for label, t, members in groups:
        for i, (u, f) in enumerate(members or [("—", None)]):
            rows.append([label if i == 0 else "", t if i == 0 else "", u, "—" if f is None else fmt(f)])
    return rows


def disagreements():
    """The a20 comparison (test split) and the a31 disagreements, joined with the frozen randomisation
    table."""
    att = C.read_csv(A20)
    tested = [r for r in att if r["verdict"] in ("kind_agrees", "kind_differs")]
    agree = [r for r in tested if r["verdict"] == "kind_agrees"]
    res = C.read_csv(A31)
    assert len(res) == len(tested) - len(agree), "a31 and a20 disagree on the number of disagreements"
    frozen = C.read_csv(FROZEN)
    fz = {r["loinc"] for r in frozen}
    allowed = [r for r in res if r["resolution"] == "RANDOMISE"]
    assert fz <= {r["loinc"] for r in allowed}
    kept_ph = [r for r in allowed if r["loinc"] not in fz]
    assert kept_ph and all(r["component"] == "pH" for r in kept_ph)
    return SimpleNamespace(att=att, tested=tested, agree=agree, toto=sum(int(r["n_obs"]) for r in tested),
                           res=res, frozen=frozen, fz=fz, kept_ph=kept_ph)


def _share(n, total):
    """Percent to one decimal. A share that would print as 0.0 prints as <0.1, as in the Chapter 4 tables."""
    p = 100 * n / total
    return "<0.1" if 0 < p < 0.05 else f"{p:.1f}"


def ph_share(d):
    """The share of the MIMIC-IV `[arb'U]` observations (test split) that are pH. This share is why
    `[arb'U]` and `[pH]` both mark pH and why randomizing between them brings no benefit (see the
    caption of Table A.2)."""
    arb = [r for r in d.att if r["mimic_unit"] == "[arb'U]"]
    ph = {r["loinc"] for r in d.kept_ph}
    assert ph <= {r["loinc"] for r in arb}
    return sum(int(r["n_obs"]) for r in arb if r["loinc"] in ph) / sum(int(r["n_obs"]) for r in arb)


def resolution_rows(d):
    """The rows of Table A.2, one per a31 rule in RULES order, then the total. The example in each row is
    the largest code under the rule. tables.py draws them."""
    assert {r["rule"] for r in d.res} == set(RULES), "a31 has a rule Table A.2 does not describe"
    rows = []
    for rule, (label, fits) in RULES.items():
        rs = sorted((r for r in d.res if r["rule"] == rule), key=lambda r: -int(r["n_obs"]))
        assert all(fits(r["mimic_unit"], r["external_unit"]) for r in rs), rule
        if all(r["loinc"] in d.fz for r in rs):
            done = DONE["RANDOMISE"]
        elif rs[0]["resolution"] == "RANDOMISE":       # allowed, but left out of the frozen table: pH
            assert all(r in d.kept_ph for r in rs)
            assert ph_share(d) >= 0.95, "the caption says both spellings mark pH"
            done = "Kept (no benefit)"
        else:
            done = DONE[rs[0]["resolution"]]
        rows.append([label, rs[0]["mimic_unit"], rs[0]["external_unit"], C.num(len(rs)),
                     _share(sum(int(r["n_obs"]) for r in rs), d.toto), done])
    rows.append(["Total", "", "", C.num(len(d.res)), _share(sum(int(r["n_obs"]) for r in d.res), d.toto), ""])
    return rows


def randomized_rows(frozen):
    """The rows of Table A.3, grouped by analyte (LOINC component) and sorted by long common name within
    each, and the indices of the rows that start a new analyte. Every code keeps its official name, as
    the licence requires."""
    rand = sorted(frozen, key=lambda r: (r["component"], C.long_name(r["loinc"]), r["loinc"]))
    rows = [[r["loinc"], C.long_name(r["loinc"])] for r in rand]
    starts = {i for i in range(1, len(rand)) if rand[i]["component"] != rand[i - 1]["component"]}
    return rows, starts


def values():
    amap = load_unit_alias_map(C.UNIT_ALIAS_JSON)
    canon = lambda u: us.canonical_unit(u, amap)  # noqa: E731
    assert canon("mEq/L") == "meq/L" and canon("#/hpf") == "{#}/[HPF]"
    assert us.ALIAS_MAP_CORRECTIONS == {"[S]": "s", "AU": "[arb'U]"}
    assert canon("sec") == "s" and canon("units") == "[arb'U]" and canon("s") == "S"
    from input_preprocessing_utils import normalize_unit       # the alias map before correction
    assert normalize_unit("sec", amap) == ("[S]", True) and normalize_unit("units", amap) == ("AU", True)
    assert us.standardise_unit("log10 copies/ml", amap) == ("log10 copies/ml", 1.0)
    raw_units = {r["unit"] for r in C.read_csv(A20_UNITS) if r["level"] == "raw"}
    assert "sec" in raw_units and "s" not in raw_units and "units" in raw_units

    # -- the examples in A.2, as the frozen table shows them: eq stays apart from mol, [IU] from U and
    # from plain counts, and an annotation-only unit from % (each is its own target) --
    assert all(us.STANDARD_UNIT[u][0] == u for u in ("meq/L", "mmol/L", "[IU]/L", "U/mL", "10*3/uL", "{score}", "%"))

    # -- Table A.1: the frozen conversion table --
    groups = conversion_groups()
    rows = conversion_rows(groups, C.factor)
    assert not any("`" in c or "*" in c.replace("10*", "") for r in rows for c in r)
    targets = {t for _, t, _ in groups}

    # -- agreement with the terminologies (a20, test split) and Table A.2 (a31) --
    d = disagreements()
    att, tested, agree, toto, res, frozen = d.att, d.tested, d.agree, d.toto, d.res, d.frozen
    obs_agree = sum(int(r["n_obs"]) for r in agree)
    assert len({r["loinc"] for r in att}) == len(att)      # one row per label-space LOINC code
    rows = resolution_rows(d)
    # the share in the total row and the agreeing share, as printed, add up to 100
    assert round(100 - float(C.pct(obs_agree, toto).rstrip("%")), 1) == float(rows[-1][4])
    assert not any("`" in c for r in rows for c in r)

    # -- Table A.3: the randomized codes --
    assert all(r["notation_a"] == "mEq/L" and r["notation_b"] == "mmol/L" for r in frozen)
    comps = {r["component"] for r in frozen}
    assert comps <= {"Sodium", "Potassium", "Chloride", "Bicarbonate", "Carbon dioxide", "Anion gap",
                     "Base excess"}, comps
    rows3, _ = randomized_rows(frozen)
    assert all(name for _, name in rows3)


    v = {
        "n_alias": C.num(len(amap)),
        "n_units": C.num(len(us.STANDARD_UNIT)),
        "n_targets": C.num(len(targets)),
        "n_converted": C.num(sum(len(m) for _, _, m in groups)),
        "n_lab": C.num(len(att)), "n_tested": C.num(len(tested)),
        "n_agree": C.num(len(agree)),
        "pct_agree": C.pct(len(agree), len(tested)),
        "pct_agree_obs": C.pct(obs_agree, toto),
        "n_disagree": C.num(len(res)),
        "n_rand": C.num(len(frozen)),
        }
    return v
