"""Tiered perturbation coverage over the quantitative label space: how much of each analyte class each
kind of evidence covers (thesis Chapter 4).

Tier 1a  named analyte, observed between-lab magnitude from a study
Tier 1b  named analyte, CLIA/RCPAQAP allowable total error. This gives the shape
         (absolute below a breakpoint, relative above) and a ceiling, but no CV
Tier 2   analyte class default
Tier 0   not an analytical measurement (device settings, timings): no perturbation

Matching uses the LOINC LONG_COMMON_NAME, because the coagulation codes all share
COMPONENT='Coagulation'. They can only be told apart by the long name.

Reads data/labelspace/labelspace.csv, data/variability/clia_2026_limits.csv and
UPSTREAM_DIR/Loinc.csv. Prints the coverage and writes it to
results/analysis/a21/class_coverage.csv (aggregate output only).

  python analysis/a21_perturbation_coverage.py
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
LABELSPACE = ROOT / "data" / "labelspace" / "labelspace.csv"
CLIA_CSV = ROOT / "data" / "variability" / "clia_2026_limits.csv"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = ROOT / "results" / "analysis" / "a21"

#: Tier 1a comes from the canonical register (pipeline/variability_register.py). No copy is
#: kept here. a27 verifies the register and gives the canonical coverage numbers. Use those in
#: preference to the tier-1a columns of this script.
sys.path.insert(0, str(locations.PIPELINE))
import variability_register as vr  # noqa: E402

TIER1A = {a: (vr.regex_for(a),
              vr.resolve(a, "value_cv") is not None,
              vr.resolve(a, "range_pctV") is not None)
          for a in sorted({r[0] for r in vr.RECORDS})}

#: Tier 1b: CLIA analytes, regex on LONG_COMMON_NAME. Keys mirror clia_2026_limits.csv.
TIER1B = {
    "Albumin": r"^Albumin \[", "Blood gas pCO2": r"^Carbon dioxide \[Partial pressure",
    "Blood gas pO2": r"^Oxygen \[Partial pressure", "Blood gas pH": r"^pH of (Arterial |Venous |Capillary |Mixed venous )?[Bb]lood",
    "B-natriuretic peptide (BNP)": r"Natriuretic peptide B \[",
    "Pro B-natriuetic peptide (proBNP)": r"Natriuretic peptide\.B prohormone",
    "Carbon dioxide": r"^Carbon dioxide, total \[", "Cholesterol, total": r"^Cholesterol \[",
    "Cholesterol, high density liprotein": r"^Cholesterol in HDL",
    "Ferritin": r"^Ferritin \[", "Iron, total": r"^Iron \[",
    "Phosphorus": r"^Phosphate \[", "Triglycerides": r"^Triglyceride \[",
    "Troponin I": r"^Troponin I", "Troponin T": r"^Troponin T",
    "Urea Nitrogen": r"^Urea nitrogen \[", "Total Iron Binding Capacity (direct)": r"Iron binding capacity",
    "Complement C3": r"^Complement C3 \[", "Complement C4": r"^Complement C4 \[",
    "C-reactive protein (HS)": r"^C reactive protein", "IgA": r"^IgA \[", "IgE": r"^IgE \[",
    "IgG": r"^IgG \[", "IgM": r"^IgM \[", "Cancer antigen (CA) 125": r"^Cancer Ag 125",
    "Cortisol": r"^Cortisol \[", "Estradiol": r"^Estradiol \[", "Folate, serum": r"^Folate \[",
    "Follicle stimulating hormone": r"^Follitropin \[", "Free throxine": r"^Thyroxine \(T4\) free \[",
    "Human chorionic gonadotropin": r"^Choriogonadotropin", "Luteinizing hormone": r"^Lutropin \[",
    "Parathyroid hormone": r"^Parathyrin", "Progesterone": r"^Progesterone \[",
    "Prolactin": r"^Prolactin \[", "Testosterone": r"^Testosterone \[",
    "Triiodothyronine": r"^Triiodothyronine \[", "Thyroxine": r"^Thyroxine \[Mass",
    "Vitamin B12": r"^Cobalamin \[", "Acetaminophen": r"^Acetaminophen \[",
    "Alcohol, blood": r"^Ethanol \[", "Blood lead": r"^Lead \[", "Carbamazepine": r"^Carbamazepine",
    "Digoxin": r"^Digoxin \[", "Gentamicin": r"^Gentamicin \[", "Lithium": r"^Lithium \[",
    "Phenobarbital": r"^Phenobarbital \[", "Phenytoin": r"^Phenytoin", "Salicylate": r"^Salicylate",
    "Theophylline": r"^Theophylline \[", "Tobramycin": r"^Tobramycin \[",
    "Valproic acid": r"^Valproate \[", "Vancomycin": r"^Vancomycin \[",
    "Erythrocyte count": r"^Erythrocytes \[#/volume\]", "Hematocrit": r"^Hematocrit \[",
    "Leukocyte count": r"^Leukocytes \[#/volume\]", "Fibrinogen": r"^Fibrinogen \[",
    "Prothrombin time": r"^Prothrombin time \(PT\)|^INR in ",
}

#: Tier 0: quantitative but not an analytical measurement of a specimen.
NOT_ANALYTICAL = re.compile(
    r"^Specimen collection|Inhaled oxygen concentration|Ventilat|Positive end expiratory|"
    r"^Body temperature|Tidal volume|airway pressure|Respiratory (rate|system)|"
    r"Oxygen/Total gas setting|Volume of Urine|collection duration", re.I)

ELECTROLYTE = re.compile(
    r"^(Sodium|Potassium|Chloride|Bicarbonate|Magnesium|Calcium|Phosphate|Anion gap|"
    r"Carbon dioxide, total)[\s\[.]", re.I)
HORMONE = re.compile(
    r"Thyrotropin|Thyroxine|Triiodothyronine|Cortisol|Prostate specific|Choriogonadotropin|"
    r"Cancer Ag|Ferritin|Folate|Cobalamin|Vitamin|Parathyrin|Aldosterone|Renin|Insulin|"
    r"Troponin|Natriuretic peptide|Procalcitonin|Prolactin|Testosterone|Estradiol|"
    r"Follitropin|Lutropin|Progesterone", re.I)
DERIVED_PROP = {"NFr", "MFr", "VFr", "MRto", "ArVRat", "Ratio", "CFr", "SFr", "AFr"}


def classify(r):
    """(class, is_derived) from LOINC metadata alone."""
    cls, prop, lcn = r["CLASS"], r["PROPERTY"], r["LONG_COMMON_NAME"]
    if NOT_ANALYTICAL.search(lcn) or cls in ("PULM", "CLIN", "BDYTMP.ATOM", "SPEC"):
        return "not an analytical measurement", False
    if prop in DERIVED_PROP:
        return "derived fraction / ratio", True
    if cls == "COAG":
        return "coagulation", False
    if cls == "UA":
        return "urinalysis", False
    if cls == "DRUG/TOX":
        return "drug / toxicology", False
    # "pH of" alone would swallow urine and body-fluid pH, which are test-strip
    # ladders with a completely different error model.
    if prop == "PPres" or re.match(r"pH of (Arterial |Venous |Capillary |Mixed venous )?[Bb]lood", lcn):
        return "blood gas", False
    if prop.endswith("CCnc"):
        return "enzyme", False
    if prop == "ACnc" or HORMONE.search(lcn):
        return "hormone / immunoassay", False
    if cls == "HEM/BC":
        return "haematology index", False
    if cls == "CHEM":
        return "electrolyte" if ELECTROLYTE.match(lcn) else "substrate / core chemistry", False
    return "other (cell marker, micro, sero)", False


def main():
    occ = {r["loinc"]: int(r["train_occ"]) for r in csv.DictReader(open(LABELSPACE))}
    clia = {r["analyte"]: r for r in csv.DictReader(open(CLIA_CSV))}
    unknown = set(TIER1B) - set(clia)
    if unknown:
        sys.exit(f"TIER1B names not in clia_2026_limits.csv: {sorted(unknown)}")

    qn = {}
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r["LOINC_NUM"] in occ and r["SCALE_TYP"] == "Qn":
                qn[r["LOINC_NUM"]] = r
    TOT = sum(occ[c] for c in qn)

    t1v, t1r, t1b, cls_of = {}, {}, {}, {}
    for c, r in qn.items():
        lcn = r["LONG_COMMON_NAME"]
        cls_of[c] = classify(r)
        for name, (rx, hv, hr) in TIER1A.items():
            if re.search(rx, lcn, re.I):
                if hv: t1v[c] = name
                if hr: t1r[c] = name
                break
        for name, rx in TIER1B.items():
            if re.search(rx, lcn, re.I):
                t1b[c] = name
                break

    agg = defaultdict(lambda: {"codes": 0, "obs": 0, "v": 0, "r": 0, "b": 0,
                               "anchors": set(), "derived": False})
    for c, r in qn.items():
        k, der = cls_of[c]
        a = agg[k]
        a["codes"] += 1; a["obs"] += occ[c]; a["derived"] = der
        if c in t1v: a["v"] += occ[c]; a["anchors"].add(t1v[c])
        if c in t1r: a["r"] += occ[c]; a["anchors"].add(t1r[c])
        if c in t1b: a["b"] += occ[c]; a["anchors"].add(t1b[c])

    order = sorted(agg, key=lambda k: -agg[k]["obs"])
    print(f"quantitative labelspace: {len(qn)} codes, {TOT:,} train observations\n")
    hdr = (f"{'class':32s} {'codes':>5s} {'obs share':>9s} {'1a val':>7s} {'1a rng':>7s} "
           f"{'1b CLIA':>8s} {'anch':>5s}")
    print(hdr); print("-" * len(hdr))
    for k in order:
        a = agg[k]
        f = lambda x: f"{100*x/a['obs']:6.1f}%"
        print(f"{k:32s} {a['codes']:5d} {100*a['obs']/TOT:8.2f}% {f(a['v'])} {f(a['r'])} "
              f"{f(a['b']):>8s} {len(a['anchors']):5d}")
    tv = sum(a["v"] for a in agg.values()); tr = sum(a["r"] for a in agg.values())
    tb = sum(a["b"] for a in agg.values())
    t0 = agg["not an analytical measurement"]["obs"]
    either = sum(occ[c] for c in qn if c in t1v or c in t1b)
    print("-" * len(hdr))
    print(f"{'ALL':32s} {len(qn):5d} {100.0:8.2f}% {100*tv/TOT:6.1f}% {100*tr/TOT:6.1f}% "
          f"{100*tb/TOT:7.1f}%")
    print(f"\nvalue side : tier 1a {100*tv/TOT:.1f}%, and with tier 1b (CLIA) the named-analyte total is "
          f"{100*either/TOT:.1f}% of quantitative obs")
    print(f"range side : tier 1a {100*tr/TOT:.1f}%  (no CLIA tier, as CLIA limits apply "
          "to values only)")
    print(f"tier 0     : {100*t0/TOT:.2f}% not analytical, excluded from perturbation")
    print(f"tier 2     : the remainder, {100*(TOT-either-t0)/TOT:.1f}% of obs on the value side")
    nov = [k for k in order if not agg[k]["anchors"] and not agg[k]["derived"]
           and k != "not an analytical measurement"]
    print("\nclasses with no named anchor inside them:")
    print("   " + ("none (every perturbable class contains at least one anchored analyte)"
                   if not nov else
                   "\n   ".join(f"{k:32s} {100*agg[k]['obs']/TOT:5.2f}%" for k in nov)))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "class_coverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "codes", "obs", "obs_share_pct", "t1a_value_pct", "t1a_range_pct",
                    "t1b_clia_pct", "n_anchors", "anchors", "derived"])
        for k in order:
            a = agg[k]
            w.writerow([k, a["codes"], a["obs"], round(100*a["obs"]/TOT, 4),
                        round(100*a["v"]/a["obs"], 2), round(100*a["r"]/a["obs"], 2),
                        round(100*a["b"]/a["obs"], 2), len(a["anchors"]),
                        "|".join(sorted(a["anchors"])), int(a["derived"])])
    print(f"\nwrote {OUT/'class_coverage.csv'}")


if __name__ == "__main__":
    sys.exit(main())
