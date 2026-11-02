"""Test whether one divisor k converts the CLIA allowable total error (TEa) to a coefficient of
variation (thesis Chapter 4 and Appendix B).

CLIA TEa is a regulatory ceiling on total error (bias plus imprecision), while a perturbation needs
a dispersion parameter. The textbook conversion is CV = TEa / k, with k around 3. The script checks
whether a single k is supported empirically, on the analytes that have both a CLIA limit and an
observed between-laboratory CV (Adeli 2017, Paal 2021).

Absolute and hybrid CLIA limits are converted to a percentage at the MIMIC median concentration of
the code, taken from a17_T1_per_code.csv (analysis/a17). Reads data/variability/clia_2026_limits.csv.
Prints the table and writes it to results/analysis/a22/clia_divisor.csv (aggregate values only).

  python analysis/a22_clia_divisor.py
"""
import csv, re, sys
from statistics import median
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
CLIA = ROOT / "data" / "variability" / "clia_2026_limits.csv"
T1 = ROOT / "results" / "analysis" / "a17" / "a17_T1_per_code.csv"
OUT = ROOT / "results" / "analysis" / "a22"

#: analyte -> (CLIA analyte name, observed CV_BL %, source, regex on a17 `component`,
#:             expected unit of the MIMIC median)
PAIRS = [
    ("Sodium",     "Sodium",                             1.1,  "Adeli 2017 T2", r"^Sodium \[Moles/volume\] in Serum", "mmol/L"),
    ("Calcium",    "Calcium, total",                     2.6,  "Adeli 2017 T2", r"^Calcium \[Mass/volume\] in Serum",  "mg/dL"),
    ("Creatinine", "Creatinine",                         5.0,  "Adeli 2017 T2", r"^Creatinine \[Mass/volume\] in Serum", "mg/dL"),
    ("ALT",        "Alanine aminotransferase (ALT/SGPT)", 24.6, "Adeli 2017 T2", r"^Alanine aminotransferase \[Enzymatic activity/volume\] in Serum", "U/L"),
    ("ALP",        "Alkaline Phosphatase",               6.6,  "Adeli 2017 T2", r"^Alkaline phosphatase \[Enzymatic activity/volume\] in Serum", "U/L"),
    ("FT4",        "Free throxine",                      14.3, "Adeli 2017 T2", r"^Thyroxine \(T4\) free \[Mass/volume\] in Serum", "ng/dL"),
    ("Calcium",    "Calcium, total",                     3.1,  "Paal 2021 EQA", r"^Calcium \[Mass/volume\] in Serum",  "mg/dL"),
    ("Creatinine", "Creatinine",                         4.7,  "Paal 2021 EQA", r"^Creatinine \[Mass/volume\] in Serum", "mg/dL"),
    ("HbA1c",      "Hemoglobin A1c",                     5.1,  "Paal 2021 EQA", r"^Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    ("PSA",        "Prostate Specific Antigen, total",   18.8, "Paal 2021 EQA", r"^Prostate specific Ag \[Mass/volume\] in Serum", "ng/mL"),
]


def tea_pct(row, conc):
    """CLIA limit expressed as a percentage at concentration `conc`."""
    pct = float(row["pct"]) if row["pct"] else None
    ab = float(row["abs"]) if row["abs"] else None
    ab_pct = 100.0 * ab / conc if (ab and conc) else None
    shape = row["shape"]
    if shape == "relative":
        return pct, f"{pct:g}%"
    if shape == "absolute":
        return ab_pct, f"±{ab:g} {row['abs_unit']} at median {conc:g} = {ab_pct:.1f}%"
    # hybrid: CLIA says "greater of"
    if ab_pct is None:
        return pct, f"{pct:g}% (no median)"
    g = max(pct, ab_pct)
    which = "abs" if ab_pct > pct else "pct"
    return g, f"greater of {pct:g}% and ±{ab:g} at median {conc:g} ({ab_pct:.1f}%) -> {g:.1f}% [{which}]"


def main():
    clia = {r["analyte"]: r for r in csv.DictReader(open(CLIA))}
    med = {}
    for r in csv.DictReader(open(T1)):
        comp = r["component"]
        n = int(r["n"])
        if not r["median"]:
            continue
        # keep the highest-volume code matching each component string
        prev = med.get(comp)
        if prev is None or n > prev[1]:
            med[comp] = (float(r["median"]), n)

    print(f"{'analyte':11s} {'source':15s} {'CV_BL':>6s} {'TEa%':>6s} {'k':>5s}   how TEa% was obtained")
    print("-" * 110)
    ks = []
    rows = []
    for name, cn, cv, src, rx, unit in PAIRS:
        if cn not in clia:
            print(f"{name:11s} !! '{cn}' not in CLIA table"); continue
        hits = [(c, v) for c, v in med.items() if re.search(rx, c, re.I)]
        if not hits:
            print(f"{name:11s} !! no a17 median matching {rx}"); continue
        comp, (m, n) = max(hits, key=lambda x: x[1][1])
        t, how = tea_pct(clia[cn], m)
        k = t / cv
        ks.append(k)
        rows.append({"analyte": name, "source": src, "cv_bl_pct": cv, "tea_pct": round(t, 2),
                     "k": round(k, 3), "mimic_median": m, "median_unit": unit,
                     "n_obs": n, "clia_shape": clia[cn]["shape"], "derivation": how})
        print(f"{name:11s} {src:15s} {cv:6.1f} {t:6.1f} {k:5.2f}   {how}")
    print("-" * 110)
    lo, hi = min(ks), max(ks)
    print(f"\nk = TEa% / observed CV_BL%   over {len(ks)} analyte-source pairs")
    print(f"   min {lo:.2f}   median {median(ks):.2f}   max {hi:.2f}   spread {hi/lo:.1f}x")
    print("   pairs where CLIA is tighter than the observed CV (k < 1): "
          f"{[r['analyte'] for r in rows if r['k'] < 1]}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "clia_divisor.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT/'clia_divisor.csv'}")


if __name__ == "__main__":
    sys.exit(main())
