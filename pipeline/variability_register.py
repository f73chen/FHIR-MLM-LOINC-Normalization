"""The between-laboratory variability register (thesis Chapter 4 and Appendix B). It holds every
literature magnitude the perturbation uses, with its provenance.

The magnitudes are imported from here by pipeline/build_perturbation_spec.py, by analysis/a21, a25,
a27 and a30, and by analysis/appendix/appendix_b.py.

Each record carries:
  quantity      value_cv   = between-laboratory CV of the measured value
                range_pctV = spread of the stated reference limit between labs
  tier          observed       a measured between-laboratory CV
                specification  a performance specification (what labs should achieve)
                regulatory     an allowable-total-error ceiling (not a CV)
  verification  image          read from a rendered page image by eye
                pdf_text       read from the PDF text layer
                derived        computed from figures in the paper (not quoted)
                abstract       from the abstract or indexing record; full text not obtained

Tiers are not interchangeable. `observed` outranks `specification` where both exist for the same
analyte. `specification` still outranks a class default, because it is a number for the right
analyte. `regulatory` is never used as a CV.
"""
from collections import defaultdict

#: source key -> (full citation, what it measures, n laboratories)
SOURCES = {
    "adeli2017_t2": (
        "Adeli K, et al. Clin Biochem 2017;50:925-935, Table 2 (p.933).",
        "CV_BL and %V by instrument manufacturer, Canadian laboratories (CSCC survey)", None),
    "friedberg2007_t5": (
        "Friedberg RC, et al. Arch Pathol Lab Med 2007;131:348-357, Table 5. CAP Q-Probes.",
        "distribution of stated reference limits across US laboratories", 163),
    "ricos2019": (
        "Ricos C, Perich C, Boned B, et al. Biochem Med 2019;29(1):010701, Tables 3-5, "
        "APS_IL column.", "inter-laboratory performance SPECIFICATION (not an observation)", None),
    "paal2021": (
        "Paal M, Habler K, Vogeser M. Biochem Med 2021;31(3):030902, Table 1.",
        "median CV of all RfB (Germany) EQA participants, labelled CVA in the paper; the "
        "authors state it estimates analytical variation IN AN INTER-LABORATORY SETTING",
        810),
    "vu2022_t5": (
        "Vu HQ, et al. Hematol Rep 2022;14(2):73-84, Table 5, 'All methods CV (%)' row, "
        "level 1 (N=78).", "all-method CV across six analyser groups, CBC EQA material", 78),
    "cwiklinska2020": (
        "Cwiklinska A, et al. Biochem Med 2020;30(2):020707.",
        "median inter-laboratory CV, MANUAL urine microscopy method groups", 183),
    "meijer2018_t2": (
        "Meijer P, Kynde K, van den Besselaar AMHP, Van Blerk M, Woods TAL. "
        "Clin Chem Lab Med 2018;56(10):1698-1703, Table 2.",
        "between-laboratory CV of INR, mean over 4 EQA samples, Quick reagent group "
        "(60% of participants, the higher of the two groups)", 605),
    "cunningham2012": (
        "Cunningham MT, Olson JD, Chandler WL, et al. Arch Pathol Lab Med "
        "2012;136(7):789-795.", "all-method CV, CAP 2008 coagulation survey", 3500),
}

#: (analyte, loinc LONG_COMMON_NAME regex, quantity, magnitude, tier, source, locator, verification)
RECORDS = [
    # ---------- observed between-laboratory value CVs ----------
    ("Sodium", r"^Sodium \[", "value_cv", 1.1, "observed", "adeli2017_t2", "All row", "image"),
    ("Calcium", r"^Calcium \[|^Calcium\.ionized", "value_cv", 2.6, "observed", "adeli2017_t2", "All row", "image"),
    ("Creatinine", r"^Creatinine \[", "value_cv", 5.0, "observed", "adeli2017_t2", "All row", "image"),
    ("ALT", r"^Alanine aminotransferase \[", "value_cv", 24.6, "observed", "adeli2017_t2", "All row", "image"),
    ("ALP", r"^Alkaline phosphatase \[", "value_cv", 6.6, "observed", "adeli2017_t2", "All row", "image"),
    ("FT4", r"^Thyroxine \(T4\) free \[", "value_cv", 14.3, "observed", "adeli2017_t2", "All row", "image"),
    ("HbA1c", r"^Hemoglobin A1c", "value_cv", 5.1, "observed", "paal2021", "Table 1, CVA col (n=810 labs)", "pdf_text"),
    ("PSA", r"^Prostate specific Ag \[", "value_cv", 18.8, "observed", "paal2021", "Table 1, CVA col (n=804 labs)", "pdf_text"),
    ("Haemoglobin", r"^Hemoglobin \[Mass/volume\]", "value_cv", 1.5, "observed", "vu2022_t5", "L1", "image"),
    ("Haematocrit", r"^Hematocrit \[", "value_cv", 2.2, "observed", "vu2022_t5", "L1", "image"),
    ("Erythrocytes", r"^Erythrocytes \[#/volume\]", "value_cv", 1.7, "observed", "vu2022_t5", "L1", "image"),
    ("MCV", r"^MCV \[", "value_cv", 1.7, "observed", "vu2022_t5", "L1", "image"),
    ("MCH", r"^MCH \[", "value_cv", 1.8, "observed", "vu2022_t5", "L1", "image"),
    ("MCHC", r"^MCHC \[", "value_cv", 2.2, "observed", "vu2022_t5", "L1", "image"),
    ("RDW", r"^Erythrocyte \[DistWidth\]|^Erythrocyte distribution width", "value_cv", 2.1, "observed", "vu2022_t5", "L1", "image"),
    ("Leukocytes", r"^Leukocytes \[#/volume\] in Blood", "value_cv", 2.8, "observed", "vu2022_t5", "L1", "image"),
    ("Platelets", r"^Platelets \[#/volume\]", "value_cv", 5.2, "observed", "vu2022_t5", "L1", "image"),
    ("Urine WBC", r"^Leukocytes \[#/area\] in Urine sediment", "value_cv", 71.0, "observed", "cwiklinska2020", "Abstract+Results: manual groups 53-71%, max taken", "pdf_text"),
    ("Urine RBC", r"^Erythrocytes \[#/area\] in Urine sediment", "value_cv", 70.0, "observed", "cwiklinska2020", "Abstract+Results: manual groups 55-70%, max taken", "pdf_text"),
    ("INR", r"^INR in ", "value_cv", 8.1, "observed", "meijer2018_t2", "Table 2, Quick mean", "image"),
    ("Fibrinogen", r"^Fibrinogen \[Mass/volume\]", "value_cv", 7.7, "observed", "cunningham2012", "Abstract", "pdf_text"),
    # ---------- specification-tier value CVs (Ricos APS_IL) ----------
    ("Potassium", r"^Potassium \[", "value_cv", 1.5, "specification", "ricos2019", "APS_IL", "image"),
    ("Chloride", r"^Chloride \[", "value_cv", 1.4, "specification", "ricos2019", "APS_IL", "image"),
    ("Magnesium", r"^Magnesium \[", "value_cv", 4.5, "specification", "ricos2019", "APS_IL", "image"),
    ("Glucose", r"^Glucose \[", "value_cv", 5.9, "specification", "ricos2019", "APS_IL", "image"),
    ("Urate", r"^Urate \[", "value_cv", 5.2, "specification", "ricos2019", "APS_IL", "image"),
    ("Total protein", r"^Protein \[Mass/volume\] in (Serum|Ser)", "value_cv", 3.2, "specification", "ricos2019", "APS_IL", "image"),
    ("Bilirubin", r"^Bilirubin", "value_cv", 9.6, "specification", "ricos2019", "APS_IL", "image"),
    ("AST", r"^Aspartate aminotransferase \[", "value_cv", 6.0, "specification", "ricos2019", "APS_IL", "image"),
    ("LDH", r"^Lactate dehydrogenase \[", "value_cv", 6.1, "specification", "ricos2019", "APS_IL", "image"),
    ("GGT", r"^Gamma glutamyl transferase \[", "value_cv", 12.0, "specification", "ricos2019", "APS_IL", "image"),
    ("Amylase", r"^Amylase \[", "value_cv", 12.0, "specification", "ricos2019", "APS_IL", "image"),
    ("CK", r"^Creatine kinase \[", "value_cv", 4.9, "specification", "ricos2019", "APS_IL", "image"),
    # These five also have an observed CV, so the tier rule discards the specification. They
    # are recorded anyway, so that the register is a complete transcription of Ricos Tables 3–5
    # (all 17 APS_IL values) and a reviewer can see the discarded figure. Three are higher than
    # the observation they lose to (creatinine 7.0 vs 5.0, calcium 3.5 vs 2.6, sodium 2.1 vs 1.1).
    # This is where the tier rule (observed wins, for realism) and the max rule (take the
    # largest, for conservatism) disagree. The tier rule governs.
    ("Sodium", r"^Sodium \[", "value_cv", 2.1, "specification", "ricos2019", "APS_IL", "image"),
    ("Calcium", r"^Calcium \[|^Calcium\.ionized", "value_cv", 3.5, "specification", "ricos2019", "APS_IL", "image"),
    ("ALP", r"^Alkaline phosphatase \[", "value_cv", 6.4, "specification", "ricos2019", "APS_IL", "image"),
    ("Creatinine", r"^Creatinine \[", "value_cv", 7.0, "specification", "ricos2019", "APS_IL", "image"),
    ("ALT", r"^Alanine aminotransferase \[", "value_cv", 8.7, "specification", "ricos2019", "APS_IL", "image"),
    # ---------- reference-range %V ----------
    # Adeli reports %V directly. Its %V columns are for a 14-year-old female (table legend).
    ("Sodium", r"^Sodium \[", "range_pctV", 1.1, "observed", "adeli2017_t2", "All, %V URL", "image"),
    ("Calcium", r"^Calcium \[|^Calcium\.ionized", "range_pctV", 3.3, "observed", "adeli2017_t2", "All, %V LRL", "image"),
    ("Creatinine", r"^Creatinine \[", "range_pctV", 15.9, "observed", "adeli2017_t2", "All, %V URL", "image"),
    ("ALT", r"^Alanine aminotransferase \[", "range_pctV", 30.2, "observed", "adeli2017_t2", "All, %V URL", "image"),
    ("ALP", r"^Alkaline phosphatase \[", "range_pctV", 41.9, "observed", "adeli2017_t2", "All, %V URL", "image"),
    ("FT4", r"^Thyroxine \(T4\) free \[", "range_pctV", 16.6, "observed", "adeli2017_t2", "All, %V URL", "image"),
    # Friedberg does not report %V. It is derived as (p90 - p10) / p50 from its Table 5
    # percentile grid. analysis/a27 re-derives all 16 bounds from the grid and asserts these.
    ("Potassium", r"^Potassium \[", "range_pctV", 8.6, "observed", "friedberg2007_t5", "LRL", "derived"),
    ("Calcium", r"^Calcium \[|^Calcium\.ionized", "range_pctV", 5.9, "observed", "friedberg2007_t5", "LRL", "derived"),
    ("Magnesium", r"^Magnesium \[", "range_pctV", 20.8, "observed", "friedberg2007_t5", "URL", "derived"),
    ("TSH", r"^Thyrotropin \[", "range_pctV", 48.6, "observed", "friedberg2007_t5", "LRL", "derived"),
    ("Haemoglobin", r"^Hemoglobin \[Mass/volume\]", "range_pctV", 8.6, "observed", "friedberg2007_t5", "URL male", "derived"),
    ("Platelets", r"^Platelets \[#/volume\]", "range_pctV", 14.1, "observed", "friedberg2007_t5", "LRL", "derived"),
    ("aPTT", r"^aPTT ", "range_pctV", 21.7, "observed", "friedberg2007_t5", "LRL", "derived"),
]

TIER_RANK = {"observed": 2, "specification": 1, "regulatory": 0}


def by_analyte():
    """analyte -> {quantity: [record, ...]}"""
    out = defaultdict(lambda: defaultdict(list))
    for a, rx, q, mag, tier, src, loc, ver in RECORDS:
        out[a][q].append({"regex": rx, "magnitude": mag, "tier": tier,
                          "source": src, "locator": loc, "verification": ver})
    return out


def resolve(analyte, quantity):
    """The magnitude to use, and why.

    The best tier available wins (the tier rule). Within that tier, take the largest
    magnitude (the max rule), because over-perturbing is the safe direction.
    Returns (magnitude, tier, [sources]) or None.
    """
    recs = by_analyte().get(analyte, {}).get(quantity, [])
    if not recs:
        return None
    best = max(TIER_RANK[r["tier"]] for r in recs)
    same = [r for r in recs if TIER_RANK[r["tier"]] == best]
    mag = max(r["magnitude"] for r in same)
    return mag, same[0]["tier"], sorted({r["source"] for r in same if r["magnitude"] == mag})


def regex_for(analyte):
    for a, rx, *_ in RECORDS:
        if a == analyte:
            return rx
    raise KeyError(analyte)


def anchors_for_matching():
    """analyte -> (regex, value_cv or None, value_tier or None, range_pctV or None, source),
    the shape analysis/a25 consumes."""
    out = {}
    for a in sorted({r[0] for r in RECORDS}):
        v = resolve(a, "value_cv")
        g = resolve(a, "range_pctV")
        out[a] = (regex_for(a), v[0] if v else None,
                  {"observed": "obs", "specification": "spec"}.get(v[1]) if v else None,
                  g[0] if g else None,
                  "; ".join(v[2]) if v else ("; ".join(g[2]) if g else ""))
    return out
