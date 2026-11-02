"""Compare candidate partitions of the quantitative codes into groups, on how defensible each is and on
coverage. The groups carry the group magnitude defaults of the perturbation (thesis Chapter 4,
Appendix B).

Hand-written classes are a judgement and are not published anywhere. The partitions differ in how
much of them is hand-made:

  P0  a21's hand-written classes            judgement over LOINC fields
  P1  the LOINC CLASS axis, verbatim        published, no judgement
  P2  LOINC CLASS x PROPERTY group          published, finer (PROP_GROUP below)
  P3  P2, CHEM split by CLIA subspecialty   CLIA's published headings (Routine Chemistry /
                                            Immunology / Endocrinology / Toxicology / Hematology)
  P4  P2, CHEM split by RCPAQAP programme   Endocrine and Tumour Markers vs general chemistry

They are compared on two things:
  coverage   share of observations in a group holding at least one literature anchor
  cohesion   how much the anchors inside a group disagree (max/min). A group whose anchors span an
             order of magnitude is not one class, however it was derived.

PROP_GROUP is also read by pipeline/build_perturbation_spec.py and by Table B.3 in Appendix B.

Reads UPSTREAM_DIR/Loinc.csv, data/labelspace/labelspace.csv, the CLIA and RCPAQAP tables in
data/variability/ and the literature register (pipeline/variability_register.py). Writes
results/analysis/a25/partition_comparison.csv. The output is aggregate only.

  python analysis/a25_partition_comparison.py
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = ROOT / "results" / "analysis" / "a25"

#: PROPERTY codes collapsed into physically meaningful groups. This is a judgement,
#: but a small one that can be checked. It only merges properties that measure the same
#: kind of thing (all concentration units together, all counts together).
PROP_GROUP = {
    "MCnc": "concentration", "SCnc": "concentration", "ACnc": "concentration",
    "Osmol": "concentration", "MSCnc": "concentration",
    "CCnc": "catalytic", "RelCCnc": "catalytic",
    # The group is "cell index". "count" would not fit, because EntMeanVol is MCV (a volume),
    # EntMass is MCH (a mass) and DistWidth is RDW. They are all per-cell quantities
    # reported by the same instrument as the counts, and that makes them one class default.
    # This differs from the `count_poisson` scale class in a30, which is `Naric` only. The
    # scale class asks whether Poisson sqrt(N) applies. This grouping asks which family it is.
    "NCnc": "cell index", "EntMass": "cell index", "EntMCnc": "cell index",
    "EntVol": "cell index", "EntMeanVol": "cell index", "DistWidth": "cell index",
    "Naric": "cell index",
    "NFr": "fraction", "MFr": "fraction", "VFr": "fraction", "CFr": "fraction",
    "SFr": "fraction", "AFr": "fraction", "MRto": "fraction", "Ratio": "fraction",
    "ArVRat": "fraction", "RelMCnc": "fraction", "RelSCnc": "fraction",
    "Time": "time", "RelTime": "time",
    "PPres": "partial pressure", "Pres": "partial pressure",
    "Mass": "mass", "Vol": "volume", "Temp": "temperature", "Visc": "other",
}

#: The anchor set comes from pipeline/variability_register.py, the same source as the
#: canonical CSV. analysis/a27_register_verification.py verifies the register. Run it
#: before trusting anything here.
sys.path.insert(0, str(ROOT / "pipeline"))
import variability_register as vr  # noqa: E402

ANCHORS = vr.anchors_for_matching()


def clia_subspecialty_map():
    """CLIA analyte name -> CLIA subspecialty. P3 matches the names to codes through the
    CLIA analyte-name regexes in a21.TIER1B.

    The CLIA headings (Routine Chemistry / Immunology / Endocrinology / Toxicology /
    Hematology) are a published partition from the regulator that supplies the
    tolerances. Splitting a LOINC class by them therefore adds no judgement.
    """
    sub = {r["analyte"]: r["clia_subspecialty"]
           for r in csv.DictReader(open(ROOT / "data/variability/clia_2026_limits.csv"))}
    return sub


def main():
    import importlib.util
    _s = importlib.util.spec_from_file_location(
        "a21", Path(__file__).resolve().parent / "a21_perturbation_coverage.py")
    a21 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a21)

    occ = {r["loinc"]: int(r["train_occ"])
           for r in csv.DictReader(open(ROOT / "data/labelspace/labelspace.csv"))}
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        qn = {r["LOINC_NUM"]: r for r in csv.DictReader(fh)
              if r["LOINC_NUM"] in occ and r["SCALE_TYP"] == "Qn"}
    TOT = sum(occ[c] for c in qn)

    PARTITIONS = {
        "P0 a21 hand-written": lambda r: a21.classify(r)[0],
        "P1 LOINC CLASS": lambda r: r["CLASS"],
        "P2 LOINC CLASS x PROPERTY-group":
            lambda r: f"{r['CLASS']} / {PROP_GROUP.get(r['PROPERTY'], r['PROPERTY'])}",
        "P3 P2, CHEM split by CLIA subspecialty": lambda r: p3(r),
        "P4 P2, CHEM split by RCPAQAP Endocrine/Tumour-Marker": lambda r: p4(r),
    }

    sub = clia_subspecialty_map()
    # a21.TIER1B leaves out the analytes that TIER1A already covers. On its own it would
    # leave sodium, creatinine and ALT unassigned and split those codes away from their
    # own anchors. So the TIER1A analytes are mapped onto their CLIA row names too.
    T1A_TO_CLIA = {
        "Sodium": "Sodium", "Potassium": "Potassium", "Chloride": "Chloride",
        "Calcium": "Calcium, total", "Magnesium": "Magnesium", "Creatinine": "Creatinine",
        "Glucose": "Glucose (excluding FDA home use)", "Urate": "Uric Acid",
        "Total protein": "Total Protein", "Bilirubin": "Bilirubin, total",
        "ALT": "Alanine aminotransferase (ALT/SGPT)", "AST": "AST",
        "ALP": "Alkaline Phosphatase", "LDH": "Lactate dehydrogenase (LDH)",
        "GGT": "Gamma glutamyl transferase", "Amylase": "Amylase",
        "CK": "Creatine kinase (CK)", "HbA1c": "Hemoglobin A1c",
        "PSA": "Prostate Specific Antigen, total", "FT4": "Free throxine",
        "TSH": "Thyroid stimulating hormone", "Haemoglobin": "Hemoglobin",
        "Platelets": "Platelet count",
    }
    clia_rx = [(nm, rx) for nm, rx in a21.TIER1B.items()]
    clia_rx += [(T1A_TO_CLIA[nm], a21.TIER1A[nm][0])
                for nm in T1A_TO_CLIA if nm in a21.TIER1A]
    unknown = {nm for nm, _ in clia_rx} - set(sub)
    if unknown:
        sys.exit(f"CLIA names not in the subspecialty table: {sorted(unknown)}")

    def p3(r):
        base = f"{r['CLASS']} / {PROP_GROUP.get(r['PROPERTY'], r['PROPERTY'])}"
        if r["CLASS"] != "CHEM":
            return base
        for nm, rx in clia_rx:
            if re.search(rx, r["LONG_COMMON_NAME"], re.I) and sub.get(nm):
                return f"{base} [{sub[nm]}]"
        return f"{base} [unassigned]"

    # ---- P4: publication-backed sub-split of CHEM ----------------------------
    # RCPAQAP runs separate EQA programmes for Endocrine and Tumour Markers, apart
    # from General Serum Chemistry. These are programme memberships, so an analyte
    # can sit in several. For example, cortisol is in both Endocrine and General
    # Serum Chemistry. On a tie, membership of Endocrine or Tumour Markers wins,
    # because that is where RCPAQAP assesses it as a hormone.
    rcpa = {}
    for row in csv.DictReader(open(ROOT / "data/variability/rcpaqap_categories.csv")):
        rcpa.setdefault(row["analyte"].strip().lower(), set()).add(row["category"])
    #: the endocrine and tumour-marker set, as one regex on the start of the LOINC LONG_COMMON_NAME.
    #: P4 uses this regex alone. The `rcpa` memberships read above are not used.
    ENDO_RX = re.compile(
        r"^(Thyrotropin|Thyroxine|Triiodothyronine|Cortisol|Prostate specific|"
        r"Choriogonadotropin|Cancer Ag|Calcitonin|Alpha-1-Fetoprotein|Ferritin|Folate|"
        r"Cobalamin|Vitamin D|Calcidiol|Parathyrin|Aldosterone|Insulin|Testosterone|"
        r"Estradiol|Progesterone|Prolactin|Follitropin|Lutropin|Somatotropin|"
        r"Androstenedione|Dehydroepiandrosterone|Sex hormone|Homocysteine|"
        r"17-Hydroxyprogesterone|Natriuretic peptide|Carcinoembryonic)", re.I)

    def p4(r):
        base = f"{r['CLASS']} / {PROP_GROUP.get(r['PROPERTY'], r['PROPERTY'])}"
        if r["CLASS"] != "CHEM":
            return base
        return base + (" [endocrine/tumour marker]" if ENDO_RX.match(r["LONG_COMMON_NAME"])
                       else " [general chemistry]")

    # which codes carry an anchor
    anch = {}
    for name, (rx, v, tier, rg, src) in ANCHORS.items():
        for c, r in qn.items():
            if re.search(rx, r["LONG_COMMON_NAME"], re.I):
                anch.setdefault(c, (name, v, rg, src, tier))

    summary = []
    for pname, fn in PARTITIONS.items():
        grp_obs, grp_anch = defaultdict(int), defaultdict(list)
        for c, r in qn.items():
            g = fn(r)
            grp_obs[g] += occ[c]
            if c in anch:
                grp_anch[g].append(anch[c])
        cov_v = sum(o for g, o in grp_obs.items()
                    if any(a[1] is not None for a in grp_anch.get(g, [])))
        cov_r = sum(o for g, o in grp_obs.items()
                    if any(a[2] is not None for a in grp_anch.get(g, [])))
        spreads = []
        for g in grp_obs:
            vs = sorted({a[1] for a in grp_anch.get(g, []) if a[1] is not None})
            if len(vs) >= 2:
                spreads.append((max(vs) / min(vs), g, vs))
        summary.append((pname, len(grp_obs), 100*cov_v/TOT, 100*cov_r/TOT, spreads,
                        grp_obs, grp_anch))
        print(f"\n{'='*104}\n{pname}  ({len(grp_obs)} groups)")
        print(f"   value coverage {100*cov_v/TOT:5.1f}%      range coverage {100*cov_r/TOT:5.1f}%")
        worst = sorted(spreads, reverse=True)[:3]
        if worst:
            print("   least cohesive groups (anchor max/min):")
            for s, g, vs in worst:
                print(f"      {s:5.1f}x  {g[:44]:44s} {vs}")
        unc = sorted(((o, g) for g, o in grp_obs.items()
                      if not any(a[1] is not None for a in grp_anch.get(g, []))), reverse=True)
        if unc:
            print("   groups with no value anchor, largest first:")
            for o, g in unc[:6]:
                print(f"      {100*o/TOT:5.2f}%  {g}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "partition_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["partition", "group", "obs", "obs_share_pct", "n_anchors",
                    "anchor_value_cvs", "anchor_spread"])
        for pname, _, _, _, _, grp_obs, grp_anch in summary:
            for g, o in sorted(grp_obs.items(), key=lambda x: -x[1]):
                vs = sorted({a[1] for a in grp_anch.get(g, []) if a[1] is not None})
                w.writerow([pname, g, o, round(100*o/TOT, 4), len(grp_anch.get(g, [])),
                            " ".join(f"{x:g}" for x in vs),
                            round(max(vs)/min(vs), 2) if len(vs) >= 2 else ""])
    print(f"\nwrote {OUT/'partition_comparison.csv'}")


if __name__ == "__main__":
    sys.exit(main())
