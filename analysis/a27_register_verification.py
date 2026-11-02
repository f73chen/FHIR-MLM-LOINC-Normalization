"""Verify the variability register (pipeline/variability_register.py) end to end, before the
perturbation is built (thesis Chapter 4, Appendix B).

Checks, in order:
  1. every Friedberg %V is re-derived from the Table 5 grid in source_tables.py
     and must match the register. They are the only derived figures in the
     register. All the others are quoted
  1b. the register agrees with the verbatim transcriptions of the source tables
     (data/variability/source_tables.py)
  2. every register regex matches at least one code in the quantitative label space
  3. analytes with more than one magnitude for the same quantity are listed, with
     the magnitude that the max rule keeps
  4. the verification status of every number is reported, which flags any number
     taken only from an abstract
  5. the canonical coverage numbers, per analyte and per class
  6. writes the canonical CSV the build reads, data/variability/canonical_register.csv

Coverage is counted over the training observations that carry a numeric value
(data/variability/numeric_obs_counts.csv). Aggregate output only.

  python analysis/a27_register_verification.py
"""
import csv, importlib.util, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
import variability_register as vr  # noqa: E402
sys.path.insert(0, str(ROOT / "data" / "variability"))
import source_tables as st  # noqa: E402

LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = ROOT / "results" / "analysis" / "a27"
REGISTER_CSV = ROOT / "data" / "variability" / "canonical_register.csv"

#: register analyte -> the FRIEDBERG2007_T5 key(s) it was derived from. Under the max rule the
#: register keeps the larger of the two bounds, so the check takes the max over both bounds.
FRIEDBERG_MAP = {
    # 2823-3 Potassium SerPl-sCnc; 17861-6 Calcium SerPl-mCnc; 2601-3 Magnesium SerPl-sCnc
    "Potassium":   (["2823-3"],), "Calcium": (["17861-6"],), "Magnesium": (["2601-3"],),
    # 3016-3 TSH SerPl-aCnc; 777-3 Platelet # Bld Auto; 3173-2 aPTT Bld
    "TSH":         (["3016-3"],), "Platelets": (["777-3"],), "aPTT": (["3173-2"],),
    "Haemoglobin": (["718-7", "718-7F"],),  # Hgb Bld-mCnc
}


def derive_pctV(limits):
    """(p90 - p10) / p50 as a percentage: how Friedberg's grid becomes a %V."""
    d = dict(zip(st.FRIEDBERG2007_T5_PCTS, limits))
    return 100.0 * (d[90] - d[10]) / d[50]


def main():
    fail = []

    # ---- 1. re-derive Friedberg ------------------------------------------------
    cap = st.FRIEDBERG2007_T5

    print("=" * 78)
    print("1. Friedberg %V re-derived from the Table 5 grid (source_tables.py)")
    print("   rule: (p90 - p10) / p50 x 100, from the table's 10th, 50th and 90th percentiles")
    print("=" * 78)
    print(f"   {'CAP row':30s} {'LRL %V':>8s} {'URL %V':>8s}")
    derived = {}
    for key, (label, unit, n_labs, lows, highs) in cap.items():
        lo, hi = derive_pctV(lows), derive_pctV(highs)
        derived[key] = (lo, hi)
        print(f"   {label[:30]:30s} {lo:8.2f} {hi:8.2f}   (n={n_labs} labs)")

    print("\n   checking the register against those derivations:")
    reg = vr.by_analyte()
    for analyte, (keys,) in FRIEDBERG_MAP.items():
        recs = [r for r in reg[analyte]["range_pctV"] if r["source"] == "friedberg2007_t5"]
        if not recs:
            fail.append(f"{analyte}: no Friedberg range record in the register"); continue
        stored = max(r["magnitude"] for r in recs)
        cands = [v for k in keys for v in derived[k]]
        best = max(cands)
        ok = abs(stored - round(best, 1)) < 0.051
        print(f"   {analyte:13s} register {stored:5.1f}   max derived bound {best:5.2f}   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            fail.append(f"{analyte}: register {stored} vs derived max {best:.2f}")

    # ---- 1b. the register agrees with the verbatim source transcriptions --------
    print("\n" + "=" * 78)
    print("1b. register values vs the verbatim source transcriptions")
    print("    source_tables.py holds each published table as printed; the register holds")
    print("    only what we use. These must agree or the appendix and the analysis have")
    print("    drifted.")
    print("=" * 78)
    vu = {c: v for c, v in zip(st.VU2022_T5_COLUMNS, st.VU2022_T5["Level 1 (N=78)"][0][2])}
    expect = {
        "Erythrocytes": vu["RBC"], "Haemoglobin": vu["Hb"], "Haematocrit": vu["Hct"],
        "MCH": vu["MCH"], "MCHC": vu["MCHC"], "MCV": vu["MCV"], "RDW": vu["RDW-CV"],
        "Leukocytes": vu["WBC"], "Platelets": vu["PLT"],
        "INR": round(sum(r[2] for r in st.MEIJER2018_T2) / 4, 1),
        "Fibrinogen": st.CUNNINGHAM2012["all_method_cv_pct"],
        "Urine WBC": max(r[1] for r in st.CWIKLINSKA2020["manual"]),
        "Urine RBC": max(r[3] for r in st.CWIKLINSKA2020["manual"]),
    }
    for a, u, rng, med, cva, cvi, n in st.PAAL2021_T1:
        if a in ("HbA1c", "PSA"):
            expect[a] = cva
    ok = 0
    for a, want in sorted(expect.items()):
        got = vr.resolve(a, "value_cv")
        if got is None:
            fail.append(f"{a}: in a source table but not in the register"); continue
        if abs(got[0] - want) > 0.051:
            fail.append(f"{a}: register {got[0]} vs source table {want}")
            print(f"   {a:14s} ✗ register {got[0]:5.1f}  source {want:5.1f}")
        else:
            ok += 1
    print(f"   {ok}/{len(expect)} register values match their verbatim transcription")
    print("   (Ricos and Adeli are checked by 1 and 2 instead: their tables are the register /")
    print("    a26.TABLE2 constants, so they cannot drift. Friedberg's grid is checked by 1.)")

    # ---- 2. every regex matches something --------------------------------------
    # The denominator. The perturbation acts on observations that carry a number,
    # whatever their LOINC SCALE_TYP. The two populations differ both ways: 4.9% of 'Qn'
    # observations have no numeric value, and 47.4% of 'SemiQn' ones do (urine test
    # strips, blood pH). So coverage is counted over the observations with a number.
    NUMCSV = ROOT / "data" / "variability" / "numeric_obs_counts.csv"
    if not NUMCSV.exists():
        sys.exit(f"missing {NUMCSV}: run pipeline/build_numeric_obs_counts.py first")
    occ, scale = {}, {}
    for r in csv.DictReader(open(NUMCSV)):
        n = int(r["train_obs_numeric"])
        if n:
            occ[r["loinc"]] = n
            scale[r["loinc"]] = r["scale_typ"]
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        qn = {r["LOINC_NUM"]: r for r in csv.DictReader(fh) if r["LOINC_NUM"] in occ}
    TOT = sum(occ[c] for c in qn)
    n_nonqn = sum(occ[c] for c in qn if qn[c]["SCALE_TYP"] != "Qn")
    print(f"\n   denominator: {TOT:,} training observations carrying a numeric value, "
          f"across {len(qn)} codes")
    print(f"      of which SCALE_TYP is not 'Qn': {n_nonqn:,} "
          f"({100*n_nonqn/TOT:.1f}%), such as urine strips and blood pH")

    print("\n" + "=" * 78)
    print("2. every register regex matches at least one quantitative labelspace code")
    print("=" * 78)
    unmatched = []
    matched_obs = {}
    for a, (rx, v, tier, g, src) in vr.anchors_for_matching().items():
        hits = [c for c, r in qn.items() if re.search(rx, r["LONG_COMMON_NAME"], re.I)]
        matched_obs[a] = sum(occ[c] for c in hits)
        if not hits:
            unmatched.append(a)
    if unmatched:
        print(f"   ✗ {len(unmatched)} analyte(s) match no code: {unmatched}")
        fail.append(f"unmatched register analytes: {unmatched}")
    else:
        print(f"   OK: all {len(matched_obs)} analytes match at least one code")
    print(f"   register reaches {sum(matched_obs.values()):,} observations "
          f"({100*sum(matched_obs.values())/TOT:.1f}% of the quantitative total), "
          "counting each code once per analyte")

    # ---- 3. within-analyte conflicts -------------------------------------------
    print("\n" + "=" * 78)
    print("3. analytes carrying more than one magnitude for the same quantity")
    print("   (the max rule resolves these; listed so the discarded value is visible)")
    print("=" * 78)
    n_conf = 0
    for a, qs in sorted(reg.items()):
        for q, recs in qs.items():
            mags = {(r["magnitude"], r["source"], r["tier"]) for r in recs}
            if len(mags) > 1:
                n_conf += 1
                used = vr.resolve(a, q)
                print(f"   {a:14s} {q:11s} uses {used[0]:5.1f} ({used[1]}, {used[2][0]})")
                for m, s, t in sorted(mags, key=lambda x: -x[0]):
                    mark = "  <-- used" if m == used[0] and t == used[1] else ""
                    print(f"       {m:5.1f}  {t:13s} {s}{mark}")
    if not n_conf:
        print("   none")

    # ---- 4. verification status -------------------------------------------------
    print("\n" + "=" * 78)
    print("4. how each number was verified")
    print("=" * 78)
    from collections import Counter
    c = Counter(r[7] for r in vr.RECORDS)
    for k in ("image", "pdf_text", "derived", "abstract"):
        if c.get(k):
            print(f"   {k:10s} {c[k]:3d} records")
    weak = sorted({(r[0], r[2], r[5]) for r in vr.RECORDS if r[7] == "abstract"})
    if weak:
        print("\n   ✗ resting on an abstract or indexing record (full text not obtained):")
        for a, q, s in weak:
            print(f"      {a:14s} {q:11s} {s}")

    # ---- 5. canonical coverage numbers ------------------------------------------
    _a = importlib.util.spec_from_file_location("a21", HERE / "a21_perturbation_coverage.py")
    a21 = importlib.util.module_from_spec(_a); _a.loader.exec_module(a21)
    #: the RCPAQAP endocrine/tumour-marker boundary that defines P4
    a25_endo_rx = re.compile(
        r"^(Thyrotropin|Thyroxine|Triiodothyronine|Cortisol|Prostate specific|"
        r"Choriogonadotropin|Cancer Ag|Calcitonin|Alpha-1-Fetoprotein|Ferritin|Folate|"
        r"Cobalamin|Vitamin D|Calcidiol|Parathyrin|Aldosterone|Insulin|Testosterone|"
        r"Estradiol|Progesterone|Prolactin|Follitropin|Lutropin|Somatotropin|"
        r"Androstenedione|Dehydroepiandrosterone|Sex hormone|Homocysteine|"
        r"17-Hydroxyprogesterone|Natriuretic peptide|Carcinoembryonic)", re.I)

    lit_v, lit_r, spec_v, clia = set(), set(), set(), set()
    for a, (rx, v, tier, g, src) in vr.anchors_for_matching().items():
        for c, r in qn.items():
            if not re.search(rx, r["LONG_COMMON_NAME"], re.I):
                continue
            if v is not None:
                (lit_v if tier == "obs" else spec_v).add(c)
            if g is not None:
                lit_r.add(c)
    for nm, rx in a21.TIER1B.items():
        for c, r in qn.items():
            if re.search(rx, r["LONG_COMMON_NAME"], re.I):
                clia.add(c)
    f = lambda S: 100 * sum(occ[c] for c in S) / TOT

    print("\n" + "=" * 78)
    print("5. canonical coverage numbers")
    print(f"   denominator: {TOT:,} training observations carrying a numeric value")
    print("=" * 78)
    print("   per-analyte (a code's own analyte is named in a source):")
    print(f"      observed CV                      {f(lit_v):5.1f}%  ({len(lit_v)} codes)")
    print(f"      + Ricos specification            {f(lit_v | spec_v):5.1f}%  cumulative")
    print(f"      + CLIA per-analyte tolerance     {f(lit_v | spec_v | clia):5.1f}%  cumulative")
    print(f"      class default must cover         {100 - f(lit_v | spec_v | clia):5.1f}%")
    print(f"   range %V, per-analyte               {f(lit_r):5.1f}%  ({len(lit_r)} codes)")

    # ---- 5b. class-level coverage on the same numeric denominator ----------------
    _b = importlib.util.spec_from_file_location("a25", HERE / "a25_partition_comparison.py")
    a25 = importlib.util.module_from_spec(_b); _b.loader.exec_module(a25)

    from collections import defaultdict
    print("\n   class-level (the class contains >=1 anchored analyte), a weaker claim than")
    print("   per-analyte coverage:")
    anchored_v = {c for c in qn for a, (rx, v, t, g, src) in vr.anchors_for_matching().items()
                  if v is not None and re.search(rx, qn[c]["LONG_COMMON_NAME"], re.I)}
    anchored_r = {c for c in qn for a, (rx, v, t, g, src) in vr.anchors_for_matching().items()
                  if g is not None and re.search(rx, qn[c]["LONG_COMMON_NAME"], re.I)}
    for pname, fn in (("P0 hand-written", lambda r: a21.classify(r)[0]),
                      ("P1 LOINC CLASS", lambda r: r["CLASS"]),
                      ("P4 CLASSxPROPERTY+RCPAQAP", None)):
        if fn is None:
            groups = {}
            for c, r in qn.items():
                base = f"{r['CLASS']} / {a25.PROP_GROUP.get(r['PROPERTY'], r['PROPERTY'])}"
                if r["CLASS"] == "CHEM":
                    base += (" [endo]" if a25_endo_rx.match(r["LONG_COMMON_NAME"])
                             else " [genchem]")
                groups[c] = base
        else:
            groups = {c: fn(r) for c, r in qn.items()}
        gv, gr, go = defaultdict(bool), defaultdict(bool), defaultdict(int)
        for c in qn:
            g = groups[c]; go[g] += occ[c]
            gv[g] |= c in anchored_v; gr[g] |= c in anchored_r
        cv = sum(o for g, o in go.items() if gv[g]) / TOT * 100
        cr = sum(o for g, o in go.items() if gr[g]) / TOT * 100
        print(f"      {pname:28s} {len(go):3d} groups   value {cv:5.1f}%   range {cr:5.1f}%")

    # ---- 6. emit the canonical CSV ----------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    REGISTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTER_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["analyte", "quantity", "magnitude_pct", "tier", "source_key", "locator",
                    "verification", "used_by_max_rule", "train_obs_reached", "citation"])
        for a, rx, q, mag, tier, src, loc, ver in sorted(vr.RECORDS):
            used = vr.resolve(a, q)
            w.writerow([a, q, mag, tier, src, loc, ver,
                        int(used is not None and mag == used[0] and tier == used[1]),
                        matched_obs.get(a, 0), vr.SOURCES[src][0]])
    print(f"\nwrote {REGISTER_CSV}")

    print("\n" + "=" * 78)
    if fail:
        print(f"✗ {len(fail)} CHECK(S) FAILED")
        for f in fail:
            print(f"   {f}")
        return 1
    print("✓ ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
