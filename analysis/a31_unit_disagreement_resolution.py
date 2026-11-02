"""Can the codes whose MIMIC unit disagrees with the unit in the external terminologies be resolved
code-blind? These are the disagreements that analysis/a20 finds. Each is classified as a notation to
merge, a notation to randomise, or not a unit issue (thesis Chapter 4, Appendix A).

The prior question is whether a disagreement needs randomising. If it is only notation,
expanding the unit-type definition resolves it and there is nothing to randomise. Each disagreeing
code is classified as:

  MERGE           same quantity, and the equivalence holds for every analyte, so it can
                  be applied from the unit string alone. Safe to canonicalise.
  RANDOMISE       same quantity here, but the equivalence depends on which analyte it
                  is. A merge rule would need the code, which is what is being
                  predicted, so it cannot be used at inference. Both strings are
                  presented at training, drawn per site, with the value unchanged.
  NOT_A_UNIT_ISSUE  MIMIC and the terminology disagree about what is being measured.
                  This is a code-mapping question.
  (There is no NOT_CONVERTIBLE class. By the LOINC rules a mass-vs-molar disagreement
   is a different code, so it goes in NOT_A_UNIT_ISSUE with the others.)

The MERGE/RANDOMISE split depends on one question. Can the rule be applied without
knowing the code? Preprocessing must be code-blind because at inference the code is the
label. A rule keyed on the valence of an analyte, or on its LOINC PROPERTY, is circular
there, because it needs the answer to prepare the input.

Reads results/analysis/a20/a20_external_attestation.csv and UPSTREAM_DIR/Loinc.csv. Writes
results/analysis/a31/disagreement_resolution.csv, which pipeline/build_unit_randomisation.py
and Table A.2 in Appendix A read. The output is aggregate only.

  python analysis/a31_unit_disagreement_resolution.py
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
ATT = ROOT / "results" / "analysis" / "a20" / "a20_external_attestation.csv"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = ROOT / "results" / "analysis" / "a31"

#: an equivalent (eq) is one mole of charge, so eq == mol only for a monovalent species.
#: The merge is therefore not safe for every analyte. The note records, for each code, the
#: valence of its component.
MONOVALENT = {"Potassium", "Sodium", "Chloride", "Bicarbonate", "Anion gap",
              "Base excess", "Carbon dioxide", "Lithium", "Ammonium"}
DIVALENT = {"Calcium", "Magnesium", "Sulfate", "Phosphate"}


def verdict(mimic, external, component, prop):
    """Return (class, rule, note)."""
    m, e = (mimic or "").strip(), (external or "").strip()

    if m == "meq/L" and e == "mmol/L":
        # An equivalent is one mole of charge. eq == mol for a monovalent species and
        # 0.5 mol for a divalent one, so the factor depends on the analyte. Every such
        # code in this label space happens to be monovalent, but a converter would still
        # have to know that, and at inference it does not know the code.
        val = ("monovalent" if component in MONOVALENT else
               "divalent" if component in DIVALENT else "unknown valence")
        return ("RANDOMISE", "eq vs mol: the factor depends on valence",
                f"{component} is {val}; factor 1 here, but not derivable from the unit")

    if {m, e} <= {"[IU]/L", "[arb'U]/L", "[IU]/mL", "[arb'U]/mL", "U/L", "U/mL"} and m != e:
        return ("MERGE", "IU == U == arb'U, an assay-defined activity unit",
                "holds for every analyte; no scale factor, no analyte knowledge needed")

    if prop == "LsCnc" and e == "[pH]":
        # MIMIC's raw string here is the literal word "units", which it also uses for
        # ACnc and PrThr codes. So mapping "units" to [pH] cannot be decided from the string.
        return ("RANDOMISE", 'MIMIC\'s "units" is ambiguous across properties',
                "the same raw string means pH on 4 codes and arbitrary units on 3")

    if m == "[arb'U]" and e.startswith("{") and e.endswith("}"):
        return ("MERGE", "a UCUM annotation is not meaning-bearing",
                f"{e} is an annotation; UCUM says drop it, no analyte knowledge needed")

    if m == "%" and ("10*3/uL" in e or "10*9/L" in e):
        return ("NOT_A_UNIT_ISSUE", "different QUANTITY, not different units",
                "MIMIC reports a fraction; the code is an absolute count")

    if m == "[arb'U]" and e.endswith("/mL"):
        return ("NOT_A_UNIT_ISSUE", "MIMIC has no volume denominator",
                "a bare arbitrary unit vs a concentration")

    # LOINC separates mass concentration (MCnc) from substance concentration (SCnc)
    # into different codes. That is what classify_pair in a20 means by "mass vs molar".
    # So when MIMIC reports mass and the terminology says molar, the unit does not need
    # converting. The observation is mapped to the wrong code, and the right code almost
    # always exists. Converting would change what is being measured.
    return ("NOT_A_UNIT_ISSUE", "mass vs molar is a different LOINC code by definition",
            f"MIMIC reports {m} against a {prop} code; the mass-concentration code exists "
            "separately")


def main():
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        meta = {r["LOINC_NUM"]: r for r in csv.DictReader(fh)}
    rows = [r for r in csv.DictReader(open(ATT)) if r["verdict"] == "kind_differs"]
    TOT_TESTABLE = sum(int(r["n_obs"]) for r in csv.DictReader(open(ATT))
                       if r["verdict"] in ("kind_agrees", "kind_differs"))

    out, agg = [], defaultdict(lambda: [0, 0])
    for r in rows:
        md = meta.get(r["loinc"], {})
        comp, prop = md.get("COMPONENT", ""), md.get("PROPERTY", "")
        cls, rule, note = verdict(r["mimic_unit"], r["external_unit"], comp, prop)
        n = int(r["n_obs"])
        agg[cls][0] += 1; agg[cls][1] += n
        out.append({"loinc": r["loinc"], "component": comp, "property": prop,
                    "n_obs": n, "mimic_unit": r["mimic_unit"],
                    "external_unit": r["external_unit"],
                    "a20_class": r["disagreement_class"],
                    "resolution": cls, "rule": rule, "note": note})
    out.sort(key=lambda x: (x["resolution"], -x["n_obs"]))

    print(f"{len(rows)} disagreeing codes, {sum(int(r['n_obs']) for r in rows):,} observations "
          f"({100*sum(int(r['n_obs']) for r in rows)/TOT_TESTABLE:.1f}% of the testable split)\n")
    print(f"{'resolution':18s} {'codes':>5s} {'observations':>13s} {'% of testable':>14s}")
    print("-" * 56)
    for k in ("MERGE", "RANDOMISE", "NOT_A_UNIT_ISSUE"):
        if k in agg:
            c, n = agg[k]
            print(f"{k:18s} {c:>5d} {n:>13,} {100*n/TOT_TESTABLE:13.2f}%")

    for k in ("RANDOMISE", "MERGE", "NOT_A_UNIT_ISSUE"):
        sel = [r for r in out if r["resolution"] == k]
        if not sel:
            continue
        print(f"\n{'='*104}\n{k}: {len(sel)} codes, {sum(r['n_obs'] for r in sel):,} obs\n{'='*104}")
        byrule = defaultdict(list)
        for r in sel:
            byrule[r["rule"]].append(r)
        for rule, rs in sorted(byrule.items(), key=lambda x: -sum(r["n_obs"] for r in x[1])):
            print(f"\n  RULE: {rule}   ({len(rs)} codes, {sum(r['n_obs'] for r in rs):,} obs)")
            for r in rs[:8]:
                print(f"     {r['loinc']:9s} {r['n_obs']:>8,}  {r['mimic_unit']:10s} vs "
                      f"{r['external_unit'][:16]:16s} {r['component'][:26]}")
            if len(rs) > 8:
                print(f"     ... and {len(rs)-8} more")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "disagreement_resolution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nwrote {OUT/'disagreement_resolution.csv'}")


if __name__ == "__main__":
    sys.exit(main())
