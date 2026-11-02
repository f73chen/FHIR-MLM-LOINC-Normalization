"""Add the CLIA subspecialty to data/variability/clia_2026_limits.csv, parsed from the CLIA 2026
proficiency-testing PDF in SOURCE_DOCS_DIR (thesis Appendix B).

CLIA organises its proficiency-testing analyte list under subspecialty headings. The script looks
for the five in HEADINGS (Routine Chemistry, Immunology, Endocrinology, Toxicology, Hematology),
each as a table-header line that reads exactly "<heading> CLIA <year>". The partition is
published by the same regulator that supplies the tolerances, so it can back a grouping of codes
without a judgement call.

Rewrites the CSV in place with a `clia_subspecialty` column and prints the count per subspecialty.
Needs `pdftotext`.

  python pipeline/parse_clia_subspecialty.py
"""
import csv, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
PDF = locations.SOURCE_DOCS / "2026 CLIA Requirements for Proficiency Testing.pdf"
CSV = ROOT / "data/variability/clia_2026_limits.csv"
HEADINGS = ["Routine Chemistry", "Immunology", "Endocrinology", "Toxicology", "Hematology"]


def main():
    txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                         capture_output=True, text=True).stdout
    # a subspecialty owns every line after its "<Name> CLIA 2025" table header
    cur, owner = None, {}
    for line in txt.splitlines():
        s = line.strip()
        for h in HEADINGS:
            if re.match(rf"^{re.escape(h)}\s+CLIA\s+20\d\d$", s):
                cur = h
        if cur and s:
            owner.setdefault(cur, []).append(s)

    rows = list(csv.DictReader(open(CSV)))
    assigned, missing, ambiguous = {}, [], []
    for r in rows:
        a = r["analyte"]
        # Compare against the table's first column, split on a run of 2+ spaces.
        # Anchoring on a single \s is not enough: "^Hemoglobin\s" also matches the
        # "Hemoglobin A1c" row and puts haemoglobin in Routine Chemistry.
        hits = [h for h, lines in owner.items()
                if any(re.split(r"\s{2,}", l)[0].strip() == a for l in lines)]
        if len(hits) == 1:
            assigned[a] = hits[0]
        elif len(hits) > 1:
            ambiguous.append((a, hits)); assigned[a] = next(h for h in HEADINGS if h in hits)
        else:
            missing.append(a)
    if ambiguous:
        print(f"ambiguous (exact name under >1 heading): {ambiguous}\n")

    for r in rows:
        r["clia_subspecialty"] = assigned.get(r["analyte"], "")
    fields = [f for f in rows[0] if f != "clia_subspecialty"] + ["clia_subspecialty"]
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    from collections import Counter
    c = Counter(r["clia_subspecialty"] or "(unassigned)" for r in rows)
    print(f"{len(rows)} CLIA analytes:")
    for k, v in c.most_common():
        print(f"   {k:22s} {v:3d}")
    if missing:
        print(f"\nunassigned ({len(missing)}): {missing}")
    print(f"\nrewrote {CSV}")


if __name__ == "__main__":
    sys.exit(main())
