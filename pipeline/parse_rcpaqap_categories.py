"""Extract RCPAQAP's published analyte categories from its Chemical Pathology analytical performance
specifications (APS), the "Allowable Limits of Performance" PDF in SOURCE_DOCS_DIR (thesis Appendix B).

The RCPAQAP (Royal College of Pathologists of Australasia Quality Assurance Programs) organises its
allowable-limits tables under category headings: General Serum Chemistry, Endocrine, Tumour Markers,
Blood Gases, Urine Chemistry, Trace Elements, therapeutic-drug groups and others. The partition is
published by a national EQA body that is independent of CLIA. Unlike CLIA, it separates hormones and
tumour markers from general chemistry, which is the split analysis/a25_partition_comparison.py needs.

Writes data/variability/rcpaqap_categories.csv (category, analyte, limit). Needs `pdftotext`.

  python pipeline/parse_rcpaqap_categories.py
"""
import csv, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
PDF = locations.SOURCE_DOCS / "RCPAQAP Allowable Limits of Performance.pdf"
OUT = ROOT / "data/variability/rcpaqap_categories.csv"

#: headings as they appear in the document body (note the source's typo,
#: "Chemsitry", and the "/ Condensed ..." variants).
HEADINGS = [
    "Alcohol / Ammonia", "Anti-Fungals", "Antibiotics", "Bile Acids", "Biogenic Amines",
    "Blood Gases", "BNP", "Body Fluids", "Co-Oximetry", "CRP", "CSF", "Endocrine",
    "Faecal Occult Blood", "General Serum Chemsitry", "General Serum Chemistry",
    "Condensed Serum Chemistry", "General Therapeutic Drugs", "Glycohaemoglobin",
    "IGF-1 / C-Peptide", "Immunosuppressants", "Neonatal Bilirubin", "Plasma Metanephrines",
    "General Serum Chemsitry / Condensed Serum Chemistry",
    "General Serum Chemsitry / Condensed Serum Chemistry (cont.)",
    "General Therapeutic Drugs / Condensed Serum Chemistry",
    "Point-of-care Troponin / Condensed Point-of-care Troponin",
    "Point-of-care Blood Gases / Condensed Point-of-care Blood Gases",
    "Point-of-care Blood Gases / Condensed Point-of-care Blood Gases (cont.)",
    "Porphyrins", "Serum Indices",
    "Special Lipids", "Special Therapeutic Drugs", "Sweat Electrolytes", "Trace Elements",
    "Tumour Markers", "Urine Chemistry", "Vitamins",
]
#: body headings -> the category name we keep. The document's spelling varies
#: ("Chemsitry") and continuation pages repeat the heading with "(cont.)".
CANON = {
    "General Serum Chemsitry / Condensed Serum Chemistry": "General Serum Chemistry",
    "General Serum Chemsitry / Condensed Serum Chemistry (cont.)": "General Serum Chemistry",
    "General Therapeutic Drugs / Condensed Serum Chemistry": "General Therapeutic Drugs",
    "Point-of-care Troponin / Condensed Point-of-care Troponin": "Point-of-care Troponin",
    "Point-of-care Blood Gases / Condensed Point-of-care Blood Gases": "Point-of-care Blood Gases",
    "Point-of-care Blood Gases / Condensed Point-of-care Blood Gases (cont.)": "Point-of-care Blood Gases",
    "General Serum Chemsitry": "General Serum Chemistry",
}

FOOTER = re.compile(r"RCPAQAP|December 2022|^Contents$|^Analyte\b|Lower Limit|^\d+$", re.I)
#: a data row is "<analyte>  <2+ spaces>  <limit ...>" and the limit starts with a sign
LIMIT = re.compile(r"^[+\-]|^\+ or")


def main():
    txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                         capture_output=True, text=True).stdout
    lines = txt.splitlines()

    # A heading only counts as a section start if an "Analyte ... Lower Limit" header
    # follows within a few lines. That skips the table of contents, where every
    # heading appears with no table under it.
    starts = []
    for i, l in enumerate(lines):
        s = l.strip()
        if s in HEADINGS and any("Lower Limit" in lines[j]
                                 for j in range(i + 1, min(i + 6, len(lines)))):
            starts.append((i, s))
    if not starts:
        sys.exit("no section headings found. The PDF layout may have changed.")

    rows = []
    for n, (i, name) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        for l in lines[i + 1:end]:
            s = l.strip()
            if not s or FOOTER.search(s):
                continue
            parts = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]
            if len(parts) < 2 or LIMIT.match(parts[0]):
                continue
            analyte, limit = parts[0], " ".join(parts[1:])
            if not LIMIT.match(limit):
                continue
            canon = CANON.get(name, name)
            rows.append({"category": canon, "analyte": analyte, "limit": limit})

    # de-duplicate (analyte, category). The same analyte can appear in several
    # categories (creatinine in serum, urine and body fluids)
    seen, uniq = set(), []
    for r in rows:
        k = (r["category"], r["analyte"])
        if k not in seen:
            seen.add(k); uniq.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "analyte", "limit"])
        w.writeheader(); w.writerows(uniq)

    from collections import Counter
    c = Counter(r["category"] for r in uniq)
    print(f"{len(uniq)} analyte-category rows across {len(c)} categories\n")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        ex = [r["analyte"] for r in uniq if r["category"] == k][:7]
        print(f"   {k:26s} {v:3d}  {', '.join(ex)}")
    print(f"\nwrote {OUT}")
    for probe in ("Sodium", "Potassium", "PSA", "Cortisol", "Creatinine", "CA125"):
        where = sorted({r["category"] for r in uniq if r["analyte"].lower() == probe.lower()})
        print(f"   {probe:12s} -> {where or '(not found)'}")


if __name__ == "__main__":
    sys.exit(main())
