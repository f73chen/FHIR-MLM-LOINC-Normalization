"""Regenerate the structured columns (pct, abs, abs_unit, shape) of
data/variability/clia_2026_limits.csv from its verbatim `clia_limit` text, the transcription of
record of the CLIA 2026 proficiency-testing limits (thesis Appendix B).

The columns are parsed from the text because the hand-filled columns had drifted from it. The text
column is never modified. Rewrites the CSV in place and prints every corrected row.

  python pipeline/parse_clia_limits.py
"""
import csv, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

CSV = locations.REPO / "data/variability/clia_2026_limits.csv"

#: limits expressed on a non-ratio scale (titre doubling, distribution SD, or a
#: qualitative call). These carry no percentage or concentration and are not usable
#: as a perturbation magnitude.
NON_RATIO = re.compile(r"dilution|positive or negative|pos or neg", re.I)

PCT = re.compile(r"±?\s*(\d+(?:\.\d+)?)\s*%")
#: a concentration unit. "%" is left out because a percent is a relative limit.
#: Matching it here would swallow the absolute arm of every "greater of" rule
#: (ALT's 6 U/L, HDL's 6 mg/dL).
ABS = re.compile(r"±?\s*(\d+(?:\.\d+)?)\s*(mg/dL|g/dL|mmol/L|mIU/L|IU/L|U/L|ng/mL|ng/dL|"
                 r"pg/mL|mcg/mL|mcg/dL|mm ?Hg|mEq/L|mg/L|umol/L|uIU/mL)\b", re.I)


def parse(text, analyte=""):
    """Return (pct, abs, abs_unit, shape), with shape in {relative, absolute, hybrid, other}."""
    t = text
    # pH is the only quantity CLIA bounds with a bare unitless number ("TV ± 0.04").
    # It is also a log scale, so the limit is in pH units and must not be read as a %.
    if analyte.strip().lower().endswith("ph"):
        m = re.search(r"±\s*(\d+(?:\.\d+)?)", t)
        if m:
            return None, float(m.group(1)), "pH", "absolute"
    if NON_RATIO.search(t):
        return None, None, None, "other"
    # "TV ± 3SD" is a distributional rule and gives no magnitude. Drop it before parsing,
    # but keep any numeric limit that sits alongside it.
    t = re.sub(r"±\s*\d*\s*SD", " ", t, flags=re.I)
    am = ABS.search(t)
    a, au = (float(am.group(1)), am.group(2)) if am else (None, None)
    pm = PCT.search(t)
    p = float(pm.group(1)) if pm else None
    if p is not None and a is not None:
        return p, a, au, "hybrid"
    if p is not None:
        return p, None, None, "relative"
    if a is not None:
        return None, a, au, "absolute"
    return None, None, None, "other"


def main():
    rows = list(csv.DictReader(open(CSV)))
    changed = []
    for r in rows:
        p, a, au, sh = parse(r["clia_limit"], r["analyte"])
        new = {"pct": f"{p:g}" if p is not None else "",
               "abs": f"{a:g}" if a is not None else "",
               "abs_unit": au or "", "shape": sh}
        def same(k):
            if k in ("pct", "abs"):
                x, y = r[k], new[k]
                return (x == "" and y == "") or (x != "" and y != "" and float(x) == float(y))
            return r[k] == new[k]
        if not all(same(k) for k in new):
            changed.append((r["analyte"], r["clia_limit"],
                            {k: r[k] for k in new}, dict(new)))
            r.update(new)
    print(f"{len(rows)} rows, {len(changed)} corrected\n")
    for a, t, old, new in changed:
        o = f"pct={old['pct'] or '-'} abs={old['abs'] or '-'}{old['abs_unit']} {old['shape']}"
        n = f"pct={new['pct'] or '-'} abs={new['abs'] or '-'}{new['abs_unit']} {new['shape']}"
        print(f"  {a[:34]:34s} | {t[:44]:44s}\n      {o:34s} -> {n}")
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nrewrote {CSV}")
    oth = [r["analyte"] for r in rows if r["shape"] == "other"]
    print(f"shape='other' (titre/qualitative, unusable as a magnitude): {len(oth)} -> {oth}")


if __name__ == "__main__":
    sys.exit(main())
