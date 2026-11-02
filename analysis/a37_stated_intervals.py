"""Freeze each code's stated reference interval, and calibrate how far a range perturbation may move it
(thesis Chapter 4, Appendix B).

pipeline/build_perturbation_spec.py resolves a range magnitude per code from the literature. It uses a
per-analyte anchor where one exists, otherwise the maximum anchor in the class of the code, otherwise a
global fallback. On a narrow interval the class maximum can be indefensible. Chloride states 96-108 and
would take the magnesium magnitude, 20.8%. That moves the interval 3.5 of its half-widths per SD, so
the chloride ranges at two sites would not overlap.

Capping that needs the width of the interval that each code states. This script freezes it, as a30
freezes the measurement scale. The spec is then resolved once and written down, and it is not
recomputed at build time.

It also calibrates the cap, so that the constant comes from measurements. For each analyte with a
measured %V (the `observed` tier), it pairs that %V with the modal MIMIC interval for the same code
and forms

    shift_ratio = %V / halfwidth_pct        halfwidth_pct = 100 * (hi - lo) / 2 / midpoint

that is, one SD of between-site shift, expressed in half-widths of the interval being shifted. The
ratio varies far less across analytes than %V itself does. The cap is its maximum, 1.16 (aPTT). A code
may be moved as far as the most mobile analyte that anyone measured.

The ratio is meaningful only where the magnitude is a percentage of the midpoint, so the calibration
and the cap are both restricted to magnitude_kind == relative_pct. Urine specific gravity reports a
nominal ratio of 9.7 and an actual interval overlap of 94%, because its excess_over_1 rule perturbs
only the distance from 1.0.

Reads the a15 cache of training-split values (WORK_DIR/analysis_cache/a15_obs_fields) and, once it
exists, data/variability/perturbation_spec.csv. Writes data/variability/stated_intervals.csv (the
modal interval and half-width of each code) and prints the calibration. The output is aggregate only.

  python analysis/a37_stated_intervals.py
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

CACHE = locations.WORK / "analysis_cache/a15_obs_fields"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
SPEC_CSV = ROOT / "data" / "variability" / "perturbation_spec.csv"
OUT = ROOT / "data" / "variability" / "stated_intervals.csv"

#: a modal interval resting on fewer than this many observations is too thin to cap against, so
#: those codes are left out of the file and keep their class default.
MIN_N = 200


def main():
    import pandas as pd
    from fhir_data_utils import build_lab_code_mappings
    a15s = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
    a15 = importlib.util.module_from_spec(a15s); a15s.loader.exec_module(a15)
    _m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        meta = {r["LOINC_NUM"]: r["LONG_COMMON_NAME"] for r in csv.DictReader(fh)}

    df = pd.read_parquet(CACHE / "train.parquet", columns=["class_idx", "rr_low", "rr_high"])
    df = df[df["rr_low"].notna() & df["rr_high"].notna()]

    rows = []
    for ci, g in df.groupby("class_idx"):
        ln = c2l.get(ci)
        if ln is None or len(g) < MIN_N:
            continue
        counts = g.groupby(["rr_low", "rr_high"]).size()
        (lo, hi), n_modal = counts.idxmax(), int(counts.max())
        mid = (lo + hi) / 2.0
        hw = 100.0 * (hi - lo) / 2.0 / mid if (hi > lo and mid > 0) else None
        rows.append({"loinc": ln, "name": meta.get(ln, "?"), "n_with_range": int(len(g)),
                     "n_distinct_intervals": int(len(counts)),
                     "modal_share_pct": round(100.0 * n_modal / len(g), 2),
                     "rr_low": lo, "rr_high": hi,
                     "halfwidth_pct": "" if hw is None else round(hw, 3)})
    rows.sort(key=lambda r: -r["n_with_range"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    W = sum(r["n_with_range"] for r in rows)
    usable = [r for r in rows if r["halfwidth_pct"] != ""]
    print(f"{len(rows)} codes with >= {MIN_N} ranged observations, {W:,} observations")
    print(f"  {len(usable)} have a usable half-width "
          f"({sum(r['n_with_range'] for r in usable)/W*100:.1f}% of observations); the rest state "
          "a degenerate interval (low == high, or a non-positive midpoint)")
    print(f"  modal interval covers {sum(r['modal_share_pct']*r['n_with_range'] for r in rows)/W:.1f}% "
          "of a code's ranged observations on average (a15's 'MIMIC states ~one interval per "
          "code', measured here directly)")

    # -- the calibration that fixes the cap ------------------------------------
    if not SPEC_CSV.exists():               # a first run: build_perturbation_spec.py reads this file to make the spec
        print(f"  no {SPEC_CSV.name} yet: run build_perturbation_spec.py, then this again for the calibration")
        return
    spec = {r["loinc"]: r for r in csv.DictReader(open(SPEC_CSV))}
    hwof = {r["loinc"]: r["halfwidth_pct"] for r in rows}
    cal = {}
    for c, s in spec.items():
        if (s["range_tier"] != "observed" or s["magnitude_kind"] != "relative_pct"
                or not s["range_pctV"] or not hwof.get(c)):
            continue
        pv, hw = float(s["range_pctV"]), float(hwof[c])
        n = next(r["n_with_range"] for r in rows if r["loinc"] == c)
        # one row per analyte. The same %V is attached to every specimen variant of an analyte,
        # so a row per specimen would weight by specimen count.
        key = s["range_source"]
        if key not in cal or n > cal[key][0]:
            cal[key] = (n, c, lo := float(spec[c]["range_pctV"]), hw, pv / hw)

    print(f"\nCALIBRATION: the {len(cal)} analytes with a measured %V, against MIMIC's own interval")
    print(f"  {'analyte':14s} {'obs':>10s} {'halfwidth%':>10s} {'%V':>6s} {'ratio':>6s}")
    for k, (n, c, pv, hw, ra) in sorted(cal.items(), key=lambda kv: kv[1][4]):
        print(f"  {k[:14]:14s} {n:>10,} {hw:9.1f}% {pv:6.1f} {ra:6.2f}")
    q = np.array([v[4] for v in cal.values()])
    p = np.array([v[2] for v in cal.values()])
    print(f"\n  %V alone       max/min {p.max()/p.min():5.1f}x")
    print(f"  %V / halfwidth max/min {q.max()/q.min():5.1f}x   median {np.median(q):.2f}   "
          f"max {q.max():.2f}  <- the cap")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
