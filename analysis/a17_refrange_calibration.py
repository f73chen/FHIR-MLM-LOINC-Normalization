"""
Per-code statistics of MIMIC's lab values against each code's stated reference range (thesis
Appendix B).

The script covers every LOINC code with at least MIN_N observations that carry both a numeric value
and a stated range. For each code it reports the value distribution (quantiles and median) and how
the stated range sits in it. The range measures are the share of values inside the range, the
empirical percentiles at which the stated bounds fall, the width ratio and the centre offset (see
`interval_stats`). Each code is summarised under two weightings: every observation, and each
patient's first observation of the code.

analysis/a22_clia_divisor.py takes each analyte's MIMIC median concentration from the output, to
express the absolute CLIA limits as a percentage.

Reads the a15 cache (all three splits) and UPSTREAM_DIR/Loinc.csv. Writes
results/analysis/a17/a17_T1_per_code.csv with aggregates per code. stdout shows aggregates only.

  python analysis/a17_refrange_calibration.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
a15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a15)

OUT = locations.REPO / "results/analysis/a17"
SPLITS = ("train", "val", "test")
MIN_N = 1000          # per-cell floor for any reported statistic, and for any aggregate written out


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = loinc_names()
    df = load_all()
    print(f"loaded {len(df):,} observations carrying both a value and a stated range "
          f"({df.y.nunique()} LOINC codes, {df.pid.nunique():,} patients) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    bad = int(((df.rr_low > df.rr_high) & df.rr_low.notna() & df.rr_high.notna()).sum())
    one = int((df.rr_low.isna() ^ df.rr_high.isna()).sum())
    print(f"  data-quality canaries: {bad:,} inverted ranges (low>high), {one:,} one-sided ranges")
    fp = first_per_patient(df)
    print(f"first-per-(patient, code): {len(fp):,} observations", flush=True)

    frames = {"per_observation": df, "per_patient_first": fp}
    cdfs = {tag: test_C(d, names, tag, []) for tag, d in frames.items()}
    pd.concat(cdfs.values()).to_csv(OUT / "a17_T1_per_code.csv", index=False)
    print(f"{len(cdfs['per_observation'])} codes with n >= {MIN_N:,} (per observation), "
          f"{len(cdfs['per_patient_first'])} (first per patient)")
    print(f"wrote {OUT / 'a17_T1_per_code.csv'}   [{time.time()-t0:.0f}s total]")


# --------------------------- statistics ---------------------------
def interval_stats(x, lo, hi):
    """The statistics reported for one set of values against one stated range.

    What each number means (they are easy to misread):
      containment      P(lo <= x <= hi). A reference interval is defined as the central 95% of a
                       reference population, so 95% is the target only if this population is that
                       population. In MIMIC it is not, so low containment alone is expected.
      att_pct_lo/hi    the empirical percentile at which each stated bound sits. Ideal (2.5, 97.5).
                       This pair is the most informative. It tells a range that is correctly centred
                       but too narrow (2.5 -> 15 and 97.5 -> 85) apart from one that has the right
                       width but is shifted (2.5 -> 0.1 and 97.5 -> 60).
      width_ratio      empirical 95% width / stated width. Above 1, the data are more dispersed.
      centre_offset    (median - midpoint) / stated width. Signed. It is how far the population
                       centre sits from the middle of the stated range, in stated-range units.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return None
    xs = np.sort(x)
    q = np.quantile(xs, [0.025, 0.25, 0.5, 0.75, 0.975])
    r = {"n": n, "stated_low": lo, "stated_high": hi,
         "p2_5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p97_5": q[4]}
    have = np.isfinite(lo) and np.isfinite(hi)
    r["containment_pct"] = 100.0 * np.mean((xs >= lo) & (xs <= hi)) if have else np.nan
    # np.searchsorted on the sorted values == the empirical CDF, exactly and without interpolation
    r["att_pct_lo"] = 100.0 * np.searchsorted(xs, lo, "left") / n if np.isfinite(lo) else np.nan
    r["att_pct_hi"] = 100.0 * np.searchsorted(xs, hi, "right") / n if np.isfinite(hi) else np.nan
    w = (hi - lo) if have else np.nan
    r["stated_width"] = w
    r["width_ratio"] = (q[4] - q[0]) / w if have and w > 0 else np.nan
    r["centre_offset"] = (q[2] - (lo + hi) / 2) / w if have and w > 0 else np.nan
    return r


# --------------------------- data ---------------------------
def loinc_names(path=locations.UPSTREAM / "Loinc.csv"):
    """loinc -> (display name, system, property).

    COMPONENT is not a usable display name for every code. LOINC records it as the literal string
    "Observation" for MCV (787-2, MCV RBC Auto), osmolality (2692-2, Osmolality SerPl) and urine
    specific gravity (5811-5, Sp Gr Ur Strip). For INR (6301-6, INR PPP) and PT (5902-2,
    Prothrombin time) it is "Coagulation". In these codes the analyte is in the challenge or method
    part of the name. Tables labelled from COMPONENT alone are unreadable in the places a reader most
    wants to check, so LONG_COMMON_NAME comes first and COMPONENT is the fallback."""
    import csv
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            comp = (r.get("COMPONENT") or "").strip()
            long = (r.get("LONG_COMMON_NAME") or "").strip()
            out[r["LOINC_NUM"]] = (long or comp, (r.get("SYSTEM") or "").strip(),
                                   (r.get("PROPERTY") or "").strip())
    return out


def load_all():
    """Every split, restricted to observations that carry both a numeric value and a stated range.
    Returns one frame. Nothing patient-identifying survives."""
    from fhir_data_utils import build_lab_code_mappings
    _, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    arr = np.array([c2l.get(i, f"?{i}") for i in range(1623)], dtype=object)
    frames = []
    for split in SPLITS:
        base = a15.load_cache(split)[["patient_uuid", "obs_ordinal", "class_idx", "item_code",
                                      "val", "val_unit", "rr_low", "rr_high", "rr_low_unit"]]
        base = base.reset_index(drop=True)
        base["split"] = split
        frames.append(base)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["y"] = arr[df.class_idx.to_numpy()]
    keep = df.val.notna() & (df.rr_low.notna() | df.rr_high.notna())
    df = df[keep].reset_index(drop=True)
    # a patient key that is not the uuid, so nothing identifying reaches any downstream frame
    df["pid"] = pd.factorize(df.patient_uuid)[0].astype("int32")
    df = df.drop(columns=["patient_uuid", "class_idx"])
    return df


def first_per_patient(df):
    """One observation per (patient, code): the earliest in the record.

    Per-observation weighting is what the model sees, but it is not a population estimate. A patient
    with 200 potassiums contributes 200 correlated values and pulls the empirical interval toward
    their own physiology. Every statistic is reported under both weightings, and the two are not
    mixed in one table."""
    return df.sort_values(["pid", "y", "obs_ordinal"], kind="mergesort") \
             .drop_duplicates(["pid", "y"], keep="first")


# --------------------------- per-code statistics ---------------------------
def test_C(df, names, tag, rows):
    """Per code, the stated range against the empirical distribution of every value under it.

    A gap between the two is expected and is not evidence on its own, because MIMIC is a hospital
    population and a reference interval describes a healthy one."""
    out = []
    for y, g in df.groupby("y", observed=True, sort=False):
        lo, hi = g.rr_low.to_numpy(), g.rr_high.to_numpy()
        # `constant_range` marks the codes whose stated range does not vary. For the other codes,
        # the median-range statistics describe no range that an observation carried
        u = pd.DataFrame({"lo": lo, "hi": hi}).drop_duplicates()
        const = len(u) == 1
        r = interval_stats(g.val.to_numpy(), np.nanmedian(lo), np.nanmedian(hi))
        if r is None or r["n"] < MIN_N:
            continue
        # For a code with one stated range this equals containment_pct. For a code with several
        # ranges it judges each observation against the range it was given, while containment_pct
        # uses the median range.
        v = g.val.to_numpy()
        ok = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
        r["containment_own_pct"] = (100.0 * np.mean((v[ok] >= lo[ok]) & (v[ok] <= hi[ok]))
                                    if ok.any() else np.nan)
        r["n_two_sided"] = int(ok.sum())
        r["n_inverted_range"] = int((lo[ok] > hi[ok]).sum())      # data-quality canary
        c, sysm, prop = names.get(y, ("", "", ""))
        r.update(loinc=y, component=c, system=sysm, property=prop,
                 n_ranges=len(u), constant_range=const, weighting=tag)
        out.append(r)
    d = pd.DataFrame(out).sort_values("n", ascending=False)
    rows.append(d)
    return d


if __name__ == "__main__":
    main()
