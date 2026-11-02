"""How often the final perturbation changes FHIR-MLM's percentile bin (thesis Chapters 4 and 7).

The sample is the training observations in the a15 cache, at most 4,000 per code (random_state 0).
Each value and range bound is perturbed the way the datasets were built:
  * perturbation.perturb_number with perturbation_spec.csv gives the final magnitude, kind, guards and
    re-rounding of each code, including the class defaults, the range cap and the scale rules.
  * perturbation.site_z gives the same draws as the build, with the same salt. There is one draw per
    (patient, code) for the value and one shared by both range bounds.
Two things are left out because they need the whole patient record. One is the 3 derived codes, which are
recomputed from their perturbed parents and are not perturbed themselves. The other is the renormalisation
of the 5 differential codes, which are counted with their plain perturbation. Tier-0 codes are never
perturbed. The headline is observation-weighted over the ordinary scales (ratio, integer grid, bounded
fraction).

The script also reports, per code and over all its training observations, how often the perturbed value
still differs after re-rounding. Chapter 4 quotes this for serum creatinine.

Writes aggregates only, to results/analysis/a32/bin_change_final.{csv,json}.

  python analysis/a32_bin_change_final.py
"""
import csv, importlib.util, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
import perturbation as pert  # noqa: E402

CACHE = locations.WORK / "analysis_cache/a15_obs_fields"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
OUT = ROOT / "results" / "analysis" / "a32"
SAMPLE_PER_CODE = 4000          # at most this many observations per code
ORDINARY = ("ratio", "integer_grid", "bounded_frac")


def main():
    import pyarrow.parquet as pq
    from fhir_data_utils import build_lab_code_mappings
    _s = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
    a15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a15)
    _m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    numeric_bins = a15.load_bins()
    spec = pert.load_spec()
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        names = {r["LOINC_NUM"]: r["LONG_COMMON_NAME"] for r in csv.DictReader(fh)}

    df = pq.read_table(CACHE / "train.parquet",
                       columns=["patient_uuid", "class_idx", "item_code", "val", "rr_low", "rr_high"]).to_pandas()
    df = df[df["val"].notna()]

    rows, skipped = [], {"no_spec": 0, "tier0": 0, "derived": 0}
    for ci, g in df.groupby("class_idx"):
        ln = c2l.get(ci)
        if ln is None:
            continue
        item = g["item_code"].iloc[0]
        if not numeric_bins.get(item):
            continue                      # no bin table, so FHIR-MLM emits no number token
        s = spec.get(ln)
        if s is None:
            skipped["no_spec"] += 1; continue
        if s["tier0"]:
            skipped["tier0"] += 1; continue
        gg = g.sample(min(len(g), SAMPLE_PER_CODE), random_state=0) if len(g) > SAMPLE_PER_CODE else g
        nv = nrv = nr = nrr = 0
        n_all = nvc = 0                                     # changed after re-rounding, over every observation
        if s["value_cv"] is not None and not s["derived_role"]:
            for u, v in zip(g["patient_uuid"], g["val"]):
                n_all += 1
                nvc += (pert.perturb_number(v, pert.site_z(u, ln, "value"), s, s["value_cv"]) != v)
        for u, v, lo, hi in zip(gg["patient_uuid"], gg["val"], gg["rr_low"], gg["rr_high"]):
            b0 = a15.per_analyte_percentile(v, item, numeric_bins)
            if b0 is not None and s["value_cv"] is not None and not s["derived_role"]:
                vp = pert.perturb_number(v, pert.site_z(u, ln, "value"), s, s["value_cv"])
                nv += 1
                nrv += (a15.per_analyte_percentile(vp, item, numeric_bins) != b0)
            if s["range_pctV"] is None:
                continue
            z = pert.site_z(u, ln, "range")                # one draw for both bounds, as the build does
            for b in (lo, hi):
                if b is None or b != b:
                    continue
                c0 = a15.per_analyte_percentile(b, item, numeric_bins)
                if c0 is None:
                    continue
                bp = pert.perturb_number(b, z, s, s["range_pctV"])
                nr += 1
                nrr += (a15.per_analyte_percentile(bp, item, numeric_bins) != c0)
        if s["derived_role"]:
            skipped["derived"] += 1
        if nv or nr:
            rows.append({"loinc": ln, "name": names.get(ln, ""), "scale": s["scale"], "kind": s["kind"],
                         "n_obs_total": len(g), "value_cv_pct": s["value_cv"], "range_pctV": s["range_pctV"],
                         "n_bins": len(numeric_bins[item]),
                         "value_change_pct": round(100 * nvc / n_all, 2) if n_all else "",
                         "value_bin_change_pct": round(100 * nrv / nv, 2) if nv else "",
                         "range_bin_change_pct": round(100 * nrr / nr, 2) if nr else ""})
    rows.sort(key=lambda r: -r["n_obs_total"])

    ordinary = [r for r in rows if r["scale"] in ORDINARY]
    rv = [r for r in ordinary if r["value_bin_change_pct"] != ""]
    rr = [r for r in ordinary if r["range_bin_change_pct"] != ""]
    W, WR = sum(r["n_obs_total"] for r in rv), sum(r["n_obs_total"] for r in rr)
    wv = sum(r["value_bin_change_pct"] * r["n_obs_total"] for r in rv) / W
    wr = sum(r["range_bin_change_pct"] * r["n_obs_total"] for r in rr) / WR
    head = {"value_bin_change_pct": round(wv, 1), "range_bin_change_pct": round(wr, 1),
            "codes_value": len(rv), "obs_value": W, "codes_range": len(rr), "obs_range": WR,
            "codes_with_bins_and_spec": len(rows), "skipped_codes": skipped,
            "weighting": "observation-weighted over the ordinary scales (ratio, integer_grid, bounded_frac)",
            "sample": f"training split, at most {SAMPLE_PER_CODE} observations per code (a32's)"}
    print(f"{len(rows)} codes with a bin table and a spec row; skipped {skipped}")
    print("observation-weighted, ordinary scales:")
    print(f"   VALUE  bin changes on {wv:5.1f}% of observations ({len(rv)} codes, {W:,} obs)")
    print(f"   RANGE  bin changes on {wr:5.1f}% of bounds ({len(rr)} codes, {WR:,} obs)")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bin_change_final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "bin_change_final.json").write_text(json.dumps(head, indent=2) + "\n")
    print(f"wrote {OUT / 'bin_change_final.csv'} and .json")


if __name__ == "__main__":
    sys.exit(main())
