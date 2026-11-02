"""Count, per LOINC code, the training observations that carry a numeric value (thesis Chapter 4 and
Appendix B).

The perturbation acts on observations with a numeric valueQuantity, whatever their LOINC SCALE_TYP.
The two populations differ in both directions. Some SCALE_TYP='Qn' observations have no numeric value,
and many 'SemiQn' observations have one (urine test strips, blood pH). Coverage percentages are
therefore weighted by this count. The train_occ column of the label space would also count
observations that carry no number.

Reads the training-split observation cache of analysis/a15_field_identifiability.py
(WORK_DIR/analysis_cache/a15_obs_fields) and UPSTREAM_DIR/Loinc.csv. Writes
data/variability/numeric_obs_counts.csv, aggregate counts only.

  python pipeline/build_numeric_obs_counts.py
"""
import csv, importlib.util, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
OUT = ROOT / "data" / "variability" / "numeric_obs_counts.csv"
CACHE = locations.WORK / "analysis_cache/a15_obs_fields"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"



def main():
    import pyarrow.parquet as pq
    from fhir_data_utils import build_lab_code_mappings
    _s = importlib.util.spec_from_file_location(
        "a15", ROOT / "analysis" / "a15_field_identifiability.py")
    a15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a15)
    m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)

    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        meta = {r["LOINC_NUM"]: r for r in csv.DictReader(fh)}

    df = pq.read_table(CACHE / "train.parquet",
                       columns=["class_idx", "val", "rr_low", "rr_high"]).to_pandas()
    # Several MIMIC class_idx map to the same LOINC code (three creatinine items, for
    # example). Counts are summed by LOINC, because a per-code lookup would keep only the
    # last item and drop the observations of the others.
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0])   # loinc -> [obs, numeric, with_refrange, n_items]
    for ci, g in df.groupby("class_idx"):
        ln = c2l.get(ci)
        if ln is None:
            continue
        # an observation has a reference range if either bound is stated: a
        # one-sided range (upper limit only) is still a range to perturb
        has_rr = g["rr_low"].notna() | g["rr_high"].notna()
        a = agg[ln]
        a[0] += len(g); a[1] += int(g["val"].notna().sum())
        a[2] += int((g["val"].notna() & has_rr).sum()); a[3] += 1
    rows = []
    for ln, (o, n, rr, k) in agg.items():
        r = meta.get(ln, {})
        rows.append({"loinc": ln, "scale_typ": r.get("SCALE_TYP", ""),
                     "long_common_name": r.get("LONG_COMMON_NAME", ""),
                     "n_mimic_items": k, "train_obs": o, "train_obs_numeric": n,
                     "train_obs_numeric_with_refrange": rr})
    rows.sort(key=lambda x: -x["train_obs_numeric"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    tot = sum(r["train_obs"] for r in rows)
    num = sum(r["train_obs_numeric"] for r in rows)
    qn = sum(r["train_obs_numeric"] for r in rows if r["scale_typ"] == "Qn")
    print(f"{len(rows)} codes")
    print(f"   all train observations            {tot:>12,}")
    print(f"   carrying a numeric value          {num:>12,}  = {100*num/tot:.1f}%")
    print(f"      of which SCALE_TYP='Qn'        {qn:>12,}  = {100*qn/num:.1f}% of those")
    print(f"      of which not 'Qn'              {num-qn:>12,}  = {100*(num-qn)/num:.1f}%")
    rr = sum(r["train_obs_numeric_with_refrange"] for r in rows)
    print(f"   ...and carrying a reference range {rr:>12,}  = {100*rr/num:.1f}% of those")
    print("      (the population the range perturbation can act on)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
