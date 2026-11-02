"""Classify every perturbable code by measurement scale, for values and reference ranges alike, and
record the evidence that puts each one in its class (thesis Chapter 4, Appendix B).

A percentage is the wrong kind of perturbation for some quantities. pH is on a log scale, and adding
16% to a pH of 7.4 gives 8.6, which is not a survivable blood pH. Adding 8% to a urine protein strip
reading of 30 mg/dL gives 32.4, which no strip can report. Adding 10% to a microscopy count of 2
cells/HPF rounds back to 2.

Detection is evidence-based, in this precedence order:

  log            a log-scale LOINC PROPERTY: LsCnc (pH, "negative log of substance
                 concentration"), LnCnc / LaCnc (log viral loads) or LogRto (log
                 ratio). This is a published LOINC axis, so no judgement is involved.
  ratio_near_1   LOINC PROPERTY = SpGrav. Values sit at 1.00-1.05, so a percentage
                 of the value bears no relation to the physiological range.
  ladder         <= LADDER_MAX distinct values covering >= LADDER_SHARE of observations.
                 Reporting ladders: test strips, titres.
  count_poisson  LOINC PROPERTY = Naric, "number per area": a literal count of
                 objects seen in a microscope field. The reported number is the
                 number counted, so Poisson sqrt(N) applies and the result must
                 re-round to an integer.
  integer_grid   >= 99% of values are integers but the number is not a raw count
                 (a concentration in scaled units, or a quantity reported to 0 d.p.).
                 Poisson does not apply. Take the magnitude from the normal tier and
                 re-round to an integer afterwards.
  bounded_frac   a fraction of a whole reported in %, PROPERTY in NFr/MFr/VFr/CFr/SFr/AFr.
                 Relative perturbation is fine but must clamp to [0, 100].
  ratio          everything else: ordinary relative perturbation is meaningful.

Reads the training-split values from the analysis/a15 cache (WORK_DIR/analysis_cache/a15_obs_fields),
UPSTREAM_DIR/Loinc.csv and data/variability/numeric_obs_counts.csv. Writes
data/variability/measurement_scales.csv, which pipeline/build_perturbation_spec.py reads. It has one
row per code with at least MIN_OBS numeric observations. Each row gives the scale, the evidence, the
native precision and, for small value sets, every reported value. Prints aggregate counts and a
rounding check.

  python analysis/a30_measurement_scales.py
"""
import csv, importlib.util, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = Path(__file__).resolve().parent

CACHE = locations.WORK / "analysis_cache/a15_obs_fields"
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
NUMCSV = ROOT / "data" / "variability" / "numeric_obs_counts.csv"
OUT = ROOT / "results" / "analysis" / "a30"
SCALES_CSV = ROOT / "data" / "variability" / "measurement_scales.csv"

LOG_PROPS = {"LsCnc", "LnCnc", "LaCnc", "LogRto"}
NEAR1_PROPS = {"SpGrav"}
FRAC_PROPS = {"NFr", "MFr", "VFr", "CFr", "SFr", "AFr"}
#: "number per area", a count of objects in a microscope field. The reported number
#: is the number counted, so Poisson applies.
POISSON_PROPS = {"Naric"}
#: Not Poisson, although also integer-valued. NCnc is number per volume, i.e. a
#: concentration in scaled units. A platelet count of "218" means 218 x10^3/uL, from
#: an instrument that counted far more than 218 particles, so sqrt(218) does not
#: describe its counting error. EntMeanVol (MCV, fL) is a volume reported to 0 d.p.
#: and is not a count.
LADDER_MAX, LADDER_SHARE = 20, 95.0
MIN_OBS = 200                      # below this the value distribution says little


def classify(prop, unit, n_distinct, top12_share, pct_integer, vmin, vmax):
    """Return (scale, evidence). The checks run in precedence order, so a log scale
    stays log even if it is also reported on a coarse grid, as urine pH is."""
    if prop in LOG_PROPS:
        return "log", f"LOINC PROPERTY={prop} (log family)"
    if prop in NEAR1_PROPS:
        return "ratio_near_1", f"LOINC PROPERTY={prop}"
    if n_distinct <= LADDER_MAX and top12_share >= LADDER_SHARE:
        return "ladder", f"{n_distinct} distinct values, top 12 cover {top12_share:.1f}%"
    if prop in POISSON_PROPS:
        return "count_poisson", f"PROPERTY={prop} (number per AREA = per field)"
    if pct_integer >= 99.0:
        return "integer_grid", f"{pct_integer:.1f}% integer, PROPERTY={prop} (not a raw count)"
    if prop in FRAC_PROPS and unit.strip() in ("%", "% of total", ""):
        return "bounded_frac", f"PROPERTY={prop}, unit={unit!r}, range {vmin:g}-{vmax:g}"
    return "ratio", f"PROPERTY={prop}"


def main():
    import pyarrow.parquet as pq
    from fhir_data_utils import build_lab_code_mappings
    _s = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
    a15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a15)
    m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        meta = {r["LOINC_NUM"]: r for r in csv.DictReader(fh)}
    rr = {r["loinc"]: int(r["train_obs_numeric_with_refrange"])
          for r in csv.DictReader(open(NUMCSV))}

    df = pq.read_table(CACHE / "train.parquet",
                       columns=["class_idx", "val", "val_unit"]).to_pandas()
    df = df[df["val"].notna()]

    def decimals(x):
        t = f"{x!r}"
        if "e" in t or "E" in t:
            return 6
        return len(t.split(".")[1].rstrip("0")) if "." in t else 0

    vals = defaultdict(Counter)
    units = defaultdict(Counter)
    decs = defaultdict(Counter)
    for ci, g in df.groupby("class_idx"):
        ln = c2l.get(ci)
        if ln is None:
            continue
        v = g["val"].tolist()
        vals[ln].update(v)
        decs[ln].update(decimals(x) for x in v)
        units[ln].update(g["val_unit"].dropna().tolist())

    rows = []
    for ln, c in vals.items():
        n = sum(c.values())
        if n < MIN_OBS:
            continue
        r = meta.get(ln, {})
        mc = c.most_common(12)
        top12 = 100.0 * sum(v for _k, v in mc) / n
        integral = 100.0 * sum(v for k, v in c.items() if float(k).is_integer()) / n
        unit = units[ln].most_common(1)[0][0] if units[ln] else ""
        # native reporting precision: the number of decimal places the site uses. A
        # perturbation smaller than half a step rounds away and leaves the value
        # unchanged.
        nd = max(decs[ln], key=lambda d: (decs[ln][d], -d))
        step = 10.0 ** (-nd)
        med = sorted(c.elements())[n // 2] if n < 2_000_000 else None
        if med is None:
            cum, med = 0, None
            for k in sorted(c):
                cum += c[k]
                if cum >= n / 2:
                    med = k; break
        scale, why = classify(r.get("PROPERTY", ""), unit, len(c), top12, integral,
                              min(c), max(c))
        rows.append({"loinc": ln, "name": r.get("LONG_COMMON_NAME", ""),
                     "property": r.get("PROPERTY", ""), "unit": unit,
                     "scale": scale, "evidence": why, "n_obs": n,
                     "n_obs_with_refrange": rr.get(ln, 0),
                     "n_distinct": len(c), "top12_share_pct": round(top12, 2),
                     "pct_integer": round(integral, 2),
                     "min": min(c), "max": max(c), "median": med,
                     "native_decimals": nd, "reporting_step": step,
                     "min_cv_pct_to_survive_rounding":
                         round(100.0 * (step / 2) / med, 3) if med else "",
                     "example_values": " ".join(f"{k:g}" for k, _v in sorted(mc)[:8]),
                     # The one-step rule (perturbation.py) needs the full rung list, because
                     # "+-1 rung" is undefined without it. The list is written for any code
                     # with few enough distinct values, including codes not classed "ladder".
                     # Urine pH is classed `log` (PROPERTY precedence) but is still a 9-rung
                     # grid, and a +-0.014 absolute offset on it would round to zero.
                     "ladder_values": (" ".join(f"{k:g}" for k in sorted(c))
                                       if len(c) <= 200 else "")})
    rows.sort(key=lambda r: -r["n_obs"])
    TOT = sum(r["n_obs"] for r in rows)
    TOT_RR = sum(r["n_obs_with_refrange"] for r in rows)

    print(f"{len(rows)} codes with >= {MIN_OBS} numeric observations, "
          f"{TOT:,} observations\n")
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a = agg[r["scale"]]
        a[0] += 1; a[1] += r["n_obs"]; a[2] += r["n_obs_with_refrange"]
    print(f"{'scale':14s} {'codes':>5s} {'observations':>14s} {'share':>7s} "
          f"{'w/ ref range':>13s} {'share':>7s}")
    print("-" * 70)
    for k in ("ratio", "bounded_frac", "integer_grid", "count_poisson", "ladder",
              "log", "ratio_near_1"):
        if k not in agg:
            continue
        n, o, w = agg[k]
        print(f"{k:14s} {n:>5d} {o:>14,} {100*o/TOT:6.2f}% {w:>13,} {100*w/TOT_RR:6.2f}%")

    for k in ("log", "ratio_near_1", "ladder", "count_poisson", "integer_grid"):
        sel = [r for r in rows if r["scale"] == k]
        if not sel:
            continue
        hdr = ("a relative (%) perturbation is wrong for these"
               if k != "integer_grid" else
               "relative % is fine, but the result must be re-rounded to an integer")
        print(f"\n{'='*96}\n{k.upper()}: {hdr}\n{'='*96}")
        print(f"   {'obs':>10s} {'w/ range':>9s} {'distinct':>8s}  {'evidence':44s} code")
        for r in sel[:14]:
            print(f"   {r['n_obs']:>10,} {r['n_obs_with_refrange']:>9,} "
                  f"{r['n_distinct']:>8d}  {r['evidence'][:44]:44s} {r['name'][:44]}")
        if len(sel) > 14:
            print(f"   ... and {len(sel)-14} more (see the CSV)")

    # ---- saturation check: will a perturbation of the register magnitude survive? ----
    sys.path.insert(0, str(ROOT / "pipeline"))
    import variability_register as vr
    import re as _re
    print(f"\n{'='*96}")
    print("SATURATION CHECK: is the perturbation bigger than half a reporting step?")
    print("  a perturbation smaller than step/2 rounds straight back to the original value")
    print("  The threshold shown is the CV% needed to move the median value by")
    print("  half a step; a code whose magnitude is below it is effectively unperturbed.")
    print("=" * 96)
    anchored = {}
    for a in sorted({r[0] for r in vr.RECORDS}):
        got = vr.resolve(a, "value_cv")
        if got:
            anchored[a] = (vr.regex_for(a), got[0])
    risky = []
    for r in rows:
        thr = r["min_cv_pct_to_survive_rounding"]
        if thr == "":
            continue
        mag = next((m for a, (rx, m) in anchored.items()
                    if _re.search(rx, r["name"], _re.I)), None)
        if mag is None:
            mag = 16.2 if False else None      # class default resolved at build time
        if mag is not None and mag < thr:
            risky.append((r, mag, thr))
    if risky:
        # The threshold uses the median, but the perturbation is a random draw and
        # higher-valued observations move further in absolute terms. So the simulation
        # runs over the observed values of each code, to get the fraction that changes
        # after rounding to native precision.
        import random
        random.seed(0)
        print(f"   {'obs':>10s} {'magnitude':>9s} {'thresh':>7s} {'% actually':>10s}  code")
        print(f"   {'':>10s} {'':>9s} {'':>7s} {'changed':>10s}")
        for r, mag, thr in sorted(risky, key=lambda x: -x[0]["n_obs"])[:12]:
            c = vals[r["loinc"]]
            pool = list(c.elements())
            if len(pool) > 20000:
                pool = random.sample(pool, 20000)
            nd = r["native_decimals"]
            changed = sum(1 for v in pool
                          if round(v + random.gauss(0, abs(v) * mag / 100), nd) != round(v, nd))
            print(f"   {r['n_obs']:>10,} {mag:8.1f}% {thr:6.2f}% {100*changed/len(pool):9.1f}%  "
                  f"{r['name'][:44]}")
    else:
        print("   none of the anchored codes is at risk: every register magnitude exceeds")
        print("   half a reporting step at that code's median value.")

    SCALES_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SCALES_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {SCALES_CSV}")


if __name__ == "__main__":
    sys.exit(main())
