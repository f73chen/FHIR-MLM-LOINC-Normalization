"""
Which observation fields identify the lab code? A lookup table keyed on a field, or a combination of
fields, measures how much of the code each one gives away, and how much of that survives when the
values stop being exact. analysis/refrange_diagnostic.py measures one such lookup, on (reference
range, unit), but caches only its predictions. This script caches the fields themselves.

One extraction pass writes the raw fields of every LOINC-mappable lab observation to a Parquet cache,
and the analyses are cheap passes over it. Caching the raw numeric bounds makes the coarsening ladder
possible without re-scanning. `obs_ordinal` follows the convention of refrange_diagnostic.py, where
unmappable lab observations do not consume an ordinal. The cache therefore joins the evaluation dumps
of pipeline/15_eval_runner.py position for position. The perturbation and unit analyses (a16, a17, a19,
a20, a30, a32, a37), pipeline/build_numeric_obs_counts.py and the Appendix B code read the cache
(thesis Chapter 4 and Appendices A and B).

Stages:
  extract  : per-patient NDJSON -> WORK_DIR/analysis_cache/a15_obs_fields/{train,val,test}.parquet,
             ~21 min at 16 jobs for 112k patients
  gate     : reproduce refrange_diagnostic.py's results and the evaluation-dump join, from the cache
  analyse  : the four analyses below -> results/analysis/a15/

Analyses:
  A  field-set identifiability : key count, coverage, unique-key rate, H(label|key) and
                                 lookup top-k (covered and overall), per feature set
  B  coarsening ladder         : reference range at exact / 3sf / 2sf / 1sf / magnitude / unit-only
  C  value encodings           : raw / per-analyte percentile (the tokenizer's bin table) /
                                 pooled-global percentile / log-magnitude, and the bare has-bins
                                 absence signal
  D  field inventory           : what each model receives per lab observation, measured

Deterministic: every table is a frequency count over a fixed patient set. No seeds.

The cache holds patient IDs and stays in WORK_DIR. stdout and every file under results/ are aggregate
only. --smoke runs end to end on a small patient subset.

  python analysis/a15_field_identifiability.py --stage all --jobs 16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

for _p in (str(locations.THIRD_PARTY), str(locations.COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NDJSON_DIR = locations.PATIENTS
SPLITS_DIR = locations.REPO / "data/splits"
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
LABEVENT_BINS_PATH = f"{locations.TOKENIZER}/labevent_bins.pkl"
EVAL_BASE = str(locations.WORK / "eval")
CACHE_DIR = locations.WORK / "analysis_cache/a15_obs_fields"
RESULTS_DIR = locations.REPO / "results/analysis/a15"
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
TOP_KS = (1, 3, 5, 10)

# Results of refrange_diagnostic.py at n_train=100000. The gate checks against these.
PUBLISHED = {"overall_top1": 0.8267161244143743, "rrunit_top1": 0.972216389265493,
             "rr_constant_pct": 74.13793103448276, "n_scored": 3847594}


# --------------------------- extraction ---------------------------
def read_uuids(path):
    return [l.strip()[:-7] if l.strip().endswith(".ndjson") else l.strip()
            for l in open(path) if l.strip()]


def is_lab_obs(obs):
    """Matches pipeline/06_tfidf_dataset.py, refrange_diagnostic.py and the model head's code system."""
    if obs.get("resourceType") != "Observation":
        return False
    c = obs.get("code", {}).get("coding", [])
    return bool(c) and c[0].get("system") == LAB_CODE_SYSTEM and bool(c[0].get("code"))


def _f(x):
    """float or None. Reference-range bounds arrive as int or float."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _unit(q):
    """Unit string for a FHIR Quantity.

    In a sample of 54,360 lab observations, `unit` and `code` were either both absent or both present
    and equal. It therefore does not matter here whether `unit or code` (pipeline/06_tfidf_dataset.py)
    or `code or unit` (fhir_tokenizer.py) is used. This order matches refrange_diagnostic.py, so the
    gate reproduces it bit for bit.
    """
    if not q:
        return None
    return (q.get("unit") or q.get("code") or "").strip().lower() or None


def extract_patient(uuid, mimic_code_to_class, mappable):
    """Lab observations in the same order the deep models see them, with raw fields kept.

    The ordinal convention matches refrange_diagnostic.enumerate_patient. A lab observation whose
    class is unmappable is skipped and does not consume an ordinal. The join gate checks this.
    """
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    p = NDJSON_DIR / f"{uuid}.ndjson"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            resources = ndjson.load(f)
    except Exception:
        return []                      # skipped, and no patient ID or exception text is printed
    enc = get_encounters(resources)
    resources = sorted(resources, key=lambda r: sort_key(r, enc))

    out, ordinal = [], 0
    for r in resources:
        if not is_lab_obs(r):
            continue
        item_code = str(r["code"]["coding"][0].get("code"))
        cls = mimic_code_to_class.get(item_code)
        if cls is None or cls not in mappable:
            continue

        rr_lo = rr_hi = rr_lo_u = rr_hi_u = None
        rrs = r.get("referenceRange") or []
        if rrs:
            # MIMIC lab observations carry exactly one range with both bounds. To be safe, take the
            # first entry that yields any bound.
            for rr in rrs:
                lo, hi = rr.get("low"), rr.get("high")
                if lo or hi:
                    rr_lo, rr_lo_u = _f((lo or {}).get("value")), _unit(lo)
                    rr_hi, rr_hi_u = _f((hi or {}).get("value")), _unit(hi)
                    break

        val = val_u = val_str = None
        if (vq := r.get("valueQuantity")):
            kind = "quantity"
            val, val_u = _f(vq.get("value")), _unit(vq)
        elif (vs := r.get("valueString")):
            kind = "string"
            val_str = str(vs).strip().lower() or None
        elif (vcc := r.get("valueCodeableConcept")):
            kind = "codeable"
            raw = vcc.get("text") or ((vcc.get("coding") or [{}])[0].get("display") or "")
            val_str = str(raw).strip().lower() or None
        else:
            kind = "none"

        priority = None
        if (ext := r.get("extension")):
            priority = (ext[0].get("valueString") or "").strip().lower() or None

        out.append((ordinal, int(cls), item_code, rr_lo, rr_lo_u, rr_hi, rr_hi_u,
                    val, val_u, kind, val_str, priority))
        ordinal += 1
    return out


COLS = ["patient_uuid", "obs_ordinal", "class_idx", "item_code", "rr_low", "rr_low_unit",
        "rr_high", "rr_high_unit", "val", "val_unit", "val_kind", "val_string", "priority"]


def _frame(flat):
    import pandas as pd
    df = pd.DataFrame(flat, columns=COLS)
    for c, t in (("obs_ordinal", "int32"), ("class_idx", "int16")):
        df[c] = df[c].astype(t)
    for c in ("patient_uuid", "item_code", "rr_low_unit", "rr_high_unit", "val_unit",
              "val_kind", "val_string", "priority"):
        df[c] = df[c].astype("string")
    for c in ("rr_low", "rr_high", "val"):
        df[c] = df[c].astype("float64")
    return df


def stage_extract(args, mimic_code_to_class, mappable, chunk=10000):
    """Parallel NDJSON scan that writes one Parquet file per split.

    It writes in patient chunks through a single ParquetWriter so that peak memory stays bounded. The
    100k-patient train split is ~38M observations, and the intermediate list of tuples alone would
    take several GB if materialised at once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from joblib import Parallel, delayed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, limit in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        uu = read_uuids(SPLITS_DIR / f"{split}_patients.txt")
        if limit:
            uu = uu[:limit]
        t0, n_obs, writer, schema = time.time(), 0, None, None
        out = CACHE_DIR / f"{split}.parquet"
        try:
            for i in range(0, len(uu), chunk):
                part = uu[i:i + chunk]
                rows = Parallel(n_jobs=args.jobs, verbose=0)(
                    delayed(extract_patient)(u, mimic_code_to_class, mappable) for u in part)
                flat = [(u,) + rec for u, recs in zip(part, rows) for rec in recs]
                del rows
                if not flat:
                    continue
                tbl = pa.Table.from_pandas(_frame(flat), preserve_index=False)
                del flat
                if writer is None:
                    schema = tbl.schema
                    writer = pq.ParquetWriter(out, schema, compression="zstd")
                writer.write_table(tbl.cast(schema))
                n_obs += tbl.num_rows
                del tbl
                if len(uu) > chunk:
                    print(f"    {split}: {min(i+chunk, len(uu)):>7,}/{len(uu):,} patients, "
                          f"{n_obs:>11,} obs  ({time.time()-t0:.0f}s)", flush=True)
        finally:
            if writer is not None:
                writer.close()
        dt, mb = time.time() - t0, out.stat().st_size / 1e6
        print(f"  {split:5s}: {len(uu):>7,} patients -> {n_obs:>11,} obs  ({dt:.0f}s, {mb:.0f} MB)",
              flush=True)
        manifest[split] = {"n_patients": len(uu), "n_obs": int(n_obs),
                           "seconds": round(dt, 1), "mb": round(mb, 1)}
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_cache(split):
    import pandas as pd
    p = CACHE_DIR / f"{split}.parquet"
    if not p.exists():
        sys.exit(f"cache missing: {p}\nrun --stage extract first")
    return pd.read_parquet(p)


# --------------------------- key builders ---------------------------
def sigfig(x, n):
    """Round to n significant figures. None-safe; 0.0 maps to itself.

    It multiplies, rounds and divides, and does not use round(x, decimals). The two disagree on
    decimal-tie values because binary floats are not exactly their decimal literals. On the 264
    reference-range bounds in MIMIC, round(x, decimals) differs from this on 7.35 and 7.45 at n=2
    (7.3/7.5 vs 7.4/7.4). `_sigfig_vec` below uses np.round on the scaled value and produces the
    ladder's numbers. This scalar version matches it exactly, so that the unit tests check the
    implementation that is used.
    """
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if x == 0:
        return 0.0
    f = 10.0 ** ((n - 1) - math.floor(math.log10(abs(x))))
    return round(x * f) / f


def magnitude(x):
    """Order of magnitude, floor(log10|x|). 0.0 -> sentinel -99, a real bucket. None and non-finite
    values give None."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if x == 0:
        return -99
    return int(math.floor(math.log10(abs(x))))


def rr_key_exact(lo, hi, lu, hu):
    """refrange_diagnostic.py's key shape: bounds rounded to 4 dp so float noise does not fragment it.

    refrange_diagnostic.obs_features builds a sorted tuple of (lo, lo_unit, hi, hi_unit) over the
    referenceRange list. MIMIC lab observations carry exactly one range, so the sort is a no-op and
    this single tuple is identical. The gate checks it.
    """
    lo4 = None if lo is None else round(lo, 4)
    hi4 = None if hi is None else round(hi, 4)
    if lo4 is None and hi4 is None and lu is None and hu is None:
        return None
    return (lo4, lu, hi4, hu)


def rr_key_sf(lo, hi, lu, hu, n):
    if lo is None and hi is None:
        return None
    return (sigfig(lo, n), lu, sigfig(hi, n), hu)


def rr_key_mag(lo, hi, lu, hu):
    if lo is None and hi is None:
        return None
    return (magnitude(lo), lu, magnitude(hi), hu)


def strip_units(k):
    """Drop the unit slots from a 4-tuple reference-range key."""
    return None if k is None else (k[0], k[2])


def units_only(lu, hu, vu):
    u = lu or hu or vu
    return (u,) if u else None


# --------------------------- scoring core ---------------------------
def fit_table(keys, classes):
    """key -> Counter(class), over rows where the key is not None."""
    t = defaultdict(Counter)
    for k, c in zip(keys, classes):
        if k is not None:
            t[k][c] += 1
    return t


def topk_list(counter, k, tie="insertion"):
    """Top-k labels for one key.

    tie="insertion"  Counter.most_common: equal counts keep first-seen order. This is what
                     refrange_diagnostic.py does, so the gate uses it to reproduce its numbers
                     exactly.
    tie="label"      equal counts are broken by ascending label id. This is deterministic, unlike
                     insertion order, which depends on the order in which patient files are read.
                     The vectorised path does the same. The tests use it to check
                     score_vec == score_single.
    """
    if tie == "insertion":
        return [c for c, _ in counter.most_common(k)]
    return [c for c, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def score_single(train_keys, train_cls, test_keys, test_cls, prior_top, tie="insertion"):
    """Lookup accuracy for one key family, with prior backoff for missing/unseen keys.

    Returns covered-only and overall top-k, plus the table's structural statistics.
    """
    tbl = fit_table(train_keys, train_cls)
    cache = {k: topk_list(c, max(TOP_KS), tie) for k, c in tbl.items()}
    hit_cov = np.zeros(len(TOP_KS))
    hit_all = np.zeros(len(TOP_KS))
    n_cov = 0
    for k, true_c in zip(test_keys, test_cls):
        preds = cache.get(k) if k is not None else None
        covered = preds is not None
        if covered:
            n_cov += 1
            p = (preds + prior_top)[: max(TOP_KS)]
        else:
            p = prior_top
        for i, kk in enumerate(TOP_KS):
            ok = true_c in p[:kk]
            hit_all[i] += ok
            if covered:
                hit_cov[i] += ok
    n = len(test_cls)
    n_uniq = sum(1 for c in tbl.values() if len(c) == 1)
    # H(label | key), in bits, over the training mass that has a key
    tot = sum(sum(c.values()) for c in tbl.values())
    hcond = 0.0
    for c in tbl.values():
        m = sum(c.values())
        p = np.array(list(c.values()), dtype=float) / m
        hcond += (m / tot) * float(-(p * np.log2(p)).sum()) if tot else 0.0
    return {
        "n_keys": len(tbl),
        "unique_key_pct": 100.0 * n_uniq / len(tbl) if tbl else float("nan"),
        "test_coverage_pct": 100.0 * n_cov / n if n else float("nan"),
        "H_label_given_key_bits": hcond,
        **{f"top{k}_covered": (hit_cov[i] / n_cov if n_cov else float("nan"))
           for i, k in enumerate(TOP_KS)},
        **{f"top{k}_overall": hit_all[i] / n for i, k in enumerate(TOP_KS)},
    }


def label_entropy(classes):
    c = Counter(classes)
    tot = sum(c.values())
    p = np.array(list(c.values()), dtype=float) / tot
    return float(-(p * np.log2(p)).sum())


def cascade_floor(tr, te, prior_top, arr):
    """An exact reimplementation of refrange_diagnostic.py's rr+unit -> rr -> unit -> prior cascade.

    Key shapes follow obs_features(). The reference-range key already contains the units of the
    bounds, and `unit` is the valueQuantity unit. The three levels are therefore
    (rr_key, value_unit) -> rr_key -> value_unit. This is the regression gate. It must reproduce the
    results of refrange_diagnostic.py (PUBLISHED) from the cache.
    """
    t_rr_unit = fit_table(tr["k_rr_unit"], tr["cls"])
    t_rr = fit_table(tr["k_rr"], tr["cls"])
    t_unit = fit_table(tr["k_unit"], tr["cls"])
    c_rr_unit = {k: topk_list(c, max(TOP_KS)) for k, c in t_rr_unit.items()}
    c_rr = {k: topk_list(c, max(TOP_KS)) for k, c in t_rr.items()}
    c_unit = {k: topk_list(c, max(TOP_KS)) for k, c in t_unit.items()}
    levels = ("rr+unit", "rr", "unit", "prior")
    hit = {lv: np.zeros(len(TOP_KS)) for lv in levels}
    hit["all"] = np.zeros(len(TOP_KS))
    cnt = Counter()
    for kru, kr, ku, true_c in zip(te["k_rr_unit"], te["k_rr"], te["k_unit"], te["cls"]):
        true_l = arr[true_c]
        if kru is not None and kru in c_rr_unit:
            src, preds = "rr+unit", c_rr_unit[kru]
        elif kr is not None and kr in c_rr:
            src, preds = "rr", c_rr[kr]
        elif ku is not None and ku in c_unit:
            src, preds = "unit", c_unit[ku]
        else:
            src, preds = "prior", None
        p = prior_top if preds is None else (preds + prior_top)[: max(TOP_KS)]
        pl = [arr[x] for x in p]          # rank classes, compare LOINC, as refrange_diagnostic.py does
        for i, k in enumerate(TOP_KS):
            ok = true_l in pl[:k]
            hit[src][i] += ok
            hit["all"][i] += ok
        cnt[src] += 1
        cnt["all"] += 1
    return cnt, hit, levels


# --------------------------- value encodings ---------------------------
def load_bins():
    import pickle
    with open(LABEVENT_BINS_PATH, "rb") as f:
        return pickle.load(f)


def per_analyte_percentile(value, item_code, numeric_bins):
    """An exact replica of fhir_tokenizer._get_numeric_token(numeric_encoding='percentile_bins').

    Returns the p_XX string, or None when the item has no bin table. In that case FHIR-MLM emits no
    value token, and BCM and Llama get no ValueQuantity line.
    """
    if value is None:
        return None
    bins = numeric_bins.get(item_code)
    if not bins:
        return None
    n_bins = len(bins)                       # 4 or 19 cut points, as the tokenizer counts them
    scale = 100 / n_bins
    bucket = np.digitize(value, bins, right=True)
    bucket = bucket * scale - (scale / 2)
    percentile = round(float(np.clip(bucket, 0, 100)) + 1e-6)
    return f"p_{percentile}"


def build_global_percentile_edges(values, n=100):
    """Pooled bin edges over all lab values, label-independent by construction."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    return np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])


def global_percentile(value, edges):
    if value is None or not math.isfinite(value):
        return None
    return f"g_{int(np.digitize(value, edges, right=True))}"


# --------------------------- stages ---------------------------
def prep(df, numeric_bins, loinc_of_class, g_edges=None, only=None):
    """Materialise every key family once, as plain Python lists (dict keys must be hashable).

    There are two label columns:
      "cls"  the MIMIC item class (1,623 values). Only the cascade gate uses it. The gate replicates
             the behaviour of refrange_diagnostic.py, which ranks classes and then compares LOINC.
      "y"    the LOINC code, which is the label the models are scored on. Several MIMIC items map
             to the same LOINC, so class-space scoring is strictly harsher and would not reproduce
             the results of refrange_diagnostic.py. Every analysis uses this one.
    """
    # Parquet float columns return missing values as NaN. If they are left as NaN,
    # `lo is None` is False and every missing bound becomes a distinct key object (NaN != NaN). The
    # reference-range table then breaks into ~100k phantom singleton keys, where
    # refrange_diagnostic.py finds a few hundred.
    import pandas as _pd
    def _nn(seq):
        return [None if (x is None or x is _pd.NA or x != x) else x for x in seq]
    lo = _nn(df["rr_low"].tolist()); hi = _nn(df["rr_high"].tolist())
    lu = _nn(df["rr_low_unit"].astype(object).tolist())
    hu = _nn(df["rr_high_unit"].astype(object).tolist())
    vu = _nn(df["val_unit"].astype(object).tolist())
    vs = _nn(df["val_string"].astype(object).tolist())
    pr = _nn(df["priority"].astype(object).tolist())
    ic = df["item_code"].astype(object).tolist()
    val = _nn(df["val"].tolist())
    _cls = df["class_idx"].tolist()
    d = {
        "cls": _cls,
        "y": [loinc_of_class[c] for c in _cls],
        "item_code": ic,
        "val": val,
        "k_rr": [rr_key_exact(a, b, c, e) for a, b, c, e in zip(lo, hi, lu, hu)],
        "k_unit": vu,
        "k_prio": [(x,) if x else None for x in pr],
        "k_vstr": [(x,) if x else None for x in vs],
    }
    d["k_rr_unit"] = [None if (a is None or b is None) else (a, b)
                      for a, b in zip(d["k_rr"], d["k_unit"])]   # strict, see _combo()
    if only is not None:
        # The gate needs three key families. Building all twenty as tuples over 38M rows would cost
        # tens of GB.
        return {k: v for k, v in d.items() if k in only or k in ("cls", "y", "item_code", "val")}
    d["k_rr_nounit"] = [strip_units(k) for k in d["k_rr"]]
    for n in (3, 2, 1):
        d[f"k_rr{n}sf"] = [rr_key_sf(a, b, c, e, n) for a, b, c, e in zip(lo, hi, lu, hu)]
        d[f"k_rr{n}sf_nounit"] = [strip_units(k) for k in d[f"k_rr{n}sf"]]
    d["k_rrmag"] = [rr_key_mag(a, b, c, e) for a, b, c, e in zip(lo, hi, lu, hu)]
    d["k_rrmag_nounit"] = [strip_units(k) for k in d["k_rrmag"]]
    d["k_rrunitonly"] = [units_only(a, b, None) for a, b in zip(lu, hu)]
    d["k_anyunit"] = [units_only(a, b, c) for a, b, c in zip(lu, hu, vu)]
    # value encodings
    d["k_vraw"] = [None if v is None else (round(v, 4),) for v in val]
    d["k_vpct"] = [(t,) if (t := per_analyte_percentile(v, c, numeric_bins)) else None
                   for v, c in zip(val, ic)]
    d["k_vmag"] = [None if (m := magnitude(v)) is None else (m,) for v in val]
    d["k_hasbins"] = [(c in numeric_bins,) for c in ic]
    if g_edges is not None:
        d["k_vglob"] = [(t,) if (t := global_percentile(v, g_edges)) else None for v in val]
    return d


def _combo(d, parts):
    """Cross two or three key families. Strict conjunction: the combined key exists only when every
    part is present.

    This matches how refrange_diagnostic.py builds its tables (`if rr is not None and unit is not
    None`), so the cascade gate reproduces its numbers. A partial key would absorb rows that the
    cascade in refrange_diagnostic.py backs off to a shorter key, and the coverage of the top level
    would be inflated.
    """
    cols = [d[p] for p in parts]
    out = []
    for tup in zip(*cols):
        out.append(None if any(x is None for x in tup) else tup)
    return out


def stage_gate(tr, te, manifest, arr):
    print(f"\n{'='*74}\nGATE 1: reproduce refrange_diagnostic.py from the cache\n{'='*74}")
    prior_top = topk_list(Counter(tr["cls"]), max(TOP_KS))
    cnt, hit, levels = cascade_floor(tr, te, prior_top, arr)
    print(f"{'level':10s} {'n':>11s} {'share':>7s} " + " ".join(f"{'top'+str(k):>8s}" for k in TOP_KS))
    for lv in levels:
        if cnt[lv]:
            print(f"{lv:10s} {cnt[lv]:11,d} {100*cnt[lv]/cnt['all']:6.1f}% " +
                  " ".join(f"{100*hit[lv][i]/cnt[lv]:7.2f}%" for i in range(len(TOP_KS))))
    print(f"{'OVERALL':10s} {cnt['all']:11,d} {100.0:6.1f}% " +
          " ".join(f"{100*hit['all'][i]/cnt['all']:7.2f}%" for i in range(len(TOP_KS))))

    full = manifest["train"]["n_patients"] == 100000 and manifest["test"]["n_patients"] == 10000
    ok = True
    if full:
        for name, got, want in (("overall top1", hit["all"][0] / cnt["all"], PUBLISHED["overall_top1"]),
                                ("rr+unit top1", hit["rr+unit"][0] / cnt["rr+unit"], PUBLISHED["rrunit_top1"]),
                                ("n_scored", cnt["all"], PUBLISHED["n_scored"])):
            d = abs(got - want)
            good = d < (1 if name == "n_scored" else 5e-4)
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  {name}: got {got:.6f} want {want:.6f}"
                  if name != "n_scored" else
                  f"  {'PASS' if good else 'FAIL'}  {name}: got {got:,} want {want:,}")
        # reference-range constancy
        per_class = defaultdict(set)
        for k, c in zip(tr["k_rr"], tr["cls"]):
            if k is not None:
                per_class[c].add(k)
        n_rr = np.array([len(v) for v in per_class.values()])
        pct = 100.0 * (n_rr == 1).mean()
        good = abs(pct - PUBLISHED["rr_constant_pct"]) < 0.05
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  rr constancy: got {pct:.4f}% "
              f"want {PUBLISHED['rr_constant_pct']:.4f}% ({len(n_rr)} classes)")
    else:
        print("  (subset run, so the published targets do not apply and only the structure is checked)")
    return bool(ok), {"cnt": {k: int(v) for k, v in cnt.items()},
                      "overall_top1": float(hit["all"][0] / cnt["all"])}


def stage_join_gate(te_df, n_check=200000):
    """Gate 2: the cache's (uuid, obs_ordinal) must agree with an evaluation dump's true_class."""
    import pandas as pd
    print(f"\n{'='*74}\nGATE 2: join against the BCM eval dump\n{'='*74}")
    p = Path(EVAL_BASE) / "bcm__nodesc__seed0__mimictest" / "part-0000.parquet"
    if not p.exists():
        print(f"  SKIP: no dump at {p}")
        return None
    bcm = pd.read_parquet(p, columns=["patient_uuid", "obs_ordinal", "true_class"])
    mine = te_df[["patient_uuid", "obs_ordinal", "class_idx"]]
    m = bcm.merge(mine, on=["patient_uuid", "obs_ordinal"], how="inner")
    if len(m) > n_check:
        m = m.sample(n_check, random_state=0)
    agree = int((m["true_class"] == m["class_idx"]).sum())
    pct = 100.0 * agree / max(len(m), 1)
    good = pct == 100.0 and len(m) > 0
    print(f"  overlapping keys checked: {len(m):,}")
    print(f"  true_class agreement: {agree:,}/{len(m):,} = {pct:.4f}%")
    print(f"  {'PASS' if good else 'FAIL'}  obs_ordinal convention matches the eval dumps")
    return good




# =============== vectorised scoring: the production path ===============
# score_single() above is the reference implementation, with one Python loop per row. It is easy to
# check, and the gate and the tests use it. It is too slow for the analyses, because 38M training
# rows x ~30 key families is ~10^9 Python iterations. Everything below is the pandas/numpy
# equivalent, and the test suite asserts that the two agree exactly on real data.

def _sigfig_vec(a, k):
    out = np.full(a.shape, np.nan)
    m = np.isfinite(a) & (a != 0)
    if m.any():
        f = np.power(10.0, (k - 1 - np.floor(np.log10(np.abs(a[m])))))
        out[m] = np.round(a[m] * f) / f
    out[np.isfinite(a) & (a == 0)] = 0.0
    return out


def _mag_vec(a):
    out = np.full(a.shape, np.nan)
    m = np.isfinite(a) & (a != 0)
    out[m] = np.floor(np.log10(np.abs(a[m])))
    out[np.isfinite(a) & (a == 0)] = -99.0
    return out


def components(df, numeric_bins, g_edges):
    """Raw component columns and per-field presence masks.

    A component may be missing inside a present field, for example a reference range whose bounds
    carry no unit. That is still a usable key value, and the key in refrange_diagnostic.py keeps it as
    None inside the tuple. Missing sub-parts therefore get their own code and do not void the key. A
    key is absent only when the field is absent, and the presence masks record that.
    """
    lo = df["rr_low"].to_numpy(dtype="float64", na_value=np.nan)
    hi = df["rr_high"].to_numpy(dtype="float64", na_value=np.nan)
    val = df["val"].to_numpy(dtype="float64", na_value=np.nan)
    item = df["item_code"].astype("string").fillna("").to_numpy()

    c = {}
    c["rr_lo"] = np.round(lo, 4); c["rr_hi"] = np.round(hi, 4)
    for k in (3, 2, 1):
        c[f"rr_lo{k}"] = _sigfig_vec(lo, k); c[f"rr_hi{k}"] = _sigfig_vec(hi, k)
    c["rr_lom"] = _mag_vec(lo); c["rr_him"] = _mag_vec(hi)
    for name, col in (("rr_lo_u", "rr_low_unit"), ("rr_hi_u", "rr_high_unit"),
                      ("val_u", "val_unit"), ("prio", "priority"), ("vstr", "val_string")):
        c[name] = df[col].astype("string").fillna("\x00").to_numpy()
    c["val_raw"] = np.round(val, 4)
    c["val_mag"] = _mag_vec(val)
    c["hasbins"] = np.isin(item, np.array(sorted(numeric_bins), dtype=object)).astype("int8")

    # per-analyte percentile, one loop step per binned item (530 of them)
    vp = np.full(len(df), np.nan)
    order = np.argsort(item, kind="stable")
    it_sorted = item[order]
    starts = np.searchsorted(it_sorted, sorted(numeric_bins), side="left")
    ends = np.searchsorted(it_sorted, sorted(numeric_bins), side="right")
    for (it, bins), a, b in zip(sorted(numeric_bins.items()), starts, ends):
        if a == b:
            continue
        idx = order[a:b]
        v = val[idx]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        nb = len(bins); scale = 100 / nb
        buck = np.digitize(v[ok], bins, right=True) * scale - (scale / 2)
        vp[idx[ok]] = np.round(np.clip(buck, 0, 100) + 1e-6)
    c["val_pct"] = vp
    gv = np.full(len(df), np.nan)
    if g_edges is not None:
        m = np.isfinite(val)
        gv[m] = np.digitize(val[m], g_edges, right=True)
    c["val_glob"] = gv

    present = {
        "rr":      np.isfinite(lo) | np.isfinite(hi),
        "vu":      df["val_unit"].notna().to_numpy(),
        "v":       np.isfinite(val),
        "vpct":    np.isfinite(vp),
        "vglob":   np.isfinite(gv),
        "vmag":    np.isfinite(val),
        "prio":    df["priority"].notna().to_numpy(),
        "vstr":    df["val_string"].notna().to_numpy(),
        "hasbins": np.ones(len(df), dtype=bool),
        "anyunit": (df["rr_low_unit"].notna() | df["rr_high_unit"].notna()
                    | df["val_unit"].notna()).to_numpy(),
    }
    return c, present


def joint_codes(ctr, cte):
    """Factorise every component once over train+test, so the two sides share a code space and no
    key column carries NaN. A float NaN does not equal itself in a merge, so NaN keys would fail to
    match and drop out of the coverage."""
    import pandas as pd
    n = len(next(iter(ctr.values())))
    out_tr, out_te = {}, {}
    for col in ctr:
        a = np.concatenate([ctr[col], cte[col]])
        codes, _ = pd.factorize(pd.Series(a), use_na_sentinel=True)   # -1 for NaN
        codes = codes.astype("int32")
        out_tr[col], out_te[col] = codes[:n], codes[n:]
    return out_tr, out_te


# group name -> (component columns, presence key)
GROUPS = {
    "rr":        (["rr_lo", "rr_hi", "rr_lo_u", "rr_hi_u"],   "rr"),
    "rr_nou":    (["rr_lo", "rr_hi"],                         "rr"),
    "rr3sf":     (["rr_lo3", "rr_hi3", "rr_lo_u", "rr_hi_u"], "rr"),
    "rr3sf_nou": (["rr_lo3", "rr_hi3"],                       "rr"),
    "rr2sf":     (["rr_lo2", "rr_hi2", "rr_lo_u", "rr_hi_u"], "rr"),
    "rr2sf_nou": (["rr_lo2", "rr_hi2"],                       "rr"),
    "rr1sf":     (["rr_lo1", "rr_hi1", "rr_lo_u", "rr_hi_u"], "rr"),
    "rr1sf_nou": (["rr_lo1", "rr_hi1"],                       "rr"),
    "rrmag":     (["rr_lom", "rr_him", "rr_lo_u", "rr_hi_u"], "rr"),
    "rrmag_nou": (["rr_lom", "rr_him"],                       "rr"),
    "rrunit":    (["rr_lo_u", "rr_hi_u"],                     "rr"),
    "unit":      (["val_u"],                                  "vu"),
    "anyunit":   (["rr_lo_u", "rr_hi_u", "val_u"],            "anyunit"),
    "prio":      (["prio"],                                   "prio"),
    "vstr":      (["vstr"],                                   "vstr"),
    "vraw":      (["val_raw"],                                "v"),
    "vpct":      (["val_pct"],                                "vpct"),
    "vglob":     (["val_glob"],                               "vglob"),
    "vmag":      (["val_mag"],                                "vmag"),
    "hasbins":   (["hasbins"],                                "hasbins"),
}


def score_vec(ktr, ptr, ytr, kte, pte, yte, groups, prior_top):
    """Lookup accuracy for the key family formed by conjoining `groups`.

    Vectorised equivalent of score_single(). The key exists only where every group's field is
    present, and unseen or absent keys back off to the global prior. Ties are broken by ascending
    label id. This is deterministic, unlike the insertion order of Counter.most_common, which depends
    on the order in which patient files are read. tests/test_a15_vec_equiv.py asserts
    score_vec == score_single(tie="label") exactly.
    """
    import pandas as pd
    K = max(TOP_KS)
    cols, mtr, mte = [], np.ones(len(ytr), bool), np.ones(len(yte), bool)
    for gname in groups:
        gc, pk = GROUPS[gname]
        cols += [x for x in gc if x not in cols]
        mtr = mtr & ptr[pk]; mte = mte & pte[pk]

    sub = pd.DataFrame({c: ktr[c][mtr] for c in cols})
    sub["y"] = ytr[mtr]
    g = sub.groupby(cols + ["y"], observed=True, sort=False).size().reset_index(name="n")
    del sub
    if len(g) == 0:
        return {"n_keys": 0, "unique_key_pct": float("nan"), "test_coverage_pct": 0.0,
                "H_label_given_key_bits": float("nan"),
                **{f"top{k}_covered": float("nan") for k in TOP_KS},
                **{f"top{k}_overall": float(np.mean(np.isin(yte, prior_top[:k]))) for k in TOP_KS}}

    gk = g.groupby(cols, observed=True, sort=False)
    per_key = gk.size()
    n_keys = int(len(per_key))
    n_uniq = int((per_key == 1).sum())
    nn = g["n"].to_numpy(dtype="float64")
    tot = gk["n"].transform("sum").to_numpy(dtype="float64")
    N = nn.sum()
    hcond = float(-(nn * np.log2(nn / tot)).sum() / N)      # H(label|key) = -1/N * sum n log(n/n_k)

    g = g.sort_values(cols + ["n", "y"], ascending=[True] * len(cols) + [False, True],
                      kind="mergesort")
    g["r"] = g.groupby(cols, observed=True, sort=False).cumcount()
    piv = g.loc[g["r"] < K].set_index(cols + ["r"])["y"].unstack("r")
    piv.columns = [f"p{i}" for i in piv.columns]
    for i in range(K):
        if f"p{i}" not in piv.columns:
            piv[f"p{i}"] = np.nan
    piv = piv[[f"p{i}" for i in range(K)]]
    del g, gk

    right = pd.DataFrame({c: kte[c][mte] for c in cols})
    right["_y"] = yte[mte]
    res = right.merge(piv, left_on=cols, right_index=True, how="left", sort=False)
    del right, piv
    P = res[[f"p{i}" for i in range(K)]].to_numpy(dtype="float64")
    Y = res["_y"].to_numpy(dtype="float64")[:, None]
    covered = ~np.isnan(P[:, 0])
    # A key with fewer than K distinct labels leaves trailing empty slots. score_single fills them
    # by concatenating the global prior (`(preds + prior_top)[:K]`), and this code does the same.
    # Otherwise top-3/5/10 would disagree with the reference on every sparse key.
    prior_arr = np.asarray(prior_top, dtype="float64")
    ncand = (~np.isnan(P)).sum(axis=1)
    J = np.arange(K)[None, :]
    pad = J >= ncand[:, None]
    if pad.any():
        src = np.clip(J - ncand[:, None], 0, len(prior_arr) - 1)
        P = np.where(pad, prior_arr[src], P)
    eq = (P == Y)

    n_all, n_cov = len(yte), int(covered.sum())
    cov_full = np.zeros(n_all, bool)
    cov_full[np.flatnonzero(mte)[covered]] = True
    prior = np.asarray(prior_top)
    out = {"n_keys": n_keys, "unique_key_pct": 100.0 * n_uniq / n_keys,
           "test_coverage_pct": 100.0 * n_cov / n_all, "H_label_given_key_bits": hcond}
    for k in TOP_KS:
        cov_hits = int(eq[:, :k][covered].any(axis=1).sum())
        pri_hits = int(np.isin(yte[~cov_full], prior[:k]).sum())
        out[f"top{k}_covered"] = cov_hits / n_cov if n_cov else float("nan")
        out[f"top{k}_overall"] = (cov_hits + pri_hits) / n_all
    return out


# --------------------------- analyses ---------------------------
FIELD_SETS = [
    ("unit (of the value)",       ["unit"]),
    ("unit (any: range or value)", ["anyunit"]),
    ("priority",                  ["prio"]),
    ("value (raw)",               ["vraw"]),
    ("value (raw) + unit",        ["vraw", "unit"]),
    ("value (qualitative text)",  ["vstr"]),
    ("refrange (bounds only)",    ["rr_nou"]),
    ("refrange (+ own units)",    ["rr"]),
    ("refrange + value unit",     ["rr", "unit"]),
    ("refrange + unit + value",   ["rr", "unit", "vraw"]),
]

LADDER = [
    ("exact (4 dp)",       "rr",        "rr_nou"),
    ("3 significant figs", "rr3sf",     "rr3sf_nou"),
    ("2 significant figs", "rr2sf",     "rr2sf_nou"),
    ("1 significant fig",  "rr1sf",     "rr1sf_nou"),
    ("order of magnitude", "rrmag",     "rrmag_nou"),
    ("units only",         "rrunit",    None),
]

VALUE_ENCODINGS = [
    ("raw value",                     "vraw"),
    ("per-analyte percentile",        "vpct"),
    ("pooled-global percentile",      "vglob"),
    ("log-magnitude",                 "vmag"),
    ("has-bins flag only",            "hasbins"),
]

_HDR = (f"{'keys':>9s} {'uniq%':>7s} {'cov%':>6s} {'H|k':>6s} "
        f"{'top1cov':>8s} {'top1all':>8s} {'top5all':>8s}")


def _row(r):
    return (f"{r['n_keys']:9,d} {r['unique_key_pct']:6.1f}% {r['test_coverage_pct']:5.1f}% "
            f"{r['H_label_given_key_bits']:6.3f} {100*r['top1_covered']:7.2f}% "
            f"{100*r['top1_overall']:7.2f}% {100*r['top5_overall']:7.2f}%")


def analysis_A(K, rows):
    print(f"\n{'='*96}\nA: field-set identifiability   (H(label) = {K['H0']:.3f} bits, "
          f"{K['n_test']:,} test observations)\n{'='*96}")
    print(f"{'field set':30s} {_HDR}")
    for name, groups in FIELD_SETS:
        r = score_vec(K["ktr"], K["ptr"], K["ytr"], K["kte"], K["pte"], K["yte"],
                      groups, K["prior"])
        print(f"{name:30s} {_row(r)}", flush=True)
        rows.append({"analysis": "A_field_sets", "variant": name,
                     "H_label_bits": K["H0"], **r})


def analysis_B(K, rows):
    print(f"\n{'='*96}\nB: reference-range coarsening ladder\n{'='*96}")
    print(f"{'precision':22s} {'units':>7s} {_HDR}")
    for name, wu, nu in LADDER:
        for tag, g in (("yes", wu), ("no", nu)):
            if g is None:
                continue
            r = score_vec(K["ktr"], K["ptr"], K["ytr"], K["kte"], K["pte"], K["yte"],
                          [g], K["prior"])
            print(f"{name:22s} {tag:>7s} {_row(r)}", flush=True)
            rows.append({"analysis": "B_coarsening", "variant": f"{name} | units={tag}",
                         "H_label_bits": K["H0"], **r})
    r = {"n_keys": 0, "unique_key_pct": float("nan"), "test_coverage_pct": 0.0,
         "H_label_given_key_bits": K["H0"]}
    for k in TOP_KS:
        r[f"top{k}_covered"] = float("nan")
        r[f"top{k}_overall"] = float(np.mean(np.isin(K["yte"], K["prior"][:k])))
    print(f"{'no reference range':22s} {'-':>7s} {0:9,d} {'    n/a':>7s} {0.0:5.1f}% "
          f"{K['H0']:6.3f} {'    n/a':>8s} {100*r['top1_overall']:7.2f}% "
          f"{100*r['top5_overall']:7.2f}%")
    rows.append({"analysis": "B_coarsening", "variant": "no reference range",
                 "H_label_bits": K["H0"], **r})


def analysis_C(K, rows):
    print(f"\n{'='*96}\nC: value encodings\n{'='*96}")
    print(f"{'encoding':30s} {_HDR}")
    for name, g in VALUE_ENCODINGS:
        for suffix, groups in ((" ", [g]), (" + unit", [g, "unit"])):
            r = score_vec(K["ktr"], K["ptr"], K["ytr"], K["kte"], K["pte"], K["yte"],
                          groups, K["prior"])
            label = (name + suffix).strip()
            print(f"{label:30s} {_row(r)}", flush=True)
            rows.append({"analysis": "C_value_encodings", "variant": label,
                         "H_label_bits": K["H0"], **r})


def analysis_D(te_df, pte, hasbins, out_rows):
    """What each model receives per lab observation, measured on the test split."""
    n = len(te_df)
    hb = hasbins.astype(bool)

    def pct(m):
        return round(100.0 * float(np.mean(m)), 4)

    has_val, has_unit = pte["v"], pte["vu"]
    has_rr, has_prio, has_vstr = pte["rr"], pte["prio"], pte["vstr"]
    rows = [
        # field | raw presence | FHIR-MLM | BCM+Llama | TF-IDF   (after each model's gate)
        ("valueQuantity",     pct(has_val),  pct(has_val & hb),            pct(has_val & hb), pct(has_val)),
        ("value unit",        pct(has_unit), pct(has_unit & has_val & hb), pct(has_val & hb), pct(has_val)),
        ("referenceRange",    pct(has_rr),   pct(has_rr & hb),             pct(has_rr),       pct(has_rr)),
        ("priority",          pct(has_prio), 0.0,                          0.0,               pct(has_prio)),
        ("qualitative value", pct(has_vstr), pct(has_vstr),                pct(has_vstr),     pct(has_vstr)),
    ]
    print(f"\n{'='*96}\nD: what each model receives, % of {n:,} test observations\n{'='*96}")
    print(f"{'field':20s} {'present':>9s} {'FHIR-MLM':>10s} {'BCM/Llama':>11s} {'TF-IDF':>9s}")
    for f, p, m, b, t in rows:
        print(f"{f:20s} {p:8.2f}% {m:9.2f}% {b:10.2f}% {t:8.2f}%")
        out_rows.append({"field": f, "present_pct": p, "fhirmlm_pct": m,
                         "bcm_llama_pct": b, "tfidf_pct": t})
    print(f"\n  test observations whose item has a bin table: {pct(hb):.2f}%  "
          f"({int(hb.sum()):,} of {n:,})")
    return pct(hb)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "gate", "analyse", "all"], required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--force-extract", action="store_true",
                    help="re-extract even if the cache is already complete")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end correctness run on 2000/200/400 train/val/test patients")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_val, args.n_test = 2000, 200, 400

    import pandas as pd
    from fhir_data_utils import build_lab_code_mappings
    mimic_code_to_class, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH,
                                                                  SEEN_CODES_PATH)
    mappable = set(class_to_loinc)
    print(f"LOINC-mappable classes: {len(mappable)} of {len(mimic_code_to_class)} MIMIC items")
    numeric_bins = load_bins()
    print(f"bin tables: {len(numeric_bins)} items "
          f"({sum(1 for v in numeric_bins.values() if len(v)==19)} x 19 cut points, "
          f"{sum(1 for v in numeric_bins.values() if len(v)==4)} x 4)", flush=True)

    if args.stage in ("extract", "all"):
        if (CACHE_DIR / "manifest.json").exists() and not args.force_extract:
            print(f"\ncache already complete at {CACHE_DIR}, skipping extract "
                  "(--force-extract to rebuild)", flush=True)
        else:
            print(f"\n{'='*96}\nEXTRACT -> {CACHE_DIR}\n{'='*96}", flush=True)
            stage_extract(args, mimic_code_to_class, mappable)
    if args.stage == "extract":
        return

    manifest = json.loads((CACHE_DIR / "manifest.json").read_text())
    tr_df, te_df = load_cache("train"), load_cache("test")
    print(f"\ncache loaded: train {len(tr_df):,} obs | test {len(te_df):,} obs", flush=True)
    arr = np.array([class_to_loinc.get(i, f"?{i}") for i in range(1623)], dtype=object)

    gate_ok = None
    if args.stage in ("gate", "all"):
        t0 = time.time()
        gate_keys = {"k_rr", "k_rr_unit", "k_unit"}
        tr_ref = prep(tr_df, numeric_bins, arr, only=gate_keys)
        te_ref = prep(te_df, numeric_bins, arr, only=gate_keys)
        gate_ok, _ = stage_gate(tr_ref, te_ref, manifest, arr)
        join_ok = stage_join_gate(te_df)
        gate_ok = bool(gate_ok) and (join_ok is not False)
        print(f"  (gate took {time.time()-t0:.0f}s)")
        del tr_ref, te_ref
    if args.stage == "gate":
        sys.exit(0 if gate_ok else 1)

    t0 = time.time()
    g_edges = build_global_percentile_edges(tr_df["val"].dropna().to_numpy())
    ctr, ptr = components(tr_df, numeric_bins, g_edges)
    cte, pte = components(te_df, numeric_bins, g_edges)
    hb_te = cte["hasbins"].astype(bool)          # keep the raw flag; codes are opaque
    ktr, kte = joint_codes(ctr, cte)
    del ctr, cte
    y_codes, y_uniq = pd.factorize(pd.Series(np.concatenate(
        [arr[tr_df["class_idx"].to_numpy()], arr[te_df["class_idx"].to_numpy()]])))
    ytr = y_codes[:len(tr_df)].astype("int32"); yte = y_codes[len(tr_df):].astype("int32")
    prior = np.array([c for c, _ in Counter(ytr).most_common(max(TOP_KS))], dtype="int32")
    K = {"ktr": ktr, "ptr": ptr, "ytr": ytr, "kte": kte, "pte": pte, "yte": yte,
         "prior": prior, "H0": label_entropy(yte), "n_test": len(te_df)}
    print(f"  (components + codes: {time.time()-t0:.0f}s, {len(y_uniq)} LOINC labels)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, drows = [], []
    analysis_A(K, rows)
    analysis_B(K, rows)
    analysis_C(K, rows)
    hasbins_pct = analysis_D(te_df, pte, hb_te, drows)

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "a15_identifiability.csv", index=False)
    pd.DataFrame(drows).to_csv(RESULTS_DIR / "a15_field_inventory.csv", index=False)
    (RESULTS_DIR / "a15_manifest.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache": str(CACHE_DIR), "cache_manifest": manifest,
        "n_train_obs": int(len(tr_df)), "n_test_obs": int(len(te_df)),
        "n_loinc_labels": int(len(y_uniq)), "H_label_bits": K["H0"],
        "gate_passed": gate_ok, "pct_test_obs_with_bin_table": hasbins_pct,
        "top_ks": list(TOP_KS), "smoke": bool(args.smoke),
    }, indent=2))
    print(f"\nwrote {RESULTS_DIR}/{{a15_identifiability.csv, a15_field_inventory.csv, "
          f"a15_manifest.json}}")


if __name__ == "__main__":
    main()
