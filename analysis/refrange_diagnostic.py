"""
Measure how far trivial, memoryless signal gets on the LOINC task. This is a diagnostic. It is not one
of the compared models. It measures how much of the accuracy is available from lookup tables over
fields the models see unmasked.

`referenceRange` is close to a per-item constant (most classes have exactly one reference range in the
training data), and it reaches all three deep models unmasked. A model could score well by memorising
which range belongs to which test. This diagnostic measures how far that alone gets.

Method:
  * Enumerate lab observations as the deep models do: sorted(resources, key=sort_key) ->
    is_lab_obs -> keep LOINC-mappable classes. That index is `obs_ordinal`, so the results join the
    pipeline/15_eval_runner.py dumps on (patient_uuid, obs_ordinal).
  * Fit frequency lookup tables on training patients only.
  * Predict with explicit backoff and report the coverage at each level:
        refrange+unit -> refrange -> unit -> global prior
  * Score in the same 828-class LOINC space with the same top-k semantics as every model.

Deterministic (no seeds): the tables are frequency counts over a fixed patient set.

With --write, the per-observation Parquet goes to WORK_DIR/eval/refrange_diagnostic__mimictest and
the metrics to results/analysis/refrange_diagnostic.json. The Parquet holds patient uuids, and stdout
shows aggregates only. The Parquet also gives the catalogue analyses of thesis Chapter 6 their
observation-to-code table (lib/codes.py).

The thesis used 100,000 training patients, which the a15 gate checks (the default is 20,000):

  python analysis/refrange_diagnostic.py --n-train 100000 --write
"""
import argparse
import json
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
EVAL_BASE = str(locations.WORK / "eval")
RESULTS_DIR = locations.REPO / "results/analysis"
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
TOP_KS = (1, 3, 5, 10)


def read_uuids(path):
    return [l.strip()[:-7] if l.strip().endswith(".ndjson") else l.strip()
            for l in open(path) if l.strip()]


def is_lab_obs(obs):
    """Matches pipeline/06_tfidf_dataset.py's `is_lab_obs` and the model head's code system."""
    if obs.get("resourceType") != "Observation":
        return False
    c = obs.get("code", {}).get("coding", [])
    return bool(c) and c[0].get("system") == LAB_CODE_SYSTEM and bool(c[0].get("code"))


def _num(x):
    """Round a reference-range bound so float noise doesn't fragment the key."""
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def obs_features(o):
    """(refrange_key, unit_key) for one lab observation. Both may be None."""
    unit = None
    if (vq := o.get("valueQuantity")):
        unit = (vq.get("unit") or vq.get("code") or "").strip().lower() or None
    rr_key = None
    if (rrs := o.get("referenceRange")):
        parts = []
        for rr in rrs:
            lo, hi = rr.get("low", {}), rr.get("high", {})
            parts.append((_num(lo.get("value")), (lo.get("unit") or "").strip().lower() or None,
                          _num(hi.get("value")), (hi.get("unit") or "").strip().lower() or None))
        parts = [p for p in parts if any(v is not None for v in p)]
        if parts:
            rr_key = tuple(sorted(parts, key=lambda t: tuple(str(x) for x in t)))
    return rr_key, unit


def enumerate_patient(uuid, mimic_code_to_class, mappable):
    """Lab observations in the same order the deep models see them.
    Returns [(obs_ordinal, class_idx, rr_key, unit_key)] over LOINC-mappable classes only."""
    import ndjson
    from fhir_formatter import get_encounters, sort_key
    p = NDJSON_DIR / f"{uuid}.ndjson"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            resources = ndjson.load(f)
    except Exception:
        return []
    enc = get_encounters(resources)
    resources = sorted(resources, key=lambda r: sort_key(r, enc))
    out, ordinal = [], 0
    for r in resources:
        if not is_lab_obs(r):
            continue
        code = r["code"]["coding"][0].get("code")
        cls = mimic_code_to_class.get(str(code))
        if cls is None or cls not in mappable:      # unmappable classes get label -100 and are never scored
            continue
        rr, unit = obs_features(r)
        out.append((ordinal, cls, rr, unit))
        ordinal += 1
    return out


def topk_from_counter(counter, k):
    return [c for c, _ in counter.most_common(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=20000, help="train patients used to fit the tables")
    ap.add_argument("--n-test", type=int, default=None, help="limit test patients (default: all)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="write per-observation parquet + metrics json")
    args = ap.parse_args()

    from fhir_data_utils import build_lab_code_mappings
    from joblib import Parallel, delayed

    mimic_code_to_class, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    mappable = set(class_to_loinc)
    arr = np.array([class_to_loinc.get(i, f"?{i}") for i in range(1623)], dtype=object)
    print(f"LOINC-mappable classes: {len(mappable)}")

    train_u = read_uuids(SPLITS_DIR / "train_patients.txt")[: args.n_train]
    test_u = read_uuids(SPLITS_DIR / "test_patients.txt")
    if args.n_test:
        test_u = test_u[: args.n_test]
    print(f"fitting on {len(train_u):,} train patients | scoring {len(test_u):,} test patients")

    t0 = time.time()
    fn = delayed(enumerate_patient)
    train_rows = Parallel(n_jobs=args.jobs, verbose=0)(fn(u, mimic_code_to_class, mappable) for u in train_u)
    print(f"  train parsed in {time.time()-t0:.0f}s")
    t0 = time.time()
    test_rows = Parallel(n_jobs=args.jobs, verbose=0)(fn(u, mimic_code_to_class, mappable) for u in test_u)
    print(f"  test parsed in {time.time()-t0:.0f}s")

    # -- fit lookup tables on the training patients only --
    t_rr_unit, t_rr, t_unit, t_prior = (defaultdict(Counter), defaultdict(Counter),
                                        defaultdict(Counter), Counter())
    n_fit = 0
    for rows in train_rows:
        for _, cls, rr, unit in rows:
            t_prior[cls] += 1
            if rr is not None and unit is not None:
                t_rr_unit[(rr, unit)][cls] += 1
            if rr is not None:
                t_rr[rr][cls] += 1
            if unit is not None:
                t_unit[unit][cls] += 1
            n_fit += 1
    print(f"fit on {n_fit:,} train observations | keys: rr+unit={len(t_rr_unit):,} "
          f"rr={len(t_rr):,} unit={len(t_unit):,}")

    # -- how constant is the reference range per class? --
    per_class_rr = defaultdict(set)
    for rows in train_rows:
        for _, cls, rr, _ in rows:
            if rr is not None:
                per_class_rr[cls].add(rr)
    n_rr = np.array([len(v) for v in per_class_rr.values()])
    print(f"\nreference-range constancy: {(n_rr==1).mean()*100:.1f}% of {len(n_rr)} classes have exactly "
          f"one reference range (mean {n_rr.mean():.2f} distinct/class)")
    uniq_unit = sum(1 for u, c in t_unit.items() if len(c) == 1)
    print(f"unit uniqueness: {uniq_unit}/{len(t_unit)} units map to exactly one class")

    # -- predict on the test patients with explicit backoff --
    levels = ("rr+unit", "rr", "unit", "prior")
    hit = {k: np.zeros(len(TOP_KS)) for k in ("all",) + levels}
    cnt = Counter()
    dump = []
    prior_top = topk_from_counter(t_prior, max(TOP_KS))
    for u, rows in zip(test_u, test_rows):
        for ordinal, cls, rr, unit in rows:
            if rr is not None and unit is not None and (rr, unit) in t_rr_unit:
                src, c = "rr+unit", t_rr_unit[(rr, unit)]
            elif rr is not None and rr in t_rr:
                src, c = "rr", t_rr[rr]
            elif unit is not None and unit in t_unit:
                src, c = "unit", t_unit[unit]
            else:
                src, c = "prior", None
            preds = prior_top if c is None else topk_from_counter(c, max(TOP_KS))
            preds = (preds + prior_top)[: max(TOP_KS)]
            true_l = arr[cls]
            pl = [arr[p] for p in preds]
            for i, k in enumerate(TOP_KS):
                ok = true_l in pl[:k]
                hit["all"][i] += ok
                hit[src][i] += ok
            cnt[src] += 1
            cnt["all"] += 1
            if args.write:
                dump.append((u, ordinal, int(cls), true_l, src, [int(x) for x in preds]))

    print(f"\n{'='*70}\nREFERENCE-RANGE DIAGNOSTIC: {cnt['all']:,} scored test observations\n{'='*70}")
    print(f"{'backoff level':12s} {'n':>10s} {'share':>7s} " + " ".join(f"{'top'+str(k):>8s}" for k in TOP_KS))
    for lv in levels:
        if cnt[lv]:
            print(f"{lv:12s} {cnt[lv]:10,d} {100*cnt[lv]/cnt['all']:6.1f}% " +
                  " ".join(f"{100*hit[lv][i]/cnt[lv]:7.2f}%" for i in range(len(TOP_KS))))
    print(f"{'OVERALL':12s} {cnt['all']:10,d} {100.0:6.1f}% " +
          " ".join(f"{100*hit['all'][i]/cnt['all']:7.2f}%" for i in range(len(TOP_KS))))

    if args.write:
        import pandas as pd
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dump, columns=["patient_uuid", "obs_ordinal", "true_class",
                                         "true_loinc", "backoff_level", "topk_class"])
        out = Path(EVAL_BASE) / "refrange_diagnostic__mimictest"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "part-0000.parquet", index=False)
        metrics = {"n_scored": int(cnt["all"]), "n_train_patients": len(train_u),
                   "n_fit_observations": int(n_fit),
                   "refrange_constant_pct": float((n_rr == 1).mean() * 100),
                   "coverage": {lv: int(cnt[lv]) for lv in levels},
                   "topk_overall": {f"top{k}": float(hit["all"][i] / cnt["all"])
                                    for i, k in enumerate(TOP_KS)},
                   "topk_by_level": {lv: {f"top{k}": float(hit[lv][i] / cnt[lv]) if cnt[lv] else None
                                          for i, k in enumerate(TOP_KS)} for lv in levels}}
        (RESULTS_DIR / "refrange_diagnostic.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {out/'part-0000.parquet'} and {RESULTS_DIR/'refrange_diagnostic.json'}")


if __name__ == "__main__":
    main()
