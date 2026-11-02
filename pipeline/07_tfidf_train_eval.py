"""Train and evaluate the TF-IDF baseline for one condition over 5 downsample seeds (thesis Chapter 5).

For each seed:
  - The condition picks the fields. "para" keeps all of them and "nodesc" excludes the description.
    Both use the same dataset via utils.extract_structured_features_single(exclude_fields=...).
  - Train is threshold-downsampled to the 2M budget with the rule of
    utils.load_and_prepare_data_single_threshold, but through an explicit seeded RNG, so the 5 seeds
    give a real variance estimate.
  - The pinned recipe from pipeline/tfidf_recipe.py is fitted. It has a tokenizer built for lab text,
    no stop words, no top-N cap and liblinear one-vs-rest with C=6.59, and a term must occur in >= 5
    training observations.
  - The model is evaluated on the full natural test split (all observations, real frequencies) with
    batched predict_proba. The metrics are micro top-1/3/5/10 over the 679-code label space, macro
    top-1 recall and F1 over the 491-code core, and ECE and Brier calibration.
With --eval-sets or --eval-arm both, each fitted v3 model is also scored on other test sets: the
split of the sibling train arm, the perturbation replicates, severities and ablations.

The class space is the fixed primary label space (LabelEncoder fit on it). Reads the TF-IDF dataset in
WORK_DIR. Writes the 5-seed mean and std to results/07_tfidf_train_eval/tfidf_<condition><arm>.json
(aggregate metrics only).

  python pipeline/07_tfidf_train_eval.py --condition para_xsite --train-arm pert --eval-arm both
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (CONDITIONS, dataset_suffix, has_description,  # noqa: E402
                        parse_condition, refrange_mode, resolve_arm_suffix,
                        resolve_test_arm, representation)
# the single TF-IDF recipe. make_logreg lives there and is re-exported here for 08 and the tests.
from tfidf_recipe import PINNED, make_logreg, make_pipeline  # noqa: E402,F401

COMMON_DIR = str(locations.COMMON)
#: The dataset depends on the representation (v1, v2, v3). The suffix comes from the condition slug,
#: so a v2 arm cannot be trained on v1 text by forgetting a flag.
DATA_DIR_BASE = str(locations.WORK / "tfidf/dataset")
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
RESULTS_DIR = locations.REPO / "results/07_tfidf_train_eval"

SUBSET_SIZE = 2_000_000
DOWNSAMPLE_MIN = 500
SEEDS = [0, 1, 2, 3, 4]
EVAL_BATCH = 50_000
EVAL_JOBS = 1          # default sequential; pass --eval-jobs N to parallelize the test eval
FIT_JOBS = 1           # parallelize liblinear's 679 OvR binary fits across cores (~7.6x at 2M)
LOAD_JOBS = 1          # parallelize the per-file dataset load (json.loads + feature extraction)
TOPK = [1, 3, 5, 10]


# --- Pure, unit-testable logic ----------------------------------------------
def threshold_targets(counts: dict, subset_size: int, downsample_min: int) -> dict:
    """Per-class sample targets, as in utils.load_and_prepare_data_single_threshold. Classes
    with <= `downsample_min` samples keep all of them. Larger ones are scaled uniformly to the
    remaining budget, with a floor of `downsample_min`."""
    take_all = {c: n for c, n in counts.items() if n <= downsample_min}
    big = {c: n for c, n in counts.items() if n > downsample_min}
    remaining = subset_size - sum(take_all.values())
    total_big = sum(big.values())
    target = dict(take_all)
    for c, n in big.items():
        if total_big > 0:
            target[c] = min(max(int((n / total_big) * remaining), downsample_min), n)
        else:
            target[c] = n
    return target


def downsample_indices(labels: list, subset_size: int, downsample_min: int, seed: int) -> list:
    """Stratified threshold-downsample (as in utils.load_and_prepare_data_single_threshold) with an
    explicit seeded RNG."""
    counts = Counter(labels)
    target = threshold_targets(counts, subset_size, downsample_min)
    by_label = defaultdict(list)
    for i, l in enumerate(labels):
        by_label[l].append(i)
    rng = random.Random(seed)
    out = []
    for l, idxs in by_label.items():
        t = target.get(l, len(idxs))
        out.extend(idxs if len(idxs) <= t else rng.sample(idxs, t))
    rng.shuffle(out)
    return out


def topk_hits(proba: np.ndarray, y: np.ndarray, ks: list) -> dict:
    """Per-batch count of obs whose true class is within top-k of proba (micro numerator).
    Rank = #classes with strictly greater proba. Ties favor the true class, but the effect is
    negligible for continuous proba. A top-k hit means rank < k."""
    true_p = proba[np.arange(len(y)), y]
    n_greater = (proba > true_p[:, None]).sum(axis=1)
    return {k: int((n_greater < k).sum()) for k in ks}


def brier_sum(proba: np.ndarray, y: np.ndarray) -> float:
    """Sum over obs of the multiclass Brier score sum_c (p_c - onehot_c)^2 (= ||p||^2 - 2 p_y + 1)."""
    return float((np.sum(proba * proba, axis=1) - 2 * proba[np.arange(len(y)), y] + 1.0).sum())


def ece_from_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error on max-prob confidence (10 equal-width bins)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b > 0 else (conf >= edges[0]) & (conf <= edges[1])
        if m.sum():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _eval_batch_metrics(proba_m, clf_classes, n_full, yb, topk):
    """Map model-proba (model-class cols) to the full n_full space, return
    (hits_dict, y_pred, conf, partial_brier_sum) for this batch."""
    proba = np.zeros((proba_m.shape[0], n_full), dtype=np.float64)
    proba[:, clf_classes] = proba_m
    return topk_hits(proba, yb, topk), proba.argmax(1), proba.max(1), brier_sum(proba, yb)


def _eval_chunk(pipe, texts, y_chunk, clf_classes, n_full, topk, eval_batch):
    """Worker (picklable, module-level): evaluate one contiguous test chunk in sub-batches."""
    m = len(texts)
    hits = {k: 0 for k in topk}
    y_pred = np.empty(m, dtype=np.int64)
    conf = np.empty(m, dtype=np.float64)
    bsum = 0.0
    for s in range(0, m, eval_batch):
        sl = slice(s, min(s + eval_batch, m))
        h, yp, cf, bs = _eval_batch_metrics(pipe.predict_proba(texts[sl]), clf_classes, n_full, y_chunk[sl], topk)
        for k in topk:
            hits[k] += h[k]
        y_pred[sl] = yp
        conf[sl] = cf
        bsum += bs
    return hits, y_pred, conf, bsum


def eval_on_test(pipe, Xte, yte_enc, n_full, topk, eval_batch, eval_jobs, progress=None):
    """Natural-test eval accumulating micro hits, argmax preds, max-prob confidence, Brier sum.
    With eval_jobs == 1 the eval is sequential (exact). With eval_jobs > 1 the test is split into
    that many contiguous chunks, and predict_proba runs on them across processes. The results match
    the sequential path up to the floating-point summation order of the Brier sum (~1e-9).
    predict_proba does not mutate X, so there is no loky read-only-memmap issue as there is with
    the liblinear fit. Returns (hits, y_pred, conf, bsum)."""
    n = len(Xte)
    clf_classes = pipe.named_steps["clf"].classes_  # le-indices present in train
    hits = {k: 0 for k in topk}
    y_pred = np.empty(n, dtype=np.int64)
    conf = np.empty(n, dtype=np.float64)
    bsum = 0.0
    if eval_jobs == 1:
        nb = (n + eval_batch - 1) // eval_batch
        for bi, s in enumerate(range(0, n, eval_batch)):
            sl = slice(s, min(s + eval_batch, n))
            h, yp, cf, bs = _eval_batch_metrics(pipe.predict_proba(Xte[sl]), clf_classes, n_full, yte_enc[sl], topk)
            for k in topk:
                hits[k] += h[k]
            y_pred[sl] = yp
            conf[sl] = cf
            bsum += bs
            if progress and (bi % 10 == 0 or bi == nb - 1):
                progress(f"eval batch {bi + 1}/{nb}")
    else:
        from joblib import Parallel, delayed
        chunks = [c for c in np.array_split(np.arange(n), eval_jobs) if len(c)]
        if progress:
            progress(f"eval start: {n:,} obs over {len(chunks)} chunks x {eval_jobs} jobs")
        results = Parallel(n_jobs=eval_jobs, backend="loky")(
            delayed(_eval_chunk)(pipe, [Xte[i] for i in idxs], yte_enc[idxs], clf_classes, n_full, topk, eval_batch)
            for idxs in chunks)
        for idxs, (h, yp, cf, bs) in zip(chunks, results):
            for k in topk:
                hits[k] += h[k]
            y_pred[idxs] = yp
            conf[idxs] = cf
            bsum += bs
        if progress:
            progress("eval done (all chunks merged)")
    return hits, y_pred, conf, bsum


# --- Real-data IO (main only) -----------------------------------------------
def resolve_data_dir(condition: str, arm: str, data_dir: str | None = None) -> str:
    """The TF-IDF dataset for (condition, train arm), refusing a representation or train-arm
    mismatch. Shared with 08 and 08b, so every script that reads TF-IDF text applies the same checks."""
    rep = representation(condition)
    d = data_dir or (DATA_DIR_BASE + dataset_suffix(condition) + arm)
    if not Path(d).exists():
        sys.exit(f"dataset for representation {rep} not found at {d}\n"
                 f"build it first: python pipeline/06_tfidf_dataset.py --representation {rep}")
    _mf = Path(d) / "manifest.json"
    if _mf.exists():
        _m = json.loads(_mf.read_text())
        _got = _m.get("representation", "v1")
        if _got != rep:
            sys.exit(f"representation mismatch: condition {condition!r} needs {rep}, but "
                     f"{d} was built as {_got!r}.")
        # the same check for the train arm. The two v3 arms share the representation and the
        # condition and differ only in whether the training data was perturbed.
        _want_arm = arm.lstrip("_") or None
        if _m.get("train_arm") != _want_arm:
            sys.exit(f"train-arm mismatch: asked for {_want_arm!r}, but {d} was built as "
                     f"{_m.get('train_arm')!r}.")
    return d


def _load_files(fps, extract_fn, exclude, coarsen=None):
    """Picklable worker: read + feature-extract a list of patient JSONL files.

    `coarsen` is the reference-range precision setting (conditions.py). None means no
    coarsening."""
    texts, labels = [], []
    for fp in fps:
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                texts.append(extract_fn(d["prompt"], exclude, coarsen))
                labels.append(d["label"])
    return texts, labels


def load_split(data_dir, split, extract_fn, exclude, max_files=None, jobs=1, coarsen=None):
    """Load (prompt-features, label) for a split. With jobs > 1, files are read in that many
    parallel chunks and concatenated in file order, so the result is bit-identical to the
    sequential path."""
    files = sorted(glob.glob(f"{data_dir}/{split}/samples/*.jsonl"))
    if max_files:
        files = files[:max_files]
    if jobs == 1 or len(files) <= 1:
        return _load_files(files, extract_fn, exclude, coarsen)
    from joblib import Parallel, delayed
    chunks = [list(c) for c in np.array_split(np.array(files, dtype=object), min(jobs, len(files))) if len(c)]
    texts, labels = [], []
    for t, l in Parallel(n_jobs=jobs, backend="loky")(
            delayed(_load_files)(ch, extract_fn, exclude, coarsen) for ch in chunks):
        texts.extend(t)
        labels.extend(l)
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True,
                    help="see pipeline/conditions.py for the slug grammar")
    ap.add_argument("--eval-sets", default=None,
                    help="v3 only: comma-separated eval sets to score each fitted model on, in "
                         "addition to its home split, e.g. "
                         "'clean,pert_r1,pert_r2,pert_sev0.5,pert_value'. The replicates, the "
                         "severity band and the ablation then all come from one fit. "
                         "Overrides --eval-arm.")
    ap.add_argument("--eval-arm", choices=["home", "both"], default="home",
                    help="v3 only. 'home' evaluates on the training arm's own test split (only "
                         "the diagonal of the 2x2). 'both' also evaluates on the other arm's test "
                         "split, which gives cells B and C.")
    ap.add_argument("--train-arm", choices=["pert", "clean"], default=None,
                    help="v3 (xsite) only, and required there: which training arm. 'pert' = "
                         "trained on perturbed data, 'clean' = the v3 representation with no "
                         "perturbation. No default.")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny subset budget")
    ap.add_argument("--eval-jobs", type=int, default=EVAL_JOBS,
                    help="parallel processes for the natural-test eval (1=sequential, exact)")
    ap.add_argument("--fit-jobs", type=int, default=FIT_JOBS, help="parallel jobs for the liblinear OvR fit")
    ap.add_argument("--load-jobs", type=int, default=LOAD_JOBS, help="parallel jobs for the dataset load")
    ap.add_argument("--data-dir", default=None,
                    help="override the dataset directory (default: derived from the condition's "
                         "representation, so a v2 condition reads the v2 dataset)")
    args = ap.parse_args()

    arm = resolve_arm_suffix(args.condition, args.train_arm, "07_tfidf_train_eval")
    sys.path.insert(0, COMMON_DIR)
    from utils import extract_structured_features_single
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import precision_recall_fscore_support

    exclude, coarsen = parse_condition(args.condition)
    rep = representation(args.condition)
    DATA_DIR = resolve_data_dir(args.condition, arm, args.data_dir)
    print(f"  representation {rep} | dataset {DATA_DIR}")
    core491 = {r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV)) if r["in_macro_core_ge20test"] == "1"}
    primary679 = {r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV)) if r["in_primary_ge10occ"] == "1"}

    print(f"loading dataset ({args.condition}: exclude={sorted(exclude) if exclude else None}, "
          f"coarsen_reference_range={coarsen!r})...")
    mf = 800 if args.smoke else None
    Xtr, ytr = load_split(DATA_DIR, "train", extract_structured_features_single, exclude, mf,
                          args.load_jobs, coarsen)
    Xte, yte = load_split(DATA_DIR, "test", extract_structured_features_single, exclude, mf,
                          args.load_jobs, coarsen)
    # -- Cross-arm test split (the 2x2 of train arm by test arm) ------------------
    # Without this a run gives only the diagonal: train-clean/test-clean (A) and
    # train-pert/test-pert (D). The main comparisons all need the off-diagonal cells:
    #   B against A: is the shortcut brittle?
    #   D against B: does perturbation training fix it?
    #   C against A: does it cost anything at home?
    # None of them can be computed from A and D alone.
    #
    # The test splits of the two arms are the same observations of the same patients in the same
    # order. They differ only in the perturbed numbers, so the comparison is paired.
    # An evaluation takes far less time than the fit, so every model is scored on the
    # perturbation replicates, a severity band and the value/range ablation, all from one fit.
    eval_names = []
    if arm and args.eval_sets:
        eval_names = [n.strip() for n in args.eval_sets.split(",") if n.strip()]
    elif arm and args.eval_arm == "both":
        eval_names = ["clean" if arm == "__pert" else "pert"]
    extra = {}                                   # name -> (X, y_encoded placeholder)
    for nm in eval_names:
        suffix, _tag = resolve_test_arm(args.condition, arm, nm, "07_tfidf_train_eval")
        if suffix == arm:
            continue                             # that is the home split, already loaded
        d = DATA_DIR_BASE + dataset_suffix(args.condition) + suffix
        if not Path(d).exists():
            sys.exit(f"eval set {nm!r} not found at {d}. Build it first.")
        Xn, yn = load_split(d, "test", extract_structured_features_single, exclude, mf,
                            args.load_jobs, coarsen)
        extra[nm] = (Xn, yn)
        print(f"  eval set {nm:<14} {len(Xn):>10,} obs   {d}")
    print(f"  train obs={len(Xtr):,}  test obs={len(Xte):,}  train classes={len(set(ytr))}")
    assert set(ytr) <= primary679, "train labels escape the 679 space"

    le = LabelEncoder().fit(sorted(primary679))
    n_full = len(le.classes_)
    yte_enc = le.transform(yte)
    # every eval set must be the same observations in the same order as the home split, or the
    # comparison stops being paired. The asserts below check this.
    extra_enc = {}
    for nm, (Xn, yn) in extra.items():
        assert len(Xn) == len(Xte), f"{nm}: {len(Xn)} rows against the home split's {len(Xte)}"
        e = le.transform(yn)
        assert (e == yte_enc).all(), f"{nm}: labels differ from the home split, so the comparison is not paired"
        extra_enc[nm] = (Xn, e)
    core_ids = sorted(le.transform(sorted(core491)))
    budget = 50_000 if args.smoke else SUBSET_SIZE
    seeds = args.seeds[:1] if args.smoke else args.seeds

    # Sidecar progress log, for logging only (it does not affect metrics). conda run buffers
    # stdout until exit, so this lets a poller see per-seed progress mid-run.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prog = RESULTS_DIR / f"progress_{args.condition}{arm}.log"
    prog.write_text(f"{args.condition}: train={len(Xtr):,} test={len(Xte):,}; {len(seeds)} seeds; budget={budget:,}\n")

    per_seed = []
    for seed in seeds:
        idx = downsample_indices(ytr, budget, DOWNSAMPLE_MIN, seed)
        Xs = [Xtr[i] for i in idx]
        ys = le.transform([ytr[i] for i in idx])
        pipe = make_pipeline(PINNED, args.fit_jobs)
        pipe.fit(Xs, ys)

        # Natural-test eval, parallel across cores when --eval-jobs > 1 (same results).
        n = len(Xte)
        def _seedprog(msg):
            with open(prog, "a") as _pf:
                _pf.write(f"seed {seed}: {msg}\n")
        hits, y_pred, conf, bsum = eval_on_test(pipe, Xte, yte_enc, n_full, TOPK, EVAL_BATCH,
                                                args.eval_jobs, progress=_seedprog)

        micro = {f"top{k}": hits[k] / n for k in TOPK}
        correct = (y_pred == yte_enc).astype(np.float64)
        # macro top-1 over the 491 core (mean of per-class recall, core classes only)
        p, r, f1, sup = precision_recall_fscore_support(yte_enc, y_pred, labels=sorted(core_ids),
                                                        average=None, zero_division=0)
        macro = {"top1_recall_core491": float(np.mean(r)), "macro_f1_core491": float(np.mean(f1))}
        cal = {"ece": ece_from_bins(conf, correct), "brier": bsum / n}
        rec = {"seed": seed, "micro": micro, "macro": macro, "calibration": cal,
               "train_obs": len(Xs), "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_)}
        for nm, (Xn, en) in extra_enc.items():
            h2, yp2, cf2, bs2 = eval_on_test(pipe, Xn, en, n_full, TOPK, EVAL_BATCH,
                                             args.eval_jobs, progress=None)
            m2 = {f"top{k}": h2[k] / n for k in TOPK}
            c2 = (yp2 == en).astype(np.float64)
            _p2, r2, f2, _s2 = precision_recall_fscore_support(
                en, yp2, labels=sorted(core_ids), average=None, zero_division=0)
            rec.setdefault("eval_sets", {})[nm] = {
                "micro": m2,
                "macro": {"top1_recall_core491": float(np.mean(r2)),
                          "macro_f1_core491": float(np.mean(f2))},
                "calibration": {"ece": ece_from_bins(cf2, c2), "brier": bs2 / n}}
            print(f"  seed {seed}: on[{nm}] micro_top1={m2['top1']:.4f} "
                  f"macro_core491={np.mean(r2):.4f}")
            with open(prog, "a") as _pf:
                _pf.write(f"seed {seed} on[{nm}]: micro_top1={m2['top1']:.4f} "
                          f"macro_core491={np.mean(r2):.4f}\n")
        per_seed.append(rec)
        print(f"  seed {seed}: micro_top1={micro['top1']:.4f} top5={micro['top5']:.4f} "
              f"macro_core491={macro['top1_recall_core491']:.4f} ece={cal['ece']:.4f}")
        with open(prog, "a") as _pf:
            _pf.write(f"seed {seed} done: micro_top1={micro['top1']:.4f} top5={micro['top5']:.4f} "
                      f"macro_core491={macro['top1_recall_core491']:.4f} ece={cal['ece']:.4f}\n")

    def agg(path):
        vals = [s for sd in per_seed for s in [_get(sd, path)]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "condition": args.condition, "n_seeds": len(seeds), "test_obs": len(Xte),
        "has_description": has_description(args.condition),
        "refrange_mode": refrange_mode(args.condition),
        # numbers from different representations are not comparable
        "representation": rep, "dataset_dir": str(DATA_DIR),
        "exclude_fields": sorted(exclude) if exclude else None,
        "coarsen_reference_range": coarsen,
        "micro": {f"top{k}": agg(f"micro.top{k}") for k in TOPK},
        "macro_core491": {"top1_recall": agg("macro.top1_recall_core491"), "f1": agg("macro.macro_f1_core491")},
        "calibration": {"ece": agg("calibration.ece"), "brier": agg("calibration.brier")},
        # the eval sets aggregated across seeds, so the summary alone gives the replicate,
        # severity and ablation results without re-deriving them from per_seed. `train_arm` is
        # recorded so that each number can be traced to the model that produced it.
        "train_arm": arm.lstrip("_") or None,
        "eval_sets": {
            nm: {"micro": {f"top{k}": {
                     "mean": float(np.mean([sd["eval_sets"][nm]["micro"][f"top{k}"]
                                            for sd in per_seed])),
                     "std": float(np.std([sd["eval_sets"][nm]["micro"][f"top{k}"]
                                          for sd in per_seed]))} for k in TOPK},
                 "macro_core491_top1_recall": {
                     "mean": float(np.mean([sd["eval_sets"][nm]["macro"]["top1_recall_core491"]
                                            for sd in per_seed])),
                     "std": float(np.std([sd["eval_sets"][nm]["macro"]["top1_recall_core491"]
                                          for sd in per_seed]))},
                 "ece_mean": float(np.mean([sd["eval_sets"][nm]["calibration"]["ece"]
                                            for sd in per_seed]))}
            for nm in (per_seed[0].get("eval_sets") or {})},
        "per_seed": per_seed, "pinned_params": PINNED, "label_space": len(primary679), "core": len(core491),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"tfidf_{args.condition}{arm}{'_smoke' if args.smoke else ''}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nmicro top1 = {summary['micro']['top1']['mean']:.4f} ± {summary['micro']['top1']['std']:.4f}")
    print(f"wrote {out}")


def _get(d, dotted):
    for p in dotted.split("."):
        d = d[p]
    return d


if __name__ == "__main__":
    main()
