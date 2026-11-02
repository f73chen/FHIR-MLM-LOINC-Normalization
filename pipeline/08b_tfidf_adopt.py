"""Decide whether the best trial of pipeline/08_tfidf_optuna_b0.py replaces the default TF-IDF recipe
(thesis Chapter 5 and Appendix D).

The default recipe (tfidf_recipe.PINNED) and the best trial of 08 are both fitted on the real training
sample, which is the 2M downsample of 07 (seed 0, from para_xsite clean train). Both are scored on the
full v3 val split (unseen patients and paraphrases). The tuned recipe is adopted only if its micro
top-1 is more than 0.5 pp higher. One recipe then serves all four conditions. Deciding at full scale
removes a caveat about the search sample. 08 searches on a sample of about 264k observations where
every code is capped at 500, so a vocabulary rule can behave differently in the 2M fit. Picking the
best of 101 trials on the same val split inflates its val score by about the noise level (about
0.1 pp). This is well below the 0.5 pp margin, so val can be re-used here.

Applying the decision is an edit to tfidf_recipe.PINNED. It is a reviewed change, and this script
only prints it. `--check` is the gate before the TF-IDF runs. It exits non-zero unless PINNED matches
the recorded decision.

Reads the v3 TF-IDF dataset in WORK_DIR and the study summary from 08. Writes
results/08_tfidf_optuna_b0/tfidf_adopt_<tag>.json (aggregate metrics only).

  python pipeline/08b_tfidf_adopt.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import parse_condition, resolve_arm_suffix  # noqa: E402
from tfidf_recipe import PINNED, TUNABLE, make_pipeline  # noqa: E402

COMMON_DIR = str(locations.COMMON)
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
RESULTS_DIR = locations.REPO / "results/08_tfidf_optuna_b0"
CONDITION, TRAIN_ARM = "para_xsite", "clean"
MARGIN = 0.005            # adopt only if the tuned recipe is > 0.5 pp better
FIT_JOBS = 24
LOAD_JOBS = 16
EVAL_JOBS = 8

_spec = importlib.util.spec_from_file_location(
    "tte", Path(__file__).resolve().parent / "07_tfidf_train_eval.py")
tte = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tte)
_spec8 = importlib.util.spec_from_file_location(
    "tune08", Path(__file__).resolve().parent / "08_tfidf_optuna_b0.py")
tune08 = importlib.util.module_from_spec(_spec8)
_spec8.loader.exec_module(tune08)


def as_recipe(params: dict) -> dict:
    """Convert a search point to recipe keys. NO_CAP in 08 stands for max_features=None (unigrams only)."""
    p = {k: params[k] for k in TUNABLE}
    if p["max_features"] == tune08.NO_CAP and p["ngram_max"] == 1:
        p["max_features"] = None
    return p


def decide(default_top1: float, best_top1: float | None, margin: float = MARGIN) -> bool:
    """True iff the tuned recipe is adopted: strictly more than `margin` better than the default.
    Rounded to 1e-12 so that a difference of exactly 0.5 pp (e.g. 0.9550 - 0.9500, which is
    0.0050000000000000044 in floating point) is not adopted by rounding error."""
    return best_top1 is not None and round(best_top1 - default_top1, 12) > margin


def check(adopt_json: Path) -> int:
    """Gate: return 0 iff tfidf_recipe.PINNED matches the recorded decision."""
    if not adopt_json.exists():
        print(f"GATE FAIL: {adopt_json} not found. Run 08b first.")
        return 1
    rec = json.loads(adopt_json.read_text())
    want = rec["adopted_recipe"]
    got = {k: PINNED[k] for k in TUNABLE}
    if got != want:
        diff = {k: (got[k], want[k]) for k in TUNABLE if got[k] != want[k]}
        print(f"GATE FAIL: tfidf_recipe.PINNED disagrees with {adopt_json.name} "
              f"(adopt={rec['adopt']}) on (current, decided): {diff}. Apply the decision first.")
        return 1
    print(f"GATE OK: tfidf_recipe.PINNED matches the decision in {adopt_json.name} (adopt={rec['adopt']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3", help="which 08 study to read and which output to write")
    ap.add_argument("--maxfiles", type=int, default=None, help="limit train and val files (smoke)")
    ap.add_argument("--subset", type=int, default=tte.SUBSET_SIZE)
    ap.add_argument("--fit-jobs", type=int, default=FIT_JOBS)
    ap.add_argument("--load-jobs", type=int, default=LOAD_JOBS)
    ap.add_argument("--eval-jobs", type=int, default=EVAL_JOBS)
    ap.add_argument("--check", action="store_true", help="only verify PINNED against the decision")
    args = ap.parse_args()
    out_path = RESULTS_DIR / f"tfidf_adopt_{args.tag}.json"
    if args.check:
        return check(out_path)
    if args.maxfiles and args.tag == "v3":
        sys.exit("--maxfiles is a smoke setting; give it its own --tag so it cannot write the v3 decision")

    tune = json.loads((RESULTS_DIR / f"tfidf_tune_{args.tag}.json").read_text())
    default = {k: PINNED[k] for k in TUNABLE}
    if as_recipe(tune["default_params"]) != default:
        sys.exit(f"08's default trial {tune['default_params']} is not the current PINNED {default}, "
                 "so the tuning ran against a different default. Re-run 08.")
    best = as_recipe(tune["best_params"])
    print(f"default: {default}\nbest (08 trial {tune['best_trial']}, search val top-1 "
          f"{tune['best_val_top1']:.5f} vs default {tune['default_val_top1']:.5f}): {best}")

    sys.path.insert(0, COMMON_DIR)
    from utils import extract_structured_features_single as fx
    from sklearn.preprocessing import LabelEncoder

    t0 = time.time()
    arm = resolve_arm_suffix(CONDITION, TRAIN_ARM, "08b_tfidf_adopt")
    exclude, coarsen = parse_condition(CONDITION)
    data_dir = tte.resolve_data_dir(CONDITION, arm)
    primary679 = sorted({r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV))
                         if r["in_primary_ge10occ"] == "1"})
    core491 = {r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV)) if r["in_macro_core_ge20test"] == "1"}
    le = LabelEncoder().fit(primary679)
    Xtr, ytr = tte.load_split(data_dir, "train", fx, exclude, args.maxfiles, args.load_jobs, coarsen)
    idx = tte.downsample_indices(ytr, args.subset, tte.DOWNSAMPLE_MIN, seed=0)   # 07's seed-0 sample
    Xs, ys = [Xtr[i] for i in idx], le.transform([ytr[i] for i in idx])
    del Xtr, ytr
    Xva, yva = tte.load_split(data_dir, "val", fx, exclude, args.maxfiles, args.load_jobs, coarsen)
    y_va = le.transform(yva)
    core_ids = sorted(le.transform(sorted(core491)))
    print(f"train sample {len(Xs):,} | val {len(Xva):,} | loaded in {time.time() - t0:.0f}s")

    from sklearn.metrics import recall_score

    def run(name, params):
        t = time.time()
        pipe = make_pipeline(params, args.fit_jobs)
        pipe.fit(Xs, ys)
        fit_sec = time.time() - t
        hits, y_pred, _c, _b = tte.eval_on_test(pipe, Xva, y_va, len(le.classes_), [1, 5],
                                                tte.EVAL_BATCH, args.eval_jobs)
        r = {"params": params, "val_top1": hits[1] / len(Xva), "val_top5": hits[5] / len(Xva),
             "val_macro_top1_core491": float(recall_score(y_va, y_pred, labels=core_ids,
                                                          average="macro", zero_division=0)),
             "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
             "n_not_converged": tune08.n_not_converged(pipe.named_steps["clf"], params["max_iter"]),
             "fit_sec": round(fit_sec, 1), "total_sec": round(time.time() - t, 1)}
        print(f"  {name}: val top-1 {r['val_top1']:.5f} | macro core491 {r['val_macro_top1_core491']:.5f} "
              f"| vocab {r['vocab_size']:,} | fit {fit_sec:.0f}s")
        return r

    res_default = run("default", default)
    same = best == default
    res_best = res_default if same else run("best", best)
    delta = res_best["val_top1"] - res_default["val_top1"]
    adopt = (not same) and decide(res_default["val_top1"], res_best["val_top1"])
    out = {
        "condition": CONDITION, "train_arm": TRAIN_ARM, "tag": args.tag,
        "rule": f"adopt the tuned recipe only if its full-scale val micro top-1 is > {MARGIN} above the default's",
        "train_obs": len(Xs), "val_obs": len(Xva),
        "default": res_default, "best": res_best, "best_is_default": same,
        "best_trial_08": tune["best_trial"], "delta_top1": delta, "adopt": adopt,
        "adopted_recipe": best if adopt else default,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nΔ val top-1 (best − default) = {100 * delta:+.3f} pp -> "
          f"{'adopt the tuned recipe' if adopt else 'keep the default recipe'}")
    if adopt:
        print("Apply by setting these keys of tfidf_recipe.PINNED, then commit:")
        print(json.dumps(best, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
