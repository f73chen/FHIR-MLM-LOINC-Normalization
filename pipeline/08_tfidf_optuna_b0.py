"""Tune the TF-IDF recipe with Optuna on a held-out validation split (thesis Chapter 5 and
Appendix D).

Trials are scored on the validation split. Cross-validation inside the training sample is not used,
because it cannot measure what the recipe is chosen for. A vectorizer fitted before the folds are
split has already seen the validation fold (the vocabulary, the minimum count and the IDF weights).
The folds also draw their descriptions from the same training paraphrase pool. About 80% of the
validation-fold observations carry a description string that the training folds contain, against about
10% on the real test set, and a patient keeps one paraphrase per code across repeat tests. CV
therefore rewards memorising paraphrase-specific words, which is the lookup behaviour the recipe
should avoid.

The search:
  * search sample: the downsample rule of 07 with a 100k budget and seed 0, from para_xsite clean
    train. 679 codes x a floor of 500 exceeds the budget, so every code keeps min(n, 500), about 264k
    observations in total.
  * each trial fits the vectorizer and the classifier on the search sample only. It is scored on the
    v3 val split, whose patients and paraphrases are both unseen, as at test. The paraphrases come
    from the val pool of the 70/15/15 paraphrase split.
  * objective: micro top-1 on val (topk_hits from 07, called through its eval_on_test), with TPE
    seed 42. Trials run serially, so an uninterrupted run is reproducible. Trial 0 is the default
    recipe (tfidf_recipe.PINNED). It is enqueued so that it is scored the same way and the search
    starts from it.
  * 100 searched trials plus that default.
  * search space: C, solver, max_iter, ngram, sublinear_tf and max_df, plus the minimum count
    {1, 2, 3, 5, 10, 20} and max_features {5k ... 25k, 40k, 100k}. 100k is above every unigram
    vocabulary in this data (31,823 clean / 39,228 pert at the 2M scale), so for unigrams it means
    "no cap". The default max_features=None is enqueued as 100k, which keeps the same vocabulary.
    For 2- and 3-grams it bounds the coefficient matrix (679 x 100k), because an uncapped n-gram
    vocabulary has millions of terms. Tokenisation is fixed (tfidf_recipe.TOKEN_PATTERN, no stop
    words). It is a correctness decision and is not tuned.
  * resumable: the study lives in an SQLite file in WORK_DIR/tfidf/optuna. Re-running continues until
    101 trials are complete. A trial that was running when the process died is ignored.
The adoption decision is not made here. pipeline/08b_tfidf_adopt.py fits the default and the best
trial on the real 2M training sample and scores both on the full val split. The tuned recipe is
adopted only if it is more than 0.5 pp better.

Reads the v3 TF-IDF dataset in WORK_DIR. Writes results/08_tfidf_optuna_b0/tfidf_tune_<tag>.json and
progress_<tag>.log (aggregate metrics only).

  python pipeline/08_tfidf_optuna_b0.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import parse_condition, resolve_arm_suffix  # noqa: E402
from tfidf_recipe import PINNED, TUNABLE, make_pipeline  # noqa: E402

COMMON_DIR = str(locations.COMMON)
LABELSPACE_CSV = locations.REPO / "data/labelspace/labelspace.csv"
RESULTS_DIR = locations.REPO / "results/08_tfidf_optuna_b0"
STUDY_DIR = locations.WORK / "tfidf/optuna"
CONDITION, TRAIN_ARM = "para_xsite", "clean"   # one recipe for all four conditions
SUBSET = 100_000          # with this budget, the downsample rule of 07 keeps min(n, 500) per code
N_TRIALS = 100            # searched trials, on top of the enqueued default
DOWNSAMPLE_MIN = 500
FIT_JOBS = 24             # liblinear OvR fits across cores (tfidf_recipe.make_logreg)
LOAD_JOBS = 16
EVAL_JOBS = 8             # val scoring processes (07.eval_on_test)
TPE_SEED = 42
NO_CAP = 100_000          # stands in for max_features=None in the search (see the docstring)
SPACE = {
    "max_features": [5000, 10000, 15000, 20000, 25000, 40000, NO_CAP],
    "ngram_max": [1, 2, 3],
    "sublinear_tf": [True, False],
    "min_df": [1, 2, 3, 5, 10, 20],
    "max_df": [0.90, 0.95, 1.0],
    "solver": ["saga", "lbfgs", "liblinear"],
}
C_RANGE = (0.01, 10.0)    # log-uniform
MAX_ITER = (500, 2000, 500)

# reuse the loaders, downsample and evaluation of 07 (its name starts with a digit, so importlib)
_spec = importlib.util.spec_from_file_location(
    "tte", Path(__file__).resolve().parent / "07_tfidf_train_eval.py")
tte = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tte)


def suggest(trial) -> dict:
    """One point of the search space, as a dict of TUNABLE recipe keys."""
    p = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
    p["C"] = trial.suggest_float("C", *C_RANGE, log=True)
    p["max_iter"] = trial.suggest_int("max_iter", MAX_ITER[0], MAX_ITER[1], step=MAX_ITER[2])
    return {k: p[k] for k in TUNABLE}


def default_params() -> dict:
    """tfidf_recipe.PINNED as a point of the search space (None -> NO_CAP, same vocabulary)."""
    p = {k: PINNED[k] for k in TUNABLE}
    if p["max_features"] is None:
        assert p["ngram_max"] == 1, "NO_CAP only equals 'no cap' for unigrams"
        p["max_features"] = NO_CAP
    return p


def n_not_converged(clf, max_iter: int) -> int:
    """How many of the fitted binary problems (OvR) or the single multinomial fit stopped at max_iter."""
    ests = getattr(clf, "estimators_", [clf])
    return int(sum(int(np.max(e.n_iter_)) >= max_iter for e in ests))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="searched trials (the default is extra)")
    ap.add_argument("--subset", type=int, default=SUBSET)
    ap.add_argument("--maxfiles", type=int, default=None, help="limit train and val files (smoke)")
    ap.add_argument("--fit-jobs", type=int, default=FIT_JOBS)
    ap.add_argument("--load-jobs", type=int, default=LOAD_JOBS)
    ap.add_argument("--eval-jobs", type=int, default=EVAL_JOBS)
    ap.add_argument("--tag", default="v3", help="output/study name; anything but 'v3' is a side run")
    args = ap.parse_args()
    if args.maxfiles and args.tag == "v3":
        sys.exit("--maxfiles is a smoke setting; give it its own --tag so it cannot write the v3 study")

    # Silences ConvergenceWarning in this process only. The OvR fits run in loky workers, and their
    # warnings still reach the log, because a PYTHONWARNINGS filter cannot name a sklearn category
    # at interpreter start-up. Each trial records how many fits did not converge, in
    # `n_not_converged`.
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    sys.path.insert(0, COMMON_DIR)
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState
    from utils import extract_structured_features_single as fx
    from sklearn.preprocessing import LabelEncoder
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()
    arm = resolve_arm_suffix(CONDITION, TRAIN_ARM, "08_tfidf_optuna_b0")
    exclude, coarsen = parse_condition(CONDITION)
    data_dir = tte.resolve_data_dir(CONDITION, arm)
    primary679 = sorted({r["loinc"] for r in csv.DictReader(open(LABELSPACE_CSV))
                         if r["in_primary_ge10occ"] == "1"})
    le = LabelEncoder().fit(primary679)
    n_classes = len(le.classes_)

    Xtr, ytr = tte.load_split(data_dir, "train", fx, exclude, args.maxfiles, args.load_jobs, coarsen)
    n_train_all = len(Xtr)
    idx = tte.downsample_indices(ytr, args.subset, DOWNSAMPLE_MIN, seed=0)
    texts = [Xtr[i] for i in idx]
    y = le.transform([ytr[i] for i in idx])
    del Xtr, ytr
    Xva, yva = tte.load_split(data_dir, "val", fx, exclude, args.maxfiles, args.load_jobs, coarsen)
    y_va = le.transform(yva)               # raises if a val label escapes the 679 space
    print(f"{CONDITION}{arm}: search sample {len(texts):,} of {n_train_all:,} train obs | "
          f"val {len(Xva):,} obs | {n_classes} classes | {args.trials} trials + the default "
          f"({time.time() - t_start:.0f}s to load)")

    def score(params: dict):
        t = time.time()
        pipe = make_pipeline(params, args.fit_jobs)
        pipe.fit(texts, y)
        hits, *_ = tte.eval_on_test(pipe, Xva, y_va, n_classes, [1], tte.EVAL_BATCH, args.eval_jobs)
        return (hits[1] / len(Xva), len(pipe.named_steps["tfidf"].vocabulary_),
                n_not_converged(pipe.named_steps["clf"], params["max_iter"]), time.time() - t)

    def objective(trial):
        params = suggest(trial)
        val, vocab, nnc, sec = score(params)
        trial.set_user_attr("vocab_size", vocab)
        trial.set_user_attr("n_not_converged", nnc)
        trial.set_user_attr("sec", round(sec, 1))
        return val

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{STUDY_DIR / f'tune_{args.tag}.db'}"
    study = optuna.create_study(study_name=f"tfidf_tune_{args.tag}", storage=storage,
                                direction="maximize", sampler=TPESampler(seed=TPE_SEED),
                                load_if_exists=True)
    study.enqueue_trial(default_params(), user_attrs={"role": "default recipe"}, skip_if_exists=True)
    target = args.trials + 1
    done = sum(t.state == TrialState.COMPLETE for t in study.trials)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prog = RESULTS_DIR / f"progress_{args.tag}.log"
    with open(prog, "a") as pf:
        pf.write(f"[{time.strftime('%F %T')}] start: {done}/{target} trials complete | search "
                 f"{len(texts):,} obs | val {len(Xva):,} obs\n")

    def _cb(_st, tr):  # logging only: per-trial progress for a poller
        with open(prog, "a") as pf:
            pf.write(f"[{time.strftime('%F %T')}] trial {tr.number}: value={tr.value} "
                     f"state={tr.state.name} {tr.user_attrs} {tr.params}\n")

    if target - done > 0:
        # MemoryError is caught. That trial is marked FAIL and the study goes on. Any other error
        # stops the run, because it is more likely a bug than a property of the hyperparameters.
        study.optimize(objective, n_trials=target - done, callbacks=[_cb], catch=(MemoryError,))

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    defaults = [t for t in complete if t.user_attrs.get("role") == "default recipe"]
    if not defaults:
        sys.exit("the default recipe's trial is not complete, so there is nothing to compare. "
                 "Re-run to finish it.")
    best, dflt = study.best_trial, defaults[0]
    out = {
        "condition": CONDITION, "train_arm": TRAIN_ARM, "tag": args.tag,
        "objective": "micro top-1 on the v3 val split (unseen patients and unseen paraphrases)",
        "search_obs": len(texts), "train_obs_total": n_train_all, "val_obs": len(Xva),
        "n_classes": n_classes, "trials_searched_target": args.trials, "tpe_seed": TPE_SEED,
        "n_complete": len(complete),
        "n_failed": sum(t.state == TrialState.FAIL for t in study.trials),
        "tokenizer": {k: PINNED[k] for k in ("token_pattern", "stop_words", "lowercase")},
        "space": {**SPACE, "C": {"log_uniform": list(C_RANGE)}, "max_iter": list(MAX_ITER)},
        "no_cap_stands_for_none": NO_CAP,
        "default_params": dflt.params, "default_trial": dflt.number,
        "default_val_top1": dflt.value, "default_user_attrs": dflt.user_attrs,
        "best_trial": best.number, "best_params": best.params, "best_val_top1": best.value,
        "best_user_attrs": best.user_attrs,
        "best_minus_default": best.value - dflt.value,
        "best_is_default": best.number == dflt.number,
        "trials": [{"number": t.number, "params": t.params, "value": t.value, **t.user_attrs}
                   for t in complete],
        "note": "adoption is decided by 08b at the 2M scale, not here",
        "wall_sec_this_invocation": round(time.time() - t_start, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = RESULTS_DIR / f"tfidf_tune_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\ndefault (trial {dflt.number}) val top-1 = {dflt.value:.5f} | best (trial {best.number}) "
          f"= {best.value:.5f} | Δ = {100 * out['best_minus_default']:+.3f} pp")
    print(f"best params: {best.params}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
