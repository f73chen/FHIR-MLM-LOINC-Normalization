#!/usr/bin/env python
"""
T-25 on v3: what each model costs to train and to run (thesis Chapter 6).

  --stage collect   training cost from the v3 training manifests (deep models) and the manifests of
                    the persisted pipelines (TF-IDF). Inference throughput, batch size, memory and
                    parameters from the cell-D eval manifests. The cost of dumping full logits, and
                    the TF-IDF tuning search
  --stage tfidf     time TF-IDF end to end on CPU over held-out test prompts (feature extraction, then
                    predict_proba in 16's batches), since its manifests carry no timing. This is a
                    benchmark, so run it while the machine is otherwise quiet
  --stage report    T-25's rows from the outputs of the two stages above

Units and caveats:
  * Throughput is observations per second. The manifests call them "positions", and each is one scored
    observation.
  * Every deep cell-D eval dumped full logits. FHIR-MLM's perturbation-variant cells ran on the same
    positions without logits, so the dumping cost is measured there.
  * Batch sizes differ (FHIR-MLM 2, BCM 8, Llama 2), so throughput is not a controlled benchmark. Batch
    size and max length are reported beside it. Each eval process uses one GPU.
  * Training: both arms are recorded, and T-25 uses the pert arm (the model of the headline cell). Only
    full runs of the registry's seeds count, and smoke runs are excluded. Wall clock includes the
    periodic validation evals. Every deep model trained on 2 GPUs (torchrun --nproc_per_node=2).
    For TF-IDF, training is the 2M-sample fit of the persisted pipeline (`fit_sec`, 8 jobs). Its one-off
    hyperparameter search (pipeline/08) is reported separately.
  * Parameters: T-25 reuses the counts of T-08 (ch5_model_sizes_table.py), so the thesis gives each
    model one size. Llama's eval manifest counts 4-bit packed weights, and T-08 derives base + head +
    LoRA adapters. TF-IDF's count is taken from T-08 (clean arm, seed 0). The pert model's vocabulary is
    larger, so its count is recorded beside the benchmark.

Writes aggregate CSVs to results/analysis/v3/cost/.

  python analysis/v3_cost.py --stage collect
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import cells as CL  # noqa: E402
from lib import registry as R  # noqa: E402

ANALYSIS = "cost"
_T8 = importlib.util.spec_from_file_location("t08", HERE / "ch5_model_sizes_table.py")
T08 = importlib.util.module_from_spec(_T8); _T8.loader.exec_module(T08)
BASE = CL.EVAL_BASE.parent
RESULTS = HERE.parent / "results"
TRAIN = {"mlm": RESULTS / "12_mlm_train", "bcm": RESULTS / "13_bcm_train", "llama": RESULTS / "14_llama_train"}
TFIDF_MODEL = BASE / "models" / "tfidf__para_xsite__pert__seed{}"
TFIDF_TEST = BASE / "tfidf" / "dataset_v3__pert" / "test" / "samples"
OPTUNA_DB = BASE / "tfidf" / "optuna" / "tune_v3.db"
TRAIN_GPUS = 2                      # 12, 13 and 14 ran under torchrun --nproc_per_node=2
FIT_JOBS = 8                        # 16_tfidf_eval_dump.py's --fit-jobs default
EVAL_BATCH = 50_000                 # 16_tfidf_eval_dump.EVAL_BATCH
N_BENCH, N_REPEAT = 200_000, 3


def cellD(m, s):
    return R.cell(m, s, condition="para_xsite", train_arm="pert", test_arm="pert")


def manifest(c) -> dict:
    return json.loads((c.path / "manifest.json").read_text())


# -- collect -----------------------------------------------------------------
def training_rows() -> pd.DataFrame:
    rows = []
    for m, d in TRAIN.items():
        for arm in ("clean", "pert"):
            for s in R.SEEDS[m]:
                f = d / f"manifest_para_xsite__{arm}_seed{s}.json"
                if not f.exists():
                    continue
                j = json.loads(f.read_text())
                if j.get("mode", "full") != "full":
                    continue
                rows.append({"model": R.LABEL[m], "arm": arm, "seed": s, "hours": j["wall_clock_h"],
                             "sec_per_step": j["sec_per_step"], "steps": j["global_step"],
                             "best_step": j["best_step"], "early_stopped": j["early_stopped"],
                             "n_gpus": j.get("n_gpus", TRAIN_GPUS)})
    for arm in ("clean", "pert"):
        for s in R.SEEDS["tfidf"]:
            f = BASE / "models" / f"tfidf__para_xsite__{arm}__seed{s}" / "manifest.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            rows.append({"model": R.LABEL["tfidf"], "arm": arm, "seed": s, "hours": j["fit_sec"] / 3600,
                         "n_train_obs": j["n_train_obs_fitted"], "vocab": j["vocab_size"], "n_cpu_jobs": FIT_JOBS})
    return pd.DataFrame(rows)


def inference_rows() -> pd.DataFrame:
    rows = []
    for m in ("mlm", "bcm", "llama"):
        for s in R.SEEDS[m]:
            c = cellD(m, s)
            if not R.has(c):
                continue
            j = manifest(c)
            rows.append({"model": R.LABEL[m], "seed": s, "full_logits": bool(j["full_logits"]),
                         "obs_per_sec": j["positions_per_sec"], "n_obs": j["n_positions"],
                         "n_patients": j["n_rows"], "wall_clock_sec": j["wall_clock_sec"],
                         "batch_size": j["batch_size"], "max_len": j["max_len"],
                         "peak_gpu_gb": j["peak_gpu_mem_gb"], "model_load_sec": j["model_load_sec"],
                         "n_params_manifest": j["n_params"]})
    return pd.DataFrame(rows)


def dump_overhead() -> pd.DataFrame:
    """FHIR-MLM per seed: cell D (full logits) against the same pert model's no-logit cells, which score
    the same positions with perturbation variants of the test values."""
    cs, _ = CL.scan(CL.EVAL_BASE)
    rows = []
    for s in R.SEEDS["mlm"]:
        jd = manifest(cellD("mlm", s))
        var = []
        for c in cs:
            if (c.model, c.condition, c.seed, c.train_arm, c.scope, c.evalset, c.split) != \
                    ("mlm", "para_xsite", s, "pert", "history", "mimictest", "test"):
                continue
            j = json.loads((c.path / "manifest.json").read_text())
            if not j["full_logits"]:
                if j["n_positions"] != jd["n_positions"]:
                    raise AssertionError(f"{c.name}: {j['n_positions']:,} positions, cell D has {jd['n_positions']:,}")
                var.append((c.test_arm_explicit, j["positions_per_sec"]))
        if var:
            no = float(np.mean([v for _, v in var]))
            rows.append({"seed": s, "obs_per_sec_with_logits": jd["positions_per_sec"],
                         "obs_per_sec_without_logits": no, "n_cells_without": len(var),
                         "test_arms_without": ",".join(sorted(a for a, _ in var)),
                         "overhead_pct": 100 * (jd["positions_per_sec"] / no - 1)})
    return pd.DataFrame(rows)


def tuning_row() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{OPTUNA_DB}?mode=ro", uri=True)
    t = pd.read_sql("select state, datetime_start, datetime_complete from trials", con,
                    parse_dates=["datetime_start", "datetime_complete"])
    done = t[t.state == "COMPLETE"]
    ev = sorted([(x, 1) for x in done.datetime_start] + [(x, -1) for x in done.datetime_complete])
    conc = np.cumsum([d for _, d in ev]).max()
    dur = (done.datetime_complete - done.datetime_start).dt.total_seconds()
    return pd.DataFrame([{"model": R.LABEL["tfidf"], "what": "hyperparameter search (08)",
                          "n_trials": len(done), "n_not_complete": len(t) - len(done),
                          "sum_trial_hours": dur.sum() / 3600, "max_concurrent": int(conc),
                          "span_hours": (done.datetime_complete.max() - done.datetime_start.min()).total_seconds() / 3600}])


def stage_collect(out: Path) -> list[str]:
    tr, inf, ov, tu = training_rows(), inference_rows(), dump_overhead(), tuning_row()
    tr.to_csv(out / "cost_training.csv", index=False)
    inf.to_csv(out / "cost_inference.csv", index=False)
    ov.to_csv(out / "cost_dump_overhead.csv", index=False)
    tu.to_csv(out / "cost_tuning.csv", index=False)
    print("training (hours; mean ± sample sd over seeds):")
    print(tr.groupby(["model", "arm"]).hours.agg(["count", "mean", "std", "min", "max"]).to_string(
        float_format=lambda v: f"{v:.3f}"))
    print("\ninference at cell D (observations/s):")
    print(inf.groupby("model")[["obs_per_sec", "batch_size", "max_len", "peak_gpu_gb", "n_params_manifest"]]
          .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:,.2f}"))
    print("\ndumping full logits (FHIR-MLM):")
    print(ov.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nTF-IDF tuning:")
    print(tu.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return [m for m in R.MODELS if R.LABEL[m] in set(tr.model) | set(inf.model)]


# -- the TF-IDF benchmark ----------------------------------------------------
def stage_tfidf(out: Path) -> list[str]:
    import joblib
    from threadpoolctl import threadpool_limits
    for p in (str(locations.COMMON), str(locations.COMMON),
              str(locations.THIRD_PARTY), str(locations.TOKENIZER)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from utils import extract_structured_features_single

    files = sorted(TFIDF_TEST.glob("*.jsonl"))
    np.random.default_rng(0).shuffle(files)
    raw = []
    for fp in files:
        with open(fp) as fh:
            raw += [json.loads(ln)["prompt"] for ln in fh if ln.strip()]
        if len(raw) >= N_BENCH:
            break
    raw = raw[:N_BENCH]
    pipe = joblib.load(TFIDF_MODEL.as_posix().format(0) + "/pipeline.joblib")
    clf = pipe.named_steps["clf"]
    est = getattr(clf, "estimators_", [clf])
    n_params = int(sum(e.coef_.size + e.intercept_.size for e in est))
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    with threadpool_limits(1):
        t0 = time.perf_counter()
        X = [extract_structured_features_single(r, None, None) for r in raw]
        t_extract = time.perf_counter() - t0
        pipe.predict_proba(X[:1000])                              # warm-up
        t_pred = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            for i in range(0, len(X), EVAL_BATCH):
                pipe.predict_proba(X[i:i + EVAL_BATCH])
            t_pred.append(time.perf_counter() - t0)
    tp = float(np.median(t_pred))
    row = {"model": R.LABEL["tfidf"], "n_obs": len(X), "threads": 1, "cpu": cpu, "n_params": n_params,
           "vocab": len(pipe.named_steps["tfidf"].vocabulary_), "extract_sec": t_extract,
           "predict_sec_median": tp, "predict_sec_all": ";".join(f"{v:.2f}" for v in t_pred),
           "obs_per_sec_predict": len(X) / tp, "obs_per_sec_end_to_end": len(X) / (t_extract + tp)}
    pd.DataFrame([row]).to_csv(out / "cost_tfidf_benchmark.csv", index=False)
    print(pd.Series(row).to_string())
    return ["tfidf"]


# -- report ------------------------------------------------------------------
def stage_report(out: Path) -> list[str]:
    tr = pd.read_csv(out / "cost_training.csv")
    inf = pd.read_csv(out / "cost_inference.csv")
    bench = out / "cost_tfidf_benchmark.csv"
    b = pd.read_csv(bench).iloc[0] if bench.exists() else None
    pert = tr[tr.arm == "pert"].groupby("model").hours.agg(["mean", "std", "count"])
    size = {"mlm": T08.manifest_params("mlm__para_xsite__clean__seed0__mimictest"),
            "bcm": T08.manifest_params("bcm__para_xsite__clean__seed0__mimictest"),
            "llama": T08.llama_counts()["total"], "tfidf": T08.tfidf_params()[0]}
    rows = []
    for m in R.ORDER:
        lab = R.LABEL[m]
        h = pert.loc[lab] if lab in pert.index else None
        if m == "tfidf":
            rows.append({"model": lab, "params_M": size[m] / 1e6, "device": "CPU",
                         "obs_per_sec": b.obs_per_sec_end_to_end if b is not None else np.nan,
                         "obs_per_sec_sd": np.nan, "batch_size": np.nan,
                         "train_hours": h["mean"] if h is not None else np.nan,
                         "train_hours_sd": h["std"] if h is not None else np.nan,
                         "n_train_seeds": int(h["count"]) if h is not None else 0})
            continue
        g = inf[inf.model == lab]
        rows.append({"model": lab, "params_M": size[m] / 1e6, "device": "GPU",
                     "obs_per_sec": g.obs_per_sec.mean() if len(g) else np.nan,
                     "obs_per_sec_sd": g.obs_per_sec.std() if len(g) > 1 else np.nan,
                     "batch_size": g.batch_size.iloc[0] if len(g) else np.nan,
                     "train_hours": h["mean"] if h is not None else np.nan,
                     "train_hours_sd": h["std"] if h is not None else np.nan,
                     "n_train_seeds": int(h["count"]) if h is not None else 0})
    t = pd.DataFrame(rows)
    t.to_csv(out / "t25_rows.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return [m for m in R.MODELS if R.LABEL[m] in set(t.dropna(subset=["obs_per_sec"]).model)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "tfidf", "report"])
    a = ap.parse_args()
    out = R.out_dir(ANALYSIS, cross_model=False)
    fn = {"collect": stage_collect, "tfidf": stage_tfidf, "report": stage_report}[a.stage]
    print("=" * 78 + f"\ncost · {a.stage}\n" + "=" * 78, flush=True)
    R.stamp(out, ANALYSIS, fn(out), stage=a.stage)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
