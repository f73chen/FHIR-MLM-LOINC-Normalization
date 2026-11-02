"""The loader for the per-observation evaluation dumps (WORK_DIR/eval), and the paired-comparison
functions the analyses share (thesis Chapters 5 and 6).

The model families write different schemas:
    deep (mlm/bcm/llama) : topk_class / topk_logit / true_class      -> 1,623-class space
    tfidf                : topk_idx  / topk_logprob / true_idx       -> 679-LOINC space
and the reference-range diagnostic writes a third. Comparisons happen at the LOINC level, so this
module maps all of them to one schema keyed on (patient_uuid, obs_ordinal).

Matched denominators: RELMA leaves out the observations with no local name, the s2 names truncate
more than s1, the model families truncate at different points, and the clean-name flags filter
further. So a comparison across runs is computed on the intersection of the runs' (patient_uuid,
obs_ordinal) keys. `matched()` builds it and raises an error if the arms have unequal length,
because an unequal paired test is a bug.

Clustered bootstrap: observations within a patient are not independent (same history, same local
names, repeated tests). An observation-level bootstrap would understate the CI, so `bootstrap_ci`
resamples patients.

Used by the v3 analyses that read dumps (calibration, catalogue, checks, observation, rq3, singleobs,
sliding) and by a1_synthesis, a2_calibration, a8_catalogue, b3_figures, b3_sliding and b4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EVAL_BASE = locations.WORK / "eval"
DATASET_BASE = locations.WORK / "datasets"
REPO = locations.REPO
CODES_TO_TOKENS = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES = str(locations.UPSTREAM / "seen_codes.csv")
KEY = ["patient_uuid", "obs_ordinal"]
DEEP = ("mlm", "bcm", "llama")

_ARR = None


def class_to_loinc_array():
    """(1623,) array of class index -> LOINC. Unmapped classes get a unique '?{i}' sentinel, so an
    unmapped truth cannot match a different unmapped prediction."""
    global _ARR
    if _ARR is None:
        from fhir_data_utils import build_lab_code_mappings
        _, c2l = build_lab_code_mappings(CODES_TO_TOKENS, SEEN_CODES)
        _ARR = np.array([c2l.get(i, f"?{i}") for i in range(1623)], dtype=object)
    return _ARR


def run_dir(model: str, condition: str, seed: int, evalset_tag: str) -> Path:
    return EVAL_BASE / f"{model}__{condition}__seed{seed}__{evalset_tag}"


def load_dump(model: str, condition: str, seed: int, evalset_tag: str, topk: int = 10) -> pd.DataFrame:
    """Returns DataFrame[patient_uuid, obs_ordinal, true_loinc, pred_loinc, correct, topk_loinc, p_top1, p_true].
    Normalised across all families, and `correct` is LOINC-level top-1. Uses the v1/v2 run naming. v3
    runs are loaded with load_cell."""
    return load_dir(run_dir(model, condition, seed, evalset_tag), model, topk)


def load_cell(cell, topk: int = 10) -> pd.DataFrame:
    """load_dump for a v3 (or any) `lib.cells.Cell`."""
    return load_dir(cell.path, cell.model, topk)


def top1_frame(cell, extra: tuple = ()) -> pd.DataFrame:
    """Returns DataFrame[patient_uuid, obs_ordinal, true_loinc, pred_loinc, correct, *extra] for a Cell.

    This is the fast path for top-1 only. It reads the top-k list column with pyarrow and takes its
    first element, where `load_dir` stacks millions of Python lists. `correct` has the same definition
    as in `load_dir`, LOINC-level top-1 under the mapping-first rule (thesis Chapter 5). Deep models map
    their top class and their true class through the same class-to-LOINC array. TF-IDF compares its top
    LOINC with the `true_loinc` column of its dump. tests/test_top1_frame.py (slow) checks that the
    two agree row for row.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    f = cell.path / "part-0000.parquet"
    if cell.model == "tfidf":
        t = pq.read_table(f, columns=KEY + ["true_loinc", "topk_idx", *extra])
        space = np.array(json.loads((cell.path / "manifest.json").read_text())["label_space_679"], dtype=object)
        pred = space[pc.list_element(t["topk_idx"], 0).to_numpy()]
        true = t["true_loinc"].to_numpy(zero_copy_only=False).astype(object)
    else:
        t = pq.read_table(f, columns=KEY + ["true_class", "topk_class", *extra])
        arr = class_to_loinc_array()
        pred = arr[pc.list_element(t["topk_class"], 0).to_numpy()]
        true = arr[t["true_class"].to_numpy()]
    out = pd.DataFrame({"patient_uuid": t["patient_uuid"].to_numpy(zero_copy_only=False),
                        "obs_ordinal": t["obs_ordinal"].to_numpy(),
                        "true_loinc": true, "pred_loinc": pred, "correct": pred == true})
    for c in extra:
        out[c] = t[c].to_numpy(zero_copy_only=False)
    return out


def topk_hits(cell, k: int = 5) -> np.ndarray:
    """Per-row top-k hit under the mapping-first rule: any of the top-k predictions maps to the gold
    LOINC. Same row order as top1_frame. With k=1 it reproduces its `correct`."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    f = cell.path / "part-0000.parquet"
    if cell.model == "tfidf":
        t = pq.read_table(f, columns=["true_loinc", "topk_idx"])
        space = np.array(json.loads((cell.path / "manifest.json").read_text())["label_space_679"], dtype=object)
        idx = np.stack(pc.list_slice(t["topk_idx"], 0, k).to_numpy(zero_copy_only=False))
        pred, true = space[idx], t["true_loinc"].to_numpy(zero_copy_only=False).astype(object)
    else:
        t = pq.read_table(f, columns=["true_class", "topk_class"])
        arr = class_to_loinc_array()
        idx = np.stack(pc.list_slice(t["topk_class"], 0, k).to_numpy(zero_copy_only=False))
        pred, true = arr[idx], arr[t["true_class"].to_numpy()]
    return (pred == true[:, None]).any(axis=1)


def relma_covered(cell, frame: pd.DataFrame) -> np.ndarray:
    """Boolean mask over the rows of `frame`, true for observations whose description was a RELMA local name.

    A code with no RELMA name keeps its MIMIC display text and is not scored on RELMA. It only gives
    context. The two families record coverage differently:
      - The TF-IDF v3 RELMA dumps hold covered observations only, so every row is covered.
        pipeline/16_tfidf_eval_dump.py skips the rest and counts them as `n_skipped_no_relma_name` in
        the manifest. This function checks that every row is covered.
      - The deep models score every masked code in the window. Their coverage comes from the assignment
        sidecar that pipeline/18_relma_dataset.py writes beside the dataset. The sidecar is joined on
        (patient, gold LOINC), as in attach_relma.
    `frame` needs patient_uuid and true_loinc and must be in the dump's row order.
    """
    if not cell.evalset.startswith("relma_"):
        raise ValueError(f"{cell.name} is not a RELMA cell")
    if cell.model == "tfidf":
        d = pd.read_parquet(cell.path / "part-0000.parquet", columns=["desc_shown"])
        cov = d.desc_shown.notna().to_numpy()
        if len(cov) != len(frame) or not cov.all():
            raise AssertionError(f"{cell.name}: expected every TF-IDF RELMA row to carry a RELMA name")
        return cov
    if cell.v3:
        side = DATASET_BASE / f"{cell.model}__{cell.evalset}__{cell.condition}__{cell.test_arm_explicit}" / "assignment_sidecar.parquet"
    else:                                                  # v1: one dataset per (model, draw)
        side = DATASET_BASE / f"{cell.model}__{cell.evalset}" / "assignment_sidecar.parquet"
    s = pd.read_parquet(side, columns=["patient_uuid", "loinc"]).drop_duplicates()
    s["_cov"] = True
    m = frame[["patient_uuid", "true_loinc"]].merge(
        s.rename(columns={"loinc": "true_loinc"}), on=["patient_uuid", "true_loinc"], how="left",
        sort=False, validate="many_to_one")
    if len(m) != len(frame):
        raise AssertionError("the sidecar join changed the row count")
    return m["_cov"].eq(True).to_numpy()                   # NaN (no sidecar row) counts as not covered


def load_dir(d: Path, model: str, topk: int = 10) -> pd.DataFrame:
    """The loader proper, for a dump directory `d` written by `model`'s family."""
    df = pd.read_parquet(d / "part-0000.parquet")
    if model == "tfidf":
        man = json.loads((d / "manifest.json").read_text())
        space = np.array(man["label_space_679"], dtype=object)
        tk = np.stack(df["topk_idx"].values)[:, :topk]
        df["true_loinc"] = df["true_loinc"] if "true_loinc" in df else space[df["true_idx"].values]
        df["topk_loinc"] = list(space[tk])
        lp = np.stack(df["topk_logprob"].values)[:, 0]
        df["p_top1"] = np.exp(lp - df["row_logsumexp"].values)
        df["p_true"] = np.exp(df["true_logprob"].values - df["row_logsumexp"].values)
    else:
        arr = class_to_loinc_array()
        tk = np.stack(df["topk_class"].values)[:, :topk]
        df["true_loinc"] = arr[df["true_class"].values]
        df["topk_loinc"] = list(arr[tk])
        lg = np.stack(df["topk_logit"].values)[:, 0]
        df["p_top1"] = np.exp(lg - df["row_logsumexp"].values)
        df["p_true"] = np.exp(df["true_class_logit"].values - df["row_logsumexp"].values)
    df["pred_loinc"] = [t[0] for t in df["topk_loinc"]]
    df["correct"] = df["pred_loinc"].values == df["true_loinc"].values
    # The description column is `desc_shown` (a paraphrase in mimictest, a local name in relma). Older
    # dumps call it `relma_name`, so accept either.
    if "relma_name" in df.columns and "desc_shown" not in df.columns:
        df = df.rename(columns={"relma_name": "desc_shown"})
    keep = KEY + ["true_loinc", "pred_loinc", "correct", "topk_loinc", "p_top1", "p_true"]
    for c in ("desc_shown", "clean_same_loinc", "clean_global",
              "true_rank", "seq_len", "truncated", "n_labeled_in_patient", "in_pretrain_4096"):
        if c in df.columns:
            keep.append(c)
    return df[keep]


def load_sidecar(model: str, strategy: str, assignment_seed: int) -> pd.DataFrame:
    p = DATASET_BASE / f"{model}__relma_{strategy}_a{assignment_seed}" / "assignment_sidecar.parquet"
    return pd.read_parquet(p)


def attach_relma(df: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Join the assignment sidecar on (patient_uuid, true_loinc). This join attaches the clean-name
    flags and also marks the observations whose LOINC has no RELMA name. Those are absent from the
    sidecar, so `has_relma` is False."""
    if "desc_shown" in df.columns:          # tfidf already carries it per observation
        out = df.copy()
        out["has_relma"] = out["desc_shown"].notna()
        return out
    s = sidecar.rename(columns={"loinc": "true_loinc", "relma_name": "desc_shown"})
    out = df.merge(s[["patient_uuid", "true_loinc", "desc_shown", "clean_same_loinc", "clean_global"]],
                   on=["patient_uuid", "true_loinc"], how="left")
    out["has_relma"] = out["desc_shown"].notna()
    return out


# -- matched-denominator primitives -----------------------------------------
def matched(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Restrict every frame to the intersection of their (patient_uuid, obs_ordinal) keys, sorted
    identically so row i is the same observation in every arm.
    Raises if the arms end up unequal, because an unequal paired comparison is a bug.

    Works across models and across conditions, since obs_ordinal does not depend on the condition.
    Pass the arms that a claim compares and no others. With fewer, the comparison would be unmatched.
    With more, observations would be dropped for no reason."""
    if not dfs:
        return []
    keysets = [set(map(tuple, d[KEY].values)) for d in dfs]
    common = set.intersection(*keysets)
    out = []
    for d in dfs:
        m = d[[tuple(x) in common for x in d[KEY].values]].sort_values(KEY).reset_index(drop=True)
        out.append(m)
    n = {len(x) for x in out}
    if len(n) != 1:
        raise AssertionError(f"matched() produced unequal arms: {sorted(n)}")
    for other in out[1:]:
        if not (out[0][KEY].values == other[KEY].values).all():
            raise AssertionError("matched() arms are not aligned row-for-row")
    return out


def coverage(df: pd.DataFrame) -> set:
    """The set of (patient_uuid, obs_ordinal) an arm scored.

    obs_ordinal is absolute per patient. It counts LOINC-mappable lab observations in chronological
    order over the whole record, independently of any model, condition or window, so this set is
    comparable across models and conditions. The a15 join gate checks the ordinal against an evaluation
    dump."""
    return set(map(tuple, df[KEY].values))


def assert_coverage_nested(less_ablated, more_ablated, names=("less", "more")) -> int:
    """Gate: removing an input field can only add observations, because shorter rows let more of a
    patient's record fit the 8192-token window. So the coverage of the less-ablated arm must be a
    subset of the coverage of the more-ablated arm.

    If it fails, the dataset builder dropped observations it should have kept. That is a bug. The gate
    raises an error so that the bug does not shrink every intersection that depends on it.

    Returns how many observations the more-ablated arm gains."""
    a, b = coverage(less_ablated), coverage(more_ablated)
    only_a = a - b
    if only_a:
        raise AssertionError(
            f"coverage NOT nested: {len(only_a):,} observations are in {names[0]} but not "
            f"{names[1]}. Removing a field must not lose observations. Check the dataset build.")
    return len(b) - len(a)


def native_vs_matched(frames: dict, keys: set | None = None) -> pd.DataFrame:
    """Each arm's native accuracy (on every row it scored) beside its accuracy on the matched set. A
    restricted comparison is then reported together with what the restriction did.

    `frames` maps an arm label to a dump. `keys` defaults to the intersection over all of them. That
    default is right only when the table compares all of them (see `matched`)."""
    if keys is None:
        keys = set.intersection(*[coverage(d) for d in frames.values()])
    rows = []
    for name, d in frames.items():
        m = d[[tuple(x) in keys for x in d[KEY].values]]
        rows.append({"arm": name, "n_native": len(d), "top1_native": float(d.correct.mean()),
                     "n_matched": len(m), "top1_matched": float(m.correct.mean()),
                     "delta_pp": 100.0 * (float(m.correct.mean()) - float(d.correct.mean()))})
    return pd.DataFrame(rows)


def micro_topk(df: pd.DataFrame, ks=(1, 3, 5, 10)) -> dict:
    tk = np.stack(df["topk_loinc"].values)
    t = df["true_loinc"].values
    return {f"top{k}": float((tk[:, :k] == t[:, None]).any(axis=1).mean()) for k in ks}


def macro_top1(df: pd.DataFrame, classes: set | None = None) -> float:
    """Per-LOINC recall, averaged over classes (each class weighted equally).
    `classes` restricts to a shared class set. This is required when two arms with different class
    coverage are compared. Otherwise the macro means are computed over different denominators."""
    g = df.groupby("true_loinc")["correct"].mean()
    if classes is not None:
        g = g[g.index.isin(classes)]
    return float(g.mean())


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Exact McNemar test on matched arms, as Dietterich (1998) recommends when a model can be run only
    once (as Llama is)."""
    from scipy.stats import binomtest
    x = int((a["correct"].values & ~b["correct"].values).sum())
    y = int((~a["correct"].values & b["correct"].values).sum())
    p = binomtest(x, x + y, 0.5).pvalue if (x + y) else 1.0
    return {"n": len(a), "a_only": x, "b_only": y, "p": float(p),
            "a_acc": float(a["correct"].mean()), "b_acc": float(b["correct"].mean()),
            "delta": float(a["correct"].mean() - b["correct"].mean())}


def mcnemar_patients(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """McNemar's test with patients as clusters (Durkalski et al. 2003) on matched arms. D_k is patient k's
    (a right, b wrong) count minus its (a wrong, b right) count. Then z = sum(D_k) / sqrt(sum(D_k^2)), with a
    two-sided normal p. With one observation per patient, z^2 is McNemar's chi-square without continuity
    correction. `mcnemar` treats observations as independent. This test does not, for the same reason that
    the observation-level intervals resample patients. Its MDE is power.mde_mcnemar_patients."""
    from scipy.stats import norm
    if not (a["patient_uuid"].values == b["patient_uuid"].values).all():
        raise AssertionError("mcnemar_patients: the arms are not aligned patient for patient")
    d = a["correct"].values.astype(np.int64) - b["correct"].values.astype(np.int64)
    s = pd.Series(d).groupby(a["patient_uuid"].values, sort=False).sum().to_numpy(dtype=float)
    den = float(np.sqrt((s ** 2).sum()))
    if den == 0:
        return {"z": 0.0, "p": 1.0, "n_patients": len(s)}
    z = float(s.sum() / den)
    return {"z": z, "p": float(2 * norm.sf(abs(z))), "n_patients": len(s)}


def label_arms(st: dict, a_name: str, b_name: str) -> dict:
    """Rename mcnemar's positional a_/b_ keys to named ones for CSV output.

    `delta` is the a arm minus the b arm. Which arm is `a` is visible only at the call site, so a
    printed or written `delta` is easily read with the wrong sign. The renamed columns state the
    direction, as in `delta_relma_minus_mimic`.
    """
    return {"n": st["n"],
            f"acc_{a_name}": st["a_acc"], f"acc_{b_name}": st["b_acc"],
            f"delta_{a_name}_minus_{b_name}": st["delta"],
            f"n_{a_name}_only": st["a_only"], f"n_{b_name}_only": st["b_only"],
            "p": st["p"]}


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Percentile CI on top-1, resampling patients (clusters). Observations within a patient share a
    history and local-name assignment, so an observation-level bootstrap understates the interval."""
    rng = np.random.default_rng(seed)
    by = df.groupby("patient_uuid")["correct"].agg(["sum", "count"])
    s, c = by["sum"].values.astype(float), by["count"].values.astype(float)
    n = len(s)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc = s[idx].sum(axis=1) / np.maximum(c[idx].sum(axis=1), 1)
    return float(np.percentile(acc, 100 * alpha / 2)), float(np.percentile(acc, 100 * (1 - alpha / 2)))
