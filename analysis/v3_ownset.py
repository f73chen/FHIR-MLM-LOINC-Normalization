#!/usr/bin/env python
"""Each trained model on all the observations it scores, against the evaluated set (thesis Chapter 7,
Limitations).

The thesis compares the trained models on one evaluated set (thesis Chapter 5; result root `v3_both`). The
root `v3` holds each model on all the observations it scores itself. For the deep models these are the first
window of every test patient they keep. This module sets the two side by side in the main scenario (perturbed
numbers in training and test, Para), per observation and per code. It also counts the local codes that the
observations of a model add to those of the evaluated set.

Two rules apply, both from the thesis:
  * Only local codes whose LOINC code is in the 679-code label space count. The per-code tables of the root
    `v3` also hold the few local codes of the deep models whose LOINC code is outside it. They are left out
    here and reported as a count.
  * Per-code top-1 is `sm_correct`, the column the per-code tables of the thesis use
    (v3_catalogue.percode_table).

It reads only stored aggregate tables of the two roots and writes nothing. v3_book.py prints `rows()` in the
numbers book.

  python analysis/v3_ownset.py          # prints the comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "analysis"
OWN_ROOT, EVAL_ROOT = RESULTS / "v3", RESULTS / "v3_both"
KEYS = {"FHIR-MLM": "mlm", "BCM": "bcm", "Llama-3.1-8B": "llama", "TF-IDF": "tfidf"}
RARE_BELOW = 1_000                     # the rare group of T-04: fewer than 1,000 training observations of the LOINC code
OBS_FILE = "rq3/rq3_primary.csv"
PC_FILE = "catalogue_per_model/percode__rq3_para__{key}.parquet"
PC_COLS = ["code", "gold", "seed", "cell", "sm_correct", "n_obs"]


# -- pure logic (tested in tests/test_v3_ownset.py) ---------------------------
def _seed_mean(pc: pd.DataFrame) -> float:
    """Per-code top-1 as the thesis reports it. This is the share of local codes right in each model seed,
    then the mean over seeds."""
    return float(pc.groupby("seed").sm_correct.mean().mean())


def compare(own_obs: pd.DataFrame, ev_obs: pd.DataFrame, own_pc: dict, ev_pc: dict, counts: dict,
            rare_below: int = RARE_BELOW) -> pd.DataFrame:
    """One row per model.

    own_obs / ev_obs: the rq3_primary tables of the two roots (model, top1_D_mean = mean over model seeds,
    n_paired). own_pc / ev_pc: model -> its per-code table of scenario D (code, gold, seed, sm_correct, n_obs),
    one row per local code and model seed. counts: LOINC code -> training observations, for the codes of the
    label space only.
    """
    space = set(counts)
    rows = []
    for m in own_pc:
        o_all, e = own_pc[m], ev_pc[m]
        outside = o_all[~o_all.gold.isin(space)]
        o = o_all[o_all.gold.isin(space)]
        assert e.gold.isin(space).all(), f"{m}: the evaluated set holds a local code outside the label space"
        ev_codes, own_codes = set(e.code), set(o.code)
        assert ev_codes <= own_codes, f"{m}: the evaluated set has local codes the model does not score on its own"
        added = o[~o.code.isin(ev_codes)]
        first = added[added.seed == added.seed.min()] if len(added) else added      # one row per added local code
        oo, eo = own_obs.set_index("model").loc[m], ev_obs.set_index("model").loc[m]
        rows.append({
            "model": m, "n_seeds": int(o.seed.nunique()),
            "obs_own": int(oo.n_paired), "obs_eval": int(eo.n_paired),
            "top1_obs_own": float(oo.top1_D_mean), "top1_obs_eval": float(eo.top1_D_mean),
            "obs_diff_pp": 100 * (float(oo.top1_D_mean) - float(eo.top1_D_mean)),
            "codes_own": len(own_codes), "codes_eval": len(ev_codes),
            "top1_code_own": _seed_mean(o), "top1_code_eval": _seed_mean(e),
            "code_diff_pp": 100 * (_seed_mean(o) - _seed_mean(e)),
            # the local codes of the evaluated set, decided from all the observations of the model
            "code_diff_same_codes_pp": 100 * (_seed_mean(o[o.code.isin(ev_codes)]) - _seed_mean(e)),
            "codes_added": int(first.code.nunique()),
            "added_rare": int(first.gold.map(counts).lt(rare_below).sum()),
            "added_one_obs": int((first.n_obs == 1).sum()),
            "added_max_obs": int(first.n_obs.max()) if len(first) else 0,
            "added_obs": int(first.n_obs.sum()),
            "top1_code_added": _seed_mean(added) if len(added) else float("nan"),
            "codes_outside_space": int(outside.code.nunique()),
        })
    return pd.DataFrame(rows)


# -- stored tables -----------------------------------------------------------
def available() -> bool:
    return all((root / OBS_FILE).exists() and all((root / PC_FILE.format(key=k)).exists() for k in KEYS.values())
               for root in (OWN_ROOT, EVAL_ROOT))


def _percode(root: Path) -> dict:
    out = {}
    for m, k in KEYS.items():
        d = pd.read_parquet(root / PC_FILE.format(key=k), columns=PC_COLS)
        out[m] = d[d.cell == "D"].drop(columns="cell")
    return out


def table() -> pd.DataFrame:
    sys.path.insert(0, str(HERE))
    import v3_observation as O                                   # the training counts of the label space (679 LOINC codes)
    return compare(pd.read_csv(OWN_ROOT / OBS_FILE), pd.read_csv(EVAL_ROOT / OBS_FILE),
                   _percode(OWN_ROOT), _percode(EVAL_ROOT), O.train_count_map())


def rows(t: pd.DataFrame | None = None) -> list[tuple]:
    """The rows of the numbers book (§, what, observation set, value). The list is empty if the tables of a
    root are missing."""
    if t is None:
        if not available():
            return []
        t = table()
    t = t.set_index("model")
    two = [m for m in ("BCM", "Llama-3.1-8B") if m in t.index]      # the two models that keep every test patient
    name = lambda m: "Llama" if m.startswith("Llama") else m
    sets = "; ".join(f"{name(m)} {int(t.loc[m, 'obs_own']):,} observations and {int(t.loc[m, 'codes_own'])} local codes "
                     f"against {int(t.loc[m, 'obs_eval']):,} and {int(t.loc[m, 'codes_eval'])}" for m in two)
    return [
        ("7.7", "per-observation top-1 on all the observations a model scores minus the evaluated set (scenario D, Para, "
                "seed means)", "each model's first-window observations against the evaluated set (observations as counted "
         "in rq3_primary.csv; local codes: those whose LOINC code is in the label space): " + sets, "; ".join(f"{name(m)} {t.loc[m, 'obs_diff_pp']:+.3f} pp ({100 * t.loc[m, 'top1_obs_own']:.2f}% against "
                           f"{100 * t.loc[m, 'top1_obs_eval']:.2f}%)" for m in two)),
        ("7.7", "per-code top-1 on all the local codes a model scores minus the evaluated set (local codes whose LOINC "
                "code is outside the label space left out)", "as above",
         "; ".join(f"{name(m)} {t.loc[m, 'code_diff_pp']:+.2f} pp ({100 * t.loc[m, 'top1_code_own']:.2f}% against "
                   f"{100 * t.loc[m, 'top1_code_eval']:.2f}%); on the evaluated set's own local codes "
                   f"{t.loc[m, 'code_diff_same_codes_pp']:+.2f} pp" for m in two)),
        ("7.7", "the local codes that a model's own observations add to the evaluated set's: how many, with a rare LOINC "
                "code, with one test observation, the most observations of one, per-code top-1 on them",
         "as above (left out as outside the label space: "
         + ", ".join(f"{name(m)} {int(t.loc[m, 'codes_outside_space'])}" for m in two) + ")",
         "; ".join(f"{name(m)} {int(t.loc[m, 'codes_added'])}, {int(t.loc[m, 'added_rare'])}, "
                   f"{int(t.loc[m, 'added_one_obs'])}, {int(t.loc[m, 'added_max_obs'])}, "
                   f"{100 * t.loc[m, 'top1_code_added']:.1f}%" for m in two)),
    ]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(table().set_index("model").T.to_string(float_format=lambda v: f"{v:.4f}"))
