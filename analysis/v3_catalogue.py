#!/usr/bin/env python
"""The catalogue unit: one decision per local code (thesis Chapters 6 and 7). Built on a8's aggregation
and a9's code scores (`a8.aggregate_cell`, `a9.code_scores`), with a10's joint code × seed statistics.

  --stage percode   stream each cell's logits once into a small per-code table (summed mass, the other
                    three aggregation rules, ranks, counts). Everything at the catalogue unit is computed
                    from these tables, so the logits are never re-read for a table.
  --stage tables    a10's statistics over the per-code tables: per family, the joint (code × seed) rows
                    (T-10, T-14, T-16, T-30, T-32), coverage at the 1/5/10% budgets
                    (T-24), paired differences (T-10's and T-16's text, T-32, Chapter 7), the
                    aggregation rules (T-12), and RQ3's per-code cells and cell differences

Families and their observation sets:
  cross-model (all four models):
    para D           the observations every present model scored at cell D (T-10, T-12, T-14, T-24, T-30)
    RELMA D a0-2     the same, restricted to RELMA-named observations (T-16, T-24)
  per model or model pair:
    para D, 2 models FHIR-MLM ∩ TF-IDF, + the single-obs FHIR-MLM arm (T-15's para column; Chapter 7)
    nodesc D         FHIR-MLM ∩ TF-IDF, the only nodesc models, + the single-obs arm (T-15; Chapter 7)
    so para / nodesc FHIR-MLM history vs single obs on FHIR-MLM's own rows (T-32, T-35)
    per-model RQ3    para / nodesc / single-obs A-D, RELMA B and D a0 (covered), each on the positions
                     that model scored in every compared cell

Reads the evaluation dumps in WORK_DIR/eval. Writes results/analysis/v3/catalogue/ (cross-model
families) and results/analysis/v3/catalogue_per_model/. The per-code tables hold only local codes and
counts, with no patient identifiers.

  python analysis/v3_catalogue.py --stage percode
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


A8 = _load("a8", "a8_catalogue.py")
A9 = _load("a9", "a9_catalogue_deployment.py")
P9 = _load("a9p", "a9_percode_dump.py")
KEY = ["patient_uuid", "obs_ordinal"]
LETTERS = {"A": ("clean", "clean"), "B": ("clean", "pert"), "C": ("pert", "clean"), "D": ("pert", "pert")}


# -- one cell -> one per-code table ------------------------------------------
def _hits(scores, gold, space, k):
    """The per-code hit of a8.topk_from_scores. It uses the same sort (argsort, default kind), so ties
    are broken the same way."""
    order = np.argsort(-scores, axis=1)[:, :k]
    return (space[order] == gold[:, None]).any(axis=1)


def percode_table(cell, space, keep_keys=None, pick_seed=0) -> pd.DataFrame:
    """One row per local code. Each row holds the summed-mass decision from a9.code_scores (the argmax,
    with confidence equal to the winner's share of the code's mass) and the rank of the gold LOINC. It also
    holds the per-rule top-1 hits for summed mass, mean log-prob, majority vote and one random observation
    (the reference in T-12). Its observation counts give observation and deployed accuracy on these same
    rows."""
    agg = A8.aggregate_cell(cell, space, keep_keys=keep_keys)
    sc = A9.code_scores(agg["mass"], agg["gold"], space)
    rank = P9.gold_rank(agg["mass"], agg["gold"], space)
    L = np.load(cell.path / "logits_centered_f16.npy", mmap_mode="r")
    so = A8.single_obs_scores_from(agg, L, cell.model, pick_seed)
    g = agg["gold"]
    return pd.DataFrame({
        "code": agg["codes"], "gold": g, "n_obs": agg["n_obs"], "n_correct_obs": agg["n_correct"],
        "sm_pred": sc["pred"], "sm_conf": sc["conf"], "sm_correct": sc["correct"],
        "sm_rank": rank, "sm_top5": rank <= 5,
        "sm_hit1": _hits(agg["mass"], g, space, 1), "sm_hit5": _hits(agg["mass"], g, space, 5),
        "mlp_hit1": _hits(agg["logmass"], g, space, 1), "mv_hit1": _hits(agg["votes"], g, space, 1),
        "mv_hit5": _hits(agg["votes"], g, space, 5), "so_hit1": _hits(so, g, space, 1)})


# -- observation sets --------------------------------------------------------
def keyset(cell, covered_only=False) -> set:
    f = pd.read_parquet(cell.path / "part-0000.parquet", columns=KEY + ["true_loinc"])
    if covered_only:
        f = f[D.relma_covered(cell, f)]
    return set(zip(f.patient_uuid.to_numpy(), f.obs_ordinal.to_numpy()))


def matched_keys(cells, covered_only=False) -> set:
    keys = None
    for c in cells:
        k = keyset(c, covered_only)
        keys = k if keys is None else keys & k
    return keys or set()


# -- stage: percode ----------------------------------------------------------
def _cells(model, condition, letter, evalset="mimictest", scope="history"):
    tr, te = LETTERS[letter]
    return R.cells(model, condition=condition, train_arm=tr, test_arm=te, evalset=evalset, scope=scope)


def families() -> list[dict]:
    """Each family: which (model, letter) cells, over which observation set. `cross` families match
    across models. The others match one model across its own cells."""
    F = []

    def add(name, cross, spec, models, letters, covered=False, so=False, so_evalset="singleobs_trim", also=None):
        ms = [m for m in models if all(R.has(c, logits=True) for L in letters for c in _cells(m, *spec(L)))]
        if so and so_evalset != "singleobs_trim":         # a single-obs arm whose dumps may be absent
            if not all(R.has(c, logits=True) for c in _cells("mlm", spec("D")[0], "D", evalset=so_evalset,
                                                             scope="singleobs")):
                return
        if ms:
            F.append(dict(name=name, cross=cross, spec=spec, models=ms, letters=letters, covered=covered, so=so,
                          so_evalset=so_evalset, also=also))

    para = R.CONDITIONS["para_xsite"]
    add("para_D", True, lambda L: ("para_xsite", L), para, "D")
    for a in R.RELMA_DRAWS:
        ev = R.EVALSET_RELMA.format(a)
        add(f"relma_a{a}_D", True, lambda L, ev=ev: ("para_xsite", L, ev), para, "D", covered=True)
    add("para_D_mlm_tfidf", False, lambda L: ("para_xsite", L), ["mlm", "tfidf"], "D", so=True)
    add("nodesc_D", False, lambda L: ("nodesc_xsite", L), R.CONDITIONS["nodesc_xsite"], "D", so=True)
    # T-15: nodesc on the rows both models also scored with descriptions, so para vs nodesc is paired
    add("nodesc_D_t15", False, lambda L: ("nodesc_xsite", L), R.CONDITIONS["nodesc_xsite"], "D",
        also=lambda: [_cells(m, "para_xsite", "D")[0] for m in R.CONDITIONS["nodesc_xsite"]])
    add("so_para__mlm", False, lambda L: ("para_xsite", L), ["mlm"], "D", so=True)
    add("so_nodesc__mlm", False, lambda L: ("nodesc_xsite", L), ["mlm"], "D", so=True)
    ev0 = R.EVALSET_RELMA.format(R.RELMA_PRIMARY)              # T-35: history vs single obs, RELMA a0
    add(f"so_relma_a{R.RELMA_PRIMARY}__mlm", False, lambda L, ev0=ev0: ("para_xsite", L, ev0), ["mlm"], "D",
        covered=True, so=True, so_evalset=f"singleobs_trim_{ev0}")
    for m in para:                                                   # RQ3, per model
        add(f"rq3_para__{m}", False, lambda L: ("para_xsite", L), [m], "ABCD")
        ev0 = R.EVALSET_RELMA.format(R.RELMA_PRIMARY)
        add(f"rq3_relma_a{R.RELMA_PRIMARY}__{m}", False, lambda L, ev0=ev0: ("para_xsite", L, ev0), [m], "BD",
            covered=True)
    for m in R.CONDITIONS["nodesc_xsite"]:
        add(f"rq3_nodesc__{m}", False, lambda L: ("nodesc_xsite", L), [m], "ABCD")
    for cond in R.CONDITIONS:
        add(f"rq3_singleobs_{cond.split('_')[0]}__mlm", False,
            lambda L, cond=cond: (cond, L, "singleobs_trim", "singleobs"), ["mlm"], "ABCD")
    return F


def _stored_models(dest: Path) -> set:
    if not dest.exists():
        return set()
    import pyarrow.parquet as pq
    return set(pq.read_table(dest, columns=["model_key"]).column("model_key").to_pylist()) - {"mlm_so"}


def stage_percode(out_cross: Path, out_model: Path, only: str | None = None, force: bool = False):
    """Streams each family once. A family whose stored table already holds the same models is skipped,
    because its observation set depends only on which models are in it. So when a model's dumps are added,
    only the families it joins are re-streamed. --force re-streams everything."""
    space = A8.A.canonical_space()
    for fam in families():
        if only and not fam["name"].startswith(only):
            continue
        dest = (out_cross if fam["cross"] else out_model) / f"percode__{fam['name']}.parquet"
        if not force and _stored_models(dest) == set(fam["models"]):
            print(f"  · {fam['name']:24s} up to date ({', '.join(fam['models'])})", flush=True)
            continue
        t0 = time.time()
        seed0 = [_cells(m, *fam["spec"](L))[0] for m in fam["models"] for L in fam["letters"]]
        seed0 += fam["also"]() if fam.get("also") else []             # cells that only narrow the set
        keep = matched_keys(seed0, covered_only=fam["covered"])
        rows = []
        for m in fam["models"]:
            for L in fam["letters"]:
                for c in _cells(m, *fam["spec"](L)):
                    t = percode_table(c, space, keep_keys=keep)
                    rows.append(t.assign(family=fam["name"], model=R.LABEL[m], model_key=m, cell=L,
                                         seed=c.seed, evalset=c.evalset, n_obs_set=len(keep)))
        # T-32, T-35 and Chapter 7: the single-observation FHIR-MLM, on the same observation set. It is
        # an extra arm and is not part of the definition of the set. Adding it to the intersection could
        # shrink every catalogue number already computed. Its positions equal FHIR-MLM's anyway.
        if fam["so"]:
            cond = fam["spec"]("D")[0]
            for c in _cells("mlm", cond, "D", evalset=fam["so_evalset"], scope="singleobs"):
                if R.has(c, logits=True):
                    t = percode_table(c, space, keep_keys=keep)
                    rows.append(t.assign(family=fam["name"], model="FHIR-MLM (single obs)", model_key="mlm_so",
                                         cell="D", seed=c.seed, evalset=c.evalset, n_obs_set=len(keep)))
        tmp = dest.with_suffix(".tmp.parquet")
        pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False)
        tmp.replace(dest)                                  # atomic: an interrupted stream leaves no half table
        print(f"  ✓ {fam['name']:24s} {len(fam['models'])} model(s) × {len(fam['letters'])} cell(s), "
              f"{len(keep):,} observations ({(time.time() - t0) / 60:.1f} min) -> {dest.name}", flush=True)
    return sorted({m for f in families() for m in f["models"]})


# -- stage: tables (a10's statistics over the per-code tables) ---------------
A10 = _load("a10", "a10_catalogue_seedjoint.py")
ORDER = [R.LABEL[m] for m in R.ORDER]                     # FHIR-MLM, TF-IDF, Llama, BCM; fixes the draw order


def joint_rows(pc: pd.DataFrame, arm: str) -> tuple[list[dict], list[dict]]:
    """The per-model rows for one arm, from a frame with columns model, code, seed, correct, top5,
    conf, n_obs (one arm only). Intervals come from a10's joint_ci. The seed SDs of the two confidence
    columns are for T-30."""
    rows, cov_rows = [], []
    for model in sorted(pc.model.unique()):
        s = pc[pc.model == model]
        piv = lambda col: s.pivot_table(index="code", columns="seed", values=col, aggfunc="first")
        m_ = piv("correct"); mat = m_.to_numpy(dtype=float); seeds = m_.columns.to_numpy()
        nobs = s.groupby("code").n_obs.first().reindex(m_.index).to_numpy(dtype=float)
        t5 = piv("top5").reindex(m_.index).to_numpy(dtype=float)
        conf = piv("conf").reindex(m_.index).to_numpy(dtype=float)
        p1, p5 = mat.mean(axis=0), t5.mean(axis=0)
        pdep = (mat * nobs[:, None]).sum(axis=0) / nobs.sum()
        (a_lo, a_hi), (b_lo, b_hi) = A10.joint_ci(mat)
        (d_lo, d_hi), _ = A10.joint_ci(mat, weights=nobs)
        (f_lo, f_hi), _ = A10.joint_ci(t5)
        wrong = ~mat.astype(bool)
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        r = {"arm": arm, "model": model, "n_seeds": mat.shape[1], "n_codes": mat.shape[0],
             "top1_seed0": p1[0], "top1_mean": p1.mean(), "top1_sd": sd(p1), "top1_min": p1.min(),
             "top1_max": p1.max(), "top1_A_lo": a_lo, "top1_A_hi": a_hi, "top1_B_lo": b_lo, "top1_B_hi": b_hi,
             "top5_seed0": p5[0], "top5_mean": p5.mean(), "top5_A_lo": f_lo, "top5_A_hi": f_hi,
             "dep_seed0": pdep[0], "dep_mean": pdep.mean(), "dep_A_lo": d_lo, "dep_A_hi": d_hi,
             "codes_wrong_seed0": int(wrong[:, 0].sum()), "codes_wrong_mean": float(wrong.sum(axis=0).mean()),
             "codes_wrong_sd": sd(wrong.sum(axis=0)), "pct_codes_wrong_mean": float(wrong.mean(axis=0).mean()),
             "pct_vol_wrong_mean": float(np.mean([nobs[w].sum() / nobs.sum() for w in wrong.T])),
             "med_obs_wrong_mean": float(np.mean([np.median(nobs[w]) for w in wrong.T if w.any()])),
             "med_obs_right_mean": float(np.mean([np.median(nobs[~w]) for w in wrong.T]))}
        if "n_correct_obs" in s.columns:            # observation accuracy on these same rows (T-10, T-16)
            nco = piv("n_correct_obs").reindex(m_.index).to_numpy(dtype=float)
            pobs = nco.sum(axis=0) / nobs.sum()
            r.update({"obs_seed0": pobs[0], "obs_mean": pobs.mean(), "obs_sd": sd(pobs)})
        cma = conf.mean(axis=0) - p1                  # conf − acc at the code level (T-30)
        r.update({"conf_minus_acc_mean": float(cma.mean()), "conf_minus_acc_sd": sd(cma)})
        eces, ccs, cws, fw = [], [], [], []
        for j in range(mat.shape[1]):
            cj, kj = conf[:, j], mat[:, j].astype(bool)
            eces.append(A10.C.ece(cj, kj, n_bins=A10.N_BINS))
            ccs.append(float(cj[kj].mean()) if kj.any() else np.nan)
            cws.append(float(cj[~kj].mean()) if (~kj).any() else np.nan)
            order = np.argsort(-cj, kind="stable")
            bad = np.flatnonzero(~kj[order])
            fw.append(int(bad[0]) + 1 if len(bad) else len(kj) + 1)
            rc = A10.C.risk_coverage(cj, kj)
            cov_rows.append({"arm": arm, "model": model, "seed": int(seeds[j]),
                             **{f"cov@{b:g}": A10.C.coverage_at_risk(rc, b) for b in A10.BUDGETS}})
        r.update({"ece_seed0": eces[0], "ece_mean": float(np.mean(eces)), "ece_sd": sd(eces),
                  "conf_correct_seed0": ccs[0], "conf_correct_mean": float(np.nanmean(ccs)),
                  "conf_correct_sd": sd(np.array(ccs)[~np.isnan(ccs)]),
                  "conf_wrong_seed0": cws[0], "conf_wrong_mean": float(np.nanmean(cws)),
                  "conf_wrong_sd": sd(np.array(cws)[~np.isnan(cws)]),
                  "firstwrong_seed0": fw[0], "firstwrong_mean": float(np.mean(fw)),
                  "firstwrong_all": "/".join(str(x) for x in fw)})
        rows.append(r)
    return rows, cov_rows


def pairwise_rows(pc: pd.DataFrame, arm: str, names: list | None = None) -> list[dict]:
    """Paired differences for one arm. Each replicate uses one shared code draw across every pair and
    resamples each model's seeds independently. The interval is on the difference. `names` defaults to
    the registry models in ORDER, then any other arm (e.g. the single-observation FHIR-MLM). Extra arms
    come last so the random draws of the four models do not depend on them."""
    if names is None:
        names = [m for m in ORDER if m in set(pc.model)] + sorted(set(pc.model) - set(ORDER))
    mats, seeds_n, codes_ref = {}, {}, None
    for m in names:
        s = pc[pc.model == m]
        if s.empty:
            continue
        piv = s.pivot_table(index="code", columns="seed", values="correct", aggfunc="first")
        if codes_ref is None:
            codes_ref = piv.index
        elif not piv.index.equals(codes_ref):
            raise AssertionError(f"{arm}/{m}: code index differs from the first model's; cannot pair")
        mats[m], seeds_n[m] = piv.to_numpy(dtype=float), piv.shape[1]
    names = [m for m in names if m in mats]
    rng = np.random.default_rng(A10.BOOT_SEED)
    nc = len(codes_ref)
    draws = rng.integers(0, nc, size=(A10.N_BOOT, nc))
    seed_pick = {m: rng.integers(0, seeds_n[m], size=(A10.N_BOOT, seeds_n[m])) for m in names}
    per = {m: np.take_along_axis(mats[m][draws].mean(axis=1), seed_pick[m], axis=1).mean(axis=1) for m in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = np.percentile(per[a] - per[b], [2.5, 97.5])
            rows.append({"arm": arm, "model_a": a, "model_b": b, "diff_mean": mats[a].mean() - mats[b].mean(),
                         "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
                         "seeds_a": seeds_n[a], "seeds_b": seeds_n[b]})
    return rows


def _v3_frame(pc: pd.DataFrame) -> pd.DataFrame:
    return pc.rename(columns={"sm_correct": "correct", "sm_top5": "top5", "sm_conf": "conf"})


RULES = {"sm_hit1": "summed mass", "mlp_hit1": "mean log-probability", "mv_hit1": "majority vote",
         "so_hit1": "one random observation"}


def rule_rows(pc: pd.DataFrame, fam: str) -> list[dict]:
    """T-12: per-code top-1 under each aggregation rule, per model and seed. `so_hit1` (one random
    observation standing for its code) is the reference in T-12."""
    g = pc.groupby(["model", "seed"])[list(RULES)].mean().reset_index()
    return [{"family": fam, **r} for r in g.to_dict("records")]


CROSS_FAMILIES = ("para_D", "relma_a0_D", "relma_a1_D", "relma_a2_D")
FINAL_FAMILIES = ("para_D_mlm_tfidf", "nodesc_D", "nodesc_D_t15", "so_para__mlm", "so_nodesc__mlm",
                  f"so_relma_a{R.RELMA_PRIMARY}__mlm")


def _family_stats(fams, root: Path):
    joint, cov, pw, pw5, rules, keys = [], [], [], [], [], set()
    for fam in fams:
        p = root / f"percode__{fam}.parquet"
        if not p.exists():
            continue
        pc = _v3_frame(pd.read_parquet(p))
        j, c = joint_rows(pc, fam); joint += j; cov += c
        pw += pairwise_rows(pc, fam)
        # the same paired bootstrap on top-5 (Chapter 7's shortlist argument), in its own pass so the
        # top-1 draws above are unchanged
        pw5 += pairwise_rows(pc.assign(correct=pc.top5), fam)
        rules += rule_rows(pc, fam)
        keys |= set(pc.model_key)
        print(f"  ✓ {fam:18s} {pc.model.nunique()} arm(s), {pc.code.nunique()} codes, "
              f"{int(pc.n_obs_set.iloc[0]):,} observations", flush=True)
    for name, rows in (("catalogue_joint", joint), ("catalogue_coverage_by_seed", cov),
                       ("catalogue_pairwise", pw), ("catalogue_pairwise_top5", pw5), ("catalogue_rules", rules)):
        pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)
    return keys


def stage_tables(out_cross: Path, out_model: Path) -> tuple[list, list]:
    mc = _family_stats(CROSS_FAMILIES, out_cross)
    mm = _family_stats(FINAL_FAMILIES, out_model)
    # RQ3: each model across its own cells (the cell letter plays the model's part in a10's statistics)
    cj, cp = [], []
    for p in sorted(out_model.glob("percode__rq3_*.parquet")):
        pc = _v3_frame(pd.read_parquet(p))
        fam = p.stem.removeprefix("percode__")
        model, key = pc.model.iloc[0], pc.model_key.iloc[0]
        cells = pc.assign(model=pc.cell)
        j, _ = joint_rows(cells, fam)
        cj += [{"family": fam, "model": model, "cell": r.pop("model"), **r} for r in j]
        cp += [{"family": fam, "model": model, **r} for r in pairwise_rows(cells, fam, names=sorted(set(pc.cell)))]
        mm.add(key)
        print(f"  ✓ {fam:28s} cells {''.join(sorted(set(pc.cell)))}, {pc.code.nunique()} codes", flush=True)
    pd.DataFrame(cj).to_csv(out_model / "catalogue_rq3_cells.csv", index=False)
    pd.DataFrame(cp).to_csv(out_model / "catalogue_rq3_pairs.csv", index=False)
    keys = lambda s: [m for m in R.MODELS if m in s]
    return keys(mc), keys(mm - {"mlm_so"} | ({"mlm"} if "mlm_so" in mm else set()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["percode", "tables"])
    ap.add_argument("--only", default=None, help="family-name prefix, e.g. para_D or rq3_")
    ap.add_argument("--force", action="store_true", help="re-stream families that are up to date")
    a = ap.parse_args()
    out_cross = R.out_dir("catalogue")
    out_model = R.out_dir("catalogue_per_model", cross_model=False)
    if a.stage == "tables":
        mc, mm = stage_tables(out_cross, out_model)
        R.stamp(out_cross, "catalogue", mc, stage="tables", note="cross-model families")
        R.stamp(out_model, "catalogue_per_model", mm, stage="tables", note="each model across its own cells")
        return
    models = stage_percode(out_cross, out_model, a.only, a.force)
    R.stamp(out_cross, "catalogue", models, stage="percode", note="cross-model families")
    R.stamp(out_model, "catalogue_per_model", models, stage="percode", note="each model across its own cells")


if __name__ == "__main__":
    main()
