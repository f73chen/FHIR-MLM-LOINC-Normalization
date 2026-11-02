#!/usr/bin/env python
"""
The tables of thesis Chapter 6, from the v3 analysis outputs in results/analysis/v3/. Each table is
written as booktabs LaTeX (results/analysis/v3/tables/<id>.tex) and drawn in the paper style
(results/analysis/v3/figures/<id>_table.{png,pdf}), both from the same rows (lib/booktabs.py).

  python analysis/v3_tables.py                 every table whose inputs exist
  python analysis/v3_tables.py --only T-19,T-41
  python analysis/v3_tables.py --list          each table's status, without writing

A table is written only when every input exists and every model it should list is present. Otherwise
the reason is printed and nothing is written.

Format: accuracy and coverage in % (per observation and deployed to 2 decimals, per code to 1),
differences in pp, ECE / Brier / confidences / temperatures as decimals. Models are always in ALL4's
order. Seed statistics are means over the declared seeds with the sample sd (Llama has one seed, so
no sd). "−" is a minus sign. The caption written into the .tex is the table ID and a short title.
Aggregate output only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import registry as R  # noqa: E402

_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
_B = importlib.util.spec_from_file_location("booktabs", HERE / "lib" / "booktabs.py")
BT = importlib.util.module_from_spec(_B); _B.loader.exec_module(BT)

ALL4 = ["FHIR-MLM", "BCM", "Llama-3.1-8B", "TF-IDF"]          # every table's row order
DEEP3 = ["FHIR-MLM", "BCM", "Llama-3.1-8B"]
NODESC = ["FHIR-MLM", "TF-IDF"]
OFFSHELF = ["Llama-3.1-8B", "MedGemma-4B", "Gemma-3-12B", "Meditron-7B", "OpenBioLLM-8B"]
ARM = {"clean": "clean", "pert": "perturbed"}
MINUS = "−"
TABLES: dict = {}


# -- formatting --------------------------------------------------------------
def f(x, d=4):
    return "—" if pd.isna(x) else f"{x:,.{d}f}".replace("-", MINUS)


def sg(x, d=2):
    return "—" if pd.isna(x) else f"{x:+,.{d}f}".replace("-", MINUS)


def n(x):
    return "—" if pd.isna(x) else f"{int(round(x)):,}"


def pc(x, d=1):
    return "—" if pd.isna(x) else f"{100 * x:.{d}f}%"


def ci(lo, hi, d=2, signed=True):
    g = sg if signed else f
    return f"[{g(lo, d)}, {g(hi, d)}]"


def pval(p):
    if pd.isna(p):
        return "—"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def with_sd(s, x, d=2):
    """A value with its seed sd in the same cell. Parentheses hold only a seed sd. A single run has no sd."""
    return s if pd.isna(x) else f"{s} ({f(x, d)})"


def po(x):
    """Per-observation (or deployed) accuracy or coverage, in %, 2 decimals."""
    return "—" if pd.isna(x) else f"{100 * x:.2f}"


def pcd(x):
    """Per-code accuracy or coverage, in %, 1 decimal."""
    return "—" if pd.isna(x) else f"{100 * x:.1f}"


def rng(lo, hi, fn):
    """A level interval, lo–hi, in the number format of the cell."""
    return f"{fn(lo)}–{fn(hi)}"


# -- sources and output ------------------------------------------------------
def stamp_of(analysis: str, file: str) -> dict | None:
    """Where `file` is and its `_run.json` record (models, newest stage's time and commit), or None if absent."""
    p = R.OUT_ROOT / analysis / file
    if not p.exists():
        return None
    run = p.parent / "_run.json"
    rec = json.loads(run.read_text()) if run.exists() else {}
    last = max(rec.get("stages", {}).values(), key=lambda s: s.get("written", ""), default={})
    return {"path": p, "models": rec.get("models", []), "written": last.get("written", "?"), "git": last.get("git", "?")}


def load(analysis: str, file: str):
    """The DataFrame, or None if the file is absent."""
    s = stamp_of(analysis, file)
    if not s:
        return None
    return pd.read_parquet(s["path"]) if s["path"].suffix == ".parquet" else pd.read_csv(s["path"])


def table(tid: str, title: str, needs: list, expected: list):
    def deco(fn):
        TABLES[tid] = {"fn": fn, "title": title, "needs": needs, "expected": expected}
        return fn
    return deco


def fid(tid: str) -> str:
    return tid.replace("-", "").lower()


def build(tid: str, write: bool = True) -> dict:
    """Build one table. Returns tid, status, complete (bool, or None while an input is missing), the rows
    (`out`) and the stamps of its sources. Writes the .tex and the drawn table when `write` is set and the
    table is complete."""
    spec = TABLES[tid]
    src, stamps = {}, []
    for a, fl in spec["needs"]:
        df = load(a, fl)
        if df is None:
            return {"tid": tid, "status": f"waiting for {a}/{fl}", "complete": None, "out": None, "stamps": stamps}
        src[fl] = df
        stamps.append({"source": f"{a}/{fl}", **stamp_of(a, fl)})
    out = spec["fn"](src)
    present = out.pop("present")
    incomplete = out.pop("incomplete", [])
    missing = [m for m in spec["expected"] if m not in present]
    complete = not (missing or incomplete)
    why = "; ".join(x for x in [f"missing {', '.join(missing)}" if missing else "",
                                f"pending {', '.join(incomplete)}" if incomplete else ""] if x)
    status = "complete" if complete else f"INCOMPLETE ({why}): not written"
    res = {"tid": tid, "status": status, "complete": complete, "out": out, "stamps": stamps}
    if write and complete:
        root = R.OUT_ROOT
        (root / "tables").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)
        kw = {k: out[k] for k in ("groups", "right", "total", "indent", "gaps") if k in out}
        kw["split"] = BT.split_cols(out["body"], out.get("right", ()))   # so the estimates line up
        (root / "tables" / f"{fid(tid)}.tex").write_text(BT.tex(
            out["header"], out["body"], caption=f"{tid}: {spec['title']}", label=f"tab:{fid(tid)}",
            comment=f"{tid} -- generated by analysis/v3_tables.py", **kw))
        png, w = BT.draw(FS, out["header"], out["body"], str(root / "figures" / f"{fid(tid)}_table"), **kw,
                         **({"pad": out["pad"]} if "pad" in out else {}))
        res["status"] += f" -> {png} ({w:.2f} in{'; WIDER than the text width' if w > FS.TEXTWIDTH else ''})"
    return res


# -- shared ------------------------------------------------------------------
def _joint(S, fam, file="catalogue_joint.csv"):
    j = S[file]
    return j[j.arm == fam].set_index("model")


def _models_in(j, order):
    return [m for m in order if m in j.index]


def _pair(pw, fam, a, b):
    """(a − b mean, lo, hi) from a catalogue_pairwise.csv row of family `fam`, whichever way it is stored."""
    q = pw[(pw.arm == fam) & (((pw.model_a == a) & (pw.model_b == b)) | ((pw.model_a == b) & (pw.model_b == a)))]
    r = q.iloc[0]
    s = 1 if r.model_a == a else -1
    lo, hi = sorted((s * r.lo, s * r.hi))
    return s * r.diff_mean, lo, hi


def _summary(s: pd.DataFrame, family: str) -> dict:
    """{model: {quantity: (mean, sd)}} for one rq3 family."""
    g = s[s.family == family]
    return {m: {q: (r["mean"], r["sd"]) for q, r in d.set_index("quantity").iterrows()} for m, d in g.groupby("model")}


CELLS = ["A", "B", "C", "D"]
# The differences read as losses and gains. A − B is what the shift costs, D − B what perturbation training
# recovers, and A − C what it costs on clean numbers. So gap closed = (D − B) / (A − B) can be read off the
# columns. It is the ratio of the seed means. The summary holds the mean of per-seed ratios instead.
CONTRASTS = [("B_minus_A", "A − B", -1), ("D_minus_B", "D − B", 1), ("C_minus_A", "A − C", -1)]
H_2X2 = ["Model", "A", "B", "C", "D", "A − B", "D − B", "A − C", "Gap closed"]
G_2X2 = [("Top-1 (%)", 1, 4), ("Difference (pp)", 5, 7)]


def _rows_2x2(q: dict, models: list, d: int = 2, label=lambda m: m) -> list:
    """One row of seed means per model, in % / pp (the seed sds go in the thesis caption)."""
    return [[label(m)] + [f(100 * q[m][c][0], d) for c in CELLS]
            + [sg(sign * 100 * q[m][k][0], d) for k, _, sign in CONTRASTS]
            + [pc(q[m]["D_minus_B"][0] / -q[m]["B_minus_A"][0], 1) if "gap_closed" in q[m] else "—"]
            for m in models if m in q]


def _arm_table(s: pd.DataFrame, cols: list, extra: list, d: int = 2) -> tuple[list, set, list]:
    """Rows grouped by training arm, clean-trained models first, as T-30 groups by names."""
    body, gaps, present = [], set(), []
    for arm in ("clean", "pert"):
        start = len(body)
        for m in ALL4:
            q = s[(s.model == m) & (s.arm == arm)].set_index("quantity")
            if q.empty:
                continue
            present.append(m)
            body.append([ARM[arm].capitalize() if len(body) == start else "", m]
                        + [f(100 * q.loc[c, "mean"], d) if c in q.index else "—" for c in cols]
                        + [sg(100 * q.loc[c, "mean"], d) if c in q.index else "—" for c in extra])
        if start and len(body) > start:
            gaps.add(start)
    return body, gaps, sorted(set(present))


# -- Overall comparison ------------------------------------------------------
@table("T-10", "accuracy at the three units (para, scenario D, matched set, means over seeds)",
       [("catalogue", "catalogue_joint.csv"), ("observation", "obs_headline.csv")], ALL4)
def t10(S):
    """Per observation top-1 and top-5, per code top-1 (with its CI) and top-5, deployed top-1 (with its CI)."""
    j = _joint(S, "para_D")
    o = S["obs_headline.csv"]; o = o[o.set == "para_xsite_D"].set_index("model")
    body = []
    for m in _models_in(j, ALL4):
        r = j.loc[m]
        ob = (lambda c: po(o.loc[m, c]) if m in o.index else "—")
        body.append([m, ob("top1_mean"), ob("top5_mean"), pcd(r.top1_mean), rng(r.top1_A_lo, r.top1_A_hi, pcd),
                     pcd(r.top5_mean), po(r.dep_mean), rng(r.dep_A_lo, r.dep_A_hi, po)])
    return {"header": ["Model", "Top-1", "Top-5", "Top-1", "95% CI", "Top-5", "Top-1", "95% CI"],
            "groups": [("Per observation (%)", 1, 2), ("Per code (%)", 3, 5), ("Deployed (%)", 6, 7)],
            "body": body, "right": set(range(1, 8)), "total": False, "present": list(j.index)}


# -- Patient history (RQ1) ---------------------------------------------------
def _so(S, cond, cell="D"):
    t = S["singleobs_obs.csv"]
    return t[(t.comparison == "history_vs_singleobs") & (t.condition == cond) & (t.cell == cell)].iloc[0]


@table("T-32", "FHIR-MLM with and without patient history at the three units (para, scenario D, 5 seeds)",
       [("singleobs", "singleobs_obs.csv"), ("catalogue_per_model", "catalogue_joint.csv"),
        ("catalogue_per_model", "catalogue_pairwise.csv")], ["FHIR-MLM"])
def t32(S):
    """Per observation with the joint patient × seed CIs, per code with the paired code × seed interval,
    and deployed, all on FHIR-MLM's own rows."""
    r = _so(S, "para_xsite")
    j = _joint(S, "so_para__mlm")
    h, s1 = j.loc["FHIR-MLM"], j.loc["FHIR-MLM (single obs)"]
    dm, lo, hi = _pair(S["catalogue_pairwise.csv"], "so_para__mlm", "FHIR-MLM", "FHIR-MLM (single obs)")
    body = [["With history", po(r.a_mean), rng(r.a_lo, r.a_hi, po), pcd(h.top1_mean), rng(h.top1_A_lo, h.top1_A_hi, pcd),
             po(h.dep_mean)],
            ["Single observation", po(r.b_mean), rng(r.b_lo, r.b_hi, po), pcd(s1.top1_mean),
             rng(s1.top1_A_lo, s1.top1_A_hi, pcd), po(s1.dep_mean)],
            ["Difference (pp)", sg(100 * r.a_minus_b, 2), ci(100 * r.diff_lo, 100 * r.diff_hi, 2), sg(100 * dm, 1),
             ci(100 * lo, 100 * hi, 1), sg(100 * (h.dep_mean - s1.dep_mean), 2)]]
    return {"header": ["Setup", "Top-1", "95% CI", "Top-1", "95% CI", "Top-1"],
            "groups": [("Per observation (%)", 1, 2), ("Per code (%)", 3, 4), ("Deployed (%)", 5, 5)],
            "body": body, "right": set(range(1, 6)), "total": True, "present": ["FHIR-MLM"]}


@table("T-34", "the contribution of patient history by training frequency (para, scenario D, 5 seeds)",
       [("singleobs", "singleobs_bands_summary.csv")], ["FHIR-MLM"])
def t34(S):
    t = S["singleobs_bands_summary.csv"].set_index("band")
    names = {"rare": "<1,000", "common": "1,000–99,999", "frequent": "≥100,000"}   # bands are [lo, hi) (T-04)
    body = [[names[b], n(r.n_obs), f"{100 * r.volume:.1f}", po(r.history), po(r.single_obs),
             f"{sg(100 * r.drop_mean, 2)} ({f(100 * r.drop_sd, 2)})"] for b, r in t.reindex(["rare", "common", "frequent"]).iterrows()]
    # parentheses hold only a seed sd, so the share has its own column
    return {"header": ["Training observations", "Observations", "Share (%)", "With history", "Single obs.",
                       "Difference (pp)"],
            "groups": [("Test set", 1, 2), ("Per-observation top-1 (%)", 3, 4)], "body": body,
            "right": set(range(1, 6)), "total": False, "present": ["FHIR-MLM"]}


@table("T-35", "the cost of removing patient history on paraphrase, RELMA and no-description names (scenario D, 5 seeds)",
       [("singleobs", "singleobs_obs.csv"), ("singleobs_relma", "singleobs_relma.csv"),
        ("catalogue_per_model", "catalogue_joint.csv"), ("catalogue_per_model", "catalogue_pairwise.csv")], ["FHIR-MLM"])
def t35(S):
    """One row per name condition (paraphrase, RELMA, no description) with the units as column groups. Each row
    is on FHIR-MLM's own rows for that condition. Per observation it has the joint patient × seed interval, and
    per code the paired code × seed interval."""
    header = ["Names", "History", "Single obs.", "Difference (pp)", "History", "Single obs.", "Difference (pp)"]
    groups = [("Per observation (%)", 1, 3), ("Per code (%)", 4, 6)]
    p, d = _so(S, "para_xsite"), _so(S, "nodesc_xsite")
    rl = S["singleobs_relma.csv"].set_index("names")
    if "relma_s1_a0" not in rl.index:
        return {"header": header, "groups": groups, "body": [["RELMA", "", "", "pending", "", "", ""]],
                "right": set(range(1, 7)), "total": False, "present": ["FHIR-MLM"],
                "incomplete": ["the single-observation RELMA dumps"]}
    r = rl.loc["relma_s1_a0"]
    pw = S["catalogue_pairwise.csv"]
    body = []
    for name, obs, fam in (("Para", (p.a_mean, p.b_mean, p.a_minus_b, p.diff_lo, p.diff_hi), "so_para__mlm"),
                           ("RELMA", (r.history, r.single_obs, r.gap, r.gap_lo, r.gap_hi), "so_relma_a0__mlm"),
                           ("NoDesc", (d.a_mean, d.b_mean, d.a_minus_b, d.diff_lo, d.diff_hi), "so_nodesc__mlm")):
        j = _joint(S, fam)
        h, s1 = j.loc["FHIR-MLM"], j.loc["FHIR-MLM (single obs)"]
        dm, lo, hi = _pair(pw, fam, "FHIR-MLM", "FHIR-MLM (single obs)")
        body.append([name, po(obs[0]), po(obs[1]), f"{sg(100 * obs[2], 2)} {ci(100 * obs[3], 100 * obs[4], 2)}",
                     pcd(h.top1_mean), pcd(s1.top1_mean), f"{sg(100 * dm, 1)} {ci(100 * lo, 100 * hi, 1)}"])
    # 4.5 pt column padding keeps the seven columns inside the text width
    return {"header": header, "groups": groups, "body": body, "right": set(range(1, 7)), "total": False,
            "pad": 4.5, "present": ["FHIR-MLM"]}


# -- Descriptions (RQ2) ------------------------------------------------------
@table("T-15", "with and without descriptions, both units, on the observations both models scored in both "
       "conditions (scenario D, means over seeds)",
       [("catalogue_per_model", "catalogue_joint.csv"), ("observation_t15", "obs_t15.csv")], NODESC)
def t15(S):
    p, d = _joint(S, "para_D_mlm_tfidf"), _joint(S, "nodesc_D_t15")
    o = S["obs_t15.csv"]
    o = o[o.table == "T-15"].groupby(["model", "setup"]).acc.mean().unstack()
    both = [m for m in NODESC if m in p.index and m in d.index and m in o.index]
    assert (p.loc[both, "n_codes"] == d.loc[both, "n_codes"]).all(), "T-15's two per-code arms differ in codes"
    # Change = NoDesc − Para from the unrounded means, so the table shows the drops the thesis text quotes
    body = [[m, po(o.loc[m, "para"]), po(o.loc[m, "nodesc"]), sg(100 * (o.loc[m, "nodesc"] - o.loc[m, "para"]), 2),
             pcd(p.loc[m, "top1_mean"]), pcd(d.loc[m, "top1_mean"]),
             sg(100 * (d.loc[m, "top1_mean"] - p.loc[m, "top1_mean"]), 1)] for m in both]
    return {"header": ["Model", "Para", "NoDesc", "Change (pp)", "Para", "NoDesc", "Change (pp)"],
            "groups": [("Per observation (%)", 1, 3), ("Per code (%)", 4, 6)], "body": body,
            "right": {1, 2, 3, 4, 5, 6}, "total": False, "present": both}


@table("T-16", "accuracy on real institutional names (RELMA a0, scenario D, matched set, means over seeds)",
       [("catalogue", "catalogue_joint.csv")], ALL4)
def t16(S):
    """Per observation top-1, per code top-1 (with its CI) and top-5, deployed top-1 (with its CI), on one set."""
    j = _joint(S, "relma_a0_D")
    body = [[m, po(r.obs_mean), pcd(r.top1_mean), rng(r.top1_A_lo, r.top1_A_hi, pcd), pcd(r.top5_mean), po(r.dep_mean),
             rng(r.dep_A_lo, r.dep_A_hi, po)] for m, r in j.reindex(_models_in(j, ALL4)).iterrows()]
    return {"header": ["Model", "Top-1", "Top-1", "95% CI", "Top-5", "Top-1", "95% CI"],
            "groups": [("Per observation (%)", 1, 1), ("Per code (%)", 2, 4), ("Deployed (%)", 5, 6)],
            "body": body, "right": set(range(1, 7)), "total": False, "present": list(j.index)}


# -- Numbers (RQ3) -----------------------------------------------------------
@table("T-41", "number perturbation, para: the 2×2 per observation (means over seeds)",
       [("rq3", "rq3_obs_summary.csv")], ALL4)
def t41(S):
    q = _summary(S["rq3_obs_summary.csv"], "para")
    return {"header": H_2X2, "body": _rows_2x2(q, ALL4), "groups": G_2X2, "right": set(range(1, 9)), "total": False,
            "present": list(q)}


@table("T-42", "number perturbation, para: the 2×2 per code, differences between scenarios (paired code × seed intervals)",
       [("catalogue_per_model", "catalogue_rq3_cells.csv"), ("catalogue_per_model", "catalogue_rq3_pairs.csv")], ALL4)
def t42(S):
    c, pr = S["catalogue_rq3_cells.csv"], S["catalogue_rq3_pairs.csv"]
    body, present = [], []
    for m in ALL4:
        fam = [x for x in set(c.family) if x.startswith("rq3_para__") and (c[c.family == x].model == m).any()]
        if not fam:
            continue
        pm = pr[pr.family == fam[0]]

        def diff(x, y):                                  # cell x minus cell y, from the catalogue pair rows
            r = pm[((pm.model_a == x) & (pm.model_b == y)) | ((pm.model_a == y) & (pm.model_b == x))].iloc[0]
            s = 1 if r.model_a == x else -1
            lo, hi = sorted((s * r.lo, s * r.hi))
            return s * r.diff_mean, lo, hi
        d = {k: diff(*k) for k in (("A", "B"), ("D", "B"), ("A", "C"))}
        body.append([m] + [f"{sg(100 * d[k][0], 1)} {ci(100 * d[k][1], 100 * d[k][2], 1)}" for k in d])
        present.append(m)
    # differences only, each with its CI in the cell, because the thesis text uses no per-code level
    return {"header": ["Model", "A − B", "D − B", "A − C"], "groups": [("Per-code difference (pp)", 1, 3)],
            "body": body, "right": {1, 2, 3}, "total": False, "present": present}


@table("T-43", "values vs reference ranges: which perturbation moves accuracy, trained on clean numbers (per observation, "
       "means over seeds)", [("rq3", "rq3_valuerange_summary.csv"), ("rq3", "rq3_obs_summary.csv")], ALL4)
def t43(S):
    """Columns are test conditions (none perturbed = scenario A). Only models trained on clean numbers are shown.
    For models trained on perturbed numbers the four columns agree closely, as T-41's C ≈ D already shows. The
    numbers book (analysis/v3_book.py) keeps that spread. Each row has one source. FHIR-MLM uses its dumps.
    TF-IDF uses the fit of pipeline/07_tfidf_train_eval.py, which alone scored the value-only and range-only
    variants (07 and the refit in the dumps differ by ≤ 1.3e-4). BCM and Llama use the positions that all the
    row's cells share (v3_rq3), a few dozen fewer than the 2×2 intersection. So the both-perturbed column is
    checked against scenario B exactly for FHIR-MLM and within a tolerance for the others."""
    vr, para = S["rq3_valuerange_summary.csv"], _summary(S["rq3_obs_summary.csv"], "para")
    body, present = [], []
    for m in ALL4:
        d = vr[(vr.model == m) & (vr.arm == "clean")].set_index("quantity")["mean"]
        if d.empty:
            continue
        tol = {"TF-IDF": 1e-3, "BCM": 1e-4, "Llama-3.1-8B": 1e-4}.get(m, 1e-12)
        if abs(d["pert_both"] - para[m]["B"][0]) > tol:
            raise AssertionError(f"{m}: the ablation's both-perturbed test is not scenario B")
        present.append(m)
        body.append([m, po(d["clean_test"]), po(d["pert_value"]), po(d["pert_range"]), po(d["pert_both"])])
    return {"header": ["Model", "None", "Values only", "Ranges only", "Both"],
            "groups": [("Test numbers perturbed: top-1 (%)", 1, 4)], "body": body,
            "right": {1, 2, 3, 4}, "total": False, "present": present}


# -- Confidence and auto-mapping (RQ4) ---------------------------------------
@table("T-24", "per-code auto-map coverage at an error budget, paraphrase and RELMA names (scenario D, means over seeds)",
       [("catalogue", "catalogue_coverage_by_seed.csv")], ALL4)
def t24(S):
    """Paraphrase and RELMA a0 side by side, with one row per model and a column group per name condition. The
    values are coverage (%) at a per-code error budget of 1%, 5% and 10%, with the seed range at 1%."""
    c = S["catalogue_coverage_by_seed.csv"]
    blocks = [c[c.arm == fam].groupby("model").agg(c1=("cov@0.01", "mean"), c1lo=("cov@0.01", "min"),
                                                   c1hi=("cov@0.01", "max"), c5=("cov@0.05", "mean"),
                                                   c10=("cov@0.1", "mean"), k=("seed", "nunique"))
              for fam in ("para_D", "relma_a0_D")]
    present = sorted(set().union(*(g.index for g in blocks)))
    body = []
    for m in [m for m in ALL4 if m in present]:
        row = [m]
        for g in blocks:
            if m not in g.index:
                row += ["—"] * 4
                continue
            r = g.loc[m]
            row += [pcd(r.c1), rng(r.c1lo, r.c1hi, pcd) if r.k > 1 else "—", pcd(r.c5), pcd(r.c10)]
        body.append(row)
    return {"header": ["Model"] + ["1%", "Range", "5%", "10%"] * 2, "groups": [("Para", 1, 4), ("RELMA", 5, 8)],
            "body": body, "right": set(range(1, 9)), "total": False, "present": present}


@table("T-21", "per-observation calibration, before and after temperature scaling (para, scenario D, matched set)",
       [("calibration", "calibration_matched.csv"), ("calibration", "temperatures.csv")], ALL4)
def t21(S):
    """Calibration at T = 1, then temperature scaling (T fitted on the perturbed validation split, seed by seed,
    with the ECE after scaling)."""
    c = S["calibration_matched.csv"]; c = c[c.condition == "para_xsite"]
    g = c[c.scaling == "T=1"].groupby("model")[["acc", "ece_mass15", "mce_mass15", "brier_top", "gap_conf_minus_acc"]].mean()
    e = c.groupby(["model", "scaling"]).ece_mass15.mean().unstack()
    T = S["temperatures.csv"]; T = T[T.arm == "pert"].groupby("model")["T"].agg(["mean", "std", "count"])
    body = []
    for m in _models_in(g, ALL4):
        r = g.loc[m]
        t = (f(T.loc[m, "mean"], 3) + (f" ({f(T.loc[m, 'std'], 3)})" if T.loc[m, "count"] > 1 else "")) if m in T.index else "—"
        body.append([m, po(r.acc), f(r.ece_mass15, 5), f(r.mce_mass15, 4), f(r.brier_top, 5), sg(r.gap_conf_minus_acc, 5), t,
                     f(e.loc[m, "T=fitted"], 5) if "T=fitted" in e.columns else "—"])
    return {"header": ["Model", "Accuracy (%)", "ECE", "MCE", "Brier", "Conf − acc", "T (seed sd)", "ECE"],
            "groups": [("At T = 1", 1, 5), ("Temperature-scaled", 6, 7)], "body": body, "right": set(range(1, 8)),
            "total": False, "present": list(g.index)}


@table("T-30", "per-code calibration, paraphrase, RELMA and no-description names (scenario D, means over seeds, seed sd "
       "in parentheses)", [("catalogue", "catalogue_joint.csv")], ALL4)
def t30(S):
    body, gaps, present = [], set(), set()
    blocks = [("Paraphrase", _joint(S, "para_D")), ("RELMA", _joint(S, "relma_a0_D"))]
    # NoDesc at the bottom: FHIR-MLM and TF-IDF on their own matched codes, as F-12's panel (c)
    pj = load("catalogue_per_model", "catalogue_joint.csv")
    if pj is not None and (pj.arm == "nodesc_D").any():
        blocks.append(("NoDesc", pj[pj.arm == "nodesc_D"].set_index("model")))
    for label, j in blocks:
        if body:
            gaps.add(len(body))
        for i, m in enumerate(_models_in(j, ALL4)):
            r = j.loc[m]; one = r.n_seeds < 2
            sd = (lambda v, d: "") if one else (lambda v, d: f" ({f(v, d)})")
            body.append([label if i == 0 else "", m, f(r.ece_mean, 4) + sd(r.ece_sd, 4),
                         sg(r.conf_minus_acc_mean, 3) + sd(r.conf_minus_acc_sd, 3),
                         f(r.conf_correct_mean, 3) + sd(r.conf_correct_sd, 3), f(r.conf_wrong_mean, 3) + sd(r.conf_wrong_sd, 3)])
            present.add(m)
    return {"header": ["Names", "Model", "ECE", "Conf − acc", "Conf, correct", "Conf, wrong"], "body": body,
            "right": {2, 3, 4, 5}, "total": False, "gaps": gaps, "present": sorted(present)}


# -- Other findings ----------------------------------------------------------
@table("T-19", "LOINC axes on which errors differ from the truth (para, scenario D, mean over seeds)",
       [("errors", "errors_axis_marginals.csv")], ALL4)
def t19(S):
    a = S["errors_axis_marginals.csv"]
    a = a[(a.subset == "all") & (a.condition == "para_xsite")]
    g = a.groupby("model").mean(numeric_only=True)
    axes = [("COMPONENT", "Component"), ("METHOD_TYP", "Method"), ("SYSTEM", "System"), ("PROPERTY", "Property"),
            ("SCALE_TYP", "Scale"), ("TIME_ASPCT", "Time")]
    body = [[m, n(g.loc[m, "n_errors"])] + [f"{g.loc[m, f'pct_{c}']:.1f}" for c, _ in axes]
            + [f(g.loc[m, "mean_axes_differing"], 2)] for m in ALL4 if m in g.index]
    return {"header": ["Model", "Errors"] + [h for _, h in axes] + ["Mean axes"],
            "groups": [("Errors that differ on the axis (%)", 2, 7)], "body": body,
            "right": set(range(1, 9)), "total": False, "present": list(g.index)}


@table("T-25", "training and inference cost", [("cost", "t25_rows.csv")], ALL4)
def t25(S):
    t = S["t25_rows.csv"].set_index("model")
    body = []
    for m in ALL4:
        if m not in t.index:
            continue
        r = t.loc[m]
        tr = "—" if pd.isna(r.train_hours) else with_sd(f"{r.train_hours:.2f}", r.train_hours_sd)
        body.append([m, f"{r.params_M:,.1f}M", r.device, n(r.obs_per_sec), tr])
    present = [m for m in t.index if pd.notna(t.loc[m, "obs_per_sec"])]
    return {"header": ["Model", "Parameters", "Device", "Observations/s", "Training time (h)"], "body": body,
            "right": {1, 3, 4}, "total": False, "present": present}


@table("T-26", "later windows of long records: coverage and accuracy (scenario D, seed 0)",
       [("sliding", "sliding_denominators.csv"), ("sliding", "sliding_summary.csv")], DEEP3)
def t26(S):
    """The first window's share of the scored patients' observations and what all windows add. Also window 0 vs
    later windows for patients with ≥ 2 windows, with a patient-clustered CI."""
    d = S["sliding_denominators.csv"].assign(label=lambda x: x.model.map(R.LABEL)).set_index("label")
    s = S["sliding_summary.csv"].assign(label=lambda x: x.model.map(R.LABEL)).set_index("label")
    incomplete = [] if "acc_window0_multi" in s.columns else ["window accuracies"]
    body = []
    for m in [m for m in DEEP3 if m in d.index and m in s.index]:
        r, q = d.loc[m], s.loc[m]
        w0, w1 = q.get("acc_window0_multi", np.nan), q.get("acc_later_multi", np.nan)
        # the patients with ≥ 2 windows (the accuracy columns' denominator) go in the caption: n_patients_multi_window
        # Share = first window ÷ the scored patients' test observations. Dividing by all windows would only
        # give 1 / Gain.
        body.append([m, f"{100 * r.n_first_window / r.obs_of_scored_patients:.1f}", f"{r.gain:.2f}×", po(w0), po(w1),
                     sg(-q.gap_patient_matched_per10k / 100, 2), ci(-q.gap_ci_hi_per10k / 100, -q.gap_ci_lo_per10k / 100, 2)])
    return {"header": ["Model", "Share (%)", "Gain", "Window 0", "Windows 1+", "Change (pp)", "95% CI"],
            "groups": [("First window", 1, 2), ("Top-1 (%)", 3, 4)], "body": body,
            "right": set(range(1, 7)), "total": False, "present": [m for m in d.index if m in s.index],
            "incomplete": incomplete}


# -- Chapter 6, further tables -----------------------------------------------
@table("T-11", "top-1 by training frequency (para, scenario D, matched set, mean over seeds)",
       [("observation", "obs_strata.csv")], ALL4)
def t11(S):
    st = S["obs_strata.csv"]
    st = st[(st.set == "para_D") & (st.scheme == "t04")]
    acc = st.groupby(["stratum", "model"]).acc.mean().unstack()
    meta = st.groupby("stratum")[["lo", "hi", "n_codes", "n_obs"]].first()
    names = {"rare": "Rare", "common": "Common", "frequent": "Frequent"}
    body = []
    for k in ("rare", "common", "frequent"):
        lo, hi = meta.loc[k, "lo"], meta.loc[k, "hi"]
        span = f"<{n(hi)}" if k == "rare" else (f"≥{n(lo)}" if k == "frequent" else f"{n(lo)}–{n(hi - 1)}")   # [lo, hi), as T-04
        body.append([f"{names[k]} ({span})", n(meta.loc[k, "n_codes"]), n(meta.loc[k, "n_obs"])]
                    + [po(acc.loc[k, m]) if m in acc.columns else "—" for m in ALL4])
    return {"header": ["Training observations", "Codes", "Observations"] + ALL4,
            "groups": [("Per-observation top-1 (%)", 3, 6)], "body": body,
            "right": set(range(1, 7)), "total": False, "present": list(acc.columns)}


def aggregation_by_band(p: pd.DataFrame) -> pd.DataFrame:
    """Per-code top-1 from one random observation vs all of a code's observations, by training-frequency group of
    the gold LOINC. The groups are those of thesis Chapter 4 (rare < 1,000 and frequent ≥ 100,000 training
    observations). The set is para, scenario D, the four-model matched set, with means over seeds and codes."""
    _O = importlib.util.spec_from_file_location("v3_observation", HERE / "v3_observation.py")
    O = importlib.util.module_from_spec(_O); _O.loader.exec_module(O)
    d = p[["code", "gold", "n_obs", "so_hit1", "sm_hit1", "model"]].copy()
    d["band"] = pd.cut(d.gold.map(O.train_count_map()).fillna(0), [-1, 999.5, 99_999.5, 1e12],
                       labels=["rare", "common", "frequent"])
    return d.groupby(["model", "band"], observed=True).agg(codes=("code", "nunique"), one_obs=("so_hit1", "mean"),
                                                           all_obs=("sm_hit1", "mean"), med=("n_obs", "median"))


@table("T-12", "what aggregating a code's observations adds, overall and by training frequency (para, scenario D, "
       "matched set, means over seeds)", [("catalogue", "catalogue_rules.csv"), ("catalogue", "percode__para_D.parquet")],
       ALL4)
def t12(S):
    """The gain from aggregating, overall and by training-frequency group, so the per-group numbers the thesis
    text gives are in the table it cites. The groups' code-weighted mean is checked against the overall gain."""
    r = S["catalogue_rules.csv"]
    g = r[r.family == "para_D"].groupby("model")[["so_hit1", "sm_hit1"]].mean()
    g["gain"] = g.sm_hit1 - g.so_hit1
    b = aggregation_by_band(S["percode__para_D.parquet"])
    b["gain"] = b.all_obs - b.one_obs
    body = []
    for m, x in g.reindex(_models_in(g, ALL4)).iterrows():
        bm = b.loc[m]
        if abs((bm.gain * bm.codes).sum() / bm.codes.sum() - x.gain) > 5e-4:
            raise AssertionError(f"{m}: the frequency groups' gains do not add up to the overall gain")
        body.append([m, pcd(x.so_hit1), pcd(x.sm_hit1), sg(100 * x.gain, 1)]
                    + [sg(100 * bm.loc[k, "gain"], 1) for k in ("rare", "common", "frequent")])
    return {"header": ["Model", "One observation", "All observations", "All codes", "Rare", "Common", "Frequent"],
            "groups": [("Per-code top-1 (%) from", 1, 2), ("Gain (pp)", 3, 6)], "body": body,
            "right": set(range(1, 7)), "total": False, "present": list(g.index)}


@table("T-14", "the codes a deployed model gets wrong (para, scenario D, matched set, means over seeds)",
       [("catalogue", "catalogue_joint.csv")], ALL4)
def t14(S):
    j = _joint(S, "para_D")
    # parentheses hold only a seed sd: the count's sd in its cell, the shares in their own columns
    body = [[m, f"{r.codes_wrong_mean:.1f}" + (f" ({f(r.codes_wrong_sd, 1)})" if r.n_seeds > 1 else ""),
             f"{100 * r.pct_codes_wrong_mean:.1f}", f"{100 * r.pct_vol_wrong_mean:.2f}", f(r.med_obs_wrong_mean, 1),
             f(r.med_obs_right_mean, 1)]
            for m, r in j.reindex(_models_in(j, ALL4)).iterrows()]
    # each column reads as a phrase without the caption: "wrong codes: % of observations",
    # "median observations per code: wrong". No label is used twice.
    return {"header": ["Model", "Number", "% of codes", "% of observations", "Wrong", "Correct"],
            "groups": [("Wrong codes", 1, 3), ("Median observations per code", 4, 5)], "body": body,
            "right": set(range(1, 6)), "total": False, "present": list(j.index)}


@table("T-17", "per-observation top-1, RELMA vs paraphrase names (paired, each model on its own rows)",
       [("observation", "obs_relma_vs_para.csv")], ALL4)
def t17(S):
    t = S["obs_relma_vs_para.csv"].set_index("model")
    body = [[m, po(r.relma_mean), po(r.para_mean), with_sd(sg(100 * r.change_mean, 2), 100 * r.change_sd)]
            for m, r in t.reindex(_models_in(t, ALL4)).iterrows()]
    return {"header": ["Model", "RELMA", "Paraphrase", "Change (pp)"],
            "groups": [("Per-observation top-1 (%)", 1, 2)], "body": body,
            "right": {1, 2, 3}, "total": False, "present": list(t.index)}


@table("T-20", "mean confidence by how far the prediction is from the truth (para, scenario D, mean over seeds)",
       [("errors", "errors_axis_marginals.csv")], ALL4)
def t20(S):
    a = S["errors_axis_marginals.csv"]
    a = a[(a.subset == "all") & (a.condition == "para_xsite")]
    g = a.groupby("model").mean(numeric_only=True)
    body = [[m, f(g.loc[m, "mean_conf_correct"], 3), f(g.loc[m, "mean_conf_1axis"], 3),
             f(g.loc[m, "mean_conf_3plus_axes"], 3)] for m in ALL4 if m in g.index]
    return {"header": ["Model", "Correct", "1 axis differs", "≥ 3 axes differ"], "body": body,
            "right": {1, 2, 3}, "total": False, "present": list(g.index)}


@table("T-22", "per-observation auto-map coverage at an error budget (para, scenario D, matched set, means over seeds)",
       [("calibration", "selective_matched.csv")], ALL4)
def t22(S):
    s = S["selective_matched.csv"]
    # no 5% column: every model's error rate is below 5%, so each auto-maps 100% there
    g = s[s.condition == "para_xsite"].groupby("model")[["cov@0.001", "cov@0.01"]].mean()
    body = [[m] + [po(r[c]) for c in ("cov@0.001", "cov@0.01")] for m, r in g.reindex(_models_in(g, ALL4)).iterrows()]
    return {"header": ["Model", "0.1%", "1%"], "groups": [("Coverage (%) at an error budget of", 1, 2)],
            "body": body, "right": {1, 2}, "total": False, "present": list(g.index)}


@table("T-28", "off-the-shelf models, few-shot: what the output was (4,952 observations)",
       [("offshelf", "t28_rows.csv")], OFFSHELF)
def t28(S):
    # perturbed numbers only, as in F-11's left panel. F-11's right panel compares the share correct with clean
    # numbers, which is all the thesis text says about them.
    t = S["t28_rows.csv"]
    body = []
    for m in OFFSHELF:
        r = t[(t.model == m) & (t.arm == "pert")]
        if r.empty:
            continue
        r = r.iloc[0]
        body.append([m, f"{r.correct_pct:.2f}", f"{r.valid_wrong_pct:.1f}", f"{r.not_a_loinc_pct:.1f}",
                     f"{r.unparseable_pct:.1f}"])
    return {"header": ["Model", "Correct", "Real LOINC, wrong", "Not a real LOINC", "No LOINC"],
            "groups": [("Share of outputs (%)", 1, 4)], "body": body, "right": {1, 2, 3, 4}, "total": False,
            "present": sorted(set(t.model))}


@table("T-45", "the 2×2 without descriptions and for single-observation FHIR-MLM (per observation, means over seeds)",
       [("rq3", "rq3_obs_summary.csv")], NODESC)
def t45(S):
    s = S["rq3_obs_summary.csv"]
    body = []
    for fam, models, name in (("nodesc", NODESC, "NoDesc {}"), ("singleobs_para", ["FHIR-MLM"], "Single obs. {}"),
                              ("singleobs_nodesc", ["FHIR-MLM"], "Single obs. NoDesc {}")):
        body += _rows_2x2(_summary(s, fam), models, label=name.format)
    # 4.5 pt column padding keeps the nine columns inside the text width
    return {"header": ["Setup"] + H_2X2[1:], "body": body, "groups": G_2X2, "right": set(range(1, 9)), "total": False,
            "pad": 4.5, "present": sorted(set(s[s.family == "nodesc"].model))}


@table("T-46", "real institutional names (RELMA a0): trained on clean (B) vs perturbed (D) numbers",
       [("rq3", "rq3_obs_summary.csv")], ALL4)
def t46(S):
    s = S["rq3_obs_summary.csv"]
    rl, pa = _summary(s, "relma_a0"), _summary(s, "para")
    body = []
    for m in [m for m in ALL4 if m in rl]:
        v = rl[m]
        body.append([m, po(v["B"][0]), po(v["D"][0]), with_sd(sg(100 * v["D_minus_B"][0], 2), 100 * v["D_minus_B"][1]),
                     with_sd(sg(100 * pa[m]["D_minus_B"][0], 2), 100 * pa[m]["D_minus_B"][1])])
    return {"header": ["Model", "B (%)", "D (%)", "D − B (pp)", "D − B, Para (pp)"],
            "groups": [("RELMA names", 1, 3)], "body": body, "right": set(range(1, 5)), "total": False,
            "present": list(rl)}


@table("T-47", "perturbation severity (per observation, means over seeds)", [("rq3", "rq3_severity_summary.csv")], ALL4)
def t47(S):
    body, gaps, present = _arm_table(S["rq3_severity_summary.csv"], ["sev0.5", "sev1.0", "sev2.0"],
                                     ["band_2x_minus_half"])
    return {"header": ["Trained on", "Model", "0.5×", "1×", "2×", "2× − 0.5× (pp)"],
            "groups": [("Top-1 (%) at a severity of", 2, 4)], "body": body, "right": {2, 3, 4, 5}, "total": False,
            "gaps": gaps, "present": present}


@table("T-48", "five perturbation draws (seed 0)", [("rq3", "rq3_replicates.csv")], ALL4)
def t48(S):
    t = S["rq3_replicates.csv"]
    body, gaps = [], set()
    for arm in ("clean", "pert"):                       # grouped by training arm, as T-47
        start = len(body)
        for m in ALL4:
            r = t[(t.model == m) & (t.arm == arm)]
            if r.empty:
                continue
            r = r.iloc[0]
            # 3 decimals here, because the draws differ by hundredths of a pp and 2 decimals would hide that.
            # "Spread" covers all six draws, as the caption says. There is no CI column. The t interval over
            # r1–r5 is about the mean over draws, but the question is how far one draw moves. The interval
            # stays in rq3_replicates.csv.
            body.append([ARM[arm].capitalize() if len(body) == start else "", m, f(100 * r.base_draw, 3),
                         f(100 * r.r_mean, 3), f(100 * r.spread_all_six, 3)])
        if start and len(body) > start:
            gaps.add(start)
    return {"header": ["Trained on", "Model", "Base draw (%)", "Mean of 5 draws (%)", "Spread (pp)"],
            "body": body, "right": {2, 3, 4}, "total": False, "gaps": gaps, "present": sorted(set(t.model))}


def ambiguity_table(a: pd.DataFrame, cell: str) -> dict:
    """T-49's layout for one scenario: each model's error rates on ambiguous and unique paraphrase strings, the
    difference with its joint patient × seed CI, and the ratio. T-49 is scenario D, and the numbers book
    (v3_book.py) shows scenario A."""
    g = a[a.cell == cell].set_index("model")
    body = [[m, f(100 * r.err_ambiguous, 2), f(100 * r.err_unique, 2), sg(100 * r.gap, 2),
             ci(100 * r.gap_lo, 100 * r.gap_hi, 2), f"{r.err_ratio:.2f}×"]
            for m, r in g.reindex(_models_in(g, ALL4)).iterrows()]
    return {"header": ["Model", "Ambiguous", "Unique", "Difference (pp)", "95% CI", "Ratio"],
            "groups": [("Error rate (%)", 1, 2)], "body": body, "right": set(range(1, 6)), "total": False,
            "present": list(g.index)}


@table("T-49", "error rates on ambiguous vs unique paraphrase strings (scenario D, means over seeds)",
       [("checks", "ambiguity_summary.csv")], ALL4)
def t49(S):
    # scenario D only. The numbers book keeps scenario A's rows, which the thesis quotes in §6.3.
    return ambiguity_table(S["ambiguity_summary.csv"], "D")


@table("T-50", "pretraining overlap: difference-in-differences against each control (scenarios A and D)",
       [("checks", "pretrain_did.csv")], ["BCM", "Llama-3.1-8B", "TF-IDF"])
def t50(S):
    """Gap = top-1 on the stage-1 patients minus the other patients. FHIR-MLM's gap is on the rows it shares with
    the control. DiD = FHIR-MLM's gap − the control's. Values are in pp with 3 decimals, because the effects are
    hundredths of a pp."""
    d = S["pretrain_did.csv"]
    body, gaps, present = [], set(), set()
    for cell in ("A", "D"):
        g = d[(d.cell == cell) & (d.rows != "own")]
        if body:
            gaps.add(len(body))
        for i, m in enumerate([m for m in ["BCM", "Llama-3.1-8B", "TF-IDF"] if m in set(g.model)]):
            q = g[g.model == m]
            excl = int(((q.did_lo > 0) | (q.did_hi < 0)).sum())
            body.append([cell if i == 0 else "", m, sg(100 * q.mlm_gap_same_rows.mean(), 3), sg(100 * q.gap.mean(), 3),
                         with_sd(sg(100 * q.did.mean(), 3), 100 * q.did.std(), 3), f"{excl} of {len(q)}"])
            present.add(m)
    return {"header": ["Scenario", "Control", "FHIR-MLM", "Control", "DiD (pp)", "Significant seeds"],
            "groups": [("Gap (pp)", 2, 3)], "body": body, "right": set(range(2, 6)),
            "total": False, "gaps": gaps, "present": sorted(present)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    ids = [x.strip() for x in a.only.split(",") if x.strip()] or list(TABLES)
    for tid in ids:
        r = build(tid, write=not a.list)
        print(f"  {tid:6s} {r['status']}")


if __name__ == "__main__":
    main()
