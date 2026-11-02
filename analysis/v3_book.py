#!/usr/bin/env python
"""
Generate the numbers book, results/analysis/v3/numbers.md (thesis Chapters 6 to 8).

The book follows the order of the thesis. It first lists every Chapter 6 table with its denominator
and seed basis, its rows and its sources. The rows come from the builders in v3_tables, which also
write the .tex. Then come the in-text numbers that have no table, and then the numbers derived for
Chapters 7 and 8. Do not edit numbers.md by hand. Re-run the script instead.

  python analysis/v3_book.py
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_T = importlib.util.spec_from_file_location("v3_tables", HERE / "v3_tables.py")
T = importlib.util.module_from_spec(_T); _T.loader.exec_module(T)

BOOK = HERE.parent / "results" / "analysis" / "v3" / "numbers.md"
SEEDS = "means over the declared seeds (5; Llama 1) with the sample sd"

SECTIONS = [                       # Chapter 6's sections, each with its tables in the thesis's order
    ("6.1 Overall comparison", ["T-10", "T-11", "T-12", "T-14"]),
    ("6.2 Patient history (RQ1)", ["T-32", "T-34", "T-35"]),
    ("6.3 Descriptions (RQ2)", ["T-15", "T-16", "T-17", "T-49"]),
    ("6.4 Numbers (RQ3)", ["T-41", "T-46", "T-45", "T-42", "T-43", "T-47", "T-48"]),
    ("6.5 Confidence and auto-mapping (RQ4)", ["T-24", "T-21", "T-22", "T-30"]),
    ("6.6 Secondary findings", ["T-19", "T-20", "T-25", "T-28", "T-26", "T-50"]),
]

BASIS = {
    "T-10": "Matched set (para, scenario D: 418,039 observations, 619 local codes); " + SEEDS + ". Per code and "
            "deployed: joint code × seed bootstrap (Llama: codes only).",
    "T-32": "FHIR-MLM's own rows (1,189,162 observations; 682 codes), scenario D. Per observation: joint patient × "
            "seed CIs; per code: joint code × seed, the difference paired.",
    "T-34": "As T-32, by training-frequency group (§4.5); difference = history − single observation (seed sd).",
    "T-35": "Each row on FHIR-MLM's own rows for that name condition (RELMA: a0, RELMA-named rows), scenario D. "
            "Per observation: joint patient × seed CIs; per code: paired code × seed.",
    "T-15": "The observations FHIR-MLM and TF-IDF both scored in both conditions (scenario D; obs_denominators' 'T-15 "
            "set'); per code on the same pair's codes; " + SEEDS + ". Change = NoDesc − Para, from the unrounded means.",
    "T-16": "RELMA a0, RELMA-named observations only (§4.10), the four-model matched set at scenario D (537 codes); "
            + SEEDS + ".",
    "T-41": "Each model on its own positions scored in all four scenarios; " + SEEDS + " (seed sds in the caption).",
    "T-42": "Each model's own observation set across its four scenarios, per code; " + SEEDS + "; differences with "
            "paired code × seed intervals (Llama: codes only).",
    "T-43": ("Models trained on clean numbers. One source per row (FHIR-MLM's dumps; TF-IDF's 07; BCM and Llama on "
             "the positions all the row's cells share); " + SEEDS + "."),
    "T-24": "Per code, para and RELMA a0 matched sets at scenario D; budgets 1 / 5 / 10%; " + SEEDS + "; range over "
            "seeds at 1%.",
    "T-21": "Para scenario-D matched set, marginalised 679-LOINC probabilities (§5.7); " + SEEDS + ". T fitted on the "
            "perturbed validation split, seed by seed.",
    "T-30": "Per code, para and RELMA a0 matched sets at scenario D, and NoDesc's FHIR-MLM/TF-IDF set (as F-12 (c)); "
            + SEEDS + " (seed sd in parentheses).",
    "T-19": "Each model's own errors at scenario D (para), native rows; " + SEEDS + ". Shares are of errors and sum to "
            "the mean number of differing axes.",
    "T-25": "Training: pert arm, full runs, wall clock incl. validation, 2 GPUs (deep models); TF-IDF = the 2M fit "
            "(8 jobs), plus a one-off 6.5 h tuning search. Inference: scenario-D eval manifests (full logits dumped); "
            "TF-IDF = the CPU benchmark (feature extraction + predict_proba, one thread). Parameters = T-08's.",
    "T-26": "Scenario D, seed 0. Share = the first window's share of the scored patients' test observations; gain = "
            "all windows ÷ first window. Top-1 for patients with ≥ 2 windows (counts in the caption); change = "
            "windows 1+ − window 0, patient-clustered CI.",
    "T-11": "Para scenario-D matched set; groups by training occurrences of the gold LOINC (§4.5); " + SEEDS + ".",
    "T-12": "Para scenario-D matched set; 'one observation' = one random observation's prediction stands for its "
            "code (seed 0 draw); " + SEEDS + ". The gain also by §4.5's groups of the gold LOINC's training count "
            "(the per-code file; the block under 'In-text numbers' has the groups' levels).",
    "T-14": "Para scenario-D matched set; deployed decisions; " + SEEDS + ".",
    "T-17": "Each model on its own paired RELMA/para rows (RELMA-named only), scenario D; " + SEEDS + ".",
    "T-20": "As T-19.",
    "T-22": "As T-21; coverage = share auto-mapped at the error budget (0.1 / 1%; at 5%, every model covers 100%).",
    "T-28": "4,952 test observations (500 patients), one observation per prompt, few-shot; perturbed numbers (clean "
            "vs perturbed: the §6.4 / §6.6.3 block below and F-11).",
    "T-45": "As T-41.",
    "T-46": "RELMA a0, RELMA-named rows; " + SEEDS + ".",
    "T-47": "As T-43.",
    "T-48": "Seed 0; five extra perturbation draws; spread = highest − lowest of all six draws (the t interval over "
            "r1–r5 is in rq3_replicates.csv).",
    "T-49": "Each model's own rows, scenario D; error rates and their difference with a joint patient × seed "
            "interval (scenario A: the §6.3 block below).",
    "T-50": "FHIR-MLM vs each control on the rows they share, scenarios A and D; gap = top-1 on the stage-1 "
            "patients − the others; per-seed CIs.",
}


def md_table(header, body, groups=()):
    hdr = list(header)
    for title, a, b in groups:
        for k in range(a, b + 1):
            hdr[k] = f"{title} · {hdr[k]}"
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    rows += ["| " + " | ".join(c if c else " " for c in r) + " |" for r in body]
    return rows


def table_block(tid) -> list[str]:
    res = T.build(tid, write=False)
    spec = T.TABLES[tid]
    lines = [f"### {tid} · {spec['title']}", ""]
    if not res["complete"]:
        lines.append(f"- *{res['status']}*")
    lines.append(f"- **Basis:** {BASIS.get(tid, '—')}")
    if res["out"] is not None:
        lines += [""] + md_table(res["out"]["header"], res["out"]["body"], res["out"].get("groups", ())) + [""]
        for s in res["stamps"]:
            lines.append(f"- **Source:** `{s['source']}` (models {', '.join(s['models']) or '—'})")
    lines.append("")
    return lines


# -- in-text numbers ---------------------------------------------------------
def _df(analysis, file):
    return T.load(analysis, file)


def text_numbers() -> list[str]:
    L = ["## In-text numbers (no table)", ""]
    d = _df("checks", "casting_agreement.csv")
    if d is not None:
        g = d.groupby(["condition", "model"]).agg(differs=("share_pred_differs", "max"), acc=("acc_diff", lambda x: x.abs().max()))
        L += ["### §5.7 mapping-first vs marginalised top-1", "",
              "| condition | model | max share of predictions that differ | max accuracy difference |", "|---|---|---|---|"]
        L += [f"| {c} | {m} | {100 * r.differs:.4f}% | {r.acc:.1e} |" for (c, m), r in g.iterrows()] + [""]
    L += training_length()
    L += aggregation_by_band()
    j = _df("catalogue", "catalogue_joint.csv")
    if j is not None and {"relma_a0_D", "relma_a1_D", "relma_a2_D"} <= set(j.arm):
        q = j[j.arm.str.startswith("relma_a")].pivot_table(index="model", columns="arm", values="top1_mean")
        L += ["### §6.3 per-code top-1 over the three RELMA name draws (scenario D, matched sets)", "",
              "| model | a0 | a1 | a2 | spread (pp) |", "|---|---|---|---|---|"]
        L += [f"| {m} | {100 * r.relma_a0_D:.1f} | {100 * r.relma_a1_D:.1f} | {100 * r.relma_a2_D:.1f} | "
              f"{100 * (r.max() - r.min()):.1f} |" for m, r in q.iterrows()] + [""]
    for fam_file, label in (("catalogue", "four-model matched set"),):
        t5 = _df(fam_file, "catalogue_pairwise_top5.csv")
        t1 = _df(fam_file, "catalogue_pairwise.csv")
        if t5 is None:
            continue
        L += [f"### §6.1 / §7.6 per-code differences, top-1 and top-5, paired code × seed intervals ({label}, scenario D)", "",
              "| names | pair | top-1 (pp) | top-5 (pp) |", "|---|---|---|---|"]
        for r in t5[t5.arm.isin(["para_D", "relma_a0_D"])].itertuples():
            q = t1[(t1.arm == r.arm) & (t1.model_a == r.model_a) & (t1.model_b == r.model_b)].iloc[0]
            star = lambda x: " *" if x.excludes_zero else ""
            L.append(f"| {r.arm} | {r.model_a} − {r.model_b} | {100 * q.diff_mean:+.1f} [{100 * q.lo:+.1f}, {100 * q.hi:+.1f}]{star(q)} | "
                     f"{100 * r.diff_mean:+.1f} [{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]{star(r)} |")
        L += ["", "- \\* = the interval excludes 0.", ""]
    p = _df("observation", "obs_pairwise.csv")
    if p is not None:
        L += ["### §6.1 per-observation differences between models, with the MDE (scenario D, matched sets, seed 0)", "",
              "| names | pair | observations | difference (pp) | MDE (pp) | design effect | p |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r.set.split('_')[0]} | {r.model_a} − {r.model_b} | {r.n:,} | {100 * r.delta_a_minus_b:+.3f} | "
              f"{100 * r.mde_80:.3f} | {r.design_effect:.1f} | {T.pval(r.p_patients)} |" for r in p.itertuples()]
        L += ["", "- MDE and p: McNemar's test with patients as clusters (Durkalski et al. 2003), the MDE at 80% power "
              "and α = 0.05. The exact, independent-observations p and MDE stay in the file.", ""]
    r = _df("observation", "obs_relma_vs_para.csv")
    if r is not None:
        L += ["### §6.3 RELMA vs paraphrase names per observation, seed 0: McNemar's p (T-17 gives the means over "
              "seeds)", "", "| model | observations | change, seed 0 (pp) | p, exact | p, patients as clusters |",
              "|---|---|---|---|---|"]
        L += [f"| {x.model} | {x.n_seed0:,} | {100 * x.change_seed0:+.3f} | {T.pval(x.mcnemar_p)} | "
              f"{T.pval(x.mcnemar_p_patients)} |" for x in r.itertuples()] + [""]
    c = _df("checks", "ambiguity_counts.csv")
    if c is not None:
        L += ["### §6.3 ambiguous paraphrase strings: how many (error rates in T-49)", ""]
        L += [f"- {r.unit}: {int(r.n_ambiguous):,} of {int(r.n):,} ambiguous ({100 * r.share:.2f}%)" for r in c.itertuples()]
        L.append("")
    a = _df("checks", "ambiguity_summary.csv")
    if a is not None:
        q = T.ambiguity_table(a, "A")
        L += ["### §6.3 error rates on ambiguous vs unique paraphrase strings, scenario A (T-49 shows scenario D only)",
              ""] + md_table(q["header"], q["body"], q["groups"]) + [""]
    f = _df("checks", "pretrain_overlap_facts.csv")
    if f is not None:
        L += ["### §6.6.5 pretraining overlap: who overlaps (the test in T-50)", ""]
        L += [f"- {r.model}: {r.n_patients_overlap:,} of {r.n_patients:,} patients in the stage-1 corpus "
              f"({100 * r.share_patients_overlap:.2f}%), {100 * r.share_obs_overlap:.2f}% of its observations"
              for r in f.itertuples()]
        L.append("")
    s = _df("rq3", "rq3_obs_summary.csv")
    if s is not None:
        L += ["### §6.4 seed sd of the 2×2 cells, per observation (pp; T-41 and T-45 give them in the "
              "caption)", "", "| setup | model | A | B | C | D |", "|---|---|---|---|---|---|"]
        for fam in ("para", "nodesc", "singleobs_para", "singleobs_nodesc"):
            g = s[(s.family == fam) & s.quantity.isin(T.CELLS) & (s.n_seeds > 1)]
            for m, d in g.groupby("model"):
                sd = d.set_index("quantity")["sd"]
                L.append(f"| {fam} | {m} | " + " | ".join(f"{100 * sd[c]:.2f}" for c in T.CELLS) + " |")
        L.append("")
    q = _df("rq3", "rq3_primary.csv")
    if q is not None and "D_minus_B_mean" in q:
        L += ["### §6.4 D − B per observation, means over seeds, with joint patient × seed 95% CIs (the main test)", "",
              "| model | observations | seeds | D (%) | B (%) | D − B (pp) | 95% CI | lowest–highest seed (pp) |",
              "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r.model} | {r.n_paired:,} | {r.n_seeds} | {T.po(r.top1_D_mean)} | {T.po(r.top1_B_mean)} | "
              f"{100 * r.D_minus_B_mean:+.2f} | [{100 * r.joint_ci_lo:+.2f}, {100 * r.joint_ci_hi:+.2f}] | "
              f"{100 * r.D_minus_B_seed_min:+.2f} to {100 * r.D_minus_B_seed_max:+.2f} |" for r in q.itertuples()]
        L += ["", "- Each model's B ∩ D positions (the same at every seed); D, B and D − B equal T-41's. Llama: one seed, so "
              "its interval resamples patients only.", ""]
    v = _df("rq3", "rq3_valuerange_summary.csv")
    if v is not None:
        L += ["### §6.4 values vs ranges, trained on perturbed numbers (T-43 shows the clean arm only)", "",
              "| model | none | values only | ranges only | both | spread (pp) |", "|---|---|---|---|---|---|"]
        q = ["clean_test", "pert_value", "pert_range", "pert_both"]
        for m in [m for m in T.ALL4 if m in set(v.model)]:
            d = v[(v.model == m) & (v.arm == "pert")].set_index("quantity")["mean"][q]
            L.append(f"| {m} | " + " | ".join(T.po(x) for x in d) + f" | {100 * (d.max() - d.min()):.3f} |")
        L.append("")
    d = _df("rq3", "rq3_fewshot_control.csv")
    if d is not None:
        lab = {"openbio-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B", "gemma3-12b": "Gemma-3-12B",
               "llama31-8b": "Llama-3.1-8B", "meditron-7b": "Meditron-7B"}
        L += ["### §6.4 / §6.6.3 off-the-shelf models, clean vs perturbed numbers (F-11 panel b)", "",
              "| model | clean (%) | perturbed (%) | difference (pp) | exact 95% CI | McNemar p |", "|---|---|---|---|---|---|"]
        L += [f"| {lab.get(r.model, r.model)} | {100 * r.correct_clean:.2f} | {100 * r.correct_pert:.2f} | "
              f"{100 * r.pert_minus_clean:+.2f} | [{100 * r.exact_ci_lo:+.2f}, {100 * r.exact_ci_hi:+.2f}] | {T.pval(r.mcnemar_p)} |"
              for r in d.itertuples()] + [""]
    d = _df("sliding", "sliding_fixed_cohort.csv")
    if d is not None:
        g = d.groupby("model").first()
        L += ["### §6.6.4 F-10a slopes (change in top-1 per extra window)", "",
              "| model | cohort patients | windows | slope (pp per window) | 95% CI |", "|---|---|---|---|---|"]
        L += [f"| {T.R.LABEL.get(m, m)} | {int(r.n_patients):,} | {int(r.depth)} | {-r.slope_per_window / 100:+.4f} | "
              f"[{-r.slope_ci_hi / 100:+.4f}, {-r.slope_ci_lo / 100:+.4f}] |" for m, r in g.iterrows()] + [""]
    d = _df("singleobs", "singleobs_obs.csv")
    if d is not None:
        L += ["### §6.2 history vs single observation per observation, scenarios A and D (FHIR-MLM's own rows)", "",
              "| condition | scenario | with history | single observation | difference | 95% CI |", "|---|---|---|---|---|---|"]
        h = d[d.comparison == "history_vs_singleobs"]
        L += [f"| {r.condition} | {r.cell} | {100 * r.a_mean:.2f} | {100 * r.b_mean:.2f} | {100 * r.a_minus_b:+.2f} pp | "
              f"[{100 * r.diff_lo:+.2f}, {100 * r.diff_hi:+.2f}] |" for r in h.itertuples()] + [""]
        s1 = d[d.comparison == "singleobs_vs_tfidf"]
        L += ["### §6.2 single-observation FHIR-MLM vs TF-IDF, per observation (scenario D)", "",
              "| condition | n | single-obs FHIR-MLM | TF-IDF | difference | 95% CI |", "|---|---|---|---|---|---|"]
        L += [f"| {r.condition} | {r.n:,} | {r.a_mean:.5f} | {r.b_mean:.5f} | {100 * r.a_minus_b:+.3f} pp | "
              f"[{100 * r.diff_lo:+.3f}, {100 * r.diff_hi:+.3f}] |" for r in s1.itertuples()]
        L.append("")
    pj = _df("catalogue_per_model", "catalogue_joint.csv")
    pw = _df("catalogue_per_model", "catalogue_pairwise.csv")
    SO, TF, H = "FHIR-MLM (single obs)", "TF-IDF", "FHIR-MLM"
    rows = []
    for fam, cond in (("para_D_mlm_tfidf", "para_xsite"), ("nodesc_D", "nodesc_xsite")):
        if pj is None or pw is None or not (pj.arm == fam).any():
            continue
        j = pj[pj.arm == fam].set_index("model")
        if not {SO, TF, H} <= set(j.index):
            continue
        a, alo, ahi = T._pair(pw, fam, SO, TF)
        b, blo, bhi = T._pair(pw, fam, H, TF)
        rows.append(f"| {cond} | {100 * a:+.2f} pp [{100 * alo:+.2f}, {100 * ahi:+.2f}] | {100 * b:+.2f} pp [{100 * blo:+.2f}, {100 * bhi:+.2f}] | "
                    f"{100 * (j.loc[SO, 'obs_mean'] - j.loc[TF, 'obs_mean']):+.3f} pp | "
                    f"{100 * (j.loc[SO, 'dep_mean'] - j.loc[TF, 'dep_mean']):+.3f} pp |")
    L += ["### §6.2 single-observation and history FHIR-MLM minus TF-IDF (FHIR-MLM ∩ TF-IDF, scenario D)", ""]
    if rows:
        L += ["| condition | single obs − TF-IDF, per code (paired CI) | history − TF-IDF, per code (paired CI) | "
              "single obs − TF-IDF, per observation | single obs − TF-IDF, deployed |", "|---|---|---|---|---|"] + rows
    else:
        L.append("- *waiting for the per-code tables*")
    L.append("")
    L += untabled()
    return L


def untabled() -> list[str]:
    """Numbers Ch 6–8 quote that no table shows, each with its observation set."""
    rows = []
    c = _df("calibration", "calibration_matched.csv")
    if c is not None:
        t1 = c[c.scaling == "T=1"]
        n_para = int(t1[t1.condition == "para_xsite"].n.iloc[0])
        for m, d in t1[t1.condition == "nodesc_xsite"].groupby("model"):
            rows.append(("6.5", f"{m}: per-observation ECE at T = 1, NoDesc", "FHIR-MLM ∩ TF-IDF, NoDesc, "
                         f"{int(d.n.iloc[0]):,} observations (T-21's Para ECEs: the 4-model set, {n_para:,})",
                         f"{d.ece_mass15.mean():.5f} ({d.ece_mass15.min():.5f}–{d.ece_mass15.max():.5f})"))
    r = _df("calibration", "relma_transfer.csv")
    if r is not None:
        d = r[r.model == "FHIR-MLM"]
        rows.append(("8.2", "FHIR-MLM: per-observation ECE at T = 1, RELMA names",
                     f"its own RELMA-named rows, {int(d.n.iloc[0]):,} observations",
                     f"{d.ece_T1.mean():.5f} ({d.ece_T1.min():.5f}–{d.ece_T1.max():.5f})"))
        g = r.groupby("model")[["ece_T1", "ece_Tind"]].mean()
        rows.append(("6.5", "per-observation ECE on RELMA names, T = 1 → the Para (in-domain) temperature",
                     "each model's own RELMA-named rows",
                     "; ".join(f"{m} {g.loc[m, 'ece_T1']:.5f} → {g.loc[m, 'ece_Tind']:.5f}" for m in T.ALL4 if m in g.index)))
    cv = _df("catalogue", "catalogue_coverage_by_seed.csv")
    j = _df("catalogue", "catalogue_joint.csv")
    if cv is not None and j is not None:
        d = cv[(cv.model == "FHIR-MLM") & (cv.arm == "para_D")]
        codes = int(j[(j.arm == "para_D") & (j.model == "FHIR-MLM")].n_codes.iloc[0])
        rows.append(("6.5 (F-26)", "FHIR-MLM: per-code coverage at a 1% / 5% / 10% budget, Para (lowest–highest seed)",
                     f"the 4-model matched set, {codes} local codes",
                     " / ".join(f"{100 * d[k].min():.1f}–{100 * d[k].max():.1f}%" for k in ("cov@0.01", "cov@0.05", "cov@0.1"))))
    if j is not None:
        d = j[j.arm == "para_D"].set_index("model")
        span = 100 * (d.loc["FHIR-MLM", "top1_max"] - d.loc["FHIR-MLM", "top1_min"])
        rows.append(("6.1, 6.4, 7.7", "per-code top-1 in scenario D, Para (lowest–highest seed; T-10 shows the mean and "
                     "the joint code × seed CI)", f"the 4-model matched set, {int(d.n_codes.iloc[0])} local codes",
                     "; ".join(f"{m} {100 * d.loc[m, 'top1_min']:.1f}–{100 * d.loc[m, 'top1_max']:.1f}%"
                               for m in ("FHIR-MLM", "BCM", "TF-IDF") if m in d.index)
                     + f" (FHIR-MLM spans {span:.1f} pp)"))
        w = j[j.arm.isin(["para_D", "relma_a0_D"])].pivot(index="model", columns="arm", values="pct_vol_wrong_mean")
        n_codes = j[j.arm.isin(["para_D", "relma_a0_D"])].groupby("arm").n_codes.first()
        rows.append(("6.3", "wrong codes' share of observations, Para → RELMA a0 (T-14 shows Para)",
                     f"the 4-model matched sets, {n_codes['para_D']} / {n_codes['relma_a0_D']} local codes",
                     "; ".join(f"{m} {100 * w.loc[m, 'para_D']:.2f}% → {100 * w.loc[m, 'relma_a0_D']:.2f}%"
                               for m in T.ALL4 if m in w.index)))
    e = _df("rq3", "rq3_ece_by_seed.csv")
    if e is not None:
        p = e[(e.condition == "para_xsite") & (e.scope == "history") & e.cell.isin(["A", "B"])]
        g = p.groupby(["model", "cell"]).gap_conf_minus_acc.mean().unstack()
        rows.append(("6.4", "per-observation confidence minus accuracy, scenario A → B, Para (+ = overconfident)",
                     "each model's own rows", "; ".join(f"{m} {100 * g.loc[m, 'A']:+.2f} → {100 * g.loc[m, 'B']:+.2f} pp"
                                                       for m in T.ALL4 if m in g.index)))
    s = _df("singleobs", "singleobs_bands_single_occurrence.csv")
    if s is not None:
        g = s.groupby("band", sort=False).agg(n=("n_obs_once", "first"), gap=("drop", "mean"), lo=("drop", "min"),
                                              hi=("drop", "max"))
        rows.append(("6.2", "history − single-observation FHIR-MLM top-1 by training-frequency group, observations whose "
                     "code appears once in the patient's scored record (T-34's artefact check)",
                     "FHIR-MLM's rows, " + " / ".join(f"{int(x.n):,}" for _, x in g.iterrows()) + " observations",
                     "; ".join(f"{b} {100 * x.gap:+.2f} pp ({100 * x.lo:+.2f} to {100 * x.hi:+.2f})"
                               for b, x in g.iterrows())))
    o = _df("rq3", "rq3_obs_by_seed.csv")
    if o is not None:
        d = o[(o.model == "FHIR-MLM") & (o.family == "para")]
        rows.append(("7.1, 7.7", "FHIR-MLM: per-observation top-1 in scenario D (lowest–highest seed)",
                     f"its own rows scored in all four scenarios, {int(d.n_matched.iloc[0]):,} observations",
                     f"{100 * d.D.min():.2f}–{100 * d.D.max():.2f}% ({100 * (d.D.max() - d.D.min()):.2f} pp)"))
    L = ["### Ch 6–8 numbers no table shows", "",
         "| § | what | observation set | value (mean; seed range) |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {s} | {v} |" for a, b, s, v in rows] + [""]
    return L


# -- aggregation by frequency group ------------------------------------------
def aggregation_by_band() -> list[str]:
    """Per-code top-1 from one random observation vs all of a code's observations. Codes are grouped by the training
    count of their gold LOINC, in the groups of Chapter 4. The set is para, scenario D, the four-model matched set,
    with means over seeds and codes."""
    st = T.stamp_of("catalogue", "percode__para_D.parquet")
    if st is None:
        return []
    g = T.aggregation_by_band(pd.read_parquet(st["path"], columns=["code", "gold", "n_obs", "so_hit1", "sm_hit1",
                                                                   "model"]))
    L = ["### §6.1 what aggregation adds, by training-frequency group (per code, para, scenario D)", "",
         "| model | group | local codes | one observation (%) | all observations (%) | gain (pp) | median test observations |",
         "|---|---|---|---|---|---|---|"]
    L += [f"| {m} | {b} | {int(r.codes)} | {100 * r.one_obs:.1f} | {100 * r.all_obs:.1f} | "
          f"{100 * (r.all_obs - r.one_obs):+.1f} | {r.med:g} |" for (m, b), r in g.iterrows()]
    return L + [""]


# -- training length ---------------------------------------------------------
RESULTS = HERE.parent / "results"
#: (label, training script's results folder, manifest stem with {arm}, registry model, per-code family in
#: catalogue_per_model)
RUNS = [
    ("FHIR-MLM", "12_mlm_train", "para_xsite__{arm}", "mlm", "rq3_para__mlm"),
    ("FHIR-MLM, single obs", "12_mlm_train", "singleobs_para_xsite__{arm}", "mlm", "rq3_singleobs_para__mlm"),
    ("FHIR-MLM, NoDesc", "12_mlm_train", "nodesc_xsite__{arm}", "mlm", "rq3_nodesc__mlm"),
    ("FHIR-MLM, single obs, NoDesc", "12_mlm_train", "singleobs_nodesc_xsite__{arm}", "mlm", "rq3_singleobs_nodesc__mlm"),
    ("BCM", "13_bcm_train", "para_xsite__{arm}", "bcm", "rq3_para__bcm"),
    ("Llama-3.1-8B", "14_llama_train", "para_xsite__{arm}", "llama", "rq3_para__llama"),
]


def _best(folder: str, stem: str, seed: int):
    """(validation top-1, best step, steps run, step cap) from a training run's manifest, or None if absent."""
    p = RESULTS / folder / f"manifest_{stem}_seed{seed}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m["best_metric"], m["best_step"], m["global_step"], m["max_steps"]


def training_length() -> list[str]:
    """Each fine-tuning run's best checkpoint beside its per-code top-1 when tested on the kind of numbers it was
    trained on (scenario A for the clean arm, D for the perturbed arm), on the rows that model scored. Early stopping
    watches per-observation validation top-1."""
    rows, mlm = [], []
    for label, folder, stem, model, fam in RUNS:
        pq = T.stamp_of("catalogue_per_model", f"percode__{fam}.parquet")
        per = None
        if pq:
            per = pd.read_parquet(pq["path"], columns=["cell", "seed", "sm_hit1"]).groupby(["cell", "seed"]).sm_hit1.mean()
        for arm, cell in (("clean", "A"), ("pert", "D")):
            for s in T.R.SEEDS[model]:
                d = _best(folder, stem.format(arm=arm), s)
                if d is None:
                    continue
                val, best, ran, cap = d
                code = per.get((cell, s), np.nan) if per is not None else np.nan
                rows.append(f"| {label} | {arm} | {s} | {best:,} | {ran:,} / {cap:,} | {100 * val:.3f} | "
                            f"{'—' if pd.isna(code) else f'{100 * code:.1f}'} |")
                if label == "FHIR-MLM" and pd.notna(code):
                    mlm.append((best, code))
    if not rows:
        return []
    L = ["### Training length: best checkpoint vs per-code top-1", "",
         "- Early stopping watches per-observation validation top-1 (patience 3 evaluations of 400 steps); the best "
         "checkpoint is the one tested. Per-code top-1 is each run's own rows at home (A for the clean arm, D for the "
         "perturbed arm), from `catalogue_per_model/percode__*.parquet`."]
    if mlm:
        early = [c for b, c in mlm if b <= 2800]; late = [c for b, c in mlm if b >= 4800]
        if early and late:
            L.append(f"- FHIR-MLM: best at ≤ 2,800 steps → {100 * min(early):.1f}–{100 * max(early):.1f}% per code "
                     f"({len(early)} runs); at ≥ 4,800 → {100 * min(late):.1f}–{100 * max(late):.1f}% ({len(late)} runs).")
    L += ["", "| model | arm | seed | best step | ran to | validation top-1 (%) | per-code top-1 at home (%) |",
          "|---|---|---|---|---|---|---|"] + rows + [""]
    return L


# -- Ch 7–8 derived numbers --------------------------------------------------
N_LOCAL, N_MAPPABLE, N_IN_SPACE = 1_623, 828, 747          # local codes; those with a LOINC; those in the 679


def derived() -> list[str]:
    L = ["## Ch 7–8 derived numbers", ""]
    rows = []
    cv = _df("catalogue", "catalogue_coverage_by_seed.csv")
    if cv is not None and (cv.arm == "relma_a0_D").any():
        m = cv[(cv.model == "FHIR-MLM") & (cv.arm.str.startswith("relma_a"))]
        a0 = m[m.arm == "relma_a0_D"]
        c5, c1 = a0["cov@0.05"], m["cov@0.01"]
        rows += [("7.5", "FHIR-MLM catalogue coverage at 5%, RELMA a0 (mean; range over seeds)",
                  f"{c5.mean():.3f} ({c5.min():.3f}–{c5.max():.3f})"),
                 ("7.5", "FHIR-MLM coverage at 1%, RELMA (mean over seeds × the three name draws; range)",
                  f"{c1.mean():.3f} ({c1.min():.3f}–{c1.max():.3f}; {len(c1)} seed × draw cells)")]
        o = cv[(cv.arm == "relma_a0_D") & (cv.model != "FHIR-MLM")].groupby("model")[["cov@0.05", "cov@0.1"]].mean()
        rows += [("7.5", f"{mdl}: coverage at 5% / 10%, RELMA a0", f"{r['cov@0.05']:.3f} / {r['cov@0.1']:.3f}")
                 for mdl, r in o.iterrows()]
        auto = c5.mean() * N_IN_SPACE
        rows += [("8.4", "share of the evaluated catalogue left to review at 5% (FHIR-MLM, RELMA a0)", f"{1 - c5.mean():.1%}"),
                 ("8.4", f"of {N_LOCAL:,} local codes: auto-mapped at 5% if the coverage holds on all {N_IN_SPACE} in-space codes",
                  f"{auto:.0f} ({auto / N_LOCAL:.0%}); {N_LOCAL - auto:.0f} left, of which {N_LOCAL - N_MAPPABLE} have no LOINC "
                  f"and {N_MAPPABLE - N_IN_SPACE} fall outside the 679"),
                 ("8.4", "expert-hours to review a 2,000-code catalogue in full vs only what is not auto-mapped "
                         "(1–2 h per 100 codes, Zunner 2013)",
                  f"20–40 h vs {(1 - c5.mean()) * 20:.1f}–{(1 - c5.mean()) * 40:.1f} h")]
    tr = _df("cost", "cost_training.csv")
    if tr is not None:
        h = tr[tr.arm == "pert"].groupby("model").hours.mean()
        rows += [("7.5", "Llama's training time ÷ FHIR-MLM's (pert arm)", f"{h['Llama-3.1-8B'] / h['FHIR-MLM']:.1f}×"),
                 ("8.4", "FHIR-MLM fine-tuning hours (pert-arm mean)", f"{h['FHIR-MLM']:.1f} h")]
    L += ["| § | what | value |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    L += ["", "- The 8.4 extrapolation applies the evaluated catalogue's coverage to all 747 in-space codes; the "
              "evaluated catalogue is the matched set's codes (T-16).", ""]
    return L


def main():
    git = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    L = ["# Numbers book", "",
         f"*Generated by `analysis/v3_book.py` on {dt.datetime.now():%Y-%m-%d %H:%M} at {git}. Do not edit by hand: "
         "re-run the script.*", ""]
    for title, tids in SECTIONS:
        L += [f"## {title}", ""]
        for tid in tids:
            L += table_block(tid)
    L += text_numbers()
    L += derived()
    BOOK.write_text("\n".join(L) + "\n")
    print(f"wrote {BOOK} ({len(L)} lines)")


if __name__ == "__main__":
    main()
