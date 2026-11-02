#!/usr/bin/env python
"""The results figures (thesis Chapter 6), drawn from the analysis outputs in
results/analysis/v3/ in the academic style of lib/figstyle.py. The style is FS.apply_paper (Computer
Modern at printed size), black data, thin dashed grey references, open axes and no grid. Titles and
explanations go in the caption, so none appear inside the figure. Models are told apart by panel, axis
label, marker or line style, and never by colour alone.

  python analysis/v3_figures.py                every figure whose inputs exist
  python analysis/v3_figures.py --only F-10a
  python analysis/v3_figures.py --list

As with the tables (v3_tables.py), a figure is written only when every input exists and every model
it should show is present. Each goes to results/analysis/v3/figures/ as PNG (to look at) and PDF
(for the thesis), named after its figure ID (F-05 -> f05). Aggregate data only.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import registry as R  # noqa: E402

_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
_T = importlib.util.spec_from_file_location("v3_tables", HERE / "v3_tables.py")
T = importlib.util.module_from_spec(_T); _T.loader.exec_module(T)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PLAIN = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))

FIGS: dict = {}
DEEP3 = T.DEEP3
GREY_REF = dict(color="0.45", lw=0.6, ls=(0, (4, 2)))
BAND = dict(color="0.85", lw=0)
#: one look per model wherever several share an axis: marker and line style, black or grey, so the
#: figures read in print and without colour (a legend names them)
SERIES = {"FHIR-MLM": dict(color="black", ls="-", marker="o", mfc="black"),
          "BCM": dict(color="black", ls=(0, (5, 2)), marker="s", mfc="white"),
          "Llama-3.1-8B": dict(color="0.45", ls="-", marker="^", mfc="0.45"),
          "TF-IDF": dict(color="black", ls=(0, (1, 1.5)), marker="D", mfc="white")}


def figure(fid, needs, expected):
    def deco(fn):
        FIGS[fid] = {"fn": fn, "needs": needs, "expected": expected}
        return fn
    return deco


def build(fid, write=True) -> dict:
    spec = FIGS[fid]
    src = {}
    for a, fl in spec["needs"]:
        df = T.load(a, fl)
        if df is None:
            return {"fid": fid, "status": f"waiting for {a}/{fl}"}
        src[fl] = df
    FS.apply_paper(9)
    fig, present = spec["fn"](src)
    missing = [m for m in spec["expected"] if m not in present]
    complete = not missing
    status = "complete" if complete else f"INCOMPLETE (missing {', '.join(missing)}): not written"
    if write and complete:
        root = R.OUT_ROOT / "figures"
        root.mkdir(parents=True, exist_ok=True)
        png = FS.save(fig, str(root / fid.replace("-", "").lower()))
        status += f" -> {png}"
    plt.close(fig)
    return {"fid": fid, "status": status}


def panel_label(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom")


# -- calibration and selective prediction (F-05, F-06) -----------------------
CONDS = [("para_xsite", "Paraphrase"), ("nodesc_xsite", "NoDesc")]


def _conf_axis(ax, lo_err, hi_err):
    """1 − confidence on a reversed log axis (more confident to the right), decade ticks."""
    from matplotlib.ticker import LogFormatterMathtext, LogLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(hi_err, lo_err)


@figure("F-05", [("calibration", "reliability_bins.csv")], T.ALL4)
def f05(S):
    """Confidence minus accuracy in each equal-mass confidence bin (15 bins, cell D, matched set). Above zero
    is over-confident. Each point is the mean over seeds of one bin, where bin k is the k-th equal-mass slice in
    every seed (Llama has one seed). Ties at confidence 1.0 can merge a seed's top slices, which all sit at
    confidence and accuracy ≈ 1."""
    b = S["reliability_bins.csv"]
    b = b[b.scaling == "T=1"]
    b = b.groupby(["model", "condition", "bin"], as_index=False)[["conf", "gap_conf_minus_acc"]].mean()
    conds = [(c, lab) for c, lab in CONDS if (b.condition == c).any()]
    fig, axes = plt.subplots(1, len(conds), figsize=(FS.TEXTWIDTH, 2.2), squeeze=False, sharey=True)
    err = 1 - b.conf
    # Llama's confidences can saturate at exactly 1.0 in stored precision, so 1 − confidence is 0 there and a
    # log axis cannot start at it. Those points sit at a floor a decade below the smallest real value.
    floor = float(err[err > 0].min()) / 10
    present = set()
    for k, (ax, (c, lab)) in enumerate(zip(axes[0], conds)):
        g = b[b.condition == c]
        for m in [m for m in T.ALL4 if m in set(g.model)]:
            q = g[g.model == m].sort_values("conf")
            st = SERIES[m]
            ax.plot((1 - q.conf).clip(lower=floor), q.gap_conf_minus_acc, color=st["color"], ls=st["ls"],
                    marker=st["marker"], mfc=st["mfc"], mec=st["color"], ms=3, lw=0.8, label=m)
            present.add(m)
        ax.axhline(0, **GREY_REF)
        _conf_axis(ax, floor / 2, err.max() * 2)
        panel_label(ax, f"({chr(97 + k)}) {lab}")
        ax.set_xlabel("1 − confidence (log scale)")
        if k == 0:
            ax.set_ylabel("Confidence − accuracy")
        FS.finish(ax)
    hs = [plt.Line2D([], [], color=SERIES[m]["color"], ls=SERIES[m]["ls"], marker=SERIES[m]["marker"],
                     mfc=SERIES[m]["mfc"], mec=SERIES[m]["color"], ms=3, label=m) for m in T.ALL4 if m in present]
    fig.legend(handles=hs, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               borderaxespad=0.0)
    fig.tight_layout(w_pad=1.0)
    return fig, sorted(present)


#: F-06 and F-26 are drawn alike: one line per model as in SERIES, dashed budget lines, coverage in percent,
#: the legend above the panels, the same size
RISK_LW = 1.0


def _risk_legend(fig, present):
    """The models' legend in one row above the panels, as in F-05. Inside panel (a) of F-06 the budget lines
    would run through its entries."""
    hs = [plt.Line2D([], [], color=SERIES[m]["color"], ls=SERIES[m]["ls"], lw=RISK_LW, label=m)
          for m in T.ALL4 if m in present]
    fig.legend(handles=hs, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               borderaxespad=0.0)


@figure("F-06", [("calibration", "risk_coverage_curves.csv")], T.ALL4)
def f06(S):
    """Selective prediction at the observation unit. The most confident predictions are accepted first, and
    the error rate among those accepted is plotted against the share accepted (cell D, each panel's matched
    set). The panels show paraphrase and RELMA names, as F-26 does per code. Each line is the median over
    seeds at each coverage, like the bold line in F-26, without the seed lines (Llama has one seed). Dashed
    lines mark the 0.1% and 1% budgets of T-22. Error is on a log axis because the errors of the deep models
    stay below 0.1% until nearly every observation is accepted."""
    r = S["risk_coverage_curves.csv"]
    conds = [(c, lab) for c, lab in (("para_xsite", "Paraphrase"), ("relma_a0", "RELMA")) if (r.condition == c).any()]
    fig, axes = plt.subplots(1, len(conds), figsize=(FS.TEXTWIDTH, 2.2), squeeze=False, sharey=True)
    present = set()
    for k, (ax, (c, lab)) in enumerate(zip(axes[0], conds)):
        g = r[r.condition == c]
        for m in [m for m in T.ALL4 if m in set(g.model)]:
            seeds = [q.sort_values("coverage") for _, q in g[g.model == m].groupby("seed")]
            grid = seeds[0].coverage.to_numpy()          # every seed's curve has the same thinned coverages
            risk = np.median([np.interp(grid, q.coverage, q.risk) for q in seeds], axis=0)
            st = SERIES[m]
            ax.plot(grid, np.clip(risk, 1e-5, None), color=st["color"], ls=st["ls"], lw=RISK_LW, label=m, zorder=3)
            present.add(m)
        for budget in R.BUDGETS_OBS[:2]:                  # T-22's columns: 0.1% and 1%
            ax.axhline(budget, **GREY_REF)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 0.5); ax.set_xlim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        panel_label(ax, f"({chr(97 + k)}) {lab}")
        ax.set_xlabel("Coverage (% of observations)")
        if k == 0:
            ax.set_ylabel("Error among accepted (%)")
        FS.finish(ax)
    _risk_legend(fig, present)
    fig.tight_layout(w_pad=1.0)
    return fig, sorted(present)


# -- sliding windows (F-10a, F-10b) ------------------------------------------
@figure("F-10a", [("sliding", "sliding_fixed_cohort.csv")], DEEP3)
def f10a(S):
    """The fitted change in top-1 per additional window, with its patient-clustered 95% interval."""
    d = S["sliding_fixed_cohort.csv"].groupby("model").first()
    d.index = d.index.map(R.LABEL)
    models = [m for m in DEEP3 if m in d.index]
    fig, ax = plt.subplots(figsize=(FS.TEXTWIDTH, 0.35 + 0.28 * len(models)))
    for i, m in enumerate(models):
        r = d.loc[m]
        y = len(models) - 1 - i
        lo, hi = -r.slope_ci_hi / 100, -r.slope_ci_lo / 100           # from error per 10k to accuracy points
        ax.plot([lo, hi], [y, y], color="black", lw=1.0)
        ax.plot([-r.slope_per_window / 100], [y], "o", color="black", ms=4)
    ax.axvline(0, **GREY_REF)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models[::-1])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Change in top-1 per additional window (percentage points)")
    ax.xaxis.set_major_formatter(PLAIN)
    FS.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig, models


@figure("F-10b", [("sliding", "sliding_fixed_cohort.csv"), ("sliding", "sliding_summary.csv")], DEEP3)
def f10b(S):
    """Top-1 by window index for a fixed cohort (patients whose record reaches the last plotted window
    and one more), with the patient-clustered 95% band. The dashed line is the cohort's mean over its windows."""
    d = S["sliding_fixed_cohort.csv"]
    d = d.assign(label=d.model.map(R.LABEL))
    models = [m for m in DEEP3 if m in set(d.label)]
    fig, axes = plt.subplots(1, len(models), figsize=(FS.TEXTWIDTH, 1.9), squeeze=False)
    for k, (ax, m) in enumerate(zip(axes[0], models)):
        g = d[d.label == m].sort_values("window_idx")
        acc = 100 - g.err_per_10k / 100
        lo, hi = 100 - g.ci_hi / 100, 100 - g.ci_lo / 100
        ax.fill_between(g.window_idx, lo, hi, **BAND)
        ax.plot(g.window_idx, acc, color="black", lw=1.0, marker="o", ms=2.5)
        mean = 100 - (g.err_per_10k * g.n_obs).sum() / g.n_obs.sum() / 100
        ax.axhline(mean, **GREY_REF)
        panel_label(ax, f"({chr(97 + k)}) {m}, {int(g.n_patients.iloc[0]):,} patients")
        ax.set_xlabel("Window index")
        if k == 0:
            ax.set_ylabel("Top-1 (%)")
        ax.set_xticks(range(0, int(g.depth.iloc[0]), 2 if g.depth.iloc[0] <= 8 else 4))
        FS.finish(ax)
    fig.tight_layout(w_pad=1.2)
    return fig, models


# -- frequency and catalogue (F-02, F-12, F-26) ------------------------------
@figure("F-02", [("observation", "obs_strata.csv")], T.ALL4)
def f02(S):
    """Top-1 by how often the true LOINC occurs in training (9 log bins), para, cell D, matched set. Each
    point is the seed mean, with the Wilson 95% interval at that mean and the bin's n (observation sampling)."""
    s = S["obs_strata.csv"]
    s = s[(s.set == "para_D") & (s.scheme == "log9")]
    models = [m for m in T.ALL4 if m in set(s.model)]
    meta = s.groupby("stratum")[["lo", "hi", "n_obs"]].first()
    x = np.sqrt(meta.lo * meta.hi).to_numpy()
    fig, ax = plt.subplots(figsize=(FS.TEXTWIDTH, 2.4))
    for k, m in enumerate(models):
        acc = s[s.model == m].groupby("stratum").acc.mean().reindex(meta.index).to_numpy()
        ci = np.array([_wilson(a, n) for a, n in zip(acc, meta.n_obs)])
        xo = x * (1 + 0.07 * (k - (len(models) - 1) / 2))
        st = SERIES[m]
        ax.errorbar(xo, acc, yerr=[acc - ci[:, 0], ci[:, 1] - acc], color=st["color"], ls=st["ls"], marker=st["marker"],
                    mfc=st["mfc"], mec=st["color"], ms=3.5, lw=0.9, elinewidth=0.6, capsize=0, label=m)
    ax.set_xscale("log")
    for b in (1_000, 100_000):                 # the frequency groups' boundaries: rare | common | frequent (T-11)
        ax.axvline(b, **GREY_REF)
    ax.set_xlabel("Code frequency in training (log scale)")
    ax.set_ylabel("Per-observation top-1 (%)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
    ax.legend(loc="lower right", handlelength=2.6)
    FS.finish(ax)
    return fig, models


def _wilson(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@figure("F-12", [("catalogue", "catalogue_joint.csv")], T.ALL4)
def f12(S):
    """Observation top-1 against catalogue top-1 for each model, per name condition, both units on the
    same rows (cell D, means over seeds). The bar is the joint code × seed 95% interval of the catalogue."""
    j = S["catalogue_joint.csv"]
    fams = [(f, lab) for f, lab in (("para_D", "Paraphrase"), ("relma_a0_D", "RELMA")) if (j.arm == f).any()]
    pj = T.load("catalogue_per_model", "catalogue_joint.csv")
    if pj is not None and (pj.arm == "nodesc_D").any():
        j = pd.concat([j, pj[pj.arm == "nodesc_D"]]); fams.append(("nodesc_D", "NoDesc"))
    fig, axes = plt.subplots(1, len(fams), figsize=(FS.TEXTWIDTH, 2.4), squeeze=False, sharey=True)
    present = set()
    for k, (ax, (fam, lab)) in enumerate(zip(axes[0], fams)):
        g = j[j.arm == fam].set_index("model")
        for m in [m for m in T.ALL4 if m in g.index]:
            r, st = g.loc[m], SERIES[m]
            ax.plot([0, 1], [r.obs_mean, r.top1_mean], color=st["color"], ls=st["ls"], lw=0.9)
            ax.plot([0, 1], [r.obs_mean, r.top1_mean], ls="none", marker=st["marker"], mfc=st["mfc"], mec=st["color"], ms=4)
            ax.plot([1, 1], [r.top1_A_lo, r.top1_A_hi], color=st["color"], lw=0.8)
            present.add(m)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Per observation", "Per code"]); ax.set_xlim(-0.3, 1.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        panel_label(ax, f"({chr(97 + k)}) {lab}")
        if k == 0:
            ax.set_ylabel("Top-1 (%)")
        FS.finish(ax)
    hs = [plt.Line2D([], [], color=SERIES[m]["color"], ls=SERIES[m]["ls"], marker=SERIES[m]["marker"],
                     mfc=SERIES[m]["mfc"], mec=SERIES[m]["color"], label=m) for m in T.ALL4 if m in present]
    axes[0][0].legend(handles=hs, loc="lower left", handlelength=2.4)   # panel (a)'s lower left is empty
    fig.tight_layout(w_pad=1.0)
    return fig, sorted(present)


@figure("F-26", [("catalogue", "catalogue_joint.csv")], T.ALL4)
def f26(S):
    """Catalogue risk-coverage. Codes are accepted most-confident first, and the error among those accepted
    is plotted against the share of the catalogue accepted. Faint lines are single seeds and the bold line is
    the median over seeds (cell D). Dashed lines mark the 1%, 5% and 10% budgets of T-24."""
    root = T.stamp_of("catalogue", "catalogue_joint.csv")["path"].parent
    fams = [(f, lab) for f, lab in (("para_D", "Paraphrase"), ("relma_a0_D", "RELMA"))
            if (root / f"percode__{f}.parquet").exists()]
    fig, axes = plt.subplots(1, len(fams), figsize=(FS.TEXTWIDTH, 2.2), squeeze=False, sharey=True)
    grid = np.linspace(0.01, 1, 100)
    present = set()
    for k, (ax, (fam, lab)) in enumerate(zip(axes[0], fams)):
        pc = pd.read_parquet(root / f"percode__{fam}.parquet", columns=["model", "seed", "sm_conf", "sm_correct"])
        for m in [m for m in T.ALL4 if m in set(pc.model)]:
            curves = []
            for _, g in pc[pc.model == m].groupby("seed"):
                o = np.argsort(-g.sm_conf.to_numpy(), kind="stable")
                wrong = ~g.sm_correct.to_numpy()[o].astype(bool)
                cov = np.arange(1, len(o) + 1) / len(o)
                risk = np.cumsum(wrong) / np.arange(1, len(o) + 1)
                ax.plot(cov, risk, color="0.8", lw=0.5, zorder=1)
                curves.append(np.interp(grid, cov, risk))
            st = SERIES[m]
            ax.plot(grid, np.median(curves, axis=0), color=st["color"], ls=st["ls"], lw=RISK_LW, label=m, zorder=3)
            present.add(m)
        for b in R.BUDGETS_CODE:
            ax.axhline(b, **GREY_REF)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        panel_label(ax, f"({chr(97 + k)}) {lab}")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        ax.set_xlabel("Coverage (% of codes)")
        if k == 0:
            ax.set_ylabel("Error among accepted (%)")
        FS.finish(ax)
    _risk_legend(fig, present)
    fig.tight_layout(w_pad=1.0)
    return fig, sorted(present)


# -- errors by LOINC axis (F-09) ---------------------------------------------
@figure("F-09", [("errors", "errors_axis_marginals.csv")], T.ALL4)
def f09(S):
    """(a) How far each error is from the truth, as the share of a model's errors by the number of LOINC
    axes that differ (para, cell D, mean over seeds). (b) The error rate, with the seed range as a bar."""
    m = S["errors_axis_marginals.csv"]
    m = m[(m.subset == "all") & (m.condition == "para_xsite")]
    models = [x for x in T.ALL4 if x in set(m.model)]
    g = m.groupby("model").mean(numeric_only=True).reindex(models)
    cols = [("pct_axes_0", "0"), ("pct_axes_1", "1"), ("pct_axes_2", "2"), ("pct_axes_3", "3")]
    shares = {lab: g[c].to_numpy() for c, lab in cols}
    shares["≥4"] = g[["pct_axes_4", "pct_axes_5", "pct_axes_6"]].sum(axis=1).to_numpy()
    shades = ["white", "0.85", "0.6", "0.35", "black"]
    if not shares["0"].any():                          # no error with all six axes equal: leave it out
        shares.pop("0"); shades.pop(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FS.TEXTWIDTH, 0.6 + 0.3 * len(models)),
                               gridspec_kw={"width_ratios": [2.2, 1]})
    ys = np.arange(len(models))[::-1]
    left = np.zeros(len(models))
    for (lab, v), shade in zip(shares.items(), shades):
        a.barh(ys, v, left=left, height=0.55, color=shade, edgecolor="black", linewidth=0.4, label=lab)
        left += v
    a.set_yticks(ys); a.set_yticklabels(models); a.set_xlim(0, 100)
    a.set_xlabel("Share of errors (%)")
    a.legend(title="Axes that differ", ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0,
             columnspacing=1.0, borderaxespad=0.2, title_fontsize=8)
    FS.finish(a); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    rate = m.groupby("model").error_rate.agg(["mean", "min", "max"]).reindex(models) * 100
    b.errorbar(rate["mean"], ys, xerr=[rate["mean"] - rate["min"], rate["max"] - rate["mean"]], fmt="o", ms=3.5,
               color="black", elinewidth=0.8, capsize=0)
    b.set_xscale("log"); b.set_yticks(ys); b.set_yticklabels([])
    ticks = [x for x in (0.03, 0.1, 0.3, 1, 3, 10, 30) if rate["min"].min() / 2 <= x <= rate["max"].max() * 2]
    b.set_xticks(ticks); b.xaxis.set_minor_locator(plt.NullLocator())
    b.set_xlim(min(ticks) / 1.2, max(ticks) * 1.2)
    b.set_xlabel("Error rate (%, log scale)")
    b.xaxis.set_major_formatter(PLAIN)
    FS.finish(b); b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    for ax in (a, b):
        ax.set_ylim(-0.6, len(models) - 0.4)
    fig.tight_layout(w_pad=0.8)
    return fig, models


# -- off-the-shelf models (F-11) ---------------------------------------------
OUTCOMES = [("correct_pct", "Correct", "black"), ("valid_wrong_pct", "Real LOINC, wrong", "0.45"),
            ("not_a_loinc_pct", "Not a real LOINC", "0.8"), ("unparseable_pct", "No LOINC", "white")]


@figure("F-11", [("offshelf", "t28_rows.csv")], T.OFFSHELF)
def f11(S):
    """(a) What each off-the-shelf model's answers were with perturbed numbers (as T-28). (b) The correct
    share alone, both arms, with Wilson 95% intervals."""
    t = S["t28_rows.csv"]
    models = [m for m in T.OFFSHELF if m in set(t.model)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FS.TEXTWIDTH, 0.55 + 0.3 * len(models)),
                               gridspec_kw={"width_ratios": [2.2, 1]})
    ys = np.arange(len(models))[::-1]
    p = t[t.arm == "pert"].set_index("model").reindex(models)
    left = np.zeros(len(models))
    for col, lab, shade in OUTCOMES:
        a.barh(ys, p[col], left=left, height=0.55, color=shade, edgecolor="black", linewidth=0.4, label=lab)
        left += p[col].to_numpy()
    a.set_yticks(ys); a.set_yticklabels(models)
    a.set_xlim(0, 100); a.set_xlabel("Share of answers (%)")
    a.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0, columnspacing=1.0,
             borderaxespad=0.2)       # two rows: one row of four runs into panel (b)'s legend
    FS.finish(a); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    for k, (arm, mk, off) in enumerate((("clean", "o", 0.12), ("pert", "s", -0.12))):
        q = t[t.arm == arm].set_index("model").reindex(models)
        b.errorbar(q.correct_pct, ys + off, xerr=[q.correct_pct - q.correct_lo_pct, q.correct_hi_pct - q.correct_pct],
                   fmt=mk, ms=3.5, color="black", mfc="black" if arm == "pert" else "white", elinewidth=0.8,
                   capsize=0, label="Perturbed" if arm == "pert" else "Clean")
    b.set_yticks(ys); b.set_yticklabels([])
    b.set_xlim(0, max(2.2, float(t.correct_hi_pct.max()) * 1.1)); b.set_xlabel("Correct (%)")
    b.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0, borderaxespad=0.2)
    FS.finish(b); b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    for ax in (a, b):
        ax.set_ylim(-0.6, len(models) - 0.4)
    fig.tight_layout(w_pad=0.8)
    return fig, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for fid in [x.strip() for x in a.only.split(",") if x.strip()] or list(FIGS):
        if a.list:
            spec = FIGS[fid]
            print(f"  {fid:6s} needs {', '.join(f'{x}/{y}' for x, y in spec['needs'])}")
        else:
            print(f"  {fid:6s} {build(fid)['status']}")


if __name__ == "__main__":
    main()
