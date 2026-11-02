#!/usr/bin/env python
"""Draw F-19: the two evaluation units and the bridge between them (thesis Chapter 5).

Every result is reported at two units, per observation (a live feed) and per local code (a catalogue
migration). *Deployed accuracy* bridges them. Each code's single decision is applied back over all of
that code's volume, which is then scored per observation. The figure is a toy schematic that computes
all three on the same twelve observations. It shows why they disagree, and why deployed accuracy can
sit above per-observation accuracy while per-code accuracy sits below both.

The data are synthetic and tiny. There are 12 observations from three local codes with counts
8 / 3 / 1 and a hand-chosen right/wrong pattern (7/8, 2/3, 0/1). No thesis numbers appear. The script
computes the three quotients from the toy pattern (asserted below).

The figure uses FS.apply_paper(9) and is drawn at the text width, so it prints 1:1 at
width=\\textwidth. It uses black ink only. Every result is an outlined square with its local code's
letter, scored with a check or a cross. In the deployed panel, the two results that turn right once
their code's decision is applied are shaded light grey. There is no title, gloss, legend or arrow
label. The caption explains grouping, applying back, the shading and that this is a toy. Panels have
names and no letters, because (a)-(c) could be confused with codes A-C.

Writes results/analysis/ch5/figures/ch5_two_units_bridge_v2.*.

  python analysis/ch5_two_units_bridge_figure.py
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
FIG = HERE.parent / "results" / "analysis" / "ch5" / "figures"
NAME = "ch5_two_units_bridge_v2"

# toy pattern of (local code, per-observation correct?), with 8 of code A, 3 of B and 1 of C
OBS = [("A", 1), ("A", 1), ("A", 1), ("A", 0), ("A", 1), ("A", 1), ("A", 1), ("A", 1),
       ("B", 1), ("B", 0), ("B", 1),
       ("C", 0)]
CODES = ["A", "B", "C"]

# geometry, in inches: the figure is the text width less savefig's 0.02 in padding on each side
W, H = FS.TEXTWIDTH - 0.04, 1.62
GAP = 0.34                        # between panels, holds the arrows
P = (W - 2 * GAP) / 3             # panel width
S, S_SMALL = 0.17, 0.12           # result squares: full size, and grouped in the per-code panel
ROWS = (1.16, 0.85, 0.54)         # the three rows (results 1-4, 5-8, 9-12; codes A, B, C)
HEAD_Y, SCORE_Y = 1.50, 0.12
SHADE = "0.85"                    # the results that turn right in the deployed panel


def compute():
    n_obs = len(OBS); obs_right = sum(r for _, r in OBS)
    # catalogue decision: majority-of-observations stands in for summed probability in the toy
    code_right = {c: (sum(r for cc, r in OBS if cc == c) * 2 > sum(1 for cc, _ in OBS if cc == c)) for c in CODES}
    n_code = len(CODES); codes_right = sum(code_right.values())
    deployed_right = sum(1 for c, _ in OBS if code_right[c])
    return n_obs, obs_right, n_code, codes_right, deployed_right, code_right


def square(ax, x, y, s, letter, size, fc="white"):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=fc, ec="black", lw=0.6))
    ax.text(x, y, letter, ha="center", va="center", fontsize=size)


def mark(ax, x, y, ok):
    # the times sign comes from CM and is thin. The check mark comes from STIX (the cm fallback) and is heavier.
    # Set the cross larger so the two read evenly.
    ax.text(x, y, r"$\checkmark$" if ok else r"$\times$", ha="center", va="center", fontsize=9 if ok else 11)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="black",
                                 shrinkA=0, shrinkB=0))


def results_grid(ax, x0, oks, shaded=()):
    """12 results in 3 rows of 4: square with the code letter, then its mark."""
    pitch = P / 4
    for i, ((code, _), ok) in enumerate(zip(OBS, oks)):
        x = x0 + (i % 4) * pitch + 0.16
        y = ROWS[i // 4]
        square(ax, x, y, S, code, 8, fc=SHADE if i in shaded else "white")
        mark(ax, x + S / 2 + 0.11, y, ok)


def build(write=True):
    n_obs, obs_right, n_code, codes_right, deployed_right, code_right = compute()
    assert (n_obs, obs_right, n_code, codes_right, deployed_right) == (12, 9, 3, 2, 11)
    flipped = [i for i, (c, r) in enumerate(OBS) if not r and code_right[c]]
    assert flipped == [3, 9], flipped                       # the caption's "two wrong results become correct"
    FS.apply_paper(9)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=-10))   # tight bbox = the full width
    xs = [k * (P + GAP) for k in range(3)]

    # per observation: each result scored on its own
    results_grid(ax, xs[0], [r for _, r in OBS])
    # per local code: each code's results combined into one decision
    # groups right-aligned, so every code's arrow into its decision is the same length
    x_dec = xs[1] + P - 0.30
    x_last = x_dec - S / 2 - 0.36                   # centre of each group's last result
    for code, y in zip(CODES, ROWS):
        n = sum(1 for c, _ in OBS if c == code)
        for j in range(n):
            square(ax, x_last - (n - 1 - j) * (S_SMALL + 0.018), y, S_SMALL, code, 6.5)
        arrow(ax, x_last + S_SMALL / 2 + 0.06, x_dec - S / 2 - 0.06, y)
        square(ax, x_dec, y, S, code, 8)
        mark(ax, x_dec + S / 2 + 0.11, y, code_right[code])
    # deployed: every result takes its code's decision
    results_grid(ax, xs[2], [code_right[c] for c, _ in OBS], shaded=flipped)

    heads = ("Per observation", "Per local code", "Deployed")
    scores = (f"{obs_right}/{n_obs} = {obs_right / n_obs:.2f}", f"{codes_right}/{n_code} = {codes_right / n_code:.2f}",
              f"{deployed_right}/{n_obs} = {deployed_right / n_obs:.2f}")
    for x, h, sc in zip(xs, heads, scores):
        ax.text(x + P / 2, HEAD_Y, h, ha="center", va="center")
        ax.text(x + P / 2, SCORE_Y, sc, ha="center", va="center")
    for k in range(2):
        arrow(ax, xs[k] + P + 0.05, xs[k + 1] - 0.05, ROWS[1])
    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        print("wrote", FS.save(fig, str(FIG / NAME)))
    return fig


if __name__ == "__main__":
    build()
