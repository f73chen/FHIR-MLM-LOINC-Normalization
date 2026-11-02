#!/usr/bin/env python
"""F-15: class imbalance in the 679-code label space (thesis Chapter 4).

One picture for "the task is extremely long-tailed". It shows every LOINC code in the primary label
space, ranked by training occurrences (linear rank, log count). The only marks besides the curve are
the bin edges of T-04, which sits beside it. They are 1,000 and 100,000 training occurrences, drawn as
identical dashed lines. They show where the curve crosses each edge, i.e. the code counts in the table.
1,000 is also the rare/common reporting boundary. No code is dropped or trained differently. The
counts per bin, the meaning of the edges and the most frequent code go in the caption.

Reads data/labelspace/labelspace.csv (columns loinc / train_occ / test_occ / in_primary_ge10occ) and
ranks the codes itself, by train_occ. The counts that the chapter quotes are recomputed and asserted
at build time, so a change in them stops the build.

The paper style (FS.apply_paper) is Computer Modern at 9 pt, black ink and thin rules, with no grid and
no title. The figure is drawn at half the thesis \\textwidth so it prints 1:1 at `width=0.5\\textwidth`.

Writes results/analysis/a1/figures/ch4_class_imbalance_v3.{png,pdf}.

  python analysis/ch4_imbalance_figure.py
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)

LABELSPACE = HERE.parent / "data" / "labelspace" / "labelspace.csv"
FIG = HERE.parent / "results" / "analysis" / "a1" / "figures"
CUT = 1000
EDGES = (1_000, 100_000)   # T-04's bin edges. CUT is also the rare/common reporting boundary.
SIZE = 9        # printed point size at half the text width


def build(write=True):
    FS.apply_paper(SIZE)
    d = pd.read_csv(LABELSPACE)
    p = d[d["in_primary_ge10occ"] == 1].sort_values("train_occ", ascending=False).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)
    rare = p[p.train_occ < CUT]; common = p[p.train_occ >= CUT]
    # checks against the counts the chapter quotes
    assert len(p) == 679, len(p)
    assert len(rare) == 347 and len(common) == 332, (len(rare), len(common))
    assert int(p.train_occ.max()) == 1_145_183 and p.loc[0, "loinc"] == "4544-3"  # Hct VFr Bld Auto
    share = rare.test_occ.sum() / p.test_occ.sum()
    assert abs(share - 0.002447) < 1e-4, share
    frequent = p[p.train_occ >= EDGES[1]]
    assert len(frequent) == 71 and frequent["rank"].max() == 71
    assert common["rank"].max() == 332                                                # crossing ranks

    # half the text width, minus the padding savefig adds on both sides
    W = FS.TEXTWIDTH / 2 - 2 * plt.rcParams["savefig.pad_inches"]
    fig, ax = plt.subplots(figsize=(W, 0.72 * W))
    ax.plot(p["rank"], p.train_occ, color="black", lw=1.0, zorder=3)
    for edge in EDGES:                                                # T-04's bin edges
        ax.axhline(edge, color="0.4", lw=0.6, ls=(0, (4, 2)), zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0, len(p) + 12)
    ax.set_ylim(p.train_occ.min() / 1.5, p.train_occ.max() * 1.5)
    ax.set_xticks([0, 200, 400, 600])
    ax.tick_params(which="minor", length=1.5, width=0.4)
    ax.set_xlabel("LOINC code rank")
    ax.set_ylabel("Training occurrences")
    FS.finish(ax)
    fig.tight_layout(pad=0)
    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        print("wrote", FS.save(fig, str(FIG / "ch4_class_imbalance_v3")))
    return fig


if __name__ == "__main__":
    build()
