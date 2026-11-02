#!/usr/bin/env python
"""Draw how closely each training paraphrase matches the LOINC name of its code. This is the companion
figure to T-06 (thesis Chapter 4).

T-06 gives a mean and one threshold. The histogram shows the whole distribution, so a reader can see
that paraphrases range from near-copies of the official name to strings that share almost nothing with
it.

The measure comes from a6 part 1, as do the T-06 numbers. It is the cosine similarity between the
character 3-5-gram TF-IDF vectors (`char_wb`, lowercased) of a paraphrase and of the LOINC long common
name of its code. The vectoriser is fitted on the 679 official names. lib/textsim.py explains why the
measure is model-free. The values are recomputed here and gated against those in T-06.

The figure has no reference lines, because the 0.2 threshold is a bin edge and the median sits next
to the 0.5 edge. Lines there would merge into the bar outlines. Both numbers, and the definition of the
measure, go in the caption and prose. It is drawn at half the thesis \\textwidth, so it prints 1:1 at
`width=0.5\\textwidth`.

Reads the training paraphrase pool (data/paraphrases/clean) and the 679-code label space. These hold
LOINC text and counts only, with no patient data. Writes
results/analysis/a6/figures/ch4_paraphrase_similarity_v2.* and prints the summary values and a few
paraphrases from the first bin, for the caption.

  python analysis/ch4_paraphrase_similarity_figure.py
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
_A6 = importlib.util.spec_from_file_location("a6", HERE / "a6_data_quality.py")
A6 = importlib.util.module_from_spec(_A6); _A6.loader.exec_module(A6)

FIG = HERE.parent / "results" / "analysis" / "a6" / "figures"
THRESH = 0.2
SIZE = 9        # printed point size at half the text width


def train_similarities(return_text=False):
    """The a6 part 1 computation, on the training split only."""
    space = list(A6.A.canonical_space())
    table = A6.LX.load_axis_table(space)
    names = A6.official_names(space, table)
    pos = {l: i for i, l in enumerate(space)}
    vec = A6.TS.fit_char_tfidf(list(names))
    tr, _va, _te, _ = A6.load_pools()
    qs, si = [], []
    for loinc, paras in tr.items():
        if loinc not in pos:
            continue
        for p in paras:
            qs.append(p); si.append(pos[loinc])
    _bi, _bs, ss = A6.TS.nearest_and_self(qs, names, np.array(si), vec)
    if return_text:
        return np.asarray(ss, dtype=float), len(set(si)), qs, [names[i] for i in si]
    return np.asarray(ss, dtype=float), len(set(si))


def build(write=True):
    ss, n_codes, qs, own = train_similarities(return_text=True)
    mean, p10, p50, p90 = float(np.mean(ss)), *map(float, np.percentile(ss, [10, 50, 90]))
    below = 100 * float((ss < THRESH).mean())
    # gates: T-06's values
    assert len(ss) == 152_015 and n_codes == 679, (len(ss), n_codes)
    assert abs(mean - 0.469) < 0.0015, mean
    assert abs(p10 - 0.094) < 0.0015 and abs(p50 - 0.495) < 0.0015, (p10, p50)
    assert abs(below - 17.42) < 0.05, below
    print(f"n={len(ss):,} over {n_codes} codes | mean {mean:.3f} | p10 {p10:.3f} | median {p50:.3f} | "
          f"p90 {p90:.3f} | below {THRESH}: {below:.2f}%")

    # what the spike in the first bin is, printed for the caption (paraphrase text only, no patient data)
    low = ss < 0.05
    words = np.array([len(q.split()) for q in qs])
    print(f"first bin (< 0.05): {int(low.sum()):,} paraphrases ({100*low.mean():.1f}%); "
          f"<= 2 words: {100*(words[low] <= 2).mean():.1f}% (vs {100*(words <= 2).mean():.1f}% overall)")
    rng = np.random.default_rng(0)
    for k in rng.choice(np.flatnonzero(low), size=8, replace=False):
        print(f"    {qs[k]!r:45.45}  <-  {own[k][:70]!r}")

    FS.apply_paper(SIZE)
    W = FS.TEXTWIDTH / 2 - 2 * plt.rcParams["savefig.pad_inches"]
    fig, ax = plt.subplots(figsize=(W, 0.72 * W))
    bins = np.linspace(0, 1, 21)                                  # 0.05-wide bins
    counts, _, _ = ax.hist(ss, bins=bins, color="0.75", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, counts.max() * 1.05)
    ax.set_yticks([0, 4000, 8000, 12000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Similarity to the LOINC name")
    ax.set_ylabel("Training paraphrases")
    FS.finish(ax)
    fig.tight_layout(pad=0)
    if write:
        FIG.mkdir(parents=True, exist_ok=True)
        out = FIG / "ch4_paraphrase_similarity_v2"
        print("wrote", FS.save(fig, str(out)))
    return fig


if __name__ == "__main__":
    build()
