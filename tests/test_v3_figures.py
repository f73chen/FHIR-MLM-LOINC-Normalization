"""Tests for analysis/v3_figures.py. Every figure whose inputs exist must build, return the models it
shows, and use only the colours of the paper style (black, greys, white). Models are told apart by
panel, axis or marker/line style.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_figures_t", HERE / "v3_figures.py")
F = importlib.util.module_from_spec(_S); _S.loader.exec_module(F)


def test_series_styles_are_greyscale():
    from matplotlib.colors import to_rgb
    for m, st in F.SERIES.items():
        r, g, b = to_rgb(st["color"])
        assert r == g == b, m
    assert len({(st["ls"], st["marker"]) for st in F.SERIES.values()}) == len(F.SERIES)


@pytest.mark.parametrize("fid", list(F.FIGS))
def test_every_buildable_figure_draws(fid):
    spec = F.FIGS[fid]
    src = {}
    for a, fl in spec["needs"]:
        df = F.T.load(a, fl)
        if df is None:
            pytest.skip(f"{fid}: {a}/{fl} not produced yet")
        src[fl] = df
    F.FS.apply_paper(9)
    fig, present = spec["fn"](src)
    assert present and set(present) <= set(spec["expected"]), (fid, present)
    for ax in fig.axes:
        assert not ax.get_title(), f"{fid}: no titles inside figures"
    F.plt.close(fig)
