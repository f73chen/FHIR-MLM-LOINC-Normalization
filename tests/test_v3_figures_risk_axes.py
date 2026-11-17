"""
Tests for the two risk-coverage figures (analysis/v3_figures.py):
  F-06 draws an error of exactly 0 on a tick "0" (a symmetric-log axis, linear below 0.001%), without clipping the data;
  F-26 draws each faint per-seed line in its model's line style, and none for a model with one seed.
Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))
_S = importlib.util.spec_from_file_location("v3_figures_risk_t", HERE / "v3_figures.py")
F = importlib.util.module_from_spec(_S); _S.loader.exec_module(F)

COV = [0.25, 0.5, 0.75, 1.0]


def curves():
    """Two conditions; FHIR-MLM with two identical seeds (zero error on half of the coverage, then an error below the
    old axis floor of 0.001%), TF-IDF with one."""
    rows = []
    for cond in ("para_xsite", "relma_a0"):
        for seed in (0, 1):
            for c, r in zip(COV, [0.0, 0.0, 2e-6, 4e-3]):
                rows.append({"model": "FHIR-MLM", "condition": cond, "seed": seed, "scaling": "T=1", "coverage": c, "risk": r})
        for c, r in zip(COV, [0.0, 8e-6, 5e-5, 2e-2]):
            rows.append({"model": "TF-IDF", "condition": cond, "seed": 0, "scaling": "T=1", "coverage": c, "risk": r})
    return pd.DataFrame(rows)


def test_f06_draws_zero_error_on_its_own_tick():
    F.FS.apply_paper(9)
    fig, present = F.f06({"risk_coverage_curves.csv": curves()})
    assert present == ["FHIR-MLM", "TF-IDF"] and len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_yscale() == "symlog"
        assert ax.get_ylim()[0] == 0
        ticks = [float(t) for t in ax.yaxis.get_majorticklocs()]
        assert ticks == [float(t) for t in F.F06_TICKS] and ticks[0] == 0.0
        fmt = ax.yaxis.get_major_formatter()
        assert [fmt(t, None) for t in ticks] == ["0", "0.001", "0.01", "0.1", "1", "10"]     # in percent
        assert len(ax.yaxis.get_minorticklocs()) == 0
        line = next(l for l in ax.lines if l.get_label() == "FHIR-MLM")
        y = [float(v) for v in line.get_ydata()]
        assert y == [0.0, 0.0, 2e-6, 4e-3]             # nothing is clipped to the old floor of 1e-5
        assert line.get_clip_on() is False             # a line on the bottom edge is drawn in full
    F.plt.close(fig)


def percode(tmp_path):
    rows = []

    def add(model, seed, conf, correct):
        for c, k in zip(conf, correct):
            rows.append({"model": model, "seed": seed, "sm_conf": c, "sm_correct": k})

    add("FHIR-MLM", 0, [0.9, 0.8, 0.7, 0.6], [True, True, True, False])
    add("FHIR-MLM", 1, [0.9, 0.8, 0.7, 0.6], [True, True, False, True])
    add("BCM", 0, [0.9, 0.8, 0.7, 0.6], [True, False, True, False])
    add("BCM", 1, [0.9, 0.8, 0.7, 0.6], [False, True, True, True])
    add("Llama-3.1-8B", 0, [0.9, 0.8, 0.7, 0.6], [True, True, False, False])
    pd.DataFrame(rows).to_parquet(tmp_path / "percode__para_D.parquet")


def test_f26_seed_lines_follow_their_model(tmp_path, monkeypatch):
    percode(tmp_path)                                  # no RELMA file: one panel
    monkeypatch.setattr(F.T, "stamp_of", lambda a, f: {"path": tmp_path / "catalogue_joint.csv"})
    F.FS.apply_paper(9)
    fig, present = F.f26({})
    assert present == ["BCM", "FHIR-MLM", "Llama-3.1-8B"] and len(fig.axes) == 1
    by = {}
    for line in fig.axes[0].lines:
        by.setdefault(line.get_gid(), []).append(line)
    if not hasattr(by["median:FHIR-MLM"][0], "_unscaled_dash_pattern"):
        pytest.skip("this matplotlib does not expose a line's dash pattern")
    for m in ("FHIR-MLM", "BCM"):
        seeds, (bold,) = by[f"seed:{m}"], by[f"median:{m}"]
        assert len(seeds) == 2
        for line in seeds:
            assert line.get_linewidth() == F.SEED_LW and line.get_color() == F.SEED_GREY
            assert line._unscaled_dash_pattern == bold._unscaled_dash_pattern      # the model's own line style
        assert bold.get_linewidth() == F.RISK_LW
    assert by["seed:FHIR-MLM"][0]._unscaled_dash_pattern != by["seed:BCM"][0]._unscaled_dash_pattern
    assert "seed:Llama-3.1-8B" not in by and len(by["median:Llama-3.1-8B"]) == 1   # one seed: no faint line
    F.plt.close(fig)
