"""Tests for analysis/v3_tables.py and the LaTeX cell translation in analysis/lib/booktabs.py.

The table layer only formats. These tests check that every symbol it emits compiles under pdfLaTeX,
and that the number formats are what the tables promise (a real minus sign, explicit signs on
differences). They also check that every table whose inputs exist is rectangular, with its column
groups inside the header.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


BT = _load("booktabs_t", HERE / "lib" / "booktabs.py")
T = _load("v3_tables_t", HERE / "v3_tables.py")


def test_tex_cell_translates_every_symbol_the_tables_use():
    got = BT._tex_cell("−0.5 ± 0.1 × 2 → ≤3 ≥4 <5 10% 1–2 —")
    assert got == r"$-$0.5 $\pm$ 0.1 $\times$ 2 $\rightarrow$ $\leq$3 $\geq$4 $<$5 10\% 1--2 ---"
    assert got.isascii()
    assert BT._tex_cell("FHIR-MLM") == "FHIR-MLM"          # a hyphen is not a minus sign


def test_group_titles_are_translated_too():
    # an unescaped % in "Top-1 (%)" would comment out the rest of the LaTeX line
    head = BT._head(["Model", "A", "B − A"], [("Top-1 (%)", 1, 2)])
    assert r"\multicolumn{2}{c}{Top-1 (\%)}" in head[1]
    assert r"B $-$ A" in head[3]


def test_split_columns_line_up_the_estimates():
    assert BT._parts("0.0501 (0.0064)") == ("0.0501", "(0.0064)")
    assert BT._parts("+2.13 [+2.03, +2.23]") == ("+2.13", "[+2.03, +2.23]")
    assert BT._parts("84.0") == ("84.0", "") and BT._parts("5 of 5") == ("5 of 5", "")
    one_run = [["FHIR-MLM", "0.0501 (0.0064)"], ["Llama-3.1-8B", "0.0448"]]
    same = [["a", "+0.9 [−4.1, +5.5]"], ["b", "+1.7 [−0.6, +3.9]"]]
    assert BT.split_cols(one_run, {1}) == {1} and BT.split_cols(same, {1}) == set()
    t = BT.tex(["Model", "ECE", "n"], [r + ["3"] for r in one_run], groups=[("Calibration", 1, 2)], right={1, 2},
               total=False, split={1})
    assert r"\begin{tabular}{lr@{\ }lr}" in t and r"\multicolumn{3}{c}{Calibration}" in t and r"\cmidrule(lr){2-4}" in t
    assert r"Model & \multicolumn{2}{r}{ECE} & n \\" in t
    assert r"FHIR-MLM & 0.0501 & (0.0064) & 3 \\" in t and r"Llama-3.1-8B & 0.0448 &  & 3 \\" in t


def test_formatters():
    assert T.f(-1.5, 2) == "−1.50" and T.f(1234.5, 1) == "1,234.5"
    assert T.sg(0.5) == "+0.50" and T.sg(-0.5) == "−0.50"
    assert T.ci(-0.1, 0.2, 2) == "[−0.10, +0.20]"
    assert T.ci(0.1, 0.2, 2, signed=False) == "[0.10, 0.20]"
    assert T.n(1189162.0) == "1,189,162" and T.pc(0.364, 1) == "36.4%"
    assert T.pval(0.0) == "<0.0001" and T.pval(0.078) == "0.078" and T.pval(0.005) == "0.0050"
    assert T.f(float("nan")) == "—"
    assert T.with_sd("+1.00", 0.5) == "+1.00 (0.50)" and T.with_sd("+1.00", float("nan")) == "+1.00"


@pytest.mark.parametrize("tid", list(T.TABLES))
def test_every_buildable_table_is_rectangular(tid):
    spec = T.TABLES[tid]
    src = {}
    for a, fl in spec["needs"]:
        df = T.load(a, fl)
        if df is None:
            pytest.skip(f"{tid}: {a}/{fl} not produced yet")
        src[fl] = df
    out = spec["fn"](src)
    assert out["body"], f"{tid} has no rows"
    for row in out["body"]:
        assert len(row) == len(out["header"]), (tid, row)
        assert all(isinstance(c, str) for c in row), (tid, row)
    for _, a, b in out.get("groups", ()):
        assert 0 <= a <= b < len(out["header"])
    assert set(out["right"]) <= set(range(len(out["header"])))
    assert set(out.get("indent", ())) <= set(range(len(out["body"])))
