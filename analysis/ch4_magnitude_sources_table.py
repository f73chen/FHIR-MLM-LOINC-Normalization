#!/usr/bin/env python
"""Write T-38: where the evidence for each perturbation magnitude comes from, for measured values and for
reference ranges (thesis Chapter 4).

The table shows how much of the perturbation rests on a number published for the right analyte, and
how much is inherited from its analyte group or the global median. It gives the codes per evidence
tier and the share of observations they carry.

The denominators differ, and the caption must say so:
    values  share of the numeric training observations
    ranges  share of the numeric training observations that carry a reference range

The rows are the tiers of `data/variability/perturbation_spec.csv` (`value_tier`, `range_tier`).
Blood and body-fluid pH (2 codes) get their own row, because their magnitude is an absolute offset
derived from a regulatory limit, the CLIA pH limit of +/-0.04 (0.04 / k 2.82). Ranges have no
specification tier, because published tolerances exist for measured values only. That cell is shown
as –.

Reads data/variability/perturbation_spec.csv and numeric_obs_counts.csv and gates the counts. Writes
results/analysis/a1/tables/ch4_magnitude_sources.tex (booktabs LaTeX) and the same table as an image,
results/analysis/a1/figures/ch4_magnitude_sources_table.{png,pdf}, which prints 1:1 at 100%.

  python analysis/ch4_magnitude_sources_table.py
"""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
_B = importlib.util.spec_from_file_location("booktabs", HERE / "lib" / "booktabs.py")
BT = importlib.util.module_from_spec(_B); _B.loader.exec_module(BT)

VAR = HERE.parent / "data" / "variability"
FIG = HERE.parent / "results" / "analysis" / "a1" / "figures"
TAB = HERE.parent / "results" / "analysis" / "a1" / "tables"

#: (row label, tier) in evidence order. The tier "" means never perturbed.
TIERS = [("Measurement naming the analyte", "observed"),
         ("Specification naming the analyte", "specification"),
         ("Derived from a CLIA limit", "derived"),
         ("Analyte group default", "class"),
         ("Global median", "fallback"),
         ("Not perturbed", "")]
GROUPS = [("Values", 1, 2), ("Reference ranges", 3, 4)]
HEADER = ["Evidence source", "Codes", "Obs. (%)", "Codes", "Obs. (%)"]
RIGHT = {1, 2, 3, 4}


def counts():
    spec = list(csv.DictReader(open(VAR / "perturbation_spec.csv")))
    with_range = {r["loinc"]: int(r["train_obs_numeric_with_refrange"])
                  for r in csv.DictReader(open(VAR / "numeric_obs_counts.csv"))}
    assert all(r["loinc"] in with_range for r in spec)
    obs_v = {r["loinc"]: int(r["n_obs"] or 0) for r in spec}
    obs_r = {r["loinc"]: with_range[r["loinc"]] for r in spec}
    TV, TR = sum(obs_v.values()), sum(obs_r.values())
    out = {}
    for label, t in TIERS:
        v = [r["loinc"] for r in spec if r["value_tier"] == t]
        g = [r["loinc"] for r in spec if r["range_tier"] == t]
        out[t] = (len(v), sum(obs_v[c] for c in v), len(g), sum(obs_r[c] for c in g))
    # gates: the tiers partition the 497 codes, and the totals are the ones Chapter 4 quotes
    assert len(spec) == 497 and TV == 32_817_457 and TR == 31_757_321
    assert [out[t][0] for _, t in TIERS] == [50, 50, 2, 262, 124, 9]
    assert [out[t][2] for _, t in TIERS] == [37, 0, 2, 185, 264, 9]
    # cross-check: with pH folded into "measurement", that row has these known counts and shares
    pct = lambda x, T: round(100 * x / T, 1)
    assert out["observed"][0] + out["derived"][0] == 52 and pct(out["observed"][1] + out["derived"][1], TV) == 47.2
    assert out["observed"][2] + out["derived"][2] == 39 and pct(out["observed"][3] + out["derived"][3], TR) == 27.2
    assert [pct(out[t][1], TV) for t in ("specification", "class", "fallback", "")] == [15.9, 33.6, 3.0, 0.2]
    assert [pct(out[t][3], TR) for t in ("class", "fallback")] == [56.1, 16.7]
    return out, len(spec), TV, TR


def cells(out, n, TV, TR):
    def share(x, T):
        p = 100 * x / T
        return "<0.1" if 0 < p < 0.05 else f"{p:.1f}"
    body = []
    for label, t in TIERS:
        nv, ov, nr, orr = out[t]
        rng = ("–", "–") if t == "specification" else (f"{nr:,}", share(orr, TR))
        body.append([label, f"{nv:,}", share(ov, TV), *rng])
    body.append(["Total", f"{n:,}", "100.0", f"{n:,}", "100.0"])
    return body


def build(write=True):
    out, n, TV, TR = counts()
    body = cells(out, n, TV, TR)
    for r in body:
        print("  " + " | ".join(r))
    if write:
        TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
        tex = TAB / "ch4_magnitude_sources.tex"
        tex.write_text(BT.tex(HEADER, body, groups=GROUPS, right=RIGHT, total=True,
                              caption="The evidence source of each perturbation magnitude",
                              label="tab:magnitude-sources",
                              comment="T-38 -- generated by analysis/ch4_magnitude_sources_table.py "
                                      "from data/variability/perturbation_spec.csv"))
        print("wrote", tex)
        png, w = BT.draw(FS, HEADER, body, str(FIG / "ch4_magnitude_sources_table"), groups=GROUPS,
                         right=RIGHT, total=True)
        print(f"wrote {png} (natural width {w:.2f} in; text width {FS.TEXTWIDTH} in)")


if __name__ == "__main__":
    build()
