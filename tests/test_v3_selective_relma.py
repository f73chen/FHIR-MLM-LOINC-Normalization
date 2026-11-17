"""
Tests for T-22's RELMA columns: `v3_calibration.relma_rows_from` gives RELMA names the same curve and
selective-prediction rows as the paraphrase conditions, under the condition "relma_a0", and `v3_tables.t22` shows
paraphrases and RELMA names side by side in T-24's layout (the mean over seeds at each error budget and the worst to
best seed at the stricter one). Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(HERE))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


V = _load("v3_calibration_relma_t", HERE / "v3_calibration.py")
T = _load("v3_tables_relma_t", HERE / "v3_tables.py")


def frame(conf, correct):
    return pd.DataFrame({"p_top1": conf, "correct": correct})


def test_relma_rows_use_the_rule_of_the_paraphrase_rows():
    # most confident first: three right, then one wrong, then one right
    df = frame([0.99, 0.95, 0.90, 0.60, 0.50], [True, True, True, False, True])
    curves, sel = V.relma_rows_from([("M", 0, df)])
    assert len(sel) == 1
    r = sel[0]
    assert (r["model"], r["condition"], r["seed"], r["scaling"]) == ("M", "relma_a0", 0, "T=1")
    rc = V.C.risk_coverage(df.p_top1.to_numpy(), df.correct.to_numpy())
    for b in V.R.BUDGETS_OBS:
        assert r[f"cov@{b:g}"] == pytest.approx(V.C.coverage_at_risk(rc, b))
    assert r["cov@0.001"] == pytest.approx(0.6)          # three of five accepted before the first error
    assert curves and {c["condition"] for c in curves} == {"relma_a0"}
    assert all(c["model"] == "M" and c["seed"] == 0 and c["scaling"] == "T=1" for c in curves)


def test_one_selective_row_per_model_and_seed():
    a = frame([0.9, 0.8], [True, True])
    b = frame([0.9, 0.8], [False, True])
    _, sel = V.relma_rows_from([("M", 0, a), ("M", 1, b), ("N", 0, a)])
    assert [(r["model"], r["seed"]) for r in sel] == [("M", 0), ("M", 1), ("N", 0)]
    assert sel[0]["cov@0.001"] == pytest.approx(1.0)      # no error: everything is accepted
    assert sel[1]["cov@0.001"] == pytest.approx(0.0)      # the most confident prediction is wrong: nothing fits the budget
    assert V.relma_rows_from([]) == ([], [])


def selective(with_relma=True):
    rows = []

    def add(model, cond, seed, c01, c1):
        rows.append({"model": model, "condition": cond, "seed": seed, "scaling": "T=1", "cov@0.001": c01, "cov@0.01": c1,
                     "cov@0.05": 1.0})

    add("FHIR-MLM", "para_xsite", 0, 0.9970, 1.0); add("FHIR-MLM", "para_xsite", 1, 0.9990, 1.0)
    add("Llama-3.1-8B", "para_xsite", 0, 0.9963, 1.0)
    add("FHIR-MLM", "nodesc_xsite", 0, 0.5, 0.6)            # another condition: never in the table
    if with_relma:
        add("FHIR-MLM", "relma_a0", 0, 0.9700, 1.0); add("FHIR-MLM", "relma_a0", 1, 1.0, 1.0)
        add("Llama-3.1-8B", "relma_a0", 0, 0.9930, 0.9990)
    return pd.DataFrame(rows)


def test_t22_shows_paraphrase_and_relma_names_side_by_side():
    out = T.t22({"selective_matched.csv": selective()})
    assert out["header"] == ["Model", "0.1%", "Range", "1%", "0.1%", "Range", "1%"]
    assert out["groups"] == [("Para", 1, 3), ("RELMA", 4, 6)]
    body = {r[0]: r[1:] for r in out["body"]}
    assert body["FHIR-MLM"] == ["99.80", "99.70–99.90", "100.00", "98.50", "97.00–100.00", "100.00"]
    assert body["Llama-3.1-8B"] == ["99.63", "—", "100.00", "99.30", "—", "99.90"]     # one seed: no range
    assert [r[0] for r in out["body"]] == ["FHIR-MLM", "Llama-3.1-8B"]                 # the tables' model order
    assert out["incomplete"] == []


def test_t22_flags_outputs_from_before_the_relma_rows():
    out = T.t22({"selective_matched.csv": selective(with_relma=False)})
    assert out["incomplete"]
    body = {r[0]: r[1:] for r in out["body"]}
    assert body["FHIR-MLM"][:3] == ["99.80", "99.70–99.90", "100.00"]
    assert body["FHIR-MLM"][3:] == ["—", "—", "—"]
