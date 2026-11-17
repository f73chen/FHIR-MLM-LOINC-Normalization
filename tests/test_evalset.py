"""Tests for the evaluated-set switch (analysis/lib/evalset.py). It is off by default. When it is on, test cells
are restricted to the main set (RELMA cells to the RELMA set), and the exempt cells (validation, sliding windows,
v1) are left whole."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AN = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(AN))
from lib.cells import Cell  # noqa: E402


def fresh_evalset(monkeypatch, tmp_path, on: bool):
    """A fresh copy of lib/evalset with a tiny key file of its own (the real one carries patient ids)."""
    monkeypatch.setenv("N8_EVALSET_DIR", str(tmp_path))
    if on:
        monkeypatch.setenv("N8_EVALSET", "both")
    else:
        monkeypatch.delenv("N8_EVALSET", raising=False)
    pd.DataFrame({"patient_uuid": ["p1", "p1", "p2"], "obs_ordinal": [0, 1, 5],
                  "relma": [True, False, True]}).to_parquet(tmp_path / "keys.parquet", index=False)
    spec = importlib.util.spec_from_file_location(f"evalset_test_{on}_{id(tmp_path)}", AN / "lib" / "evalset.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


TEST = Cell("mlm", "para_xsite", 0, evalset="mimictest", train_arm="pert")
RELMA = Cell("mlm", "para_xsite", 0, evalset="relma_s1_a0", train_arm="pert")
VAL = Cell("mlm", "para_xsite", 0, evalset="mimictest", train_arm="pert", split="val")
SLIDING = Cell("mlm", "para_xsite", 0, evalset="sliding", train_arm="pert")
V1 = Cell("mlm", "para", 0)
PATS = np.array(["p1", "p1", "p2", "p2", "p3"], dtype=object)
ORDS = np.array([0, 1, 5, 6, 0])


def test_off_by_default_changes_nothing(monkeypatch, tmp_path):
    EV = fresh_evalset(monkeypatch, tmp_path, on=False)
    assert not EV.active() and EV.suffix() == ""
    for c in (TEST, RELMA, VAL, SLIDING, V1):
        assert not EV.applies(c)
        assert EV.mask(c, PATS, ORDS).all()
    df = pd.DataFrame({"patient_uuid": PATS, "obs_ordinal": ORDS})
    assert EV.restrict(TEST, df) is df


def test_on_restricts_test_cells_to_the_main_set(monkeypatch, tmp_path):
    EV = fresh_evalset(monkeypatch, tmp_path, on=True)
    assert EV.active() and EV.suffix() == "_both"
    assert EV.mask(TEST, PATS, ORDS).tolist() == [True, True, True, False, False]   # p3 unknown -> out


def test_on_restricts_relma_cells_to_the_relma_set(monkeypatch, tmp_path):
    EV = fresh_evalset(monkeypatch, tmp_path, on=True)
    assert EV.mask(RELMA, PATS, ORDS).tolist() == [True, False, True, False, False]


def test_exempt_cells_stay_whole(monkeypatch, tmp_path):
    EV = fresh_evalset(monkeypatch, tmp_path, on=True)
    for c in (VAL, SLIDING, V1):
        assert not EV.applies(c)
        assert EV.mask(c, PATS, ORDS).all()


def test_unknown_value_is_an_error(monkeypatch, tmp_path):
    EV = fresh_evalset(monkeypatch, tmp_path, on=False)
    monkeypatch.setenv("N8_EVALSET", "bogus")
    with pytest.raises(ValueError):
        EV.active()
