"""Tests for pipeline/01_build_split.py on synthetic fixtures. They need no MIMIC data, Arrow or
datasets. enumerate_arrow_patients is imported inside main(), so importing the module is cheap.

  python -m pytest tests/test_build_split.py -q
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Load the digit-prefixed module by path.
_SPEC = importlib.util.spec_from_file_location(
    "build_split", Path(__file__).resolve().parents[1] / "pipeline" / "01_build_split.py"
)
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)


def make_eligible(n=500, n_pretrain=200):
    """Synthetic eligible frame: varied history lengths, all within eligibility bounds.
    Returns (df_eligible, pretrain_uuids) where the first n_pretrain UUIDs are 'pretrained'."""
    rows = []
    for i in range(n):
        rows.append({
            "filename": f"u{i:04d}.ndjson",
            "total_resources": 4 + (i % 400),      # spans 4..403, all in (3, 8192]
            "Encounter": 1, "Procedure": 0,
            "LabObservation": 1 + (i % 5),          # always >= 1
            "ChartObservation": 0, "Specimen": 0,
        })
    df = bs.apply_eligibility(pd.DataFrame(rows))
    pretrain = {f"u{i:04d}" for i in range(n_pretrain)}
    return df, pretrain


SMALL = dict(val_size=20, test_size=50, train_size=100, n_bins=10, seed=42)


def test_strip_ndjson():
    assert bs.strip_ndjson("abc.ndjson") == "abc"
    assert bs.strip_ndjson("abc") == "abc"
    assert bs.strip_ndjson("a.ndjson.ndjson") == "a.ndjson"  # strips one suffix only


def test_apply_eligibility_bounds():
    df = pd.DataFrame([
        {"filename": "keep.ndjson", "total_resources": 100, "Encounter": 0, "Procedure": 0,
         "LabObservation": 1, "ChartObservation": 0, "Specimen": 0},
        {"filename": "no_lab.ndjson", "total_resources": 100, "Encounter": 0, "Procedure": 0,
         "LabObservation": 0, "ChartObservation": 0, "Specimen": 0},   # 0 labs -> drop
        {"filename": "too_few.ndjson", "total_resources": 3, "Encounter": 0, "Procedure": 0,
         "LabObservation": 5, "ChartObservation": 0, "Specimen": 0},   # <=3 -> drop
        {"filename": "too_many.ndjson", "total_resources": 8193, "Encounter": 0, "Procedure": 0,
         "LabObservation": 5, "ChartObservation": 0, "Specimen": 0},   # >8192 -> drop
    ])
    out = bs.apply_eligibility(df)
    assert set(out["uuid"]) == {"keep"}


def test_build_split_sizes_and_disjointness():
    df, pretrain = make_eligible()
    s = bs.build_split(df, pretrain, **SMALL)
    train, val, test = set(s["train"]["uuid"]), set(s["val"]["uuid"]), set(s["test"]["uuid"])
    assert (len(val), len(test), len(train)) == (20, 50, 100)
    assert not (train & val) and not (train & test) and not (val & test)
    # run_asserts (with matching small sizes) must pass cleanly.
    bs.run_asserts(s, val_size=20, test_size=50, train_size=100)


def test_all_finetune_splits_are_clean():
    df, pretrain = make_eligible()
    s = bs.build_split(df, pretrain, **SMALL)
    for k in ("train", "val", "test"):
        assert not (set(s[k]["uuid"]) & pretrain), f"{k} must be contamination-free"
    # train comes from the clean pool too, so it is disjoint from pretrain (matched distribution).
    assert set(s["train"]["uuid"]) <= (set(df["uuid"]) - pretrain)


def test_determinism():
    df, pretrain = make_eligible()
    s1 = bs.build_split(df, pretrain, **SMALL)
    s2 = bs.build_split(df, pretrain, **SMALL)
    for k in ("train", "val", "test", "reserve"):
        assert set(s1[k]["uuid"]) == set(s2[k]["uuid"]), f"{k} not reproducible under fixed seed"


def test_reserve_is_clean_minus_ft_splits():
    df, pretrain = make_eligible()
    s = bs.build_split(df, pretrain, **SMALL)
    clean = set(s["clean_pool"]["uuid"])
    train, val, test, reserve = (set(s[k]["uuid"]) for k in ("train", "val", "test", "reserve"))
    assert reserve == clean - train - val - test
    assert not (reserve & pretrain)


def test_run_asserts_fires_on_injected_overlap():
    df, pretrain = make_eligible()
    s = bs.build_split(df, pretrain, **SMALL)
    # Inject a contamination hole: move a pretrained patient into test.
    bad_uuid = next(iter(pretrain))
    bad_row = s["clean_pool"].iloc[[0]].copy()
    bad_row["uuid"] = bad_uuid
    s["test"] = pd.concat([s["test"], bad_row], ignore_index=True)
    with pytest.raises(AssertionError):
        bs.run_asserts(s, val_size=20, test_size=50, train_size=100)


def test_run_asserts_fires_on_train_val_overlap():
    df, pretrain = make_eligible()
    s = bs.build_split(df, pretrain, **SMALL)
    dup = s["val"].iloc[[0]].copy()
    s["train"] = pd.concat([s["train"], dup], ignore_index=True)
    with pytest.raises(AssertionError):
        bs.run_asserts(s, val_size=20, test_size=50, train_size=100)
