"""Tests for pipeline/09_mlm_dataset.py on synthetic fixtures (no MIMIC data)."""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mlmds", Path(__file__).resolve().parents[1] / "pipeline" / "09_mlm_dataset.py")
md = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(md)


def _write_pool(tmp_path, loinc, n):
    (tmp_path / f"{loinc}.txt").write_text("\n".join(f"{loinc}_p{i}" for i in range(n)) + "\n")


def test_load_paraphrases_3way_disjoint_partition_deterministic(tmp_path):
    # 2160-0 Creat SerPl-mCnc; 718-7 Hgb Bld-mCnc; 2345-7 Glucose SerPl-mCnc
    sizes = {"2160-0": 100, "718-7": 56, "2345-7": 20}
    for lc, n in sizes.items():
        _write_pool(tmp_path, lc, n)
    tr, va, te, st = md.load_paraphrases_3way(tmp_path, verbose=False)
    assert st["n_loincs"] == 3
    for lc, n in sizes.items():
        s_tr, s_va, s_te = set(tr[lc]), set(va[lc]), set(te[lc])
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)   # pairwise disjoint
        assert len(va[lc]) >= 1 and len(te[lc]) >= 1                            # non-degenerate
        assert len(tr[lc]) + len(va[lc]) + len(te[lc]) == n                     # exact partition
    assert (len(tr["2160-0"]), len(va["2160-0"]), len(te["2160-0"])) == (70, 15, 15)
    tr2, va2, te2, _ = md.load_paraphrases_3way(tmp_path, verbose=False)
    assert (tr, va, te) == (tr2, va2, te2)                                      # deterministic (seed)


def test_load_paraphrases_3way_dedup(tmp_path):
    (tmp_path / "X.txt").write_text("a\nb\na\nc\nb\n" + "\n".join(f"u{i}" for i in range(20)))
    tr, va, te, st = md.load_paraphrases_3way(tmp_path, verbose=False)
    allp = tr["X"] + va["X"] + te["X"]
    assert len(allp) == len(set(allp))            # dedup -> no string in two pools
    assert st["n_dedup_removed"] == 2             # the 2nd 'a' and 2nd 'b'


def test_load_paraphrases_3way_degenerate_pool_raises(tmp_path):
    _write_pool(tmp_path, "tiny", 1)              # 1 line: empty test pool, so the guard fires
    with pytest.raises(AssertionError):
        md.load_paraphrases_3way(tmp_path, verbose=False)


def test_read_uuid_list_strips_dedups_sorts(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("uuidB.ndjson\nuuidA.ndjson\nuuidB.ndjson\n\nuuidC\n")
    assert md.read_uuid_list(f) == ["uuidA", "uuidB", "uuidC"]
