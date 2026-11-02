"""Tests for analysis/lib/registry.py: declared seeds, where outputs go, and the `_run.json` stamp."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from lib import cells as C  # noqa: E402
from lib import registry as R  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT_ROOT", tmp_path / "out")
    return tmp_path / "out"


def test_seeds_are_declared_not_inferred():
    assert R.SEEDS["llama"] == (0,)
    with pytest.raises(ValueError, match="no seed 1"):
        R.cell("llama", 1, condition="para_xsite", train_arm="pert")
    assert len(R.cells("mlm", condition="para_xsite", train_arm="pert")) == 5


def test_cross_model_output_needs_all_four_models(root, monkeypatch):
    monkeypatch.setattr(R, "present", lambda *a, **k: ["mlm", "bcm", "tfidf"])
    with pytest.raises(RuntimeError, match="needs all"):
        R.out_dir("x")
    monkeypatch.setattr(R, "present", lambda *a, **k: list(R.MODELS))
    d = R.out_dir("x")
    assert d == root / "x" and d.is_dir()


def test_per_model_output_needs_no_other_model(root, monkeypatch):
    monkeypatch.setattr(R, "present", lambda *a, **k: [])
    assert R.out_dir("rq3", cross_model=False) == root / "rq3"


def test_stamp_merges_stages(root):
    d = R.out_dir("x", cross_model=False)
    s = json.loads(R.stamp(d, "x", ["mlm", "tfidf"], stage="a").read_text())
    assert s["models"] == ["FHIR-MLM", "TF-IDF"] and "mode" not in s
    # a second stage merges: the union of models (registry order), both stages recorded
    s = json.loads(R.stamp(d, "x", ["llama"], stage="b").read_text())
    assert s["models"] == ["FHIR-MLM", "Llama-3.1-8B", "TF-IDF"] and set(s["stages"]) == {"a", "b"}


def test_loading_by_path_shares_the_package_s_cell_class():
    """Scripts load lib files by path and the tests import the package. Either way there must be one
    Cell class."""
    spec = importlib.util.spec_from_file_location("registry_by_path", ANALYSIS / "lib" / "registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.Cell is C.Cell


@pytest.mark.skipif(not C.EVAL_BASE.is_dir(), reason="eval volume not mounted")
def test_present_on_the_real_dumps():
    assert set(R.present(logits=True, **R.HEADLINE)) == set(R.MODELS)
    nodesc = R.present(models=R.CONDITIONS["nodesc_xsite"], logits=True,
                       condition="nodesc_xsite", train_arm="pert")
    assert nodesc == ["mlm", "tfidf"]
    with pytest.raises(FileNotFoundError, match="relma_s1_a9"):   # a draw that was never built
        R.require(["mlm"], condition="para_xsite", train_arm="pert", evalset="relma_s1_a9")
