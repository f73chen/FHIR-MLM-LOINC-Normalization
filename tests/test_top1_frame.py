"""Tests that analysis/lib/dumps.top1_frame, the fast top-1 reader, defines `correct` in the same
way as load_dir. It reads the real dumps, so it is skipped when the evaluation directory is
missing. The assertions are aggregate only.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from lib import cells as C  # noqa: E402
from lib import dumps as D  # noqa: E402
from lib import registry as R  # noqa: E402

pytestmark = pytest.mark.skipif(not C.EVAL_BASE.is_dir(), reason="eval volume not mounted")


@pytest.mark.parametrize("model,spec", [
    ("bcm", R.HEADLINE),                                                       # deep family
    ("tfidf", dict(condition="nodesc_xsite", train_arm="clean", test_arm="pert")),  # TF-IDF family
])
def test_fast_reader_equals_the_reference_loader(model, spec):
    c = R.cell(model, 0, **spec)
    fast, ref = D.top1_frame(c), D.load_cell(c)
    assert len(fast) == len(ref)
    for col in ("patient_uuid", "obs_ordinal", "true_loinc", "pred_loinc", "correct"):
        assert (fast[col].to_numpy() == ref[col].to_numpy()).all(), col


def test_relma_coverage_follows_section_4_10():
    """Deep-model RELMA dumps score every masked code, and only rows with a RELMA name count (thesis
    Chapter 4). TF-IDF's RELMA dumps hold covered rows only. The covered share is pinned loosely
    around the coverage Chapter 4 reports."""
    c = R.cell("mlm", 0, condition="para_xsite", train_arm="pert", evalset="relma_s1_a0")
    f = D.top1_frame(c)
    cov = D.relma_covered(c, f)
    assert 0.95 < cov.mean() < 0.99
    t = R.cell("tfidf", 0, condition="para_xsite", train_arm="pert", evalset="relma_s1_a0")
    assert D.relma_covered(t, D.top1_frame(t)).all()
    with pytest.raises(ValueError):
        D.relma_covered(R.cell("mlm", 0, **R.HEADLINE), f)
