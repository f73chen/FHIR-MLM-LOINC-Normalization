"""Tests for analysis/lib/cells.py, which composes and parses the names of evaluation-run directories.

The names below are copied from the real evaluation directory. If the naming grammar in
pipeline/15_eval_runner.py or pipeline/conditions.py changes and the analyses do not follow, these
tests fail. Otherwise the mismatch would first show up in a table.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from lib.cells import EVAL_BASE, Cell, parse, scan  # noqa: E402

KNOWN = [
    # (cell spec, the directory name it must produce)
    (dict(model="mlm", condition="para_xsite", seed=0, train_arm="pert"),
     "mlm__para_xsite__pert__seed0__mimictest"),
    (dict(model="mlm", condition="para_xsite", seed=0, train_arm="clean", test_arm="pert"),
     "mlm__para_xsite__clean__seed0__mimictest__onpert"),
    (dict(model="bcm", condition="para_xsite", seed=3, train_arm="pert", test_arm="clean"),
     "bcm__para_xsite__pert__seed3__mimictest__onclean"),
    (dict(model="bcm", condition="para_xsite", seed=3, train_arm="clean", split="val"),
     "bcm__para_xsite__clean__seed3__mimictest__val"),
    (dict(model="tfidf", condition="para_xsite", seed=2, evalset="relma_s1_a1",
          train_arm="clean", test_arm="pert"),
     "tfidf__para_xsite__clean__seed2__relma_s1_a1__onpert"),
    (dict(model="mlm", condition="para_xsite", seed=1, evalset="relma_s1_a2", train_arm="pert"),
     "mlm__para_xsite__pert__seed1__relma_s1_a2"),
    (dict(model="mlm", condition="nodesc_xsite", seed=4, evalset="singleobs_trim",
          train_arm="pert", test_arm="clean", scope="singleobs"),
     "mlm__nodesc_xsite__pert__seed4__singleobs_trim__trained__onclean"),
    (dict(model="mlm", condition="para_xsite", seed=1, train_arm="clean", test_arm="pert_value"),
     "mlm__para_xsite__clean__seed1__mimictest__onpert_value"),
    (dict(model="mlm", condition="para_xsite", seed=2, train_arm="pert", test_arm="pert_sev2.0"),
     "mlm__para_xsite__pert__seed2__mimictest__onpert_sev2.0"),
    (dict(model="mlm", condition="para_xsite", seed=0, train_arm="clean", test_arm="pert_r3"),
     "mlm__para_xsite__clean__seed0__mimictest__onpert_r3"),
    (dict(model="bcm", condition="para_xsite", seed=0, evalset="sliding", train_arm="pert"),
     "bcm__para_xsite__pert__seed0__sliding"),
    (dict(model="tfidf", condition="nodesc_xsite", seed=4, train_arm="pert", test_arm="clean"),
     "tfidf__nodesc_xsite__pert__seed4__mimictest__onclean"),
    # v1: no arm segment, and the arms are refused
    (dict(model="llama", condition="para", seed=0), "llama__para__seed0__mimictest"),
    (dict(model="mlm", condition="para", seed=2, evalset="singleobs_trim_relma_s1_a0",
          scope="singleobs"),
     "mlm__para__seed2__singleobs_trim_relma_s1_a0__trained"),
    (dict(model="llama", condition="nodesc", seed=0, split="val"),
     "llama__nodesc__seed0__mimictest__val"),
]


@pytest.mark.parametrize("spec,name", KNOWN)
def test_composes_the_runner_s_name(spec, name):
    assert Cell(**spec).name == name


@pytest.mark.parametrize("spec,name", KNOWN)
def test_parse_round_trips(spec, name):
    c = parse(name)
    assert c is not None and c.name == name
    assert c == Cell(**spec)


def test_diagonal_is_the_same_cell_whether_or_not_the_test_arm_is_spelled_out():
    a = Cell("mlm", "para_xsite", 0, train_arm="pert")
    b = Cell("mlm", "para_xsite", 0, train_arm="pert", test_arm="pert")
    assert a.name == b.name == "mlm__para_xsite__pert__seed0__mimictest"
    assert a.test_arm_explicit == b.test_arm_explicit == "pert"


def test_letters_follow_t39():
    base = dict(model="tfidf", condition="para_xsite", seed=0)
    assert Cell(**base, train_arm="clean").letter == "A"
    assert Cell(**base, train_arm="clean", test_arm="pert").letter == "B"
    assert Cell(**base, train_arm="pert", test_arm="clean").letter == "C"
    assert Cell(**base, train_arm="pert").letter == "D"
    # letters are defined on any evalset or scope of the clean/perturbed axis ...
    assert Cell(**base, train_arm="clean", test_arm="pert", evalset="relma_s1_a0").letter == "B"
    # ... but not for perturbed variants, nor for v1
    assert Cell(**base, train_arm="clean", test_arm="pert_value").letter is None
    assert Cell("mlm", "para", 0).letter is None


@pytest.mark.parametrize("spec", [
    dict(model="mlm", condition="para_xsite", seed=0),                       # v3 without an arm
    dict(model="mlm", condition="para", seed=0, train_arm="pert"),          # v1 with an arm
    dict(model="mlm", condition="para", seed=0, test_arm="pert"),
    dict(model="mlm", condition="para_xsite", seed=0, train_arm="pert", test_arm="perturbed"),
    dict(model="mlm", condition="para_xsite", seed=0, train_arm="pert", scope="single"),
    dict(model="mlm", condition="para_xsite", seed=0, train_arm="pert", split="train"),
    dict(model="mlm", condition="para_xsite", seed=-1, train_arm="pert"),
    dict(model="m__x", condition="para_xsite", seed=0, train_arm="pert"),
    dict(model="mlm", condition="relma", seed=0),
])
def test_bad_specs_raise_valueerror_not_systemexit(spec):
    with pytest.raises(ValueError):
        Cell(**spec)


@pytest.mark.parametrize("name", [
    "refrange_diagnostic__mimictest",                       # the floor: not in the grammar
    "mlm__para__seed0__singleobs_trim__trained__BS2REF",    # a one-off tag
    "mlm__para_xsite__pert__seed0__mimictest__smoke400",    # a partial run
    "mlm__para_xsite__pert__seed0__mimictest__onpert",      # the diagonal is never tagged
    "mlm__para__seed0__mimictest__onpert",                  # arms are v3-only
    "mlm__para_xsite__seed0__mimictest",                    # v3 without its arm segment
    "mlm__para_xsite__pert__mimictest",                     # no seed field
])
def test_names_outside_the_grammar_do_not_parse(name):
    assert parse(name) is None


def test_eval_base_matches_the_dump_loader():
    from lib import dumps
    assert EVAL_BASE == dumps.EVAL_BASE


@pytest.mark.skipif(not EVAL_BASE.is_dir(), reason="eval volume not mounted")
def test_every_real_eval_directory_round_trips():
    """Checks the real directories. Every directory that the runners wrote must parse and re-compose
    to its own name, and every v3 directory must be in the grammar. Unparsed names must be the known
    irregular ones."""
    cells, bad = scan()
    assert all(c.name == c.path.name for c in cells)
    v3_bad = [b for b in bad if "xsite" in b]
    assert not v3_bad, f"v3 directories outside the grammar: {v3_bad}"
    allowed = {"refrange_diagnostic__mimictest"}
    unexpected = [b for b in bad if b not in allowed and "__smoke" not in b and "BS2REF" not in b]
    assert not unexpected, f"unparsed eval directories: {unexpected}"
    assert sum(c.v3 for c in cells) > 300
