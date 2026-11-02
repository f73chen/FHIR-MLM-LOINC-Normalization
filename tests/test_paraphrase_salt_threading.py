"""Tests that the dataset builders 09, 10 and 11 thread `paraphrase_salt` through to the per-patient
draw, which the RELMA assignment seeds rely on.

The tests target two failures:
 1. A dropped salt. If any link in
        _build_split_dataset -> _process_chunk -> _process_one_patient
    drops the keyword, every assignment seed gives the same assignment.
    A signature-only check is not enough, because the parallel (`_process_chunk`) and sequential
    paths are separate call sites, and the parallel one is easy to miss.
 2. A changed default. salt=None must reproduce the unsalted assignment, or datasets built without
    a salt stop being reproducible.

The checks are static (AST) only. They need no patient data, no tokenizers and no heavy imports.
"""
import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"
FILES = ["09_mlm_dataset.py", "10_bcm_dataset.py", "11_llama_dataset.py"]
CHAIN = ["_process_one_patient", "_process_chunk", "_build_split_dataset"]


def _tree(fn):
    return ast.parse((SCRIPTS / fn).read_text())


def _funcs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("fn", FILES)
@pytest.mark.parametrize("func", CHAIN)
def test_function_accepts_paraphrase_salt(fn, func):
    f = _funcs(_tree(fn))[func]
    names = [a.arg for a in f.args.args + f.args.kwonlyargs]
    assert "paraphrase_salt" in names, f"{fn}:{func} missing paraphrase_salt parameter"


@pytest.mark.parametrize("fn", FILES)
@pytest.mark.parametrize("func", CHAIN)
def test_paraphrase_salt_defaults_to_none(fn, func):
    """The default must be None, so builds without a salt stay bit-reproducible."""
    f = _funcs(_tree(fn))[func]
    names = [a.arg for a in f.args.args]
    di = names.index("paraphrase_salt") - (len(names) - len(f.args.defaults))
    assert di >= 0, f"{fn}:{func} paraphrase_salt has no default"
    d = f.args.defaults[di]
    assert isinstance(d, ast.Constant) and d.value is None, f"{fn}:{func} default is not None"


@pytest.mark.parametrize("fn", FILES)
def test_every_internal_call_forwards_the_salt(fn):
    """Each call to _process_one_patient or _process_chunk inside these files must forward
    paraphrase_salt. A call that missed it would build the dataset with the unsalted assignment."""
    tree = _tree(fn)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = getattr(f, "id", None) or getattr(f, "attr", None)
        if name not in ("_process_one_patient", "_process_chunk"):
            continue
        if not any(k.arg == "paraphrase_salt" for k in node.keywords):
            bad.append(f"{name} @ line {node.lineno}")
    assert not bad, f"{fn}: calls that do not forward paraphrase_salt: {bad}"


@pytest.mark.parametrize("fn", FILES)
def test_rng_uses_salt_only_when_provided(fn):
    """salt=None must take the unsalted patient_rng(uuid) branch, byte for byte."""
    src = (SCRIPTS / fn).read_text()
    assert "patient_rng(uuid, salt=paraphrase_salt) if paraphrase_salt else patient_rng(uuid)" in src, \
        f"{fn}: rng line is not the guarded form"


def test_patient_rng_salt_actually_changes_the_draw():
    """Sanity check of the mechanism. Different salts give different per-patient RNG streams and the
    same salt gives identical ones. This is what makes assignment seeds meaningful."""
    from paraphrase_utils import patient_rng
    u = "00000000-0000-0000-0000-000000000001"
    a = [patient_rng(u, salt="relma-a0").random() for _ in range(3)]
    b = [patient_rng(u, salt="relma-a1").random() for _ in range(3)]
    c = [patient_rng(u, salt="relma-a0").random() for _ in range(3)]
    assert a == c, "same salt must be deterministic"
    assert a != b, "different salts must produce different assignments"
    assert patient_rng(u).random() != patient_rng(u, salt="relma-a0").random(), \
        "salted stream must differ from the default stream"
