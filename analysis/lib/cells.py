"""Evaluation-run addressing: turns a cell spec into the name of an evaluation-output directory (under
WORK_DIR/eval), and a directory name back into a cell spec.

The evaluation runner composes the name in one formula (pipeline/15_eval_runner.py, `run_name`), and
TF-IDF's dump script (pipeline/16_tfidf_eval_dump.py) follows the same grammar. The analyses read the
directories through this module, which builds the formula from the arm functions in conditions.py.
tests/test_cells.py requires every real evaluation directory to round-trip through it.

The grammar (15_eval_runner.py, the `run_name` assignment):
    {model}__{condition}{arm}__seed{seed}__{evalset}{scope}{cross}[__{split}]
      arm    ''  for v1/v2, '__clean' | '__pert' for v3          conditions.resolve_arm_suffix
      cross  ''  on the 2x2 diagonal, '__on{test arm}' off it    conditions.resolve_test_arm
      scope  ''  or '__trained' (the single-observation-trained checkpoint)
      split  nothing for the test split, '__val' for validation

  A v3 test arm is 'clean', 'pert', or a perturbed variant built by conditions.perturbation_tag:
  'pert_value', 'pert_range', 'pert_sev0.5', 'pert_sev2.0', 'pert_r1' .. 'pert_r5'.

The module only handles names and reads no data, so it is safe to import anywhere. lib/registry.py
imports it for the results analyses (thesis Chapters 6 and 7), and v3_checks.py and v3_cost.py use it
directly.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]                                   # the repository root
# load conditions.py by path and register it once. pipeline/ holds dozens of modules, and putting it
# on sys.path could shadow names for the analysis that imports this.
if "n8_conditions" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("n8_conditions", REPO / "pipeline" / "conditions.py")
    sys.modules["n8_conditions"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["n8_conditions"])
COND = sys.modules["n8_conditions"]

#: where 15_eval_runner and 16_tfidf_eval_dump write. Must equal lib/dumps.py's EVAL_BASE (tested).
EVAL_BASE = locations.WORK / "eval"

SCOPES = ("history", "singleobs")
SPLITS = ("test", "val")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.]*")               # one `__`-delimited field
_TEST_ARM = re.compile(r"clean|pert(_value|_range)?(_sev[0-9]+(\.[0-9]+)?)?(_r[1-9][0-9]?)?")

#: the letters of the cells in the clean/perturbed 2x2 design (T-39), keyed by (train arm, test arm)
LETTERS = {("clean", "clean"): "A", ("clean", "pert"): "B",
           ("pert", "clean"): "C", ("pert", "pert"): "D"}


@dataclass(frozen=True)
class Cell:
    """One eval run. `test_arm=None` means "the arm the model was trained on" (the 2x2 diagonal).
    `scope='singleobs'` means the single-observation-trained checkpoint (the `__trained` tag).
    Single-observation input alone does not make a run 'singleobs'."""
    model: str
    condition: str
    seed: int
    evalset: str = "mimictest"
    train_arm: str | None = None
    test_arm: str | None = None
    scope: str = "history"
    split: str = "test"

    def __post_init__(self):
        # Validate here and raise ValueError, so that a bad spec never reaches conditions.py. Its
        # checks raise SystemExit, which suits a CLI but not a library.
        if not _TOKEN.fullmatch(self.model) or "__" in self.model:
            raise ValueError(f"bad model token {self.model!r}")
        if self.condition not in COND.CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"bad seed {self.seed!r}")
        if not _TOKEN.fullmatch(self.evalset) or "__" in self.evalset:
            raise ValueError(f"bad evalset token {self.evalset!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, not {self.scope!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {self.split!r}")
        if self.v3:
            if self.train_arm not in ("clean", "pert"):
                raise ValueError(f"{self.condition} is v3: train_arm must be 'clean' or 'pert'")
            if self.test_arm is not None and not _TEST_ARM.fullmatch(self.test_arm):
                raise ValueError(f"unknown test arm {self.test_arm!r}")
        elif self.train_arm is not None or self.test_arm is not None:
            raise ValueError(f"{self.condition} is {self.representation}: arms are v3-only")

    # -- identity ------------------------------------------------------------
    @property
    def representation(self) -> str:
        return COND.representation(self.condition)

    @property
    def v3(self) -> bool:
        return self.representation == "v3"

    @property
    def test_arm_explicit(self) -> str | None:
        """The arm of the test data, spelled out even on the diagonal (None for v1/v2)."""
        return (self.test_arm or self.train_arm) if self.v3 else None

    @property
    def letter(self) -> str | None:
        """'A'..'D' when this is a cell of the clean/perturbed 2x2, whatever the evalset or scope.
        None for v1/v2 and for perturbed variants (value/range, severity, replicates)."""
        return LETTERS.get((self.train_arm, self.test_arm_explicit)) if self.v3 else None

    # -- the name ------------------------------------------------------------
    @property
    def name(self) -> str:
        arm = COND.resolve_arm_suffix(self.condition, self.train_arm, "lib/cells")
        _, cross = COND.resolve_test_arm(self.condition, arm, self.test_arm, "lib/cells")
        scope = "__trained" if self.scope == "singleobs" else ""
        split = f"__{self.split}" if self.split != "test" else ""
        return (f"{self.model}__{self.condition}{arm}__seed{self.seed}"
                f"__{self.evalset}{scope}{cross}{split}")

    @property
    def path(self) -> Path:
        return EVAL_BASE / self.name

    def exists(self) -> bool:
        """A run is complete when its manifest exists. The directory appears when a job starts, and the
        manifest when it finishes."""
        return (self.path / "manifest.json").is_file()

    def has_logits(self) -> bool:
        return (self.path / "logits_centered_f16.npy").is_file()

    def with_(self, **changes) -> "Cell":
        return replace(self, **changes)

    def __str__(self) -> str:
        return self.name


def parse(name: str) -> Cell | None:
    """The Cell a directory name encodes, or None if it is not in the grammar (the floor's
    `refrange_diagnostic__mimictest`, `__smokeN` partial runs, one-off tags like `__BS2REF`).

    Parsing goes field by field and is then checked by re-composition. A name only parses if
    `parse(n).name == n`, so an ambiguous or lossy reading cannot slip through.
    """
    tok = name.split("__")
    if len(tok) < 4:
        return None
    model, condition, i = tok[0], tok[1], 2
    train_arm = None
    if tok[i] in ("clean", "pert"):
        train_arm, i = tok[i], i + 1
    if i + 1 >= len(tok):                                 # need a seed field and an evalset field
        return None
    m = re.fullmatch(r"seed([0-9]+)", tok[i])
    if not m:
        return None
    seed, evalset, rest = int(m.group(1)), tok[i + 1], tok[i + 2:]
    scope, test_arm, split = "history", None, "test"
    if rest and rest[0] == "trained":
        scope, rest = "singleobs", rest[1:]
    if rest and rest[0].startswith("on"):
        test_arm, rest = rest[0][2:], rest[1:]
    if rest and rest[0] == "val":
        split, rest = "val", rest[1:]
    if rest:
        return None                                       # an unknown trailing tag
    try:
        cell = Cell(model=model, condition=condition, seed=seed, evalset=evalset,
                    train_arm=train_arm, test_arm=test_arm, scope=scope, split=split)
    except ValueError:
        return None
    return cell if cell.name == name else None


def scan(base: Path = EVAL_BASE) -> tuple[list[Cell], list[str]]:
    """(cells, unparsed): every directory under `base` that parses, and the names that do not."""
    cells, bad = [], []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        c = parse(p.name)
        (cells.append(c) if c is not None else bad.append(p.name))
    return cells, bad
