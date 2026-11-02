"""What the results analyses (the v3_*.py scripts, thesis Chapters 6 and 7) run over: the models, their
seeds and conditions, the headline cell, and where outputs go (results/analysis/v3/<analysis>/).

The seeds of each model are declared here. A loader that falls back to whichever seed exists
could report a single-seed model as a five-seed mean. A missing dump is therefore an error, unless
the caller asks only for the models present. The list of models a result was
computed over is recorded beside it (`stamp`).

A cross-model output compares the four models on their matched observations. So `out_dir` raises an
error for one unless every model's headline dump (with its logits) is present. A per-model output
needs only the dumps of that model.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_lib(name: str):
    """A `lib` module, loaded once per process whether the caller imported `lib` as a package (the
    tests) or loads files by path (the analysis scripts). Two copies of cells.py would mean two Cell
    classes whose instances never compare equal. Also, a dataclass module must be in sys.modules while
    it executes, and a bare `module_from_spec` does not put it there."""
    pkg = sys.modules.get(f"lib.{name}")
    if pkg is not None:
        return pkg
    key = f"n8_lib_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


try:                                                      # imported as the `lib` package
    from . import cells as CELLS
except ImportError:                                       # loaded by file path
    CELLS = load_lib("cells")
Cell = CELLS.Cell
parse = CELLS.parse
scan = CELLS.scan

REPO = CELLS.REPO                                         # the repository root

# -- models ------------------------------------------------------------------
MODELS = ("mlm", "bcm", "llama", "tfidf")
LABEL = {"mlm": "FHIR-MLM", "bcm": "BCM", "llama": "Llama-3.1-8B", "tfidf": "TF-IDF"}
#: model order of the catalogue bootstrap draws and of t25_rows.csv (tables and figures use v3_tables.ALL4)
ORDER = ("mlm", "tfidf", "llama", "bcm")
#: declared training seeds. Llama is trained once.
SEEDS = {"mlm": (0, 1, 2, 3, 4), "bcm": (0, 1, 2, 3, 4), "llama": (0,), "tfidf": (0, 1, 2, 3, 4)}
#: which v3 conditions each model was trained on. BCM and Llama have no nodesc run, and only FHIR-MLM
#: has a single-observation-trained checkpoint.
CONDITIONS = {"para_xsite": ("mlm", "bcm", "llama", "tfidf"), "nodesc_xsite": ("mlm", "tfidf")}
SINGLEOBS_MODELS = ("mlm",)

# -- the design --------------------------------------------------------------
#: the headline cell: trained and tested on perturbed values, with paraphrase descriptions
HEADLINE = dict(condition="para_xsite", train_arm="pert", test_arm="pert")
RELMA_DRAWS = (0, 1, 2)
RELMA_PRIMARY = 0                                         # the reported draw; draws 1 and 2 give the range
EVALSET_RELMA = "relma_s1_a{}"
#: error budgets for auto-mapping: per observation, and per code (T-24)
BUDGETS_OBS = (0.001, 0.01, 0.05)
BUDGETS_CODE = (0.01, 0.05, 0.10)

# -- cells -------------------------------------------------------------------
def cell(model: str, seed: int, **spec) -> Cell:
    """The Cell for `model` at `seed`, refusing a seed the model was never trained at."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if seed not in SEEDS[model]:
        raise ValueError(f"{LABEL[model]} has no seed {seed}; declared seeds are {SEEDS[model]}")
    return Cell(model=model, seed=seed, **spec)


def cells(model: str, seeds: tuple | None = None, **spec) -> list[Cell]:
    """`model`'s cells at each of its declared seeds (or the subset `seeds`)."""
    return [cell(model, s, **spec) for s in (SEEDS[model] if seeds is None else seeds)]


def has(c: Cell, logits: bool = False) -> bool:
    return c.exists() and (not logits or c.has_logits())


def present(models=MODELS, seeds: tuple | None = None, logits: bool = False, **spec) -> list[str]:
    """The models whose dumps exist for this cell spec at every declared seed (or at `seeds`)."""
    out = []
    for m in models:
        ss = SEEDS[m] if seeds is None else tuple(s for s in seeds if s in SEEDS[m])
        if ss and all(has(cell(m, s, **spec), logits) for s in ss):
            out.append(m)
    return out


def require(models, seeds: tuple | None = None, logits: bool = False, **spec) -> None:
    """Raise, naming every missing dump, unless all `models` are present for the spec."""
    missing = []
    for m in models:
        for s in (SEEDS[m] if seeds is None else seeds):
            if s in SEEDS[m] and not has(cell(m, s, **spec), logits):
                missing.append(cell(m, s, **spec).name)
    if missing:
        raise FileNotFoundError(f"{len(missing)} required dump(s) missing: {missing[:6]}"
                                + (" ..." if len(missing) > 6 else ""))


# -- outputs -----------------------------------------------------------------
OUT_ROOT = REPO / "results" / "analysis" / "v3"


def out_dir(analysis: str, cross_model: bool = True) -> Path:
    """Where `analysis` writes. A cross-model output needs every model's headline dump with its logits."""
    if cross_model:
        have = present(logits=True, **HEADLINE)
        if set(have) != set(MODELS):
            raise RuntimeError(f"a cross-model output needs all of {MODELS}; present: {sorted(have)}")
    d = OUT_ROOT / analysis
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                     # stamping must never break an analysis
        return "?"


def stamp(d: Path, analysis: str, models, note: str = "", stage: str = "all") -> Path:
    """Record in `_run.json` beside the outputs: per stage, its models, time and commit.

    Each stage is merged into the record, so an analysis run stage by stage does not end up describing
    only its last stage. The top-level models are the union."""
    p = d / "_run.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if rec.get("analysis") not in (None, analysis):
        rec = {}
    stages = rec.get("stages", {})
    stages[stage] = {"models": [LABEL[m] for m in models],
                     "written": _dt.datetime.now().isoformat(timespec="seconds"), "git": _git_head(),
                     "note": note}
    labels = {x for s in stages.values() for x in s["models"]}
    rec = {"analysis": analysis,
           "models": [LABEL[m] for m in MODELS if LABEL[m] in labels],
           "stages": stages}
    p.write_text(json.dumps(rec, indent=1) + "\n")
    return p
