"""Tests for pipeline/08_tfidf_optuna_b0.py (tuning on the validation split) and
pipeline/08b_tfidf_adopt.py (the adoption rule and its gate), on synthetic inputs: no dataset, no fit.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"


def _load(name, mod):
    s = importlib.util.spec_from_file_location(mod, _SCRIPTS / name)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d08 = _load("08_tfidf_optuna_b0.py", "t_d08")
d08b = _load("08b_tfidf_adopt.py", "t_d08b")


# -- 08: the search space ---------------------------------------------------
def test_default_is_a_point_of_the_search_space():
    """enqueue_trial must be able to replay the default exactly, or the study cannot start."""
    p = d08.default_params()
    for k, choices in d08.SPACE.items():
        assert p[k] in choices, f"default {k}={p[k]!r} not in {choices}"
    assert d08.C_RANGE[0] <= p["C"] <= d08.C_RANGE[1]
    lo, hi, step = d08.MAX_ITER
    assert lo <= p["max_iter"] <= hi and (p["max_iter"] - lo) % step == 0
    assert p["max_features"] == d08.NO_CAP          # the default's None, as the search spells it


def test_suggest_returns_exactly_the_tunable_keys():
    optuna = pytest.importorskip("optuna")
    p = d08.default_params()
    got = d08.suggest(optuna.trial.FixedTrial(p))
    assert got == {k: p[k] for k in d08.TUNABLE}


def test_search_space_covers_the_agreed_axes():
    assert d08.SPACE["min_df"] == [1, 2, 3, 5, 10, 20]
    assert d08.NO_CAP in d08.SPACE["max_features"] and 40000 in d08.SPACE["max_features"]
    assert d08.N_TRIALS == 100 and d08.TPE_SEED == 42
    assert (d08.CONDITION, d08.TRAIN_ARM) == ("para_xsite", "clean")
    assert "token_pattern" not in d08.SPACE and "stop_words" not in d08.SPACE


def test_no_cap_exceeds_every_unigram_vocabulary():
    # unigram vocabulary at the 2M scale with min_df=1: 31,823 terms (clean), 39,228 (pert)
    assert d08.NO_CAP > 39_228


# -- 08b: adoption rule and gate ---------------------------------------------
def test_adoption_needs_strictly_more_than_half_a_point():
    assert d08b.MARGIN == 0.005
    assert not d08b.decide(0.9500, 0.9550)          # exactly 0.5 pp: not adopted
    assert d08b.decide(0.9500, 0.95501)
    assert not d08b.decide(0.9500, 0.9400)
    assert not d08b.decide(0.9500, None)


def test_as_recipe_maps_no_cap_back_to_none_for_unigrams_only():
    p = d08.default_params()
    assert d08b.as_recipe(p)["max_features"] is None
    assert d08b.as_recipe({**p, "ngram_max": 2})["max_features"] == d08.NO_CAP
    assert d08b.as_recipe(p) == {k: d08b.PINNED[k] for k in d08b.TUNABLE}


def test_gate(tmp_path, capsys):
    f = tmp_path / "adopt.json"
    current = {k: d08b.PINNED[k] for k in d08b.TUNABLE}
    assert d08b.check(f) == 1                                           # no decision yet
    f.write_text(json.dumps({"adopt": False, "adopted_recipe": current}))
    assert d08b.check(f) == 0                                           # kept, and PINNED unchanged
    f.write_text(json.dumps({"adopt": True, "adopted_recipe": {**current, "min_df": 2, "C": 1.5}}))
    assert d08b.check(f) == 1                                           # adopted but not applied
    assert "min_df" in capsys.readouterr().out
