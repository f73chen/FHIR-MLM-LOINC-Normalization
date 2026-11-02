"""Tests for pipeline/tfidf_recipe.py, the TF-IDF tokenizer and recipe, on synthetic strings (no dataset,
no patient data).

The analyzer cases are failures of the default sklearn tokenizer with English stop words. Each test
asserts that the failure does not happen with the recipe tokenizer. The drift tests check that 07, 08,
08b and 16 do not go back to building their own vectorizer.
"""
import importlib.util
import pickle
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(_SCRIPTS))
import tfidf_recipe as R  # noqa: E402

pytest.importorskip("sklearn")


def an(s):
    return R.make_vectorizer().build_analyzer()(s)


# -- values -----------------------------------------------------------------
def test_values_are_whole_tokens_with_their_signs():
    assert an("1.390e+02") == ["1.390e+02"]
    bags = {tuple(an(v)) for v in ("1.390e+02", "2.390e+02", "1.390e-02", "-1.390e+02")}
    assert len(bags) == 4, "139, 239, 0.0139 and -139 must not share a token bag"


def test_every_4sf_value_keeps_its_own_token():
    vals = [f"{m / 1000:.3f}e{e:+03d}" for m in range(1000, 10000, 7) for e in (-3, 0, 2, 4)]
    assert len({tuple(an(v)) for v in vals}) == len(vals)


def test_old_default_tokenizer_really_lost_the_leading_digit():
    """Pins what the default tokenizer gets wrong, so the comparisons above are not vacuous."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    old = TfidfVectorizer(stop_words="english").build_analyzer()
    assert old("1.390e+02") == old("2.390e+02") == old("1.390e-02") == ["390e", "02"]


# -- units ------------------------------------------------------------------
@pytest.mark.parametrize("unit,token", [
    ("meq/L", "meq/l"), ("U/L", "u/l"), ("%", "%"), ("10*3/uL", "10*3/ul"), ("mg/(24.h)", "mg/(24.h)"),
    ("[IU]/L", "[iu]/l"), ("mm[Hg]", "mm[hg]"), ("s", "s"), ("U/g", "u/g"), ("fL", "fl")])
def test_units_are_whole_tokens(unit, token):
    assert an(f"x 5.000e+00 {unit}") == ["x", "5.000e+00", token]


def test_range_units_stop_at_the_comma():
    s = "Sodium 1.390e+02 meq/L reference Low: 1.350e+02 meq/L, High: 1.450e+02 meq/L priority-STAT"
    assert an(s) == ["sodium", "1.390e+02", "meq/l", "reference", "low", "1.350e+02", "meq/l",
                     "high", "1.450e+02", "meq/l", "priority", "stat"]


def test_unitless_value_does_not_swallow_the_next_field():
    assert an("Specific gravity 1.015e+00 priority-STAT") == \
        ["specific", "gravity", "1.015e+00", "priority", "stat"]
    assert an("x 1.015e+00 reference Low: 1.001e+00, High: 1.035e+00")[:3] == ["x", "1.015e+00", "reference"]


# -- comparators, grades, signs ---------------------------------------------
def test_comparator_is_its_own_token():
    assert an("troponin <1.000e-02 ng/mL") == ["troponin", "<", "1.000e-02", "ng/ml"]
    assert an("x >=1.000e+03 U/L")[:3] == ["x", ">=", "1.000e+03"]


@pytest.mark.parametrize("g", ["1+", "2+", "3+", "4+", "5+"])
def test_dipstick_grades_survive(g):
    assert an(f"urine protein {g}") == ["urine", "protein", g]


def test_percent_and_count_signs_in_descriptions():
    assert an("Lymphocytes %") != an("Lymphocytes #")
    assert "#" in an("Leukocytes [#/volume] in Blood")


# -- words ------------------------------------------------------------------
def test_single_letters_and_digits_are_kept():
    assert an("Hepatitis B surface Ag") == ["hepatitis", "b", "surface", "ag"]
    assert an("HIV-1/2 Ab") == ["hiv", "1", "2", "ab"]
    assert an("Blood group O neg") != an("Blood group A neg")


def test_no_stop_word_loss():
    assert an("not detected") == ["not", "detected"]
    assert an("not detected") != an("detected")
    assert "whole" in an("Potassium, whole blood")
    for w in ("few", "many", "none", "am"):
        assert an(w) == [w]


def test_descriptions_still_tokenise_like_words():
    assert an("C-reactive protein (serum)") == ["c", "reactive", "protein", "serum"]
    assert an("Hemoglobin A1c") == ["hemoglobin", "a1c"]


# -- recipe -----------------------------------------------------------------
def test_pinned_recipe():
    p = R.PINNED
    assert p["token_pattern"] == R.TOKEN_PATTERN and p["stop_words"] is None and p["lowercase"] is True
    assert p["min_df"] == 5 and p["max_features"] is None
    assert (p["ngram_max"], p["sublinear_tf"], p["max_df"]) == (1, False, 1.0)
    assert (p["C"], p["solver"], p["max_iter"]) == (6.58751637576249, "liblinear", 1500)
    assert set(R.TUNABLE) <= set(p) and "token_pattern" not in R.TUNABLE


def test_vectorizer_and_pipeline_follow_the_recipe():
    v = R.make_vectorizer()
    assert (v.min_df, v.max_features, v.stop_words, v.token_pattern) == (5, None, None, R.TOKEN_PATTERN)
    v2 = R.make_vectorizer({"min_df": 2, "ngram_max": 2})
    assert v2.min_df == 2 and v2.ngram_range == (1, 2) and v2.token_pattern == R.TOKEN_PATTERN
    pipe = R.make_pipeline(fit_jobs=3)
    assert type(pipe.named_steps["clf"]).__name__ == "OneVsRestClassifier"
    assert pipe.named_steps["clf"].n_jobs == 3
    assert type(R.make_pipeline({"solver": "lbfgs"}).named_steps["clf"]).__name__ == "LogisticRegression"


def test_signature_ignores_n_jobs_but_sees_the_recipe():
    a, b = R.make_pipeline(fit_jobs=1), R.make_pipeline(fit_jobs=8)
    assert R.pipeline_signature(a) == R.pipeline_signature(b)
    assert R.pipeline_signature(R.make_pipeline({"min_df": 1})) != R.pipeline_signature(a)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    old = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", min_df=5)),
                    ("clf", a.named_steps["clf"])])
    assert R.pipeline_signature(old) != R.pipeline_signature(a), "a model fitted with another recipe must be refused"


def test_fitted_pipeline_pickles_and_tokenises_identically():
    texts = ["sodium 1.390e+02 meq/l", "potassium 4.100e+00 mmol/l", "urine protein 1+"] * 6
    y = [0, 1, 2] * 6
    pipe = R.make_pipeline({"min_df": 1, "solver": "lbfgs", "max_iter": 200}).fit(texts, y)
    back = pickle.loads(pickle.dumps(pipe))
    assert back.named_steps["tfidf"].vocabulary_ == pipe.named_steps["tfidf"].vocabulary_
    assert (back.predict(texts) == pipe.predict(texts)).all()
    assert "1.390e+02" in pipe.named_steps["tfidf"].vocabulary_


def test_records_current_recipe(tmp_path):
    import json
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"pinned_params": R.PINNED}))
    assert R.records_current_recipe(f, "pinned_params")
    assert not R.records_current_recipe(f, "pinned")
    f.write_text(json.dumps({"pinned_params": {**R.PINNED, "min_df": 1}}))
    assert not R.records_current_recipe(f, "pinned_params")
    assert not R.records_current_recipe(tmp_path / "missing.json", "pinned_params")


# -- drift: no script builds its own vectorizer -----------------------------
@pytest.mark.parametrize("script", ["07_tfidf_train_eval.py", "08_tfidf_optuna_b0.py",
                                    "08b_tfidf_adopt.py", "16_tfidf_eval_dump.py"])
def test_scripts_build_from_the_recipe(script):
    src = (_SCRIPTS / script).read_text()
    assert not re.search(r"TfidfVectorizer\s*\(", src), f"{script} builds its own TfidfVectorizer"
    assert not re.search(r"^PINNED\s*=", src, re.M), f"{script} defines its own PINNED"
    assert "tfidf_recipe" in src


def _load(name, mod):
    s = importlib.util.spec_from_file_location(mod, _SCRIPTS / name)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_modules_expose_the_same_recipe_object():
    d07 = _load("07_tfidf_train_eval.py", "r_d07")
    d16 = _load("16_tfidf_eval_dump.py", "r_d16")
    assert d07.PINNED is R.PINNED or d07.PINNED == R.PINNED
    assert d16.PINNED == R.PINNED
    assert d07.make_logreg is R.make_logreg
