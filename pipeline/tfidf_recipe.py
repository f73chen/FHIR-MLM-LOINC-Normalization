"""The definition of the TF-IDF baseline: its tokenizer, its vocabulary rule and its classifier
(thesis Chapter 5 and Appendix D). pipeline/07 (training and evaluation), 08 (tuning), 08b (adoption)
and 16 (refit and per-observation dumps) all build their pipeline from here.

sklearn's default tokenization is built for English prose. TF-IDF reads one line per observation,
e.g. `<description> 1.390e+02 meq/L reference Low: 1.350e+02 meq/L, High: 1.450e+02 meq/L priority-STAT`.
The defaults would destroy most of what is not a word:
  * token_pattern (?u)\\b\\w\\w+\\b keeps only runs of two or more word characters. `1.390e+02`
    becomes `390e 02`, which loses the leading digit and the signs of the exponent and the value
    (139 = 239 = 0.0139 = -139). `%`, `U/g`, `s` and `1+` vanish, and `10*3/uL` becomes `10 ul`.
  * stop_words="english" deletes `not`, `whole`, `few`, `many`, `none` and `am`. "not detected" then
    reads the same as "detected", and "whole blood" loses "whole".

The tokenizer here is built for this clinical text. It is applied after lowercasing, and the
alternatives are tried in this order:
    unit        the whitespace-delimited string right after a value or bound, kept whole: `meq/l`,
                `u/l`, `%`, `10*3/ul`, `mg/(24.h)`, `[iu]/l`. It ends at a comma (the range's
                "Low: ... unit, High: ..."). The guard stops a unit-less value from swallowing the
                next field: `1.015e+00 priority-stat` -> `1.015e+00`, `priority`, `stat`.
    comparator  `<`, `>`, `<=`, `>=` as their own tokens, so `<1.000e-02` keeps both facts.
    value       a value or bound in the v3 rendering (4 s.f. scientific), whole, with its signs.
    grade       dipstick grades `1+` ... `5+`.
    sign        `%` and `#` anywhere (e.g. "Lymphocytes %" vs "Lymphocytes #").
    word        runs of word characters, single letters and digits included ("hepatitis b").
  There is no stop-word list, because IDF already down-weights words that occur everywhere.
  The tokenizer is a string pattern, so the pipeline that 16 pickles stays a plain sklearn object
  that loads without project code, and every manifest records the pattern verbatim.

The vocabulary rule: a term must occur in at least 5 training observations (min_df=5), with no top-N
cap (max_features=None). With this tokenizer, a 20,000-term cap would keep thousands of exact values
and drop thousands of rare words. A word seen in 1–4 training observations of one code points to that
same code in only 11% of its test occurrences, against 97% for words seen 20+ times.

The classifier: liblinear one-vs-rest logistic regression, C=6.58751637576249, max_iter=1500,
random_state=42. 08 re-tunes the recipe on the val split and 08b adopts the result only if it beats
this default by more than 0.5 pp at the 2M scale. Adoption is an edit to PINNED here.

Run directly, it prints PINNED. With `--matches <json> <key>` it exits 0 if that JSON records the
current recipe under `key`.

  python pipeline/tfidf_recipe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# guard against a unit-less value swallowing the next field that 06 and
# utils.extract_structured_features_single can write after it. Only `reference` and `priority-`
# occur in v3 text. The other two are the extractor's remaining prefixes, guarded as well.
_UNIT_GUARD = r"(?!reference\b|priority-|status-|absent-)"
TOKEN_PATTERN = (
    r"(?u)"
    rf"(?<=e[-+]\d\d ){_UNIT_GUARD}[^\s,]+"        # unit after a value/bound with a 2-digit exponent
    rf"|(?<=e[-+]\d\d\d ){_UNIT_GUARD}[^\s,]+"     # ... and with a 3-digit exponent
    r"|[<>]=?"                                    # comparator
    r"|(?<![\w.])-?\d\.\d+e[-+]\d+"               # value / bound, whole (v3: 4 s.f. scientific)
    r"|\d\+"                                      # dipstick grade
    r"|[%#]"                                      # percent / count signs
    r"|\b\w+\b"                                   # words, single letters and digits included
)
STOP_WORDS = None

#: The pinned recipe. The tokenizer fields are part of it so that every JSON that records PINNED
#: (07's `pinned_params`, 16's manifests) also records how the text was tokenised.
PINNED = dict(
    token_pattern=TOKEN_PATTERN, stop_words=STOP_WORDS, lowercase=True,
    max_features=None, ngram_max=1, sublinear_tf=False, min_df=5, max_df=1.0,
    C=6.58751637576249, solver="liblinear", max_iter=1500,
)
#: The keys 08 searches over. Tokenisation is not among them, because it is a correctness decision.
TUNABLE = ("max_features", "ngram_max", "sublinear_tf", "min_df", "max_df", "C", "solver", "max_iter")


def recipe(params: dict | None = None) -> dict:
    """PINNED with `params` laid over it (08 passes only the tunable keys)."""
    out = dict(PINNED)
    out.update(params or {})
    return out


def make_vectorizer(params: dict | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    p = recipe(params)
    return TfidfVectorizer(lowercase=p["lowercase"], token_pattern=p["token_pattern"],
                           stop_words=p["stop_words"], max_features=p["max_features"],
                           ngram_range=(1, p["ngram_max"]), sublinear_tf=p["sublinear_tf"],
                           min_df=p["min_df"], max_df=p["max_df"])


def make_logreg(C, solver, max_iter, fit_jobs=1):
    """The pinned classifier. liblinear's multiclass is internally one-vs-rest (679 binary fits) but
    single-threaded, and takes about 2.5 h on the 2M downsample. Wrapping it in
    OneVsRestClassifier(n_jobs=fit_jobs) runs the same 679 binary fits across cores, about 7.6x
    faster at 2M. The result matches the internal one-vs-rest numerically, with probabilities within
    about 1e-9, the same calibration and the same argmax. Each binary fit is a clone with
    random_state=42, so the result does not depend on fit_jobs. saga and lbfgs are natively
    multinomial. With sklearn 1.6.1 the default loky memmap avoids the read-only bug."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    base = LogisticRegression(C=C, solver=solver, max_iter=max_iter, class_weight=None, random_state=42)
    return OneVsRestClassifier(base, n_jobs=fit_jobs) if solver == "liblinear" else base


def make_pipeline(params: dict | None = None, fit_jobs: int = 1):
    """Pipeline([("tfidf", vectorizer), ("clf", classifier)]) for a recipe (default: PINNED)."""
    from sklearn.pipeline import Pipeline
    p = recipe(params)
    return Pipeline([("tfidf", make_vectorizer(p)),
                     ("clf", make_logreg(p["C"], p["solver"], p["max_iter"], fit_jobs))])


def pipeline_signature(pipe) -> dict:
    """What a TF-IDF pipeline was built with: the analyzer, the vocabulary rule and the classifier
    settings. n_jobs is left out because it does not change the result. 16 compares a cached
    model's signature with a freshly built one and refuses the cache on any difference. The cache
    key does not encode the recipe, so without this check a stale model would be reused."""
    v, c = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    base = getattr(c, "estimator", c)
    return {"lowercase": v.lowercase, "token_pattern": v.token_pattern,
            "stop_words": v.stop_words, "ngram_range": list(v.ngram_range),
            "min_df": v.min_df, "max_df": v.max_df, "max_features": v.max_features,
            "sublinear_tf": v.sublinear_tf, "C": base.C, "solver": base.solver,
            "max_iter": base.max_iter, "class_weight": base.class_weight,
            "random_state": base.random_state, "one_vs_rest": c is not base}


def records_current_recipe(path, key: str) -> bool:
    """True iff the JSON at `path` records the current PINNED under `key`. Batch runs use this as a
    recipe-aware completion marker, so a result produced under an older recipe never counts as done."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get(key) == json.loads(json.dumps(PINNED))
    except (ValueError, OSError):
        return False


if __name__ == "__main__":
    # `python tfidf_recipe.py --matches <json> <key>`: exit 0 iff that JSON records the current recipe
    if len(sys.argv) == 4 and sys.argv[1] == "--matches":
        sys.exit(0 if records_current_recipe(sys.argv[2], sys.argv[3]) else 1)
    print(json.dumps(PINNED, indent=2))
