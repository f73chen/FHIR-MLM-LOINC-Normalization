"""Model-free text similarity for the data-quality audit (analysis/a6_data_quality.py).

It uses character n-gram TF-IDF instead of an embedding model. This measure is independent of every
model trained here, including the TF-IDF classifier. Judging the data with BCM or the fine-tuned
encoders would be circular. A paraphrase would score well because the models had learned it.
Character n-grams also handle how lab names vary in abbreviation, punctuation and word order ("WBC",
"W.B.C.", "White blood cell count"), where word-level matching fails.

TfidfVectorizer L2-normalises its rows by default, so a dot product is the cosine similarity.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_char_tfidf(corpus, ngram_range=(3, 5), min_df=1):
    """`char_wb` keeps n-grams inside word boundaries, so no n-gram spans the gap between two words.
    Lowercasing makes the match case-insensitive."""
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df,
                        sublinear_tf=True, lowercase=True)
    v.fit(list(corpus))
    return v


def nearest_and_self(queries, targets, self_idx, vec, chunk: int = 5000):
    """For each query: its best-matching target, that score, and the score of its own assigned target.

    queries  : list[str]
    targets  : list[str]                (e.g. the 679 official LOINC names)
    self_idx : (n_queries,) index into `targets` for the assigned target; negative for none
    Returns (best_idx, best_score, self_score), where self_score is NaN if self_idx is negative.

    Queries are processed in chunks so the dense similarity matrix is never built whole. About 270k
    queries against 679 targets would take about 700 MB as float32.
    """
    T = vec.transform(list(targets))
    self_idx = np.asarray(self_idx)
    qs = list(queries)
    best_idx = np.empty(len(qs), dtype=np.int32)
    best_score = np.empty(len(qs), dtype=np.float32)
    self_score = np.empty(len(qs), dtype=np.float32)
    for a in range(0, len(qs), chunk):
        b = min(a + chunk, len(qs))
        S = (vec.transform(qs[a:b]) @ T.T).toarray()      # cosine: both sides are L2-normalised
        bi = S.argmax(axis=1)
        best_idx[a:b] = bi
        best_score[a:b] = S[np.arange(b - a), bi]
        si = self_idx[a:b]
        ok = si >= 0
        ss = np.full(b - a, np.nan, dtype=np.float32)
        ss[ok] = S[np.arange(b - a)[ok], si[ok]]
        self_score[a:b] = ss
    return best_idx, best_score, self_score
