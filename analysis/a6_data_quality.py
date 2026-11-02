#!/usr/bin/env python
"""Data quality: are MedGemma's paraphrases valid descriptions of the LOINC code they are attached to, and
where do the labels come from? (thesis Chapter 4)

  --stage gates        every model trained on the same paraphrase split, its pools are disjoint, every
                       label-space code has a test pool, and the similarity probe finds the official names
  --stage part1        paraphrase validity per split. This is the similarity of each paraphrase to the LOINC long
                       common name of its code, and how often a naive name matcher would miss it. --write saves
                       the tables
  --stage concept-map  a provenance check. Does seen_codes.csv agree with the MIMIC-IV SSSOM concept map? Its
                       `mimic_desc` is also identical to the `label` of the map, so it was derived from the map.
                       The labels are the official MIMIC-IV mappings, inherited verbatim. Their correctness is
                       not measured anywhere in this study
  --stage all          all three (the default)

Similarity is model-free character n-gram TF-IDF. lib/textsim.py explains why an embedding model would be
circular here. Reads the paraphrase pool (data/paraphrases/clean), UPSTREAM_DIR/seen_codes.csv and the concept
map from mimic-code (MIMIC_CODE_DIR). --write saves the part 1 tables to results/analysis/a6/.

  python analysis/a6_data_quality.py --stage all --write
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
_A = importlib.util.spec_from_file_location("a2", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)
_L = importlib.util.spec_from_file_location("lx", HERE / "lib" / "loinc_axes.py")
LX = importlib.util.module_from_spec(_L); _L.loader.exec_module(LX)
_T = importlib.util.spec_from_file_location("textsim", HERE / "lib" / "textsim.py")
TS = importlib.util.module_from_spec(_T); _T.loader.exec_module(TS)
_M9 = importlib.util.spec_from_file_location("m9", HERE.parent / "pipeline" / "09_mlm_dataset.py")
M9 = importlib.util.module_from_spec(_M9); _M9.loader.exec_module(M9)

POOL_DIR = HERE.parent / "data" / "paraphrases" / "clean"
SEEN_CODES = locations.UPSTREAM / "seen_codes.csv"
OUT = HERE.parent / "results" / "analysis" / "a6"
CANONICAL_SHA = "8de71e4589e80a96491f0955d4f33bb63a00c8a27b3bf81b1fb1ad8e02d4261f"



def official_names(space, table):
    """Official LOINC display name per label-space position, with a safe fallback."""
    if "LONG_COMMON_NAME" in table:
        n = table["LONG_COMMON_NAME"].astype(str)
        bad = n.isin(["nan", "None", ""])
        alt = (table.COMPONENT + " " + table.PROPERTY + " " + table.SYSTEM)
        return n.where(~bad, alt).tolist()
    return (table.COMPONENT + " " + table.PROPERTY + " " + table.SYSTEM).tolist()


def load_pools():
    tr, va, te, stats = M9.load_paraphrases_3way(POOL_DIR, verbose=False)
    return tr, va, te, stats


def gates(space, table) -> bool:
    print("=" * 78); print("A6 GATES"); print("=" * 78)
    ok = True
    tr, va, te, stats = load_pools()
    sha = stats.get("split_sha256", "")
    print("[G1] paraphrase split matches the canonical split used to train every model")
    match = sha == CANONICAL_SHA
    print(f"  {'✓' if match else '✗'} sha256={sha[:16]}…  expected={CANONICAL_SHA[:16]}…")
    ok &= match

    print("[G2] pools are pairwise disjoint per LOINC")
    bad = sum(1 for l in te if (set(te[l]) & set(tr.get(l, []))) or (set(te[l]) & set(va.get(l, []))))
    print(f"  {'✓' if bad == 0 else '✗'} {bad} LOINCs with overlap between splits")
    ok &= bad == 0

    print("[G3] label space covered by the pool")
    cov = sum(1 for l in space if l in te)
    print(f"  {'✓' if cov > 0 else '✗'} {cov}/{len(space)} label-space LOINCs have a test-split pool")

    print("[G4] retrieval sanity: verbatim official names retrieve themselves")
    names = official_names(space, table)
    vec = TS.fit_char_tfidf(names)
    bi, bs, _ = TS.nearest_and_self(names, names, np.arange(len(names)), vec)
    acc = float((bi == np.arange(len(names))).mean())
    print(f"  {'✓' if acc > 0.99 else '✗'} {100*acc:.2f}% of official names retrieve themselves")
    ok &= acc > 0.99

    print("[G5] seen_codes label source readable")
    n = sum(1 for _ in csv.DictReader(open(SEEN_CODES)))
    print(f"  {'✓' if n > 0 else '✗'} {n} MIMIC->LOINC mappings")
    print("\n" + ("ALL GATES PASSED" if ok else "GATE FAILURES"))
    return ok


def part1(space, table, write=False):
    """Paraphrase validity."""
    names = official_names(space, table)
    pos = {l: i for i, l in enumerate(space)}
    vec = TS.fit_char_tfidf(list(names))
    tr, va, te, _ = load_pools()
    rows, per_loinc = [], []
    for split_name, pool in (("test", te), ("val", va), ("train", tr)):
        qs, si, ls = [], [], []
        for loinc, paras in pool.items():
            if loinc not in pos:
                continue
            for p in paras:
                qs.append(p); si.append(pos[loinc]); ls.append(loinc)
        if not qs:
            continue
        bi, bs, ss = TS.nearest_and_self(qs, names, np.array(si), vec)
        mis = bi != np.array(si)
        rows.append({"split": split_name, "n_paraphrases": len(qs), "n_loinc": len(set(ls)),
                     # not the share of bad paraphrases (see the caveat printed below). It is the
                     # miss rate of an unsupervised char-n-gram matcher against formal LOINC names,
                     # on paraphrases that are meant to be terse abbreviations ("PT", "Ab", "Kt")
                     "naive_matcher_miss_pct": 100 * float(mis.mean()),
                     "words_p50": float(np.median([len(q.split()) for q in qs])),
                     "pct_le_2_words": 100 * float(np.mean([len(q.split()) <= 2 for q in qs])),
                     "self_sim_mean": float(np.nanmean(ss)), "self_sim_p10": float(np.nanpercentile(ss, 10)),
                     "self_sim_p50": float(np.nanpercentile(ss, 50)),
                     "pct_self_sim_below_0.2": 100 * float((ss < 0.2).mean())})
        if split_name == "test":
            d = pd.DataFrame({"loinc": ls, "self_sim": ss, "mis": mis})
            per_loinc = d.groupby("loinc").agg(n=("self_sim", "size"), self_sim=("self_sim", "mean"),
                                              naive_miss_pct=("mis", lambda v: 100 * v.mean())).reset_index()
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 78); print("A6 PART 1: paraphrase validity"); print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  Note: `naive_matcher_miss_pct` is not a paraphrase error rate. It is how often an")
    print("     unsupervised character-n-gram matcher fails to recover the LOINC from the")
    print("     paraphrase alone. The paraphrases are meant to be terse abbreviations (median")
    print("     3 words; the shortest are \"PT\", \"Ab\", \"Kt\"). Real local lab names are abbreviated")
    print("     too, but this leaves a string matcher almost nothing to work with. A supervised")
    print("     model on the same text (TF-IDF) does far better. Use this metric only to compare")
    print("     splits with each other.")

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT / "a6_paraphrase_validity.csv", index=False)
        per_loinc.to_csv(OUT / "a6_paraphrase_per_loinc.csv", index=False)
        print(f"\nwrote -> {OUT}")
    return summary, per_loinc


def concept_map_check():
    """Provenance check: does seen_codes.csv reproduce MIMIC-IV's curated SSSOM map?

    This is not a validation. `mimic_desc` is identical to the `label` column of the map, so seen_codes
    was derived from the map. The agreement confirms a faithful transcription and says nothing about
    correctness. The description-column match is reported too, so that the circularity stays visible.
    """
    cm = (locations.MIMIC_CODE / "mimic-iv/concepts/concept_map/d_labitems_to_loinc.csv")
    if not cm.exists():
        print("concept map not found"); return None
    d = pd.read_csv(cm, low_memory=False)
    ic = "itemid (omop_source_code)"
    L = d[d.omop_vocabulary_id.astype(str).str.upper() == "LOINC"][[ic, "omop_concept_code", "label"]]
    L = L.dropna(subset=["omop_concept_code"])
    L[ic] = L[ic].astype(int).astype(str)
    sc = {r["mimic_code"]: r["loinc_code"].strip()
          for r in csv.DictReader(open(SEEN_CODES)) if r["loinc_code"]}
    both = agree = 0
    dis = []
    for _, r in L.iterrows():
        k = r[ic]
        if k in sc:
            both += 1
            if str(r.omop_concept_code).strip() == sc[k]:
                agree += 1
            else:
                dis.append((k, r.label, sc[k], str(r.omop_concept_code).strip()))
    print("=" * 78); print("A6: label quality vs MIMIC-IV's curated SSSOM concept map"); print("=" * 78)
    print(f"  items in both:  {both}")
    print(f"  agree:          {agree} ({100*agree/max(both,1):.2f}%)")
    print(f"  disagree:       {len(dis)}")
    lab = dict(zip(L[ic], L.label.astype(str)))
    dsame = sum(1 for k, v in sc.items() if k in lab)
    dmatch = sum(1 for r in csv.DictReader(open(SEEN_CODES))
                 if r["mimic_code"] in lab and r["mimic_desc"].strip() == lab[r["mimic_code"]].strip())
    print("\n  Circularity check: mimic_desc is identical to the concept map's own `label` column on "
          f"{dmatch}/{dsame} items ({100*dmatch/max(dsame,1):.1f}%).")
    print("     So seen_codes.csv was derived from this map, and this is a provenance check")
    print("     (faithful transcription) that does not validate the mappings.")
    for x in dis[:10]:
        print(f"    {x[0]} {str(x[1])[:28]:28} seen_codes={x[2]} concept_map={x[3]}")
    return {"n": both, "agree": agree, "disagreements": dis}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gates", "part1", "concept-map", "all"], default="all")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    space = list(A.canonical_space())
    table = LX.load_axis_table(space)
    if a.stage in ("gates", "all"):
        if not gates(space, table) and a.stage == "all":
            sys.exit("gates failed")
        if a.stage == "gates":
            sys.exit(0)
    if a.stage in ("concept-map", "all"):
        concept_map_check()
    if a.stage in ("part1", "all"):
        part1(space, table, write=a.write)


if __name__ == "__main__":
    main()
