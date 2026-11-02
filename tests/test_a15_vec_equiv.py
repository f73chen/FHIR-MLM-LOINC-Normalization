"""Checks score_vec against score_single in analysis/a15_field_identifiability.py, on the real cached
data. score_vec is the vectorised version that the analysis uses, and score_single is the plain-Python
reference. Both must agree on every statistic for every key family."""
import sys, importlib.util
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
for p in (str(locations.COMMON), str(locations.THIRD_PARTY), str(locations.TOKENIZER)):
    sys.path.insert(0, p)
spec = importlib.util.spec_from_file_location("a15",str(locations.REPO / "analysis/a15_field_identifiability.py"))
a15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a15)
from fhir_data_utils import build_lab_code_mappings
m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
arr = np.array([c2l.get(i, f"?{i}") for i in range(1623)], dtype=object)
nb = a15.load_bins()

tr_df = a15.load_cache("train"); te_df = a15.load_cache("test")
# keep the reference path tractable
tr_df = tr_df.head(400000).reset_index(drop=True)
te_df = te_df.head(120000).reset_index(drop=True)

def test_a15_vec_equals_reference():
    print(f"cross-check on train {len(tr_df):,} / test {len(te_df):,}")

    g_edges = a15.build_global_percentile_edges(tr_df["val"].dropna().to_numpy())
    ctr, ptr = a15.components(tr_df, nb, g_edges)
    cte, pte = a15.components(te_df, nb, g_edges)
    ktr, kte = a15.joint_codes(ctr, cte)
    yc, yu = pd.factorize(pd.Series(np.concatenate([arr[tr_df.class_idx.to_numpy()], arr[te_df.class_idx.to_numpy()]])))
    ytr = yc[:len(tr_df)].astype("int32"); yte = yc[len(tr_df):].astype("int32")
    prior = np.array([c for c,_ in Counter(ytr).most_common(max(a15.TOP_KS))], dtype="int32")

    # reference path: build the same keys as tuples, honouring the same presence rules
    def ref_keys(d, present, groups, n):
        """Reference key: tuple of the same component columns, None where the field is absent."""
        cols, mask = [], np.ones(n, bool)
        for g in groups:
            gc, pk = a15.GROUPS[g]
            cols += [c for c in gc if c not in cols]
            mask = mask & present[pk]
        return cols, mask

    FAMILIES = [["unit"],["anyunit"],["prio"],["vraw"],["vraw","unit"],["vstr"],
                ["rr_nou"],["rr"],["rr","unit"],["rr","unit","vraw"],
                ["rr3sf"],["rr2sf"],["rr1sf"],["rrmag"],["rrmag_nou"],["rrunit"],
                ["vpct"],["vpct","unit"],["vglob"],["vglob","unit"],["vmag"],["hasbins"],["hasbins","unit"]]

    fails = []
    for fam in FAMILIES:
        fast = a15.score_vec(ktr, ptr, ytr, kte, pte, yte, fam, prior)
        # reference: build tuple keys from the same integer codes, so only the algorithm differs
        cols, mtr = ref_keys(None, ptr, fam, len(ytr))
        _,    mte = ref_keys(None, pte, fam, len(yte))
        tk = [tuple(ktr[c][i] for c in cols) if mtr[i] else None for i in range(len(ytr))]
        ek = [tuple(kte[c][i] for c in cols) if mte[i] else None for i in range(len(yte))]
        slow = a15.score_single(tk, list(ytr), ek, list(yte), list(prior), tie="label")
        diffs = []
        for k in ("n_keys","unique_key_pct","test_coverage_pct","H_label_given_key_bits",
                  "top1_covered","top1_overall","top5_overall","top10_overall"):
            a, b = fast[k], slow[k]
            same = (isinstance(a,float) and isinstance(b,float) and a != a and b != b) or abs(a-b) <= 1e-9
            if not same: diffs.append(f"{k}: vec={a} ref={b}")
        tag = "+".join(fam)
        print(("  PASS  " if not diffs else "  FAIL  ") + tag + ("" if not diffs else "  " + "; ".join(diffs)))
        if diffs: fails.append(tag)
    print()
    print("ALL FAMILIES AGREE" if not fails else f"{len(fails)} MISMATCH: {fails}")
    if fails:
        raise AssertionError(f"{len(fails)} failures: {fails}")



if __name__ == "__main__":
    test_a15_vec_equals_reference()
