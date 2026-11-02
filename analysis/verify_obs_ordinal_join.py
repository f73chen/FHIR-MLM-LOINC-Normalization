"""
Gate for the cross-model comparisons (thesis Chapter 5). It checks whether FHIR-MLM and BCM/Llama list
a patient's lab observations in the same order.

Paired cross-model tests (McNemar, paired bootstrap) compare the models on the same observation. A
token's `position_idx` cannot be the key, because the three models use different tokenizers and a
position means something different in each. The only candidate join key is
    (patient_uuid, obs_ordinal)
where obs_ordinal is the index of the lab observation in the patient's chronological order. That key is
valid only if every pipeline emits the lab observations in the same order. Otherwise each model's
observations would first have to be remapped to the chronological order.

Method (both sides reduce to the same 1623-class index space, so they compare directly):
  * MLM   : class = abs_token - LAB_TOKEN_MIN, for input_ids in [LAB_TOKEN_MIN, LAB_TOKEN_MAX], in order.
            (LAB_TOKEN_MIN = LAB_TOKEN_MIN_RAW + TEXT_VOCAB_SIZE = 4429 + 30522 = 34951.)
  * BCM   : class = labels[i] for labels[i] != -100, in order (the labels are class indices).
  * Llama : same as BCM.
  The models truncate at different points (FHIR-MLM's window holds more lab observations than BCM's),
  so the test is a prefix test. Over the LOINC-mappable subsequence, the shorter sequence must equal the
  prefix of the longer one. BCM and Llama label only LOINC-mappable positions, while MLM's input carries
  every lab token.

Reads the datasets in WORK_DIR/datasets and prints aggregates only (counts, rates, where the orders
first diverge). It prints no patient records and no uuids.

The datasets were written with the `datasets` library of the deep-models environment. The older version
in the data-analysis environment cannot read that format, so the script runs in the deep-models
environment:

  python analysis/verify_obs_ordinal_join.py --condition para --n-patients 200
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET_BASE = str(locations.WORK / "datasets")
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")

TEXT_VOCAB_SIZE = 30522
LAB_TOKEN_MIN = 34951          # 4429 + 30522
LAB_TOKEN_MAX = 36573          # 6051 + 30522


def mlm_class_seq(input_ids, attention_mask):
    """Ordered class indices of every lab token the MLM sees (attended positions only)."""
    ids = np.asarray(input_ids)
    am = np.asarray(attention_mask)
    sel = (am == 1) & (ids >= LAB_TOKEN_MIN) & (ids <= LAB_TOKEN_MAX)
    return (ids[sel] - LAB_TOKEN_MIN).tolist()


def labelled_class_seq(labels):
    """Ordered class indices at labeled positions (BCM/Llama: labels are class indices, -100 = ignore)."""
    lab = np.asarray(labels)
    return lab[lab != -100].tolist()


def prefix_report(a, b):
    """Compare two ordered class sequences as a prefix test.
    Returns (n_compared, is_prefix, first_divergence_idx or None)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0, None, None
    for i in range(n):
        if a[i] != b[i]:
            return n, False, i
    return n, True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=["para", "nodesc"], default="para")
    ap.add_argument("--n-patients", type=int, default=200)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import datasets
    from fhir_data_utils import build_lab_code_mappings

    _, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    mappable = set(class_to_loinc)          # ~828 LOINC-mappable classes
    print(f"class->LOINC mappable classes: {len(mappable)}")

    ds = {}
    for fam in ("mlm", "bcm", "llama"):
        p = f"{DATASET_BASE}/{fam}__{args.condition}"
        ds[fam] = datasets.load_from_disk(p)[args.split]
        print(f"  {fam:6s} {args.split}: {len(ds[fam]):,} rows")

    # uuid-to-row index for each family
    idx = {fam: {u: i for i, u in enumerate(ds[fam]["uuid"])} for fam in ds}
    common = set(idx["mlm"]) & set(idx["bcm"]) & set(idx["llama"])
    print(f"\ncommon uuids across all three: {len(common):,}"
          f"  (mlm {len(idx['mlm']):,} / bcm {len(idx['bcm']):,} / llama {len(idx['llama']):,})")

    sample = sorted(common)[: args.n_patients]
    print(f"comparing {len(sample)} patients\n")

    stats = Counter()
    div_positions = []
    len_mlm, len_bcm, len_llama = [], [], []

    for u in sample:
        r_mlm = ds["mlm"][idx["mlm"][u]]
        r_bcm = ds["bcm"][idx["bcm"][u]]
        r_lla = ds["llama"][idx["llama"][u]]

        s_mlm_all = mlm_class_seq(r_mlm["input_ids"], r_mlm["attention_mask"])
        s_mlm = [c for c in s_mlm_all if c in mappable]     # BCM/Llama label only mappable positions
        s_bcm = labelled_class_seq(r_bcm["labels"])
        s_lla = labelled_class_seq(r_lla["labels"])

        len_mlm.append(len(s_mlm)); len_bcm.append(len(s_bcm)); len_llama.append(len(s_lla))

        # BCM vs Llama: same pipeline, expected identical
        n, ok, d = prefix_report(s_bcm, s_lla)
        stats["bcm_llama_compared"] += 1
        stats["bcm_llama_prefix_ok"] += bool(ok)
        if ok is False:
            stats["bcm_llama_diverged"] += 1

        # MLM vs BCM: the question this gate answers
        n, ok, d = prefix_report(s_mlm, s_bcm)
        if n == 0:
            stats["mlm_bcm_empty"] += 1
            continue
        stats["mlm_bcm_compared"] += 1
        if ok:
            stats["mlm_bcm_prefix_ok"] += 1
        else:
            stats["mlm_bcm_diverged"] += 1
            div_positions.append(d)

    print("=" * 68)
    print("RESULT")
    print("=" * 68)
    c = stats["mlm_bcm_compared"]
    ok = stats["mlm_bcm_prefix_ok"]
    print(f"BCM vs Llama : {stats['bcm_llama_prefix_ok']}/{stats['bcm_llama_compared']} prefix-match"
          f"  (diverged {stats['bcm_llama_diverged']})")
    print(f"MLM vs BCM   : {ok}/{c} prefix-match ({100.0*ok/c if c else 0:.1f}%)"
          f"  (diverged {stats['mlm_bcm_diverged']}, empty {stats['mlm_bcm_empty']})")
    if div_positions:
        dp = np.array(div_positions)
        print(f"  first divergence idx: min={dp.min()} p50={int(np.percentile(dp,50))} max={dp.max()}")
        print(f"  diverged at index 0: {(dp == 0).sum()} / {len(dp)}")
    print("\nlabelled-obs per patient (mappable): "
          f"mlm mean={np.mean(len_mlm):.1f}  bcm mean={np.mean(len_bcm):.1f}  llama mean={np.mean(len_llama):.1f}")

    print("\nVERDICT:")
    if c and ok == c:
        print("  ✓ obs_ordinal is a valid cross-model join key (orders agree; shorter = prefix of longer).")
    elif c and ok / c > 0.9:
        print(f"  mostly agrees ({100.0*ok/c:.1f}%). Inspect the divergent minority before relying on it.")
    else:
        print("  ✗ orders disagree, so obs_ordinal is not valid as-is. A native-to-chronological remap is required.")


if __name__ == "__main__":
    main()
