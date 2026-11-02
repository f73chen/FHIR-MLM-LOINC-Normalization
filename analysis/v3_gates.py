#!/usr/bin/env python
"""
Structural gates for the v3 analyses behind thesis Chapter 6. Each stage checks an assumption the
tables rest on, before any table is trusted. Output is aggregate only.

  --stage codes       every TF-IDF v3 test-split dump joins to a local code for every observation
                      (lib/codes.py), and the joined gold LOINC equals the gold LOINC in the dump. Key
                      sets are identical across seeds and cells where the design says they must be.
                      TF-IDF scores every observation, so A–D share one key set, and each RELMA draw
                      shares one key set across seeds and arms
  --stage structure   every v3 cell of every model present was scored by the best checkpoint of its
                      train arm, on the data of its test arm. Cells on the same test data scored the
                      same positions

CPU only:
  python analysis/v3_gates.py --stage codes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import codes as K  # noqa: E402
from lib import registry as R  # noqa: E402

KEY = K.KEY


def _keys(c) -> pd.DataFrame:
    return pd.read_parquet(c.path / "part-0000.parquet", columns=KEY + ["true_loinc"])


def gate_codes() -> bool:
    ok = True
    print("=" * 78 + "\ncodes gate: TF-IDF local-code join (v3 test split)\n" + "=" * 78)
    lk = K.lookup()
    print(f"lookup: {len(lk):,} test observations, {lk.true_class.nunique():,} local codes")
    families = []
    for cond in R.CONDITIONS:
        families.append((f"{cond} mimictest A–D", [
            R.cell("tfidf", s, condition=cond, train_arm=tr, test_arm=te)
            for tr in ("clean", "pert") for te in ("clean", "pert") for s in R.SEEDS["tfidf"]]))
    for a in R.RELMA_DRAWS:
        ev = R.EVALSET_RELMA.format(a)
        families.append((f"para_xsite {ev} B, D", [
            R.cell("tfidf", s, condition="para_xsite", train_arm=tr, test_arm="pert", evalset=ev)
            for tr in ("clean", "pert") for s in R.SEEDS["tfidf"]]))
    for fam, cs in families:
        t0, ref, n_codes, fam_ok = time.time(), None, None, True
        for c in cs:
            if not R.has(c):
                print(f"  ✗ missing {c.name}"); fam_ok = False; continue
            d = _keys(c)
            try:
                j = K.attach(d)
            except AssertionError as e:
                print(f"  ✗ {c.name}: {e}"); fam_ok = False; continue
            ks = pd.MultiIndex.from_frame(d[KEY])
            if ref is None:
                ref, n_codes = ks, j.true_class.nunique()
            elif not ks.equals(ref):
                print(f"  ✗ {c.name}: key set differs from {cs[0].name}"); fam_ok = False
        ok &= fam_ok
        print(f"  {'✓' if fam_ok else '✗'} {fam}: {len(cs)} dumps, "
              f"{len(ref) if ref is not None else 0:,} observations each, {n_codes} local codes"
              + (", all joined with matching gold LOINC" if fam_ok else "")
              + f" ({time.time() - t0:.0f}s)", flush=True)
    print("\n" + ("codes gate PASSED" if ok else "codes gate FAILED (see ✗ above)"))
    return ok


# -- structure ---------------------------------------------------------------
RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIN_DIR = {"mlm": "12_mlm_train", "bcm": "13_bcm_train", "llama": "14_llama_train"}


def _expected_checkpoint_dir(c) -> str:
    pre = "mlm__singleobs__" if c.scope == "singleobs" else f"{c.model}__"
    return f"{pre}{c.condition}__{c.train_arm}__seed{c.seed}"


def _expected_dataset(c) -> str:
    te = c.test_arm_explicit
    if c.evalset == "mimictest":
        return f"{c.model}__{c.condition}__{te}"
    if c.evalset.startswith("singleobs_trim"):          # singleobs_trim and singleobs_trim_relma_s1_a0
        return f"mlm__{c.evalset}__{c.condition}__{te}"
    if c.evalset.startswith("relma_") or c.evalset == "sliding":
        return f"{c.model}__{c.evalset}__{c.condition}__{te}" if c.evalset.startswith("relma_") \
            else f"{c.model}__sliding__{c.condition}__{te}"
    raise ValueError(f"no expected dataset for evalset {c.evalset!r}")


def gate_structure() -> bool:
    """For every v3 cell of every model present, the cell must have been scored by the best checkpoint
    of its train arm, on the data of its test arm. Cells on the same test data must have scored the
    same positions."""
    import json
    from collections import defaultdict
    print("=" * 78 + "\nstructure gate: every v3 cell\n" + "=" * 78)
    cells, _ = R.scan()
    v3 = [c for c in cells if c.v3 and c.exists()]
    bad, groups, ckpt_by_model_seed = [], defaultdict(set), defaultdict(set)
    recipes = set()
    for c in v3:
        man = json.loads((c.path / "manifest.json").read_text())
        if man.get("run_name") != c.name:
            bad.append(f"{c.name}: manifest run_name {man.get('run_name')!r}")
        if man.get("split", "test") != c.split:
            bad.append(f"{c.name}: split {man.get('split')!r}")
        if c.model == "tfidf":
            if man.get("condition") != c.condition or man.get("seed") != c.seed:
                bad.append(f"{c.name}: condition/seed fields disagree with the name")
            recipes.add(json.dumps(man.get("pinned"), sort_keys=True))
            groups[("tfidf", c.condition, c.evalset, c.test_arm_explicit, c.split)].add(
                ("n_scored", man.get("n_scored")))
            continue
        ck, ds = Path(man["checkpoint"]), Path(man["dataset_path"].rstrip("/"))
        if ck.parent.name != _expected_checkpoint_dir(c):
            bad.append(f"{c.name}: checkpoint {ck.parent.name}, expected {_expected_checkpoint_dir(c)}")
        tm = RESULTS / TRAIN_DIR[c.model] / (f"manifest_{'singleobs_' if c.scope == 'singleobs' else ''}"
                                             f"{c.condition}__{c.train_arm}_seed{c.seed}.json")
        if not tm.exists():
            bad.append(f"{c.name}: no training manifest {tm.name}")
        elif json.loads(tm.read_text())["best_model_checkpoint"] != str(ck):
            bad.append(f"{c.name}: checkpoint is not the training run's best ({ck.name})")
        if ds.name != _expected_dataset(c):
            bad.append(f"{c.name}: dataset {ds.name}, expected {_expected_dataset(c)}")
        ckpt_by_model_seed[(c.model, c.condition, c.scope, c.train_arm, c.seed)].add(str(ck))
        groups[(c.model, str(ds), c.split)].add((man["n_positions"], man["n_rows"],
                                                 man.get("n_truncated_positions")))
    for k, v in ckpt_by_model_seed.items():
        if len(v) != 1:
            bad.append(f"{k}: {len(v)} different checkpoints across this run's cells")
    for k, v in groups.items():
        if len(v) != 1:
            bad.append(f"{k}: cells on the same test data scored different positions {sorted(v)}")
    if len(recipes) != 1:
        bad.append(f"TF-IDF: {len(recipes)} different pinned recipes across its v3 cells")
    by_model = defaultdict(int)
    for c in v3:
        by_model[c.model] += 1
    print("  cells checked: " + ", ".join(f"{R.LABEL[m]} {n}" for m, n in sorted(by_model.items())))
    print(f"  distinct test-data groups: {len(groups)}; checkpoints: {len(ckpt_by_model_seed)} runs")
    # the positions each model scored per test-data group: the facts the matched sets rest on
    for k in sorted(groups, key=str):
        if k[0] != "tfidf" and k[2] == "test":
            print(f"    {k[0]:5s} {Path(k[1]).name:42s} positions {sorted(groups[k])[0][0]:>9,}")
    for b in bad:
        print(f"  ✗ {b}")
    ok = not bad
    print("\n" + ("structure gate PASSED" if ok else f"structure gate FAILED: {len(bad)} violation(s)"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["codes", "structure"], required=True)
    a = ap.parse_args()
    ok = {"codes": gate_codes, "structure": gate_structure}[a.stage]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
