#!/usr/bin/env python
"""Off-the-shelf baselines. Open models, prompted few-shot (or zero-shot), name the LOINC code of each
sampled test observation (thesis Chapters 5 and 6, Appendix E). They are a comparator on the same
data from models that were not fine-tuned on it. They sit beside the ~52-53% complete match that
Yu et al. (2026) report for commercial LLMs.

These models were not trained on the MedGemma paraphrases, so the paraphrases cannot have leaked
into them. Llama-3.1-8B is also the base model of the QLoRA fine-tune, so its baseline against the
fine-tuned model shows what fine-tuning adds.

The outcome has four classes: correct / valid_wrong / not_a_loinc / unparseable (lib/oneshot.py).
The trained models pick from a fixed output head (the 1,623 lab items for the deep models, 679 LOINC
codes for TF-IDF), so unlike a generating model they cannot name a code that does not exist.

The models run locally from the Hugging Face cache. --stage run writes one response per sampled
observation to results/analysis/b4/raw/<tag>.jsonl. --stage score writes b4_summary.csv and
b4_per_observation.csv. The per-observation files carry patient IDs and stay in results/, which git
ignores. Stdout is aggregate only.

Run each model on each test arm, then score:
  python analysis/b4_oneshot_baselines.py --model medgemma-4b --condition para_xsite --test-arm pert --write
  python analysis/b4_oneshot_baselines.py --stage score --write
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_A = importlib.util.spec_from_file_location("a2", HERE / "a2_calibration.py")
A = importlib.util.module_from_spec(_A); _A.loader.exec_module(A)
_O = importlib.util.spec_from_file_location("oneshot", HERE / "lib" / "oneshot.py")
O = importlib.util.module_from_spec(_O); _O.loader.exec_module(O)
_L = importlib.util.spec_from_file_location("lx", HERE / "lib" / "loinc_axes.py")
LX = importlib.util.module_from_spec(_L); _L.loader.exec_module(LX)

BASE = A.EVAL.parent
TEST_DIR = BASE / "tfidf" / "dataset" / "test" / "samples"
TRAIN_DIR = BASE / "tfidf" / "dataset" / "train" / "samples"
OUT = HERE.parent / "results" / "analysis" / "b4"
RAW = OUT / "raw"

sys.path.insert(0, str(HERE.parent / "pipeline"))
from conditions import dataset_suffix, has_description, representation, resolve_test_arm  # noqa: E402


def dataset_dirs(condition: str = "para", test_arm: str | None = None):
    """(test_samples_dir, train_samples_dir) for one representation and arm.

    These models read the TF-IDF prompt files, so a representation is a directory and needs no
    rebuild. pipeline/06_tfidf_dataset.py writes `dataset`, `dataset_v2`, `dataset_v3__clean` and
    `dataset_v3__pert` side by side. Under v3 the prompts carry standardised units, 4 s.f.
    scientific values and randomised unit notation. On the pert arm they also carry perturbed values
    and ranges.

    The exemplars come from the training split. They must be in the same representation as the
    query, or the demonstration teaches a format that the query does not use. Both directories are
    returned together so that they cannot be mismatched by mistake.
    """
    suffix = dataset_suffix(condition)
    if representation(condition) == "v3":
        arm, _ = resolve_test_arm(condition, "", test_arm, "b4_oneshot_baselines")
        suffix += arm
    root = BASE / "tfidf" / f"dataset{suffix}"
    if not root.exists():
        raise SystemExit(f"b4: no prompt dataset at {root} (build it with 06_tfidf_dataset first)")
    return root / "test" / "samples", root / "train" / "samples"

N_PATIENTS, N_OBS, SEED = 500, 5000, 0

# (Hugging Face repo, prompt form, load in 4-bit). `chat` prompts through the tokenizer's chat
# template. `plain` prompts with the same content as plain text, completion-style. Meditron has no
# chat template, and forcing one on it degrades its output badly. An optional fourth field names a
# donor model. A `chat` model whose tokenizer has no chat template then uses the donor's template.
MODELS = {
    "llama31-8b":   ("meta-llama/Llama-3.1-8B-Instruct", "chat",  False),
    "gemma3-12b":   ("google/gemma-3-12b-it",            "chat",  True),   # 4-bit: 23 GB fp16
    # OpenBioLLM-8B is instruction-tuned, but its tokenizer ships no chat template, so it is prompted
    # as plain text.
    "openbio-8b":   ("aaditya/Llama3-OpenBioLLM-8B",     "plain", False),
    "meditron-7b":  ("epfl-llm/meditron-7b",             "plain", False),
    "medgemma-4b":  ("google/medgemma-4b-it",            "chat",  False),
}


def strip_reference_range(prompt: str) -> str:
    """Remove the `ReferenceRange:` line from a prompt.

    These baselines read the same prompt files as TF-IDF but do not pass through
    utils.extract_structured_features_single, so `exclude_fields` does not reach them. For the norr
    arms they drop the line here, or they would still see a field that the trained models do not get.

    Only the line is removed, and the prompt keeps its line-by-line format. The baselines therefore
    get the same fields as TF-IDF, in the prompt format they were designed for.
    """
    return "\n".join(l for l in prompt.split("\n")
                      if not re.match(r"\s*Reference\s*Range\s*:", l, re.IGNORECASE))


#: The observation shapes that the exemplars demonstrate, most common first. Counted with prompt_shape
#: over the first 400 sorted v3 pert training files (157,281 observations): numeric+range 82.1%,
#: qualitative 11.4%, numeric 3.6%, 97.2% together. The fourth shape, `no-value` at 2.8%, is left out.
#: Three exemplars already cover the shapes a model needs to see, and each extra one makes every query
#: longer.
EXEMPLAR_SHAPES = ("numeric+range", "qualitative", "numeric")


def prompt_shape(prompt: str) -> str:
    """Which of the four observation shapes this prompt is. Classified on the text the model will
    see, so a `--refrange norr` prompt is described by what is left after stripping."""
    parts = dict(s.split(":", 1) for s in prompt.split("\n") if ":" in s)
    if "Value" not in parts:
        return "no-value"
    if not any(c.isdigit() for c in parts["Value"]):
        return "qualitative"
    return "numeric+range" if "ReferenceRange" in parts else "numeric"


def pick_exemplars(train_dir, refrange="full", shapes=EXEMPLAR_SHAPES, max_files=50):
    """One training exemplar per target shape, in `shapes` order. The first match over the sorted
    files wins, so the choice is deterministic.

    A shape with no match in the scanned files is skipped. It is not replaced by a repeat of another
    shape, because that would hide the missing shape. The caller prints the count.
    """
    found = {}
    for f in sorted(train_dir.glob("*.jsonl"))[:max_files]:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                pr = d["prompt"] if refrange == "full" else strip_reference_range(d["prompt"])
                sh = prompt_shape(pr)
                if sh in shapes and sh not in found:
                    found[sh] = (pr, d["label"])
        if len(found) == len(shapes):
            break
    return [found[s] for s in shapes if s in found], [s for s in shapes if s in found]


def load_sample(refrange: str = "full", test_dir=None, train_dir=None):
    """Patient-clustered sample of test observations, plus the training exemplars.

    `refrange="norr"` strips the reference range from every prompt, matching the norr model arms.
    `test_dir`/`train_dir` default to the v1 (`para`) directories."""
    TEST_DIR_, TRAIN_DIR_ = test_dir or TEST_DIR, train_dir or TRAIN_DIR
    per = {}
    for f in sorted(TEST_DIR_.glob("*.jsonl")):
        pid = f.stem
        recs = []
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    pr = d["prompt"] if refrange == "full" else strip_reference_range(d["prompt"])
                    recs.append({"patient": pid, "prompt": pr, "gold": d["label"]})
        if recs:
            per[pid] = recs
    sample = O.sample_observations(per, N_PATIENTS, N_OBS, seed=SEED)
    # Few-shot, with one exemplar per observation shape, because a single exemplar shows only one
    # shape. Most queries are numeric with a reference range. The exemplars are stripped like the
    # queries, or they would show a field that the queries do not have.
    ex, shapes_found = pick_exemplars(TRAIN_DIR_, refrange)
    print(f"  exemplars: {len(ex)} ({', '.join(shapes_found)})")
    return sample, ex


def load_sample_relma(strategy="s1", aseed=0, refrange="full", test_dir=None, train_dir=None):
    """The same sampled observations, with the Description swapped for the real RELMA name of that
    (patient, LOINC). The pairing holds by construction. The observations and values are the same,
    and only the name differs.

    This tests the interpretation of the paraphrase results on real names. The real names are often
    more cryptic than the paraphrases ("MBI_GPT/ALT_U/L" vs "POC alanine aminotransferase"), so the
    test can refute that interpretation as well as confirm it.
    """
    _D = importlib.util.spec_from_file_location("dumps", HERE / "lib" / "dumps.py")
    D = importlib.util.module_from_spec(_D); _D.loader.exec_module(D)
    sample, ex = load_sample(refrange, test_dir, train_dir)
    sc = D.load_sidecar("mlm", strategy, aseed)[["patient_uuid", "loinc", "relma_name"]]
    m = {(r.patient_uuid, r.loinc): r.relma_name for r in sc.itertuples()}
    out = []
    for rec in sample:
        nm = m.get((rec["patient"], rec["gold"]))
        if nm is None or not str(nm).strip():
            continue                       # no RELMA name for this code: dropped. The summary's n shows it.
        out.append({**rec, "prompt": O.swap_description(rec["prompt"], str(nm))})
    return out, ex


def all_loinc_codes() -> set:
    d = pd.read_csv(LX.LOINC_CSV, low_memory=False, usecols=["LOINC_NUM"])
    return set(d.LOINC_NUM.astype(str))


def run_model(key, limit=None, write=False, zero_shot=False, batch_size=1,
              evalset="mimictest", refrange="full", condition="para", test_arm=None):
    """Generate one greedy response per sampled observation into RAW/<tag>.jsonl.

    batch_size defaults to 1 for reproducibility. Measured on medgemma-4b over 200 observations:
      same config, two runs      -> 200/200 identical (the pipeline is deterministic)
      eager vs dynamo-compiled   -> 152/200 identical
      unpadded vs left-padded batch -> 69/96 identical
    Batching and compilation are both valid execution paths. They differ only in floating-point
    reduction order. But the greedy logits of these models are near-ties, and much of their output is
    not a real LOINC. So tiny numerical differences flip the argmax on ~25-30% of observations.
    batch_size=1 with dynamo disabled is the padding-free, reproducible path, at ~0.4 s/obs. This
    sensitivity is also a sign that the models are guessing.
    """
    import torch
    import torch._dynamo
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # Every prompt has a different length, so torch.compile recompiles per shape until it hits
    # `cache_size_limit` and raises RecompileLimitExceeded. Dynamo is disabled because compiling a
    # 16-token greedy generation gains nothing, and recompiling slows every observation.
    torch._dynamo.config.disable = True
    torch._dynamo.config.cache_size_limit = 1 << 20

    cfg = MODELS[key]
    repo, style, want_4bit = cfg[0], cfg[1], cfg[2]
    template_from = cfg[3] if len(cfg) > 3 else None
    _td, _trd = dataset_dirs(condition, test_arm)
    print(f"  prompts: {_td.parent.parent.name}  (condition={condition} test_arm={test_arm})")
    sample, ex = (load_sample_relma(refrange=refrange, test_dir=_td, train_dir=_trd)
                  if evalset == "relma" else load_sample(refrange, _td, _trd))
    if limit:
        sample = sample[:limit]
    # The representation goes in the tag, or a v3 run would overwrite the v1 result of the same model.
    # `score()` labels every row by this stem. v1 adds nothing to the tag.
    _rep_tag = "" if representation(condition) == "v1" else f"__{representation(condition)}{test_arm or ''}"
    # The exemplar count goes in the tag too, so runs with different prompts never share a file name.
    # Zero-shot uses no exemplars, so it carries no `__fs`.
    _fs_tag = "" if zero_shot else f"__fs{len(ex)}"
    tag = f"{key}{'__relma' if evalset == 'relma' else ''}"\
          f"{'__norr' if refrange == 'norr' else ''}{_rep_tag}{_fs_tag}"\
          f"{'__zeroshot' if zero_shot else ''}{'__smoke' if limit else ''}"
    out_path = RAW / f"{tag}.jsonl"
    if out_path.exists() and not limit:
        print(f"  SKIP {tag} (exists)"); return out_path
    RAW.mkdir(parents=True, exist_ok=True)

    print(f"  loading {repo} ({'4-bit' if want_4bit else 'bf16'}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Batched decoder-only generation needs left padding. With right padding the model continues
    # from pad tokens and emits garbage. Checked against the single-item results.
    tok.padding_side = "left"
    if style == "chat" and getattr(tok, "chat_template", None) is None:
        if not template_from:
            raise SystemExit(f"{key}: chat style requested but no chat_template and no donor given")
        donor = AutoTokenizer.from_pretrained(template_from)
        tok.chat_template = donor.chat_template
        print(f"    (borrowed chat template from {template_from})")
    kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if want_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    model.eval()

    def render(rec):
        if style == "chat":
            msgs = O.build_fewshot_messages(rec["prompt"], ex)
            # zero-shot keeps the system turn and the query and drops every exemplar turn
            if zero_shot:
                msgs = [msgs[0], msgs[-1]]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return (O.build_plain_fewshot(rec["prompt"], ex) if not zero_shot
                else f"{O.SYSTEM}\n\n{O.build_user_turn(rec['prompt'])}")

    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as fh:
        for a0 in range(0, len(sample), batch_size):
            chunk = sample[a0:a0 + batch_size]
            texts = [render(r) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for rec, row in zip(chunk, new):
                resp = tok.decode(row, skip_special_tokens=True)
                fh.write(json.dumps({"patient": rec["patient"], "gold": rec["gold"],
                                     "response": resp}) + "\n")
            fh.flush()   # so `wc -l` shows the progress
            n_done += len(chunk)
            if n_done % (batch_size * 20) < batch_size:
                el = time.time() - t0
                print(f"    {n_done}/{len(sample)}  {el/n_done:.3f}s/obs  "
                      f"eta {(len(sample)-n_done)*el/n_done/60:.0f}m", flush=True)
    print(f"  {tag}: {n_done} responses in {(time.time()-t0)/60:.1f}m -> {out_path}")
    del model
    torch.cuda.empty_cache()
    return out_path


def score(write=False):
    allc = all_loinc_codes()
    space = set(A.canonical_space())
    rows, per_obs = [], []
    for f in sorted(RAW.glob("*.jsonl")):
        if f.stem.endswith("__smoke"):
            continue
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        outs = []
        for r in recs:
            pred = O.extract_loinc(r["response"])
            cls = O.classify(pred, r["gold"], allc)
            outs.append(cls)
            per_obs.append({"model": f.stem, "patient": r["patient"], "gold": r["gold"],
                            "pred": pred, "outcome": cls,
                            "pred_in_our_679": bool(pred in space) if pred else False})
        s = pd.Series(outs)
        n = len(s)
        rows.append({"model": f.stem, "n": n,
                     **{k: 100 * float((s == k).mean()) for k in
                        ("correct", "valid_wrong", "not_a_loinc", "unparseable")}})
    d = pd.DataFrame(rows).sort_values("correct", ascending=False)
    po = pd.DataFrame(per_obs)

    print("=" * 78); print("B4: one-shot open-model baselines (% of sampled observations)")
    print("=" * 78)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if not po.empty:
        # patient-clustered CI on exact match (lib/calib.bootstrap_clustered)
        _P = importlib.util.spec_from_file_location("calib", HERE / "lib" / "calib.py")
        C = importlib.util.module_from_spec(_P); _P.loader.exec_module(C)
        print("\n  exact-match with patient-clustered 95% CI:")
        for m, g in po.groupby("model"):
            r = C.bootstrap_clustered(g.patient.values,
                                      lambda correct: float(correct.mean()),
                                      n_boot=400, correct=(g.outcome == "correct").values)
            print(f"    {m:22s} {100*r['point']:6.2f}%  [{100*r['lo']:.2f}, {100*r['hi']:.2f}]")
        print("    Yu et al. (2026) report ~52-53% complete match for commercial LLMs.")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT / "b4_summary.csv", index=False)
        po.to_csv(OUT / "b4_per_observation.csv", index=False)
        print(f"\nwrote 2 tables -> {OUT}")
    return d, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "score"], default="run")
    ap.add_argument("--model", default=None, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--evalset", default="mimictest", choices=["mimictest", "relma"])
    ap.add_argument("--refrange", default="full", choices=["full", "norr"],
                    help="'norr' strips the ReferenceRange line from every prompt (and from the "
                         "one-shot exemplar), matching the norr model arms")
    ap.add_argument("--condition", default="para",
                    help="which representation's prompts to read. Default 'para' is the v1 "
                         "directory. 'para_xsite' reads v3 "
                         "(standardised units, 4 s.f. scientific values) and requires --test-arm.")
    ap.add_argument("--test-arm", default=None, choices=["clean", "pert"],
                    help="v3 only, and required there. 'pert' is the shared test condition the "
                         "four fine-tuned models are scored on. 'clean' is the control: these "
                         "models never trained on MIMIC, so if clean == pert for them, the "
                         "perturbation does not make the task harder in itself.")
    a = ap.parse_args()
    if a.stage == "score":
        score(write=a.write); return
    if not a.model:
        raise SystemExit("--model required for --stage run")
    _rep = representation(a.condition)
    if _rep == "v3" and a.test_arm is None:
        raise SystemExit("b4: --test-arm is required for a v3 condition ('clean' or 'pert').")
    if _rep != "v3" and a.test_arm is not None:
        raise SystemExit(f"b4: --test-arm is v3-only; --condition {a.condition!r} is {_rep}.")
    if not has_description(a.condition):
        raise SystemExit(f"b4: {a.condition!r} has no description. These models are given the test "
                         "name and asked for a LOINC; there is no nodesc prompt to hand them.")
    run_model(a.model, limit=a.limit, write=a.write, zero_shot=a.zero_shot,
              batch_size=a.batch_size, evalset=a.evalset, refrange=a.refrange,
              condition=a.condition, test_arm=a.test_arm)


if __name__ == "__main__":
    main()
