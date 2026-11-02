"""Evaluate a fine-tuned deep model (FHIR-MLM, BCM or Llama) and write one row per scored position
(thesis Chapters 5 and 6). Every analysis joins on this dump, so inference runs once. No training.

The same code scores every model, condition and evaluation set. Only load_model() and
family_logits() depend on the model family.
Evaluation sets: mimictest (the test split), relma (RELMA local names in place of the
descriptions, 18_relma_dataset.py), sliding (the windowed test split, 23_sliding_dataset.py), and,
for FHIR-MLM only, singleobs, singleobs_trim and singleobs_trim_relma (one lab Observation per row,
24_singleobs_dataset.py and 25_singleobs_train_dataset.py).

Join key: (patient_uuid, obs_ordinal), where obs_ordinal is the index of the scored lab observation
in the patient's chronological order. A token's position_idx is not comparable across models, since
their tokenizers differ. analysis/verify_obs_ordinal_join.py checks that the key lines up across model
families.

Columns: all 1623 logits for every position would be too large. The dump keeps the top-K classes and
logits, the logit of the true class and the logsumexp of the row. These give exact softmax
probabilities for those classes. Temperature scaling needs whole rows. --dump-full-logits writes them
for every position (centred float16 plus the row maximum), and --keep-full-logits keeps them for the
first N positions. The Chapter 6 analyses need full logits in every main evaluation dump.

--verify scores the model-selection split (val) and compares top-1 with the best_metric in the
training manifest (it passes within 0.02). It checks that a checkpoint loaded correctly. A randomly
initialised head still runs, so a bad load shows up only as a top-1 far below best_metric.

Reads the dataset from WORK_DIR/datasets and the best checkpoint named in the training manifest
(results/12_mlm_train, 13_bcm_train or 14_llama_train). Writes WORK_DIR/eval/<run>/part-0000.parquet
with a manifest (and the full logits when asked), and a copy of the manifest to results/15_eval/.
Patient-level rows stay in WORK_DIR. Stdout shows only aggregates.

  python pipeline/15_eval_runner.py --model mlm --condition para_xsite --train-arm pert --seed 0 \
      --evalset mimictest --dump-full-logits
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, has_description, representation,  # noqa: E402
                        resolve_arm_suffix, resolve_test_arm)
import torch

# third_party/fhir_mlm holds medair_models (FHIR-MLM's Qwen2ForMaskedLM) and the collator
for _p in (str(locations.THIRD_PARTY), str(locations.COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- Paths --
DATASET_BASE = str(locations.WORK / "datasets")
MODELS_BASE = str(locations.WORK / "models")
EVAL_BASE = str(locations.WORK / "eval")
RESULTS_DIR = locations.REPO / "results/15_eval"
REPO_DATA = locations.REPO / "data"
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
TOKENIZER_PATH = f"{locations.TOKENIZER}/bert-base-uncased-modified-tokenizer.json"  # the file 12_mlm_train.py uses

TRAIN_RESULTS = {
    "mlm": locations.REPO / "results/12_mlm_train",
    "bcm": locations.REPO / "results/13_bcm_train",
    "llama": locations.REPO / "results/14_llama_train",
}

# -- Task constants (must match training) --
N_LAB_CLASSES = 1623
TEXT_VOCAB_SIZE = 30522
LAB_TOKEN_MIN = 34951          # 4429 + 30522  (MLM absolute token space)
LAB_TOKEN_MAX = 36573          # 6051 + 30522
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
MAX_LEN = 8192
TOP_KS = (1, 3, 5, 10)
TOP_K_MAX = max(TOP_KS)

DEFAULT_BATCH = {"mlm": 2, "bcm": 8, "llama": 2}
# split the model-selection metric was computed on (used by --verify)
SELECTION_SPLIT = "val"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlm", "bcm", "llama"], required=True)
    p.add_argument("--condition", choices=MLM_CONDITIONS, required=True,
                   help="see pipeline/conditions.py for the slug grammar")
    p.add_argument("--train-arm", choices=["pert", "clean"], default=None,
                   help="v3 (xsite) only, and required there: which training arm's model to "
                        "evaluate. The test arm is --test-arm.")
    p.add_argument("--test-arm", default=None,
                   help="v3 (xsite) only: which copy of the test split to score on (--train-arm "
                        "selects the model). Defaults to the train arm (the 2x2 diagonal: A and "
                        "D). Set it to the other arm for the off-diagonal cells: train clean + "
                        "test pert = B, train pert + test clean = C. A replicate also works: "
                        "'pert_r3' scores on the third independent perturbation draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scope", choices=["full", "singleobs"], default="full",
                   help="which trained model to load: full = the 8192-window checkpoint; "
                        "singleobs = the single-observation checkpoint.")
    p.add_argument("--evalset", default="mimictest",
                   choices=["mimictest", "relma", "sliding", "singleobs", "singleobs_trim",
                            "singleobs_trim_relma"],
                   help="mimictest = the test split; relma = the RELMA-substituted test split "
                        "(18_relma_dataset.py); sliding = the windowed test split (23_sliding_dataset.py); "
                        "singleobs* = the single-observation sets (24, 25; FHIR-MLM only)")
    p.add_argument("--relma-strategy", choices=["s1", "s2"], default="s1",
                   help="relma only: s1 = local name, s2 = local name + battery name")
    p.add_argument("--assignment-seed", type=int, default=0,
                   help="relma only: which RELMA-name assignment draw to evaluate")
    p.add_argument("--split", default=None, help="override split (default: test; val when --verify)")
    p.add_argument("--limit", type=int, default=None, help="first N rows only (smoke/verify)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--verify", action="store_true",
                   help="run on the model-selection split and compare top-1 to the training manifest's best_metric")
    p.add_argument("--time-only", action="store_true", help="measure throughput; write nothing")
    p.add_argument("--dump-full-logits", action="store_true",
                   help="write the complete per-position logit matrix (centred float16 + row_max "
                        "float32), which temperature scaling and the Chapter 6 analyses need")
    p.add_argument("--keep-full-logits", type=int, default=0,
                   help="also retain full 1623-wide logits for the first N positions (temperature scaling)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    return p.parse_args()


# ==============================================================================
# Pure, unit-testable logic
# ==============================================================================
def build_class_to_loinc_array():
    """Return a (1623,) object array mapping class index -> LOINC string. An unmapped class gets a unique
    sentinel '?{i}', so an unmapped true class can only match a prediction of the same class (as in
    13_bcm_train.py)."""
    from fhir_data_utils import build_lab_code_mappings
    _, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    return np.array([class_to_loinc.get(i, f"?{i}") for i in range(N_LAB_CLASSES)], dtype=object)


def resolve_best_checkpoint(model, condition, seed, scope="full", arm=""):
    """Read the training manifest and return (checkpoint_path, best_metric, best_step). Raises when the
    manifest or the checkpoint is missing, instead of guessing a checkpoint.

    `scope="singleobs"` reads the manifest of the single-observation model (12_mlm_train.py --scope
    singleobs). That manifest has a different name, so the two models never overwrite each other's.
    """
    prefix = "singleobs_" if scope == "singleobs" else ""
    # `arm` separates the two v3 training arms, which share model family, condition and seed and
    # differ only in whether the training data was perturbed. Empty for v1 and v2.
    mf = TRAIN_RESULTS[model] / f"manifest_{prefix}{condition}{arm}_seed{seed}.json"
    if not mf.exists():
        raise FileNotFoundError(f"training manifest not found: {mf}")
    d = json.loads(mf.read_text())
    ckpt = d.get("best_model_checkpoint")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"best_model_checkpoint missing/absent for {model}/{condition}/seed{seed}: {ckpt}")
    return ckpt, d.get("best_metric"), d.get("best_step")


def center_logits(lg):
    """Return (centred float16, row_max float32). Softmax is shift-invariant: softmax((z-c)/T) ==
    softmax(z/T) for any constant c and any temperature T. Subtracting the row maximum therefore loses
    nothing and moves every value near 0, where float16 is most precise. Uncentred float16 would waste
    precision on large magnitudes, and float32 would double a 390 GB footprint."""
    rmax = lg.max(dim=1, keepdim=True).values
    return (lg - rmax).cpu().numpy().astype(np.float16), rmax.squeeze(1).cpu().numpy().astype(np.float32)


def true_rank_from_logits(lg, tc):
    """1-based rank of the true class in the full distribution (1 == top-1 correct).
    Ties count in favour of the true class (strict >), as in topk_hits in 07. topk_reduce uses
    torch.topk, which has no fixed tie order, so on an exact tie the two can disagree."""
    return (lg > lg.gather(1, tc.unsqueeze(1))).sum(dim=1) + 1


def assert_logit_dump_valid(L, RM, n_parquet, n_positions, n_classes):
    """Check the full-logit capture, which is expensive to repeat. A misaligned logit matrix would corrupt
    every calibrated number and leave accuracy untouched, so this check aborts at write time. Otherwise
    the problem would only show up during analysis."""
    if not (len(L) == len(RM) == n_parquet == n_positions):
        raise AssertionError(f"full-logit misalignment: logits={len(L)} rowmax={len(RM)} "
                             f"parquet={n_parquet} n_positions={n_positions}")
    if L.shape[1] != n_classes:
        raise AssertionError(f"expected {n_classes} classes, got {L.shape[1]}")
    mx = L.max(axis=1)
    if not np.allclose(mx, 0, atol=1e-2):
        raise AssertionError("centring broken: max(centred) should be 0, got range "
                             f"[{mx.min():.4f}, {mx.max():.4f}]")


def topk_reduce(logits_1623, labels_class):
    """logits (n,1623) float32 + labels (n,) int64 -> per-position dump arrays.
    Returns dict of numpy arrays. Softmax-exact: row_logsumexp is over the full 1623 row."""
    vals, idx = torch.topk(logits_1623, TOP_K_MAX, dim=-1)
    lse = torch.logsumexp(logits_1623.float(), dim=-1)
    true_logit = logits_1623.gather(1, labels_class.unsqueeze(1)).squeeze(1)
    return {
        "topk_class": idx.to(torch.int16).cpu().numpy(),
        "topk_logit": vals.float().cpu().numpy().astype(np.float32),
        "true_class_logit": true_logit.float().cpu().numpy().astype(np.float32),
        "row_logsumexp": lse.cpu().numpy().astype(np.float32),
    }


def accuracy_from_dump(topk_class, true_class, class_to_loinc_arr):
    """Top-K LOINC accuracy from the dumped arrays (same semantics as the training compute_metrics)."""
    pred_loincs = class_to_loinc_arr[topk_class]
    true_loincs = class_to_loinc_arr[true_class]
    out = {}
    for k in TOP_KS:
        out[f"loinc_top{k}"] = float((pred_loincs[:, :k] == true_loincs[:, None]).any(axis=1).mean())
    return out


# ==============================================================================
# Per-family model loading and logit extraction (the only family-specific code)
# ==============================================================================
def load_model(model_name, ckpt):
    """Returns (model, family_kind). Each family is saved differently:
      mlm   : full Qwen2ForMaskedLM safetensors without input_norm (training replaced it with
              Identity), so input_norm becomes Identity before the load
      bcm   : full AutoModelForTokenClassification safetensors
      llama : the QLoRA adapter only (modules_to_save includes `score`, so PEFT restores the
              1623-class head)
    """
    if model_name == "mlm":
        import torch.nn as nn
        from safetensors.torch import load_file
        from medair_models.qwen2.modeling_qwen2 import Qwen2ForMaskedLM, Qwen2Config  # third_party/fhir_mlm
        config = Qwen2Config.from_pretrained(ckpt)
        config._attn_implementation = "flash_attention_2"
        config.is_causal = False
        model = Qwen2ForMaskedLM(config)
        # the fine-tuned checkpoint has no input_norm key, because training set it to Identity after
        # loading. Here it becomes Identity before the load, the reverse of the training order.
        model.model.input_norm = nn.Identity()
        sd = load_file(os.path.join(ckpt, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "input_norm" not in k]
        if missing or unexpected:
            raise RuntimeError(f"MLM state_dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
        # training got bf16 from TrainingArguments(bf16=True). Manual eval must cast explicitly, or
        # flash-attn raises "FlashAttention only support fp16 and bf16 data type".
        model = model.to(torch.bfloat16)
        _apply_lab_logit_mask(model, config.vocab_size)
        return model, "mlm"

    if model_name == "bcm":
        from transformers import AutoModelForTokenClassification
        for attn in ("flash_attention_2", "sdpa"):
            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    ckpt, num_labels=N_LAB_CLASSES, attn_implementation=attn,
                    torch_dtype=torch.bfloat16, reference_compile=False)
                return model, "cls"
            except Exception:
                if attn == "sdpa":
                    raise
        raise RuntimeError("unreachable")

    # llama: 4-bit base + PEFT adapter (adapter carries the `score` head)
    from transformers import AutoModelForTokenClassification, BitsAndBytesConfig
    from peft import PeftModel
    base_name = json.loads(Path(ckpt, "adapter_config.json").read_text())["base_model_name_or_path"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForTokenClassification.from_pretrained(
        base_name, num_labels=N_LAB_CLASSES, quantization_config=bnb,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = PeftModel.from_pretrained(base, ckpt, is_trainable=False)
    return model, "cls"


def _apply_lab_logit_mask(model, vocab_size):
    """Forward hook on lm_head: bias non-lab-token logits to -inf (as in 12_mlm_train.py)."""
    keep = torch.zeros(vocab_size, dtype=torch.bool)
    keep[LAB_TOKEN_MIN:LAB_TOKEN_MAX + 1] = True
    bias = torch.where(keep, 0.0, float("-inf"))

    def hook(module, inp, output):
        return output + bias.to(output.device, output.dtype)

    model.lm_head.register_forward_hook(hook)


def family_logits(kind, out_logits):
    """Reduce raw model output to the common (B,S,1623) class space.
      mlm : slice the lab-token band out of the full vocab (non-lab already -inf via the hook)
      cls : already 1623-wide (BCM / Llama token-classification heads)"""
    if kind == "mlm":
        return out_logits[:, :, LAB_TOKEN_MIN:LAB_TOKEN_MAX + 1]
    return out_logits


def labels_to_class(model_name, labels):
    """MLM labels are absolute vocab token ids. BCM and Llama labels are already class indices."""
    if model_name == "mlm":
        return torch.where(labels != -100, labels - LAB_TOKEN_MIN, labels)
    return labels


def build_collator(model_name):
    if model_name == "mlm":
        from transformers import BertTokenizerFast
        from collator import DataCollatorForLabCodeMLM
        import pickle, pandas as pd
        with open(CODES_TO_TOKENS_PATH, "rb") as f:
            lab_map = pickle.load(f)[LAB_CODE_SYSTEM]
        seen = pd.read_csv(SEEN_CODES_PATH)
        seen = seen[seen.observation_type == "lab"]
        mappable = {lab_map[str(r.mimic_code)] + TEXT_VOCAB_SIZE
                    for r in seen.itertuples() if str(r.mimic_code) in lab_map and str(r.loinc_code)}
        tok = BertTokenizerFast(tokenizer_file=TOKENIZER_PATH)
        tok.bos_token, tok.bos_token_id = "[BOS]", 104
        tok.eos_token, tok.eos_token_id = "[EOS]", 105
        return DataCollatorForLabCodeMLM(tokenizer=tok, lab_token_min=LAB_TOKEN_MIN,
                                         lab_token_max=LAB_TOKEN_MAX, loinc_mappable_tokens=mappable)
    from transformers import AutoTokenizer, DataCollatorForTokenClassification
    src = "thomas-sounack/BioClinical-ModernBERT-large" if model_name == "bcm" else "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return DataCollatorForTokenClassification(tok, padding=True)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()
    # `train_arm` selects the model (which checkpoint, via its manifest). `test_arm` selects the
    # data (which copy of the test split it is scored on). The 2x2 of thesis Chapter 5 is their
    # cross product:
    #
    #        train_arm=clean, test_arm=clean -> A     train_arm=clean, test_arm=pert  -> B
    #        train_arm=pert,  test_arm=clean -> C     train_arm=pert,  test_arm=pert  -> D
    #
    # A single arm flag would make C and D indistinguishable and B and C unreachable.
    arm = resolve_arm_suffix(args.condition, args.train_arm, "15_eval_runner")
    # by default the test arm is the train arm (the diagonal of the 2x2). For v1 and v2 both are "".
    test_arm, cross_tag = resolve_test_arm(args.condition, arm, args.test_arm, "15_eval_runner")
    import datasets
    import pandas as pd

    split = args.split or (SELECTION_SPLIT if args.verify else "test")
    bs = args.batch_size or DEFAULT_BATCH[args.model]
    # The scope goes in the run name: the full-history and single-observation checkpoints are
    # evaluated on the same data and would otherwise share a directory.
    scope_tag = "__trained" if args.scope == "singleobs" else ""
    # RELMA runs, single-observation ones included, carry the strategy and the assignment seed, or
    # two draws would share a directory. `condition` stays in the name because the checkpoint is
    # still the para-trained one.
    if args.evalset == "relma":
        evalset_tag = f"relma_{args.relma_strategy}_a{args.assignment_seed}"
    elif args.evalset == "singleobs_trim_relma":
        evalset_tag = f"singleobs_trim_relma_{args.relma_strategy}_a{args.assignment_seed}"
    else:
        evalset_tag = args.evalset
    run_name = (f"{args.model}__{args.condition}{arm}__seed{args.seed}"
                f"__{evalset_tag}{scope_tag}{cross_tag}")
    # Guardrail: any split other than test goes in the run name, or a --split val run would overwrite
    # the test dump of the same model and seed.
    if split != "test":
        run_name += f"__{split}"
    # Guardrail: a --limit run is partial and must not write to the canonical path. Otherwise a
    # resumable runner would skip that run as complete, and a truncated result would become the
    # reported one.
    if args.limit and not (args.verify or args.time_only):
        run_name += f"__smoke{args.limit}"
        print(f"  [PARTIAL RUN] --limit set -> writing to '{run_name}' (not the canonical path)")
    print(f"=== 15_eval_runner | {run_name} | split={split} bs={bs} ===")

    ckpt, best_metric, best_step = resolve_best_checkpoint(args.model, args.condition, args.seed, arm=arm,
                                                           scope=args.scope)
    print(f"checkpoint: {ckpt}\n  manifest best_metric={best_metric} @step {best_step}")

    class_to_loinc_arr = build_class_to_loinc_array()
    print(f"class->LOINC: {sum(1 for x in class_to_loinc_arr if not str(x).startswith('?'))} mappable / {N_LAB_CLASSES}")

    # RELMA reads a separate test-only dataset, with descriptions replaced by real local names, using
    # the para-trained checkpoint above. The model meets local names it was not trained on.
    if args.evalset == "relma":
        ds_path = f"{DATASET_BASE}/{args.model}__relma_{args.relma_strategy}_a{args.assignment_seed}"
        # v3 RELMA sets carry the condition and the test arm in their name. Under v3 a RELMA set is
        # a site (perturbed values and ranges) as well as a set of names, so the clean and perturbed
        # builds are different data. v1 and v2 names have neither. Must match `run` in
        # 18_relma_dataset.py.
        if representation(args.condition) == "v3":
            ds_path += f"__{args.condition}{test_arm}"
        if not has_description(args.condition):
            raise ValueError("relma is an eval with descriptions, and only a para* condition has "
                             "a checkpoint trained with descriptions")
        if split != "test":
            raise ValueError(f"relma builds only a test split (got --split {split})")
    elif args.evalset == "singleobs_trim_relma":
        # Single-observation rows with RELMA local names in place of the MedGemma paraphrases
        # (25_singleobs_train_dataset.py --relma-strategy). The names come from the same
        # assign_paraphrases_per_patient, pool and salt as the full-history RELMA set, and on 60
        # patients they gave the same assignment. The coverage filter in
        # analysis/lib/dumps.attach_relma therefore applies unchanged.
        ds_path = (f"{DATASET_BASE}/mlm__singleobs_trim_relma_"
                   f"{args.relma_strategy}_a{args.assignment_seed}__{args.condition}")
        # v3: the test arm is part of the name, as in the `relma` and `singleobs_trim` branches,
        # because `assemble` in 25_singleobs_train_dataset.py writes `...__{condition}{arm}`. v1 and
        # v2 names have no arm.
        if representation(args.condition) == "v3":
            ds_path += test_arm
        if args.model != "mlm":
            raise ValueError("singleobs_trim_relma is FHIR-MLM only")
        if not has_description(args.condition):
            raise ValueError("RELMA substitutes the description; para* conditions only")
        if split != "test":
            raise ValueError(f"singleobs_trim_relma builds only a test split (got --split {split})")
    elif args.evalset == "singleobs_trim":
        # The same observations as `singleobs`, with rows trimmed to their true length
        # (25_singleobs_train_dataset.py --trim). It is faster to evaluate, and it is the form the
        # single-observation models train on, so train and eval share a padding convention. The
        # full-history checkpoint is required to reproduce its `singleobs` result here exactly.
        ds_path = f"{DATASET_BASE}/{args.model}__singleobs_trim__{args.condition}{test_arm}"
        if args.model != "mlm":
            raise ValueError("singleobs_trim is FHIR-MLM only")
        # Any split is accepted, because 25_singleobs_train_dataset.py builds train, val and test,
        # and --verify runs on the model-selection split (val). The split check at load time
        # reports a missing split.
    elif args.evalset == "singleobs":
        # Test-time history ablation: the same checkpoint, one lab Observation per row and no history
        # (24_singleobs_dataset.py). The dataset carries `obs_offset`, so the offset logic below
        # gives the patient-level chronological obs_ordinal and the dump joins the mimictest dump
        # position for position.
        ds_path = f"{DATASET_BASE}/{args.model}__singleobs__{args.condition}{test_arm}"
        if args.model != "mlm":
            raise ValueError("singleobs is built for FHIR-MLM only (24_singleobs_dataset.py)")
        if split != "test":
            raise ValueError(f"singleobs builds only a test split (got --split {split})")
    elif args.evalset == "sliding":
        # Evaluation-time sliding window: the same trained checkpoint on a windowed test set
        # (23_sliding_dataset.py). It gives a view of deployment over whole records and is not part
        # of the controlled comparison of history.
        ds_path = f"{DATASET_BASE}/{args.model}__sliding__{args.condition}{test_arm}"
    else:
        ds_path = f"{DATASET_BASE}/{args.model}__{args.condition}{test_arm}"
    if not Path(ds_path).exists():
        raise FileNotFoundError(f"dataset not found: {ds_path}"
                                + (" (run 18_relma_dataset.py first)" if args.evalset == "relma" else ""))
    _dd = datasets.load_from_disk(ds_path)
    if split not in _dd:
        raise ValueError(f"split {split!r} not in {ds_path} (has: {sorted(_dd.keys())})")
    ds = _dd[split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    uuids = ds["uuid"]
    # In a windowed or single-observation dataset a patient spans several rows. obs_ordinal must
    # stay the patient-level chronological index, or the (patient_uuid, obs_ordinal) key collides
    # across rows. The first observation of window 2 would also be ordinal 0. Such a dataset carries
    # `obs_offset`, the number of labelled observations in all earlier rows of that patient. Without
    # the column the offsets are all zero.
    has_offsets = "obs_offset" in ds.column_names
    obs_offsets = ds["obs_offset"] if has_offsets else [0] * len(ds)
    window_idxs = ds["window_idx"] if "window_idx" in ds.column_names else None
    keep = [c for c in ds.column_names
            if c not in ("uuid", "obs_offset", "window_idx", "n_windows")]
    ds_model = ds.select_columns(keep)
    print(f"dataset: {len(ds)} rows ({split})")

    collator = build_collator(args.model)
    if args.evalset in ("singleobs_trim", "singleobs_trim_relma"):
        # Trimmed rows are ragged and DataCollatorForLabCodeMLM cannot pad them. This is the wrapper
        # the training run uses, so train and eval share one padding convention (left, tokenizer pad values).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from padcollate import PadToBatchMax
        _tk = collator.tokenizer
        collator = PadToBatchMax(inner=collator, pad_token_id=_tk.pad_token_id,
                                 pad_token_type_id=_tk.pad_token_type_id, pad_to_multiple_of=8)
    t_load = time.time()
    model, kind = load_model(args.model, ckpt)
    model.eval().cuda()
    load_sec = time.time() - t_load
    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    print(f"model loaded in {load_sec:.1f}s | params={n_params:,}")

    out_dir = Path(EVAL_BASE) / run_name
    if not (args.verify or args.time_only):
        if out_dir.exists() and not args.force:
            raise FileExistsError(f"{out_dir} exists (pass --force to overwrite)")
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = {k: [] for k in ("patient_uuid", "obs_ordinal", "position_idx", "true_class",
                            "topk_class", "topk_logit", "true_class_logit", "row_logsumexp",
                            # per-position extras: rank, length, truncation, labelled count, pretraining overlap
                            "true_rank", "seq_len", "truncated", "n_labeled_in_patient",
                            "in_pretrain_4096")}
    if "window_idx" in datasets.load_from_disk(ds_path)[split].column_names:
        rows["window_idx"] = []
    full_logits_keep = []
    # full logits are stored centred: (logit - row_max) as float16 plus row_max as float32 (see
    # center_logits for why this loses nothing)
    logit_chunks, rowmax_chunks = [], []
    pretrain4096 = set()
    if args.dump_full_logits:
        p4 = REPO_DATA / "pretrain" / "pretrain_4096_patients.txt"
        pretrain4096 = {l.strip() for l in open(p4) if l.strip()}
        print(f"pretrain-4096 contamination set: {len(pretrain4096):,} patients "
              "(their observations are flagged, so the sensitivity analysis can filter on the flag)")
    n_positions = 0
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, len(ds_model), bs):
            batch_rows = [ds_model[i] for i in range(start, min(start + bs, len(ds_model)))]
            batch = collator(batch_rows)
            labels = batch.pop("labels")
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(**batch)
            logits = family_logits(kind, out.logits)                      # (B,S,1623)
            lab_cls = labels_to_class(args.model, labels).cuda()          # (B,S)

            for b in range(logits.shape[0]):
                sel = lab_cls[b] != -100
                n = int(sel.sum())
                if n == 0:
                    continue
                pos = torch.nonzero(sel, as_tuple=False).squeeze(1)
                lg = logits[b][sel].float()                               # (n,1623)
                tc = lab_cls[b][sel]                                      # (n,)
                d = topk_reduce(lg, tc)
                uuid = uuids[start + b]
                off = obs_offsets[start + b]           # 0 unless the rows carry obs_offset
                rows["patient_uuid"].extend([uuid] * n)
                rows["obs_ordinal"].extend(range(off, off + n))           # chronological within the patient
                if window_idxs is not None:
                    rows["window_idx"].extend([window_idxs[start + b]] * n)
                rows["position_idx"].extend(pos.cpu().numpy().tolist())
                rows["true_class"].extend(tc.cpu().numpy().astype(np.int16).tolist())
                rows["topk_class"].extend(d["topk_class"].tolist())
                rows["topk_logit"].extend(d["topk_logit"].tolist())
                rows["true_class_logit"].extend(d["true_class_logit"].tolist())
                rows["row_logsumexp"].extend(d["row_logsumexp"].tolist())
                # rank of the true class in the full distribution (1 = top-1 correct). It is cheap here
                # and gives top-k for any k, and MRR, beyond the stored K=10.
                rank = true_rank_from_logits(lg, tc)
                rows["true_rank"].extend(rank.cpu().numpy().astype(np.int16).tolist())
                sl = int(batch["attention_mask"][b].sum().item())
                rows["seq_len"].extend([sl] * n)
                rows["truncated"].extend([sl >= MAX_LEN] * n)
                rows["n_labeled_in_patient"].extend([n] * n)
                rows["in_pretrain_4096"].extend([uuid in pretrain4096] * n)
                if args.dump_full_logits:
                    c, rm = center_logits(lg)
                    logit_chunks.append(c); rowmax_chunks.append(rm)
                if args.keep_full_logits and len(full_logits_keep) < args.keep_full_logits:
                    take = min(args.keep_full_logits - len(full_logits_keep), n)
                    full_logits_keep.extend(lg[:take].cpu().numpy().astype(np.float16).tolist())
                n_positions += n

            if (start // bs) % 20 == 0:
                el = time.time() - t0
                done = min(start + bs, len(ds_model))
                print(f"  {done}/{len(ds_model)} rows | {n_positions:,} positions | "
                      f"{el:.1f}s | {el/max(done,1):.3f}s/row", flush=True)

    elapsed = time.time() - t0
    rate = elapsed / max(len(ds_model), 1)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    pos_per_sec = n_positions / max(elapsed, 1e-9)
    print(f"\ndone: {len(ds_model)} rows, {n_positions:,} positions in {elapsed:.1f}s "
          f"({rate:.3f}s/row, {pos_per_sec:.1f} obs/s, peak {peak_mem_gb:.1f} GB)")

    topk_arr = np.array(rows["topk_class"], dtype=np.int64)
    true_arr = np.array(rows["true_class"], dtype=np.int64)
    metrics = accuracy_from_dump(topk_arr, true_arr, class_to_loinc_arr)
    metrics["n_positions"] = int(n_positions)
    metrics["n_rows"] = int(len(ds_model))
    print("metrics: " + " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items()))

    # -- time for the whole split, extrapolated (useful with --limit) --
    full_n = len(datasets.load_from_disk(ds_path)[split])
    print(f"EXTRAPOLATION -> full {split} ({full_n} rows): {rate*full_n/3600:.2f} h")

    if args.verify:
        got, exp = metrics["loinc_top1"], best_metric
        delta = abs(got - exp) if exp is not None else None
        print("\n" + "=" * 64)
        if exp is None:
            print("VERIFY: no best_metric in the manifest to compare against")
        elif delta < 0.02:
            print(f"✓ VERIFY PASS: top1={got:.5f} vs manifest {exp:.5f} (Δ={delta:.5f})")
            print("   checkpoint loaded correctly.")
        else:
            print(f"✗ VERIFY FAIL: top1={got:.5f} vs manifest {exp:.5f} (Δ={delta:.5f})")
            print("   checkpoint probably mis-loaded. Do not trust downstream numbers.")
        print("=" * 64)
        return
    if args.time_only:
        return

    df = pd.DataFrame(rows)
    df["model"], df["condition"], df["seed"], df["evalset"] = args.model, args.condition, args.seed, args.evalset
    df["true_loinc"] = class_to_loinc_arr[true_arr]
    pq = out_dir / "part-0000.parquet"
    df.to_parquet(pq, index=False)
    print(f"wrote {pq} ({pq.stat().st_size/1e6:.1f} MB, {len(df):,} rows)")

    if args.dump_full_logits:
        L = np.concatenate(logit_chunks, axis=0)
        RM = np.concatenate(rowmax_chunks, axis=0)
        assert_logit_dump_valid(L, RM, len(df), n_positions, N_LAB_CLASSES)
        np.save(out_dir / "logits_centered_f16.npy", L)
        np.save(out_dir / "row_max_f32.npy", RM)
        print(f"wrote logits_centered_f16.npy {L.shape} "
              f"({L.nbytes/1e9:.2f} GB) and row_max_f32.npy (alignment checked)")

    if full_logits_keep:
        np.save(out_dir / "full_logits_subsample.npy", np.array(full_logits_keep, dtype=np.float16))

    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=str(locations.REPO)).decode().strip()
    except Exception:
        sha = None
    manifest = {
        "run_name": run_name, "model": args.model, "condition": args.condition, "seed": args.seed,
        "evalset": args.evalset, "evalset_tag": evalset_tag, "split": split, "checkpoint": ckpt,
        "dataset_path": ds_path,
        # RELMA provenance: without these a RELMA result cannot be traced to its draw
        "relma_strategy": args.relma_strategy if args.evalset == "relma" else None,
        "assignment_seed": args.assignment_seed if args.evalset == "relma" else None,
        "train_best_metric": best_metric, "train_best_step": best_step,
        "n_rows": len(ds_model), "n_positions": int(n_positions), "metrics": metrics,
        # deployment and efficiency metrics, captured here so the cost analysis needs no re-run
        "wall_clock_sec": elapsed, "sec_per_row": rate,
        "sec_per_position": elapsed / max(n_positions, 1), "positions_per_sec": pos_per_sec,
        "peak_gpu_mem_gb": peak_mem_gb, "model_load_sec": load_sec, "n_params": int(n_params),
        "batch_size": bs, "max_len": MAX_LEN,
        # what was captured, so an analysis can assert that its inputs exist before it starts and
        # does not find a missing column mid-run
        "full_logits": bool(args.dump_full_logits),
        "full_logits_dtype": "float16 centred (logit - row_max); row_max stored float32",
        "extra_columns": ["true_rank", "seq_len", "truncated", "n_labeled_in_patient",
                          "in_pretrain_4096"],
        "n_pretrain_4096_flagged": int(df["in_pretrain_4096"].sum()) if "in_pretrain_4096" in df else None,
        "n_truncated_positions": int(df["truncated"].sum()) if "truncated" in df else None,
        "git_sha": sha, "parquet": str(pq), "parquet_mb": round(pq.stat().st_size / 1e6, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"manifest -> {out_dir/'manifest.json'} and {RESULTS_DIR/(run_name+'.json')}")


if __name__ == "__main__":
    main()
