"""Fine-tune FHIR-MLM (Qwen2ForMaskedLM, bidirectional) on the masked-history datasets to predict each
lab observation's MIMIC lab code, scored as LOINC (thesis Chapter 5 and Appendix D).

The collator masks every lab-code token. A forward hook on the LM head restricts predictions to the
1623 lab tokens. The metric is top-K LOINC accuracy. Training uses an effective batch of 16, lr 5e-5
with a cosine schedule and 5% warmup, bf16/tf32 and FSDP shard_grad_op. With --scope singleobs the
effective batch is 3520 rows, which gives about the same number of masked positions per step. It
evaluates and saves a checkpoint every 400 steps, stops early when eval_loinc_top1 has not improved
for 3 evaluations (EarlyStoppingCallback), and stops at 8000 steps at most. Every checkpoint is kept
during training (save_total_limit=None), and a callback tracks the best step. Afterwards rank 0
deletes every checkpoint but the best. load_best_model_at_end is not used, because it deadlocks under
FSDP. --seed sets both the model seed and the data seed.

Reads WORK_DIR/datasets/mlm__{condition}{arm} (train and val, with the test split held out for
15_eval_runner.py), the pretrained checkpoint (FHIR_MLM_BASE_CHECKPOINT), the tokenizer files
(FHIR_MLM_TOKENIZER_DIR) and UPSTREAM_DIR/seen_codes.csv. With --scope singleobs it trains on the
single-observation dataset from 25_singleobs_train_dataset.py instead. Writes the best checkpoint to
WORK_DIR/models/mlm__{condition}{arm}__seed{n}/ (mlm__singleobs__... with --scope singleobs). Also
writes a progress log (the metrics of each evaluation) and a manifest to results/12_mlm_train/. The
manifest names the best checkpoint, and 15_eval_runner.py reads it. Only aggregate metrics are logged.

Modes:
  full  (default)
  short (--max-steps < --save-steps): an FSDP measurement run with no saving, best-checkpoint
        tracking or early stopping, and one evaluation at the end, to check throughput and that
        the loss falls before a full run
  dry   (--dry-run): one GPU, one train and one eval step on four rows, no FSDP and no saving

  torchrun --nproc_per_node=2 pipeline/12_mlm_train.py --condition para_xsite --train-arm pert --seed 0
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import faulthandler
import signal
# `kill -USR1 <pid>` makes a worker print every thread's Python stack to stderr (the run log). This
# shows where a stalled worker is when py-spy cannot attach (for example under ptrace_scope=1).
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
import json
import os
import shutil
import sys
import time
import pickle
from functools import partial
from pathlib import Path

import datasets
import pandas as pd
from safetensors.torch import load_file as load_safetensors
from transformers import Trainer, TrainingArguments, BertTokenizerFast, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.modeling_outputs import CausalLMOutputWithPast

# -- Model and collator (third_party/fhir_mlm) --
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, representation,  # noqa: E402
                        resolve_arm_suffix)

sys.path.insert(0, str(locations.THIRD_PARTY))
from medair_models.qwen2.modeling_qwen2 import Qwen2ForMaskedLM, Qwen2Config
sys.path.insert(0, str(locations.THIRD_PARTY))
from collator import DataCollatorForLabCodeMLM

# -- Paths --
CHECKPOINT = str(locations.FHIR_MLM_BASE)
DATASET_BASE = str(locations.WORK / "datasets")   # mlm__{condition}
MODELS_BASE = str(locations.WORK / "models")      # mlm__{condition}__seed{n}
RESULTS_DIR = locations.REPO / "results/12_mlm_train"

SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")
CODES_TO_TOKENS_PATH = str(locations.TOKENIZER / "codes_to_tokens.pkl")
TOKENIZER_PATH = str(locations.TOKENIZER / "bert-base-uncased-modified-tokenizer.json")

# -- Token space --
TEXT_VOCAB_SIZE = 30522
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
LAB_TOKEN_MIN = 34951  # 4429 + 30522
LAB_TOKEN_MAX = 36573  # 6051 + 30522
N_LAB_TOKENS = LAB_TOKEN_MAX - LAB_TOKEN_MIN + 1  # 1623

# -- Hyperparameters --
GPU_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 16
MAX_STEPS = 8000
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.05
EVAL_STEPS = 400
# -- Single-observation model (--scope singleobs) -------------------------------
# A row is one lab Observation (about 16 real tokens) with exactly one masked position. A
# full-history row has 8192 tokens and about 218 masked positions. The effective batch is matched in
# masked positions, because a learning rate is tuned against positions per step. 16 x ~218 = ~3,494
# positions per step, so about 3,500 single-observation rows per step.
SINGLEOBS_GPU_BATCH = 16       # micro-batch only. Grad accumulation holds the effective batch at
                               # SINGLEOBS_TOTAL_BATCH. The limit is the fp32 logits gradient
                               # (B x S x 112640 x 4 bytes). At 32 it needs 4.19 GiB (sequence
                               # length 311) and runs out of memory in backward a few hundred steps
                               # in. Short test runs at 32 pass, because Adam's state is not fully
                               # allocated that early. At 16 the worst case is 2.09 GiB.
SINGLEOBS_TOTAL_BATCH = 3520   # 2 GPUs x 16 x accumulation 110, or 2 x 80 x 22 with --sparse-head
SINGLEOBS_GPU_BATCH_SPARSE = 80  # micro-batch with --sparse-head. Without the full-width logits
                               # gradient, activations are the limit. In 30-step train-only timing
                               # runs, 55 took 2.02 s/step, 80 1.79 and 110 1.72, and 160 ran out of
                               # memory. 80 is 4% slower than 110 but leaves much more headroom,
                               # which matters because a short run cannot show steady-state memory.
SINGLEOBS_EVAL_BATCH = 32      # separate from the training micro-batch. Evaluation runs the stock
                               # full-width forward, because the metrics need (B,S,112640) logits.
                               # At a micro-batch of 80, accelerate's _pad_across_processes asks for
                               # 9.94 GiB. 32 fits.
SAVE_STEPS = 400          # == EVAL_STEPS, so the best evaluation step always has a checkpoint
EARLY_STOP_PATIENCE = 3   # stop if eval_loinc_top1 doesn't improve for 3 evals (= 1200 steps)
BEST_METRIC = "eval_loinc_top1"

TOP_KS = (1, 3, 5, 10)
TOP_K_MAX = max(TOP_KS)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=MLM_CONDITIONS, required=True,
                   help="see pipeline/conditions.py; paths follow the slug, so a new "
                        "arm lands in its own dataset/model directory")
    p.add_argument("--train-arm", choices=["pert", "clean"], default=None,
                   help="v3 (xsite) only, and required there: which training arm. 'pert' = "
                        "trained on perturbed data, 'clean' = the v3 representation with no "
                        "perturbation. No default.")
    p.add_argument("--so-gpu-batch", type=int, default=None,
                   help="debug/tuning: override SINGLEOBS_GPU_BATCH (per-device micro-batch). "
                        "Single-obs rows are ~24 tokens, so a small micro-batch leaves the step "
                        "kernel-launch bound rather than compute bound.")
    p.add_argument("--limit-rows", type=int, default=None,
                   help="debug: use only the first N train rows (and N/10 val), for bisecting "
                        "scale-dependent stalls on the FSDP path. --dry-run cannot do this, since it "
                        "runs on one GPU and does not work under flash-attention.")
    p.add_argument("--sparse-head", action="store_true",
                   help="in training, apply the LM head at labelled positions only. Equivalent "
                        "to the stock loss (CE ignores -100), but avoids materialising a "
                        "(B,S,112640) fp32 logits tensor and its gradient. Off by default.")
    p.add_argument("--scope", choices=["full", "singleobs"], default="full",
                   help="full = the 8192-token patient window. "
                        "singleobs = one lab Observation per row, the single-observation ablation "
                        "(dataset from 25_singleobs_train_dataset.py).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="single GPU, 1 train + 1 eval step on a 4-row slice; no FSDP/save")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="train-step cap. If < save-steps it's a short measurement run (no save/best/early-stop, "
                        "one eval at the end) for throughput + loss-decrease checks before the full run.")
    p.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="checkpoint interval (keep == eval interval)")
    p.add_argument("--eval-steps", type=int, default=EVAL_STEPS, help="eval interval")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args()


# ==============================================================================
# Pure, unit-testable logic
# ==============================================================================
def build_loinc_mappings(codes_to_tokens_path, seen_codes_path, lab_code_system, text_vocab_size):
    """Returns (loinc_mappable_tokens: set[int], mimic_token_to_loinc: dict[int,str]). Token id =
    raw lab token + text_vocab_size. Only lab codes with a LOINC mapping are included."""
    with open(codes_to_tokens_path, "rb") as f:
        codes_to_tokens = pickle.load(f)
    lab_map = codes_to_tokens[lab_code_system]
    seen_codes = pd.read_csv(seen_codes_path)
    lab_seen = seen_codes[seen_codes.observation_type == "lab"]
    loinc_mappable_tokens = set()
    mimic_token_to_loinc = {}
    for _, row in lab_seen.iterrows():
        mimic_code = str(row["mimic_code"])
        loinc_code = str(row["loinc_code"])
        if mimic_code in lab_map:
            token_id = lab_map[mimic_code] + text_vocab_size
            loinc_mappable_tokens.add(token_id)
            mimic_token_to_loinc[token_id] = loinc_code
    return loinc_mappable_tokens, mimic_token_to_loinc


def preprocess_logits_for_metrics(logits, labels, lab_token_min, lab_token_max):
    """Reduce per-batch logits to the top-K absolute lab token ids (keeps eval-time memory tractable).
    Returns (top-K abs lab ids, labels)."""
    lab_logits = logits[:, :, lab_token_min:lab_token_max + 1]
    top_k_rel = lab_logits.topk(TOP_K_MAX, dim=-1).indices  # (B,S,K)
    return top_k_rel + lab_token_min, labels


def compute_metrics(eval_pred, mimic_token_to_loinc):
    """Top-1/3/5/10 LOINC accuracy over the masked positions. Predicted and true token ids map to
    LOINC through mimic_token_to_loinc. A token without a LOINC maps to '?' and stays in the
    denominator."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_flat = preds.reshape(-1, preds.shape[-1])   # (N*S, K)
    labels_flat = labels.flatten()                     # (N*S,)
    active = labels_flat != -100
    active_preds = preds_flat[active]
    active_labels = labels_flat[active]
    n = int(active_labels.shape[0])
    if n == 0:
        return {f"loinc_top{k}": 0.0 for k in TOP_KS} | {"loinc_accuracy": 0.0, "eval_samples": 0}
    import numpy as _np
    true_loincs = _np.array([mimic_token_to_loinc.get(int(l), "?") for l in active_labels], dtype=object)
    pred_loincs = _np.array([[mimic_token_to_loinc.get(int(t), "?") for t in row] for row in active_preds], dtype=object)
    metrics = {"eval_samples": n}
    for k in TOP_KS:
        topk_match = (pred_loincs[:, :k] == true_loincs[:, None]).any(axis=1)
        metrics[f"loinc_top{k}"] = float(topk_match.mean())
    metrics["loinc_accuracy"] = metrics["loinc_top1"]
    return metrics


def apply_lab_logit_mask(model, lab_token_min, lab_token_max, vocab_size):
    """Forward hook on lm_head: bias all non-lab-token logits to -inf, so the model predicts only lab tokens."""
    keep_mask = torch.zeros(vocab_size, dtype=torch.bool)
    keep_mask[lab_token_min:lab_token_max + 1] = True
    logit_bias = torch.where(keep_mask, 0.0, float("-inf"))

    def hook_fn(module, inp, output):
        return output + logit_bias.to(output.device, output.dtype)

    model.lm_head.register_forward_hook(hook_fn)
    print(f"Logit mask: {N_LAB_TOKENS} lab tokens kept, {vocab_size - N_LAB_TOKENS} masked to -inf")


def apply_sparse_lm_head(model, lab_token_min, lab_token_max):
    """Training only: run the LM head at the labelled positions only. It gives the same loss as the
    stock forward and is much cheaper.

    The stock forward does:
        logits = self.lm_head(hidden)          # (B, S, 112640)
        logits = logits.float()                # fp32: the tensor that runs out of memory
        loss   = CrossEntropyLoss()(logits.view(-1, V), labels.view(-1))
    CrossEntropyLoss ignores positions labelled -100, so their logits are computed and thrown away.
    In the single-observation arm each row has one labelled position out of about 21. Over 20,000
    sampled rows of both splits, every row had exactly one (min == max == 1). Over 95% of that tensor
    is therefore unused, and its gradient is what runs out of memory at micro-batch 32 (4.19 GiB).

    Selecting the labelled rows before the head gives exactly the same loss, with the -inf mask
    still applied. CrossEntropyLoss(reduction="mean") averages over the non-ignored entries, which
    are the rows selected. It keeps:
      * the weights, the vocabulary and every state_dict key (nothing is re-parameterised);
      * the tied input embedding: `self.lm_head` is still the module called, so its -inf forward
        hook still fires and the gradient still reaches the tied weight;
      * the evaluation path, which stays on the stock forward (`model.training` is False there)
        because preprocess_logits_for_metrics and compute_metrics need full-width logits.
        Evaluation runs under no_grad and never allocates the gradient.

    Off unless --sparse-head is given.
    """
    if model.config.compression_method is not None:
        raise NotImplementedError(
            "--sparse-head implements only the compression_method=None branch of "
            "Qwen2ForMaskedLM.forward; this checkpoint uses "
            f"compression_method={model.config.compression_method!r}")
    orig_forward = model.forward
    ce = nn.CrossEntropyLoss()

    def sparse_forward(input_ids=None, attention_mask=None, position_ids=None,
                       past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                       output_attentions=None, output_hidden_states=None, return_dict=None,
                       token_type_ids=None, time_vals=None, time_inds=None, numeric_vals=None,
                       numeric_inds=None, compression_idx=None, numeric_lookup_id=None):
        # The signature copies Qwen2ForMaskedLM.forward, with no **kwargs. Hugging Face passes
        # `num_items_in_batch` only if the forward has a **kwargs parameter, and that argument
        # changes how the loss is normalised across gradient-accumulation steps. Adding **kwargs
        # would therefore change the loss normalisation. An unexpected argument raises TypeError.
        passthrough = dict(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            token_type_ids=token_type_ids, time_vals=time_vals, time_inds=time_inds,
            numeric_vals=numeric_vals, numeric_inds=numeric_inds, compression_idx=compression_idx)
        if (not model.training) or labels is None:
            return orig_forward(labels=labels, return_dict=return_dict,
                                numeric_lookup_id=numeric_lookup_id, **passthrough)
        outputs = model.model(return_dict=True, **passthrough)
        hidden = outputs[0]
        flat_labels = labels.reshape(-1)
        sel = flat_labels != -100
        if not bool(sel.any()):
            raise RuntimeError("--sparse-head: batch has no labelled positions; loss would be NaN")
        sel_hidden = hidden.reshape(-1, hidden.size(-1))[sel]
        logits = model.lm_head(sel_hidden).float()          # (n_labelled, V): all of the saving is here
        loss = ce(logits, flat_labels[sel].to(logits.device))
        # logits=None because nothing reads training-time logits (Trainer uses outputs["loss"]).
        # A (n_labelled, V) tensor here would have the wrong shape for anything that read it later.
        return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=None,
                                      hidden_states=None, attentions=None)

    model.forward = sparse_forward
    print("Sparse LM head: training-time head applied to labelled positions only "
          "(eval unchanged, stock forward)")


# ==============================================================================
# Trainer and sidecar progress log (rank 0 only)
# ==============================================================================
class FastLengthTrainer(Trainer):
    """Trainer whose length-grouped sampler reads lengths from a Python list instead of a lazy Arrow column.

    Hugging Face asks `datasets` for the length column, and since datasets 4.x that is a lazy
    `Column`. `LengthGroupedSampler.__iter__` then sorts each megabatch with
    `key=lambda i: lengths[i]`, so every comparison is an Arrow row lookup (about 226 us, against
    about 20 ns from a list). The sort costs ~N*log2(megabatch) lookups. On the ~12M
    single-observation rows, dataloader setup takes hours of full CPU with both GPUs idle, which
    looks like a hang at step 0.

    Materialising the column leaves the order identical, because the algorithm, the generator and
    the integer values are the same. Only the speed changes.
    """

    def _get_train_sampler(self, *a, **kw):
        sampler = super()._get_train_sampler(*a, **kw)
        lengths = getattr(sampler, "lengths", None)
        if lengths is not None and not isinstance(lengths, list):
            sampler.lengths = list(lengths)
        return sampler


class SidecarCallback(TrainerCallback):
    """Append the LOINC metrics of each evaluation to a sidecar log (rank 0 only, aggregate metrics).
    Also tracks the best (step, metric), so that after training the run can keep only the best
    checkpoint. This avoids load_best_model_at_end, which deadlocks under FSDP with save_total_limit
    at the end-of-run checkpoint load."""
    def __init__(self, path):
        self.path = path
        self.best_metric = None
        self.best_step = None

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics and BEST_METRIC in metrics:
            m = metrics[BEST_METRIC]
            if self.best_metric is None or m > self.best_metric:
                self.best_metric, self.best_step = m, state.global_step
        if state.is_world_process_zero and metrics:
            keep = {k: v for k, v in metrics.items() if "loinc" in k or k in ("eval_loss", "eval_samples")}
            with open(self.path, "a") as f:
                f.write(f"step {state.global_step}: " + " ".join(f"{k}={v}" for k, v in keep.items()) + "\n")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()
    # every path below (dataset, model dir, sidecar, manifest) carries the arm, because the two v3
    # training arms differ only by this flag
    arm = resolve_arm_suffix(args.condition, args.train_arm, "12_mlm_train")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    short = (not args.dry_run) and args.max_steps < args.save_steps   # measurement run: skip best-ckpt machinery

    loinc_mappable_tokens, mimic_token_to_loinc = build_loinc_mappings(
        CODES_TO_TOKENS_PATH, SEEN_CODES_PATH, LAB_CODE_SYSTEM, TEXT_VOCAB_SIZE)
    print(f"LOINC-mappable lab tokens: {len(loinc_mappable_tokens)} | token->LOINC entries: {len(mimic_token_to_loinc)}")

    # -- Dataset (mlm__{condition}{arm}: train and val) --
    single = args.scope == "singleobs"
    dataset_path = (f"{DATASET_BASE}/mlm__singleobs_trim__{args.condition}{arm}" if single
                    else f"{DATASET_BASE}/mlm__{args.condition}{arm}")
    print(f"\nLoading dataset: {dataset_path}")
    ds_dict = datasets.load_from_disk(dataset_path)
    if not isinstance(ds_dict, datasets.DatasetDict) or "train" not in ds_dict or "val" not in ds_dict:
        raise TypeError(f"Expected DatasetDict with train+val at {dataset_path}")
    print(f"  train={len(ds_dict['train'])}  val={len(ds_dict['val'])}  (test held out for eval block)")
    keep_cols = ["input_ids", "token_type_ids", "attention_mask", "special_tokens_mask"]
    if single:
        # `length` feeds group_by_length. Without it LengthGroupedSampler measures every row of the
        # dataset itself.
        assert "length" in ds_dict["train"].column_names, "singleobs dataset lacks a `length` column"
        keep_cols = keep_cols + ["length"]
    ds_dict = datasets.DatasetDict({k: ds_dict[k].select_columns(keep_cols) for k in ("train", "val")})

    if args.limit_rows:
        ds_dict = datasets.DatasetDict({
            "train": ds_dict["train"].select(range(min(args.limit_rows, len(ds_dict["train"])))),
            "val": ds_dict["val"].select(range(min(max(args.limit_rows // 10, 8), len(ds_dict["val"])))),
        })
        print(f"  [LIMIT-ROWS] train={len(ds_dict['train'])} val={len(ds_dict['val'])}")
    if args.dry_run:
        ds_dict = datasets.DatasetDict({
            "train": ds_dict["train"].select(range(min(4, len(ds_dict["train"])))),
            "val": ds_dict["val"].select(range(min(4, len(ds_dict["val"])))),
        })
        print(f"  [DRY-RUN] sliced to train={len(ds_dict['train'])} val={len(ds_dict['val'])}")

    # -- Tokenizer and collator --
    text_tokenizer = BertTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    text_tokenizer.bos_token, text_tokenizer.bos_token_id = "[BOS]", 104
    text_tokenizer.eos_token, text_tokenizer.eos_token_id = "[EOS]", 105
    collator = DataCollatorForLabCodeMLM(tokenizer=text_tokenizer, lab_token_min=LAB_TOKEN_MIN,
                                         lab_token_max=LAB_TOKEN_MAX, loinc_mappable_tokens=loinc_mappable_tokens)
    if single:
        # Trimmed rows are ragged, and DataCollatorForLabCodeMLM calls torch.tensor(v), so it cannot
        # pad. PadToBatchMax pads each batch to its longest row, on the left, with the pad values of
        # the tokenizer (NdjsonTokenizer.padding_side == "left"). It then delegates to the inner
        # collator. Apart from the amount of padding, the input_ids and labels are those of the
        # fixed-length rows.
        from padcollate import PadToBatchMax
        collator = PadToBatchMax(inner=collator, pad_token_id=text_tokenizer.pad_token_id,
                                 pad_token_type_id=text_tokenizer.pad_token_type_id,
                                 pad_to_multiple_of=8)

    # -- Model --
    config = Qwen2Config.from_pretrained(CHECKPOINT)
    config._attn_implementation = "flash_attention_2"
    config.is_causal = False
    state_dict = load_safetensors(os.path.join(CHECKPOINT, "model.safetensors"))
    if "model.embed_tokens.weight" in state_dict and "model.embeddings.embed_tokens.weight" not in state_dict:
        state_dict["model.embeddings.embed_tokens.weight"] = state_dict.pop("model.embed_tokens.weight")
    if "model.input_norm.weight" not in state_dict:
        state_dict["model.input_norm.weight"] = torch.ones(config.hidden_size)
    model = Qwen2ForMaskedLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.model.input_norm = nn.Identity()
    apply_lab_logit_mask(model, LAB_TOKEN_MIN, LAB_TOKEN_MAX, config.vocab_size)
    if args.sparse_head:
        apply_sparse_lm_head(model, LAB_TOKEN_MIN, LAB_TOKEN_MAX)

    # -- Output dir + collision guard --
    n_gpus = max(torch.cuda.device_count(), 1)
    _so_default = SINGLEOBS_GPU_BATCH_SPARSE if args.sparse_head else SINGLEOBS_GPU_BATCH
    gpu_bs = (args.so_gpu_batch or _so_default) if single else GPU_BATCH_SIZE
    total_bs = SINGLEOBS_TOTAL_BATCH if single else TOTAL_BATCH_SIZE
    eval_bs = SINGLEOBS_EVAL_BATCH if single else GPU_BATCH_SIZE
    grad_accum = max(total_bs // (n_gpus * gpu_bs), 1)
    if args.dry_run:
        output_dir = f"{MODELS_BASE}/dry-run__mlm__{args.condition}{arm}"
    elif short:
        output_dir = f"{MODELS_BASE}/short__mlm__{'singleobs__' if single else ''}{args.condition}{arm}"
    else:
        output_dir = (f"{MODELS_BASE}/mlm__singleobs__{args.condition}{arm}__seed{args.seed}" if single
                      else f"{MODELS_BASE}/mlm__{args.condition}{arm}__seed{args.seed}")
    if (args.dry_run or short) and Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    elif not args.dry_run and args.resume_from_checkpoint is None and Path(output_dir).exists():
        raise FileExistsError(f"{output_dir} exists (rm -rf it to rebuild, or pass --resume-from-checkpoint)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = RESULTS_DIR / (f"progress_{'singleobs_' if single else ''}{args.condition}{arm}"
                             f"_seed{args.seed}{'_short' if short else ''}.log")
    common = dict(
        output_dir=output_dir, do_train=True, do_eval=True, eval_strategy="steps",
        per_device_train_batch_size=gpu_bs, per_device_eval_batch_size=eval_bs,
        eval_accumulation_steps=eval_bs, learning_rate=LEARNING_RATE,
        seed=args.seed, data_seed=args.seed, bf16=True, bf16_full_eval=True, tf32=True, report_to=[],
    )
    if single:
        # group_by_length batches rows of similar length, so a batch pads to about 23 tokens instead
        # of about 250. Measured per step against the tokens of the full-history run, fixed 320-token
        # padding costs 8.54x, random batches 1.99x and grouped batches 0.62x.
        # remove_unused_columns=False keeps `length` available. PadToBatchMax forwards only the four
        # model arrays, so nothing else reaches torch.tensor().
        common.update(group_by_length=True, length_column_name="length", remove_unused_columns=False)

    if args.dry_run:
        training_args = TrainingArguments(**common, eval_steps=1, gradient_accumulation_steps=1, max_steps=1,
                                          logging_strategy="steps", logging_steps=1, save_strategy="no",
                                          dataloader_num_workers=0)
        print(f"\n=== DRY-RUN: max_steps=1, 1 GPU, no FSDP/save, out={output_dir} ===")
        callbacks = []
    elif short:
        training_args = TrainingArguments(
            **common, eval_steps=args.max_steps, gradient_accumulation_steps=grad_accum, max_steps=args.max_steps,
            adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1, lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=10, save_strategy="no", dataloader_num_workers=4,
            fsdp=["shard_grad_op"])
        if local_rank == 0:
            sidecar.write_text(f"# SHORT mlm {args.condition} seed{args.seed} | {args.max_steps} steps (measurement)\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [sidecar_cb]
        print(f"\n=== SHORT measurement: {args.max_steps} steps, FSDP {n_gpus}gpu, no save/best/early-stop, out={output_dir} ===")
    else:
        assert args.eval_steps == args.save_steps, "full mode: --eval-steps must == --save-steps (best eval step needs a checkpoint)"
        training_args = TrainingArguments(
            **common, eval_steps=args.eval_steps, gradient_accumulation_steps=grad_accum, max_steps=args.max_steps,
            adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1, lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=20,
            # save_total_limit=None keeps every checkpoint, so the best one survives. load_best_model_at_end
            # is not used. It deadlocks under FSDP with save_total_limit at the end-of-run checkpoint load. It is
            # also not needed, because 15_eval_runner.py loads the best checkpoint from disk (named in the
            # manifest). SidecarCallback tracks the best step, and the run prunes to it afterwards.
            # EarlyStoppingCallback still needs metric_for_best_model.
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
            metric_for_best_model=BEST_METRIC, greater_is_better=True,
            dataloader_num_workers=4, fsdp=["shard_grad_op"])
        if local_rank == 0:
            sidecar.write_text(f"# mlm {args.condition} seed{args.seed} | train={len(ds_dict['train'])} "
                               f"val={len(ds_dict['val'])} | max_steps={args.max_steps} patience={EARLY_STOP_PATIENCE}\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE), sidecar_cb]
        print(f"\n=== TRAIN mlm{'/singleobs' if single else ''} {args.condition} "
              f"seed{args.seed} | b{total_bs} "
              f"(gpu{gpu_bs}x{n_gpus}gpu x accum{grad_accum}) lr{LEARNING_RATE} cosine | "
              f"best={BEST_METRIC} patience={EARLY_STOP_PATIENCE} | out={output_dir} ===")

    trainer = FastLengthTrainer(
        model=model, args=training_args, data_collator=collator,
        train_dataset=ds_dict["train"], eval_dataset=ds_dict["val"],
        compute_metrics=partial(compute_metrics, mimic_token_to_loinc=mimic_token_to_loinc),
        preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics,
                                              lab_token_min=LAB_TOKEN_MIN, lab_token_max=LAB_TOKEN_MAX),
        callbacks=callbacks)

    if args.dry_run:
        print("\n=== 1 train step + 1 eval step ===")
        trainer.train()
        print(f"  evaluate(): {trainer.evaluate()}")
        print("\n=== DRY-RUN PASSED ===")
        return

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wall = time.time() - t0

    if trainer.is_world_process_zero():
        # keep only the best checkpoint (tracked by SidecarCallback). save_steps == eval_steps guarantees
        # that checkpoint-{best_step} exists.
        best_step = sidecar_cb.best_step
        best_ckpt = f"{output_dir}/checkpoint-{best_step}" if best_step is not None else None
        if not short and best_ckpt and Path(best_ckpt).exists():
            for ck in Path(output_dir).glob("checkpoint-*"):
                if ck.name != f"checkpoint-{best_step}":
                    shutil.rmtree(ck, ignore_errors=True)
        evals = [h for h in trainer.state.log_history if BEST_METRIC in h]
        manifest = {
            "scope": args.scope,
            "condition": args.condition, "seed": args.seed,
            # which v3 training arm and which dataset. The clean- and perturbed-trained models
            # differ only by this, and the train-arm x test-arm 2x2 depends on telling them apart.
            # It is recorded beside the dataset path, so a mismatch is visible afterwards.
            "representation": representation(args.condition),
            # derived from `arm`, the same string every path above used, so the manifest matches
            # the directory it describes
            "train_arm": arm.lstrip("_") or None,
            "dataset_path": dataset_path, "mode": "short" if short else "full",
            # the pretrained checkpoint this fine-tune started from
            "base_checkpoint": CHECKPOINT,
            "output_dir": output_dir, "best_metric_name": BEST_METRIC,
            "best_metric": sidecar_cb.best_metric, "best_step": best_step, "best_model_checkpoint": best_ckpt,
            "global_step": trainer.state.global_step, "max_steps": args.max_steps,
            "early_stopped": (not short) and trainer.state.global_step < args.max_steps,
            "wall_clock_sec": round(wall, 1), "wall_clock_h": round(wall / 3600, 3),
            "sec_per_step": round(wall / max(trainer.state.global_step, 1), 3), "n_evals": len(evals),
            "eval_curve": [{"step": h.get("step"), BEST_METRIC: h.get(BEST_METRIC), "eval_loss": h.get("eval_loss")}
                           for h in evals],
        }
        # 15_eval_runner.resolve_best_checkpoint reads this exact name. The scope prefix keeps a
        # --scope singleobs run from overwriting the manifest of the full-history run, which records
        # its best checkpoint. The arm keeps the clean- and perturbed-trained models apart.
        mf = RESULTS_DIR / (f"manifest_{'singleobs_' if single else ''}{args.condition}{arm}"
                            f"_seed{args.seed}{'_short' if short else ''}.json")
        mf.write_text(json.dumps(manifest, indent=2))
        tag = "SHORT" if short else "DONE"
        print(f"\n=== {tag} | best {BEST_METRIC}={sidecar_cb.best_metric} @ step {best_step} "
              f"(ran to {trainer.state.global_step}/{args.max_steps}) | {wall/3600:.3f}h "
              f"({wall/max(trainer.state.global_step,1):.2f}s/step) | ckpt={best_ckpt} ===")


if __name__ == "__main__":
    main()
