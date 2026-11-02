"""QLoRA fine-tune of Llama-3.1-8B-Instruct to predict each lab observation's MIMIC lab code, scored as
LOINC (thesis Chapter 5 and Appendix D). The task is token classification: a 1623-class head
(LlamaForTokenClassification) over 4-bit Llama with LoRA adapters. The labels are stored in the
dataset (the MIMIC class id at each MASK position, -100 elsewhere).

The command line, splits, progress log, manifest and run modes are those of 12_mlm_train.py and
13_bcm_train.py, and the metric (compute_metrics, preprocess_logits_for_metrics) is BCM's: top-K
LOINC accuracy over the same 1623 classes.

The recipe:
  * 4-bit NF4 QLoRA (double quantization, bf16 compute) with LoRA r=16, alpha=32 on every linear
    projection, plus the `score` head, through prepare_model_for_kbit_training and get_peft_model.
  * Two learning rates: 2e-4 for the LoRA parameters, 1e-3 for the score head (TwoGroupTrainer).
    PagedAdamW8bit for a fresh run, AdamW8bit when resuming (bitsandbytes' paged optimizer state
    does not resume correctly). betas (0.9, 0.95), weight decay 0.1, cosine schedule, 5% warmup.
  * Batch 1 per GPU, effective 16 (2 GPUs x accumulation 8), at most 8000 steps, evaluation and
    checkpoint every 400, early stopping after 3 evaluations without improvement. Attention is
    flash_attention_2, with sdpa as the fallback.
  * Gradient checkpointing is enabled inside prepare_model_for_kbit_training. TrainingArguments
    does not set it.
  * The context is 8192 tokens (--max-len), as for FHIR-MLM and BCM. The data are prepared at 8192,
    so this is a collator setting, and at 8192 the keep-oldest truncation does nothing.
  * Attention stays causal: Llama is a causal decoder, so a masked position sees only the tokens
    before it.
  * Every checkpoint is kept (a QLoRA checkpoint is only the adapter and head, about 150 MB), so
    the best can be re-selected by any metric and the trajectory inspected. The best (step, metric)
    goes in the manifest. load_best_model_at_end is not used, to avoid a PEFT/transformers
    version-compatibility crash in _load_best.
  * set_seeds() runs before the model is built, so the LoRA and head initialisation are seeded.

Reads WORK_DIR/datasets/llama__{condition}{arm} (train and val, with the test split held out for
15_eval_runner.py) and the class-to-LOINC map (FHIR_MLM_TOKENIZER_DIR/codes_to_tokens.pkl,
UPSTREAM_DIR/seen_codes.csv). Writes checkpoints to WORK_DIR/models/llama__{condition}{arm}__seed{n}/,
and a progress log and a manifest to results/14_llama_train/. Only aggregate metrics are logged.

Modes:
  full  (default)
  short (--max-steps < --save-steps): a DDP measurement run with no saving, best-checkpoint
        tracking or early stopping, and one evaluation, to probe throughput and memory at
        --max-len before the multi-day full run
  dry   (--dry-run): one GPU, one train and one eval step on four rows, no DDP and no saving.
        Run it with plain python and CUDA_VISIBLE_DEVICES=0

  accelerate launch --num_processes=2 pipeline/14_llama_train.py --condition para_xsite --train-arm pert --seed 0
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reclaims fragmentation between train and eval. Evaluation at 8192 can otherwise run out of memory
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import json
import os
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import datasets
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (MLM_CONDITIONS, representation,  # noqa: E402
                        resolve_arm_suffix)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_callback import EarlyStoppingCallback

# -- class->LOINC mapping (common/fhir_data_utils.py; same path setup as 11_llama_dataset.py) --
for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from fhir_data_utils import build_lab_code_mappings, N_LAB_CLASSES  # noqa: E402

# -- Paths --
DATASET_BASE = str(locations.WORK / "datasets")   # llama__{condition}
MODELS_BASE = str(locations.WORK / "models")      # llama__{condition}__seed{n}
RESULTS_DIR = locations.REPO / "results/14_llama_train"
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")

# -- Model / task --
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_LABELS = N_LAB_CLASSES   # 1623
MAX_LEN = 8192               # the same context as FHIR-MLM and BCM; --max-len overrides

# -- QLoRA --
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_MODULES_TO_SAVE = ["score"]

# -- Optim / schedule --
GPU_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2         # evaluation at 8192 tokens runs out of memory at 4. Batching only, so the metrics are unaffected
TOTAL_BATCH_SIZE = 16
LEARNING_RATE_LORA = 2e-4
LEARNING_RATE_HEAD = 1e-3
WARMUP_RATIO = 0.05
MAX_STEPS = 8000
EVAL_STEPS = 400
SAVE_STEPS = 400            # == EVAL_STEPS, so the best step is always a saved checkpoint
EARLY_STOP_PATIENCE = 3
BEST_METRIC = "eval_loinc_top1"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95          # as for FHIR-MLM (BCM uses 0.999)
WEIGHT_DECAY = 0.1         # as for FHIR-MLM (BCM uses 0.01)

TOP_KS = (1, 3, 5, 10)
TOP_K_MAX = max(TOP_KS)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=MLM_CONDITIONS, required=True,
                   help="see pipeline/conditions.py for the slug grammar")
    p.add_argument("--train-arm", choices=["pert", "clean"], default=None,
                   help="v3 (xsite) only, and required there: which training arm. 'pert' = "
                        "trained on perturbed data, 'clean' = the v3 representation with no "
                        "perturbation. No default.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="single GPU, 1 train + 1 eval step on a 4-row slice; no DDP/save. "
                        "Launch: CUDA_VISIBLE_DEVICES=0 python 14_llama_train.py --condition para --dry-run")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="train-step cap. If < save-steps it's a short measurement run (no save/best/early-stop, "
                        "one eval at the end), a throughput and memory probe at --max-len before the full run.")
    p.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="checkpoint interval (keep == eval interval)")
    p.add_argument("--eval-steps", type=int, default=EVAL_STEPS, help="eval interval")
    p.add_argument("--max-len", type=int, default=MAX_LEN,
                   help="train-time context length (collator truncates keep-oldest to this). Default 8192.")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args()


def set_seeds(seed):
    """Seed every RNG before the model is built, so the LoRA and score-head initialisation are
    reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Pure, unit-testable logic
# ==============================================================================
def build_class_to_loinc_array():
    """Return a (1623,) object array mapping class index -> LOINC string. An unmapped class gets a
    unique sentinel '?{i}', so an unmapped true label can only match a prediction of the same class.
    As in 13_bcm_train.py."""
    _, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    return np.array([class_to_loinc.get(i, f"?{i}") for i in range(N_LAB_CLASSES)], dtype=object)


def preprocess_logits_for_metrics(logits, labels):
    """Reduce per-batch logits (B,S,1623) to the top-K class indices (B,S,K) on the GPU, so that
    evaluation accumulates only small integer indices. As in 13_bcm_train.py."""
    return logits.topk(TOP_K_MAX, dim=-1).indices


def compute_metrics(eval_pred, class_to_loinc_arr):
    """Top-1/3/5/10 LOINC accuracy on the labeled positions (labels != -100). Predicted and true class
    indices map to LOINC through class_to_loinc_arr. An unmapped true label keeps its unique '?{i}'
    sentinel. As in 13_bcm_train.py."""
    preds = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    if isinstance(preds, tuple):
        preds = preds[0]
    preds_flat = preds.reshape(-1, preds.shape[-1])   # (N*S, K)
    labels_flat = labels.reshape(-1)                   # (N*S,)
    active = labels_flat != -100
    preds_flat = preds_flat[active]
    labels_flat = labels_flat[active]
    if len(labels_flat) == 0:
        return {f"loinc_top{k}": 0.0 for k in TOP_KS} | {"n_active_positions": 0}
    pred_loincs = class_to_loinc_arr[preds_flat]       # (n, K)
    true_loincs = class_to_loinc_arr[labels_flat]      # (n,)
    metrics = {"n_active_positions": int(len(labels_flat))}
    for k in TOP_KS:
        topk_match = (pred_loincs[:, :k] == true_loincs[:, None]).any(axis=1)
        metrics[f"loinc_top{k}"] = float(topk_match.mean())
    return metrics


def truncate_features_keep_oldest(features, max_len):
    """Keep the first max_len tokens of input_ids, attention_mask and labels in each feature, dropping
    the newest. The FHIR-MLM tokenizer truncates in the same direction. Returns new dicts. A no-op
    when max_len is at least the prepared length (8192). Other keys are kept."""
    out = []
    for f in features:
        if len(f["input_ids"]) > max_len:
            f = {**f,
                 "input_ids": f["input_ids"][:max_len],
                 "attention_mask": f["attention_mask"][:max_len],
                 "labels": f["labels"][:max_len]}
        out.append(f)
    return out


def split_param_groups(named_parameters):
    """Split the trainable parameters into (lora_params, head_params) by name: 'score' goes to the
    head, everything else to LoRA. `named_parameters` is an iterable of (name, param)."""
    lora_params, head_params = [], []
    for n, p in named_parameters:
        if not p.requires_grad:
            continue
        (head_params if "score" in n else lora_params).append(p)
    return lora_params, head_params


# ==============================================================================
# QLoRA model build
# ==============================================================================
def build_model_and_tokenizer(local_rank):
    """4-bit NF4 Llama + LoRA (all linear projections) + `score` head. Each rank loads a quantized
    copy, pinned to its GPU with device_map={"": local_rank}. Gradient checkpointing is enabled inside
    prepare_model_for_kbit_training."""
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils import prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    torch.cuda.set_device(local_rank)
    device_map = {"": local_rank}

    model = None
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS, quantization_config=bnb_config,
                attn_implementation=attn, torch_dtype=torch.bfloat16, device_map=device_map)
            print(f"  model loaded | attn_implementation={attn}")
            break
        except Exception as e:
            if attn == "sdpa":
                raise
            print(f"  flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, modules_to_save=LORA_MODULES_TO_SAVE,
        task_type=TaskType.TOKEN_CLS, bias="none")
    model = get_peft_model(model, lora_config)
    return model, tokenizer


class TwoGroupTrainer(Trainer):
    """LoRA parameters at LEARNING_RATE_LORA, the `score` head at LEARNING_RATE_HEAD. PagedAdamW8bit for a
    fresh run and the non-paged AdamW8bit on resume, because the paged optimizer state of bitsandbytes
    is corrupted when a run resumes."""
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        import bitsandbytes as bnb
        lora_params, head_params = split_param_groups(self.model.named_parameters())
        if self.args.local_rank in (-1, 0):
            print(f"  optim groups: {sum(p.numel() for p in lora_params):,} LoRA @ lr={LEARNING_RATE_LORA}, "
                  f"{sum(p.numel() for p in head_params):,} head @ lr={LEARNING_RATE_HEAD}")
        opt_cls = (bnb.optim.AdamW8bit
                   if getattr(self.args, "resume_from_checkpoint", None)
                   else bnb.optim.PagedAdamW8bit)
        if self.args.local_rank in (-1, 0):
            print(f"  optimizer: {opt_cls.__name__}")
        self.optimizer = opt_cls(
            [{"params": lora_params, "lr": LEARNING_RATE_LORA},
             {"params": head_params, "lr": LEARNING_RATE_HEAD}],
            betas=(ADAM_BETA1, ADAM_BETA2), weight_decay=WEIGHT_DECAY)
        return self.optimizer


# ==============================================================================
# Sidecar progress log (rank 0; the same format as 12_mlm_train.py and 13_bcm_train.py)
# ==============================================================================
class SidecarCallback(TrainerCallback):
    """Append the LOINC metrics of each evaluation to a sidecar log (rank 0 only, aggregate metrics). The
    best (step, metric) is derived after training from trainer.state.log_history, which survives
    --resume-from-checkpoint. A new callback instance would see only the evaluations after the resume."""
    def __init__(self, path):
        self.path = path

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if state.is_world_process_zero and metrics:
            keep = {k: v for k, v in metrics.items() if "loinc" in k or "active" in k or k == "eval_loss"}
            with open(self.path, "a") as f:
                f.write(f"step {state.global_step}: " + " ".join(f"{k}={v}" for k, v in keep.items()) + "\n")


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()
    # every path below (dataset, model dir, sidecar, manifest) carries the arm, because the two v3
    # training arms differ only by this flag
    arm = resolve_arm_suffix(args.condition, args.train_arm, "14_llama_train")
    set_seeds(args.seed)   # before the model is built: reproducible LoRA and head initialisation
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    short = (not args.dry_run) and args.max_steps < args.save_steps

    class_to_loinc_arr = build_class_to_loinc_array()
    print(f"class->LOINC array: {len(class_to_loinc_arr)} classes "
          f"({sum(1 for x in class_to_loinc_arr if not str(x).startswith('?'))} LOINC-mapped)")

    # -- Dataset (llama__{condition}{arm}: train and val, without uuid) --
    dataset_path = f"{DATASET_BASE}/llama__{args.condition}{arm}"
    print(f"\nLoading dataset: {dataset_path}")
    ds_dict = datasets.load_from_disk(dataset_path)
    if not isinstance(ds_dict, datasets.DatasetDict) or "train" not in ds_dict or "val" not in ds_dict:
        raise TypeError(f"Expected DatasetDict with train+val at {dataset_path}")
    print(f"  train={len(ds_dict['train'])}  val={len(ds_dict['val'])}  (test held out for eval block)")
    keep_cols = ["input_ids", "attention_mask", "labels"]   # labels precomputed on disk; drop uuid
    ds_dict = datasets.DatasetDict({k: ds_dict[k].select_columns(keep_cols) for k in ("train", "val")})

    if args.dry_run:
        ds_dict = datasets.DatasetDict({
            "train": ds_dict["train"].select(range(min(4, len(ds_dict["train"])))),
            "val": ds_dict["val"].select(range(min(4, len(ds_dict["val"])))),
        })
        print(f"  [DRY-RUN] sliced to train={len(ds_dict['train'])} val={len(ds_dict['val'])}")

    # -- Output dir and collision guard, before the slow 4-bit model load, to fail fast --
    n_gpus = max(torch.cuda.device_count(), 1)
    grad_accum = max(TOTAL_BATCH_SIZE // (n_gpus * GPU_BATCH_SIZE), 1)
    if not args.dry_run:
        assert grad_accum * n_gpus * GPU_BATCH_SIZE == TOTAL_BATCH_SIZE, \
            f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} not divisible by n_gpus={n_gpus} x GPU_BATCH_SIZE={GPU_BATCH_SIZE}"
    if args.dry_run:
        output_dir = f"{MODELS_BASE}/dry-run__llama__{args.condition}{arm}"
    elif short:
        output_dir = f"{MODELS_BASE}/short__llama__{args.condition}{arm}"
    else:
        output_dir = f"{MODELS_BASE}/llama__{args.condition}{arm}__seed{args.seed}"
    if (args.dry_run or short) and Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    elif not args.dry_run and args.resume_from_checkpoint is None and Path(output_dir).exists():
        raise FileExistsError(f"{output_dir} exists (rm -rf it to rebuild, or pass --resume-from-checkpoint)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = RESULTS_DIR / f"progress_{args.condition}{arm}_seed{args.seed}{'_short' if short else ''}.log"

    # -- Model + tokenizer (QLoRA) --
    print("\nBuilding model (4-bit NF4 QLoRA)…")
    model, tokenizer = build_model_and_tokenizer(local_rank)
    if local_rank == 0:
        model.print_trainable_parameters()

    # -- Collator: keep-oldest truncate to --max-len, then stock token-cls padding --
    base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                       label_pad_token_id=-100)
    def collator(features):
        return base_collator(truncate_features_keep_oldest(features, args.max_len))

    # -- TrainingArguments. gradient_checkpointing is not set here, because prepare_model_for_kbit_training
    #    already enables it and setting it here too would enable it twice. --
    common = dict(
        output_dir=output_dir, do_train=True, do_eval=True, eval_strategy="steps",
        eval_accumulation_steps=1, learning_rate=LEARNING_RATE_LORA, seed=args.seed, data_seed=args.seed,
        bf16=True, bf16_full_eval=True, tf32=True, report_to=[], label_names=["labels"],
        remove_unused_columns=True,
    )

    if args.dry_run:
        training_args = TrainingArguments(
            **common, eval_steps=1, max_steps=1, gradient_accumulation_steps=1,
            per_device_train_batch_size=GPU_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            logging_strategy="steps", logging_steps=1, save_strategy="no", dataloader_num_workers=0)
        print(f"\n=== DRY-RUN: max_steps=1, 1 GPU, QLoRA bs={GPU_BATCH_SIZE}, max_len={args.max_len}, "
              f"no DDP/save, out={output_dir} ===")
        callbacks = []
    elif short:
        training_args = TrainingArguments(
            **common, eval_steps=args.max_steps, max_steps=args.max_steps, gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=GPU_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=5, save_strategy="no",
            dataloader_num_workers=2, ddp_find_unused_parameters=False)
        if local_rank == 0:
            sidecar.write_text(f"# SHORT llama {args.condition} seed{args.seed} | {args.max_steps} steps "
                               f"max_len={args.max_len} (measurement)\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [sidecar_cb]
        print(f"\n=== SHORT measurement: {args.max_steps} steps, DDP {n_gpus}gpu bs{GPU_BATCH_SIZE} x accum{grad_accum}, "
              f"max_len={args.max_len}, no save/best, out={output_dir} ===")
    else:
        assert args.eval_steps == args.save_steps, "full mode: --eval-steps must == --save-steps (best step needs a checkpoint)"
        training_args = TrainingArguments(
            **common, eval_steps=args.eval_steps, max_steps=args.max_steps, gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=GPU_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=20,
            # every checkpoint is kept (save_total_limit=None), because a QLoRA checkpoint is only the adapter
            # and head. load_best_model_at_end is not used, to avoid a PEFT/transformers version-compatibility
            # crash in _load_best. The best (step, metric) goes in the manifest.
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
            metric_for_best_model=BEST_METRIC, greater_is_better=True,
            dataloader_num_workers=2, ddp_find_unused_parameters=False)
        if local_rank == 0 and not args.resume_from_checkpoint:   # don't clobber the log on resume
            sidecar.write_text(f"# llama {args.condition} seed{args.seed} | train={len(ds_dict['train'])} "
                               f"val={len(ds_dict['val'])} | max_steps={args.max_steps} max_len={args.max_len} "
                               f"patience={EARLY_STOP_PATIENCE}\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE), sidecar_cb]
        print(f"\n=== TRAIN llama {args.condition} seed{args.seed} | b{TOTAL_BATCH_SIZE} "
              f"(bs{GPU_BATCH_SIZE}x{n_gpus}gpu x accum{grad_accum}) LoRA={LEARNING_RATE_LORA} head={LEARNING_RATE_HEAD} "
              f"cosine β2={ADAM_BETA2} wd={WEIGHT_DECAY} max_len={args.max_len} | best={BEST_METRIC} "
              f"patience={EARLY_STOP_PATIENCE} | out={output_dir} ===")

    # Make the resume path visible to TwoGroupTrainer.create_optimizer, which uses AdamW8bit instead of PagedAdamW8bit on resume
    if not args.dry_run and args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = args.resume_from_checkpoint

    trainer = TwoGroupTrainer(
        model=model, args=training_args, data_collator=collator,
        train_dataset=ds_dict["train"], eval_dataset=ds_dict["val"],
        compute_metrics=partial(compute_metrics, class_to_loinc_arr=class_to_loinc_arr),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        processing_class=tokenizer, callbacks=callbacks)

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
        # the best comes from log_history, which survives --resume-from-checkpoint. A new SidecarCallback
        # would see only the evaluations after the resume. Every checkpoint is kept.
        evals = [h for h in trainer.state.log_history if BEST_METRIC in h]
        if evals:
            best_h = max(evals, key=lambda h: h[BEST_METRIC])
            best_metric, best_step = best_h[BEST_METRIC], best_h.get("step")
        else:
            best_metric, best_step = None, None
        best_ckpt = f"{output_dir}/checkpoint-{best_step}" if (not short and best_step is not None) else None
        if best_ckpt and not Path(best_ckpt).exists():
            print(f"  WARN: best checkpoint {best_ckpt} missing on disk (best step had no save?)")
        manifest = {
            "model": "llama", "condition": args.condition, "seed": args.seed, "mode": "short" if short else "full",
            # which v3 training arm and which dataset. The two arms differ only by this, and the
            # train-arm x test-arm 2x2 depends on telling them apart. Derived from `arm`, like every
            # path above.
            "representation": representation(args.condition),
            "train_arm": arm.lstrip("_") or None,
            "dataset_path": dataset_path,
            "output_dir": output_dir, "max_len": args.max_len, "best_metric_name": BEST_METRIC,
            "best_metric": best_metric, "best_step": best_step, "best_model_checkpoint": best_ckpt,
            "checkpoints_kept": sorted(int(p.name.split("-")[1]) for p in Path(output_dir).glob("checkpoint-*")),
            "global_step": trainer.state.global_step, "max_steps": args.max_steps,
            "n_gpus": n_gpus, "grad_accum": training_args.gradient_accumulation_steps,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGET_MODULES},
            "lr_lora": LEARNING_RATE_LORA, "lr_head": LEARNING_RATE_HEAD,
            "early_stopped": (not short) and trainer.state.global_step < args.max_steps,
            "wall_clock_sec": round(wall, 1), "wall_clock_h": round(wall / 3600, 3),
            "sec_per_step": round(wall / max(trainer.state.global_step, 1), 3), "n_evals": len(evals),
            "eval_curve": [{"step": h.get("step"), BEST_METRIC: h.get(BEST_METRIC), "eval_loss": h.get("eval_loss")}
                           for h in evals],
        }
        mf = RESULTS_DIR / f"manifest_{args.condition}{arm}_seed{args.seed}{'_short' if short else ''}.json"
        mf.write_text(json.dumps(manifest, indent=2))
        tag = "SHORT" if short else "DONE"
        print(f"\n=== {tag} | best {BEST_METRIC}={best_metric} @ step {best_step} "
              f"(ran to {trainer.state.global_step}/{args.max_steps}) | {wall/3600:.3f}h "
              f"({wall/max(trainer.state.global_step,1):.2f}s/step) | ckpt={best_ckpt} ===")


if __name__ == "__main__":
    main()
