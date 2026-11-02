"""Fine-tune the large BioClinical ModernBERT (BCM, about 396M parameters) on the BCM datasets for LOINC
lab-code prediction (thesis Chapter 5 and Appendix D). Token classification: the labels are
precomputed on disk, for a 1623-class head.

Uses the same training harness as pipeline/12_mlm_train.py (CLI, three splits, no wandb, sidecar log
and manifest, early stopping, checkpoint policy). The BCM-specific parts are
AutoModelForTokenClassification (num_labels=1623), a plain padding collator and a class-index top-K
LOINC metric. The collator does no dynamic masking, because the labels are precomputed. Training uses
gradient checkpointing with flash-attn (sdpa fallback). BCM's optimizer settings differ from
FHIR-MLM's. It uses the Hugging Face AdamW default (β2=0.999) with BERT-style weight decay (0.01),
where FHIR-MLM uses β2=0.95 and wd=0.1.

BCM trains with DDP, which also avoids the end-of-run deadlocks of FSDP. eval_steps == save_steps ==
400, and max_steps is 8000 with early stopping (patience 3). The seeds are 0–4, and the seed sets
both seed and data_seed. All checkpoints are kept during training (save_total_limit=None, no
load_best_model_at_end). SidecarCallback tracks the best (step, metric), and the other checkpoints
are pruned after training.

Reads WORK_DIR/datasets/bcm__<condition><arm> (train and val, with test held out for evaluation).
Writes the model to WORK_DIR/models/bcm__<condition><arm>__seed<n>. Writes a rank-0 progress log and
a JSON manifest to results/13_bcm_train/ (aggregate metrics only).

Modes:
  full  (default): torchrun --nproc_per_node=2 pipeline/13_bcm_train.py --condition para_xsite --train-arm pert --seed 0
  short (--max-steps < save-steps): DDP measurement run, no save/best/early-stop, one eval at the end.
  dry   (--dry-run): single GPU, 1 train + 1 eval step on a 4-row slice (GC on, bs=2), no DDP/save.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import json
import os
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
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_callback import EarlyStoppingCallback

# -- helper for the class->LOINC mapping (same path setup as 10_bcm_dataset.py) --
for _p in (str(locations.COMMON), str(locations.THIRD_PARTY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from fhir_data_utils import build_lab_code_mappings, N_LAB_CLASSES  # noqa: E402

# -- Paths --
DATASET_BASE = str(locations.WORK / "datasets")   # bcm__{condition}
MODELS_BASE = str(locations.WORK / "models")      # bcm__{condition}__seed{n}
RESULTS_DIR = locations.REPO / "results/13_bcm_train"
CODES_TO_TOKENS_PATH = f"{locations.TOKENIZER}/codes_to_tokens.pkl"
SEEN_CODES_PATH = str(locations.UPSTREAM / "seen_codes.csv")

# -- Model / task --
MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-large"
NUM_LABELS = N_LAB_CLASSES   # 1623
MAX_LEN = 8192

# -- Hyperparameters (optimizer settings kept model-specific) --
GPU_BATCH_SIZE = 8          # per-device train batch (a short run checks it fits in 24 GB)
EVAL_BATCH_SIZE = 8
TOTAL_BATCH_SIZE = 16
MAX_STEPS = 8000
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.05
EVAL_STEPS = 400
SAVE_STEPS = 400            # == EVAL_STEPS (harness prune needs best step to be a saved checkpoint)
EARLY_STOP_PATIENCE = 3
BEST_METRIC = "eval_loinc_top1"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999         # the HF AdamW default (FHIR-MLM uses 0.95)
WEIGHT_DECAY = 0.01        # BERT-style weight decay (FHIR-MLM uses 0.1)

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
                   help="single GPU, 1 train + 1 eval step on a 4-row slice (GC on, bs=2); no DDP/save")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="train-step cap. If < save-steps it's a short measurement run (no save/best/early-stop, "
                        "one eval at the end) for throughput + memory checks before the full run.")
    p.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="checkpoint interval (keep == eval interval)")
    p.add_argument("--eval-steps", type=int, default=EVAL_STEPS, help="eval interval")
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args()


# ==============================================================================
# Pure, unit-testable logic (on class indices)
# ==============================================================================
def build_class_to_loinc_array():
    """Return a (1623,) object array mapping class index -> LOINC string. Unmapped classes get a unique
    sentinel '?{i}', so an unmapped true label can only match a prediction of the same class."""
    _, class_to_loinc = build_lab_code_mappings(CODES_TO_TOKENS_PATH, SEEN_CODES_PATH)
    return np.array([class_to_loinc.get(i, f"?{i}") for i in range(N_LAB_CLASSES)], dtype=object)


def preprocess_logits_for_metrics(logits, labels):
    """Reduce per-batch logits (B,S,1623) to top-K class indices (B,S,K) on the GPU, so HF accumulates
    only small int indices at eval time. Returns the indices."""
    return logits.topk(TOP_K_MAX, dim=-1).indices


def compute_metrics(eval_pred, class_to_loinc_arr):
    """Top-1/3/5/10 LOINC accuracy on labeled positions (labels != -100). Predicted and true class indices
    map to LOINC via class_to_loinc_arr. Unmapped true labels keep their unique '?{i}' sentinel and stay
    in the denominator."""
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


def build_model(tokenizer):
    """Load BioClinical ModernBERT for 1623-way token classification. Try flash-attn-2, fall back to sdpa
    if flash-attn isn't installed. reference_compile=False is required with GC."""
    last_err = None
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS, attn_implementation=attn,
                torch_dtype=torch.bfloat16, reference_compile=False)
            print(f"  model loaded | attn_implementation={attn} | params={sum(p.numel() for p in model.parameters()):,}")
            model.config.pad_token_id = tokenizer.pad_token_id
            return model
        except Exception as e:  # flash-attn usually raises at from_pretrained if not installed
            last_err = e
            if attn == "sdpa":
                raise
            print(f"  flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to sdpa")
    raise last_err


# ==============================================================================
# Sidecar progress logging + best-step tracking (rank-0; identical shape to 12_mlm_train.py)
# ==============================================================================
class SidecarCallback(TrainerCallback):
    """Append per-eval LOINC metrics to a sidecar log that a poller can read (rank 0 only, aggregate
    metrics, no patient data). Also tracks the best (step, metric), so that after training the run can
    prune to the best checkpoint without load_best_model_at_end."""
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
    arm = resolve_arm_suffix(args.condition, args.train_arm, "13_bcm_train")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    short = (not args.dry_run) and args.max_steps < args.save_steps

    class_to_loinc_arr = build_class_to_loinc_array()
    print(f"class->LOINC array: {len(class_to_loinc_arr)} classes "
          f"({sum(1 for x in class_to_loinc_arr if not str(x).startswith('?'))} LOINC-mapped)")

    # -- Dataset (bcm__{condition}; use train+val) --
    dataset_path = f"{DATASET_BASE}/bcm__{args.condition}{arm}"
    print(f"\nLoading dataset: {dataset_path}")
    ds_dict = datasets.load_from_disk(dataset_path)
    if not isinstance(ds_dict, datasets.DatasetDict) or "train" not in ds_dict or "val" not in ds_dict:
        raise TypeError(f"Expected DatasetDict with train+val at {dataset_path}")
    print(f"  train={len(ds_dict['train'])}  val={len(ds_dict['val'])}  (test held out for eval block)")
    keep_cols = ["input_ids", "attention_mask", "labels"]   # BCM labels are precomputed on disk; drop uuid
    ds_dict = datasets.DatasetDict({k: ds_dict[k].select_columns(keep_cols) for k in ("train", "val")})

    if args.dry_run:
        ds_dict = datasets.DatasetDict({
            "train": ds_dict["train"].select(range(min(4, len(ds_dict["train"])))),
            "val": ds_dict["val"].select(range(min(4, len(ds_dict["val"])))),
        })
        print(f"  [DRY-RUN] sliced to train={len(ds_dict['train'])} val={len(ds_dict['val'])}")

    # -- Tokenizer + collator (plain padding, no masking, because labels are precomputed) --
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest", label_pad_token_id=-100)

    # -- Model --
    print("\nBuilding model…")
    set_seed(args.seed)                    # the new classification head's initial weights follow the seed
    model = build_model(tokenizer)

    # -- Output dir + collision guard --
    n_gpus = max(torch.cuda.device_count(), 1)
    if args.dry_run:
        output_dir = f"{MODELS_BASE}/dry-run__bcm__{args.condition}{arm}"
    elif short:
        output_dir = f"{MODELS_BASE}/short__bcm__{args.condition}{arm}"
    else:
        output_dir = f"{MODELS_BASE}/bcm__{args.condition}{arm}__seed{args.seed}"
    if (args.dry_run or short) and Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    elif not args.dry_run and args.resume_from_checkpoint is None and Path(output_dir).exists():
        raise FileExistsError(f"{output_dir} exists (rm -rf it to rebuild, or pass --resume-from-checkpoint)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = RESULTS_DIR / f"progress_{args.condition}{arm}_seed{args.seed}{'_short' if short else ''}.log"

    # -- TrainingArguments (DDP, GC on, token classification) --
    common = dict(
        output_dir=output_dir, do_train=True, do_eval=True, eval_strategy="steps",
        eval_accumulation_steps=1, learning_rate=LEARNING_RATE, seed=args.seed, data_seed=args.seed,
        bf16=True, bf16_full_eval=True, tf32=True, report_to=[], label_names=["labels"],
        remove_unused_columns=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if args.dry_run:
        training_args = TrainingArguments(
            **common, eval_steps=1, max_steps=1, gradient_accumulation_steps=1,
            per_device_train_batch_size=2, per_device_eval_batch_size=2,
            logging_strategy="steps", logging_steps=1, save_strategy="no", dataloader_num_workers=0)
        print(f"\n=== DRY-RUN: max_steps=1, 1 GPU, GC on bs=2, no DDP/save, out={output_dir} ===")
        callbacks = []
    elif short:
        grad_accum = max(TOTAL_BATCH_SIZE // (n_gpus * GPU_BATCH_SIZE), 1)
        training_args = TrainingArguments(
            **common, eval_steps=args.max_steps, max_steps=args.max_steps, gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=GPU_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=10, save_strategy="no",
            dataloader_num_workers=2, ddp_find_unused_parameters=False)
        if local_rank == 0:
            sidecar.write_text(f"# SHORT bcm {args.condition} seed{args.seed} | {args.max_steps} steps (measurement)\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [sidecar_cb]
        print(f"\n=== SHORT measurement: {args.max_steps} steps, DDP {n_gpus}gpu bs{GPU_BATCH_SIZE}, no save/best, out={output_dir} ===")
    else:
        assert args.eval_steps == args.save_steps, "full mode: --eval-steps must == --save-steps (best step needs a checkpoint)"
        grad_accum = max(TOTAL_BATCH_SIZE // (n_gpus * GPU_BATCH_SIZE), 1)
        training_args = TrainingArguments(
            **common, eval_steps=args.eval_steps, max_steps=args.max_steps, gradient_accumulation_steps=grad_accum,
            per_device_train_batch_size=GPU_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
            logging_strategy="steps", logging_steps=20,
            # save_total_limit=None and no load_best_model_at_end. All checkpoints are kept, the sidecar tracks
            # the best, and the run prunes to the best after training. This avoids the checkpoint-rotation and
            # FSDP-load deadlocks seen with FHIR-MLM. DDP is not expected to deadlock, so this is a second
            # safeguard.
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
            metric_for_best_model=BEST_METRIC, greater_is_better=True,
            dataloader_num_workers=2, ddp_find_unused_parameters=False)
        if local_rank == 0:
            sidecar.write_text(f"# bcm {args.condition} seed{args.seed} | train={len(ds_dict['train'])} "
                               f"val={len(ds_dict['val'])} | max_steps={args.max_steps} patience={EARLY_STOP_PATIENCE}\n")
        sidecar_cb = SidecarCallback(str(sidecar))
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE), sidecar_cb]
        print(f"\n=== TRAIN bcm {args.condition} seed{args.seed} | b{TOTAL_BATCH_SIZE} "
              f"(bs{GPU_BATCH_SIZE}x{n_gpus}gpu x accum{grad_accum}) lr{LEARNING_RATE} cosine β2={ADAM_BETA2} wd={WEIGHT_DECAY} | "
              f"best={BEST_METRIC} patience={EARLY_STOP_PATIENCE} | out={output_dir} ===")

    trainer = Trainer(
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
        best_step = sidecar_cb.best_step
        best_ckpt = f"{output_dir}/checkpoint-{best_step}" if best_step is not None else None
        if not short and best_ckpt and Path(best_ckpt).exists():
            for ck in Path(output_dir).glob("checkpoint-*"):
                if ck.name != f"checkpoint-{best_step}":
                    shutil.rmtree(ck, ignore_errors=True)
        evals = [h for h in trainer.state.log_history if BEST_METRIC in h]
        manifest = {
            "model": "bcm", "condition": args.condition, "seed": args.seed, "mode": "short" if short else "full",
            # provenance: which v3 training arm, and which dataset. The two arms differ only by
            # this, and the C and D cells of the 2x2 depend on telling them apart. It is derived
            # from `arm`, which every path above also uses.
            "representation": representation(args.condition),
            "train_arm": arm.lstrip("_") or None,
            "dataset_path": dataset_path,
            "output_dir": output_dir, "best_metric_name": BEST_METRIC,
            "best_metric": sidecar_cb.best_metric, "best_step": best_step, "best_model_checkpoint": best_ckpt,
            "global_step": trainer.state.global_step, "max_steps": args.max_steps,
            "early_stopped": (not short) and trainer.state.global_step < args.max_steps,
            "wall_clock_sec": round(wall, 1), "wall_clock_h": round(wall / 3600, 3),
            "sec_per_step": round(wall / max(trainer.state.global_step, 1), 3), "n_evals": len(evals),
            "eval_curve": [{"step": h.get("step"), BEST_METRIC: h.get(BEST_METRIC), "eval_loss": h.get("eval_loss")}
                           for h in evals],
        }
        mf = RESULTS_DIR / f"manifest_{args.condition}{arm}_seed{args.seed}{'_short' if short else ''}.json"
        mf.write_text(json.dumps(manifest, indent=2))
        tag = "SHORT" if short else "DONE"
        print(f"\n=== {tag} | best {BEST_METRIC}={sidecar_cb.best_metric} @ step {best_step} "
              f"(ran to {trainer.state.global_step}/{args.max_steps}) | {wall/3600:.3f}h "
              f"({wall/max(trainer.state.global_step,1):.2f}s/step) | ckpt={best_ckpt} ===")


if __name__ == "__main__":
    main()
