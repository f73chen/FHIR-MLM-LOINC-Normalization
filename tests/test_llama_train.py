"""Unit tests for the pure logic of pipeline/14_llama_train.py, on synthetic fixtures and the real
class->LOINC mapping. Needs the deep-models environment (torch, transformers, datasets) but no GPU and
no patient data. bitsandbytes and peft are not needed either, because the model builder imports them
lazily. Covers the metric code shared with BCM, the two Llama-specific pure helpers (keep-oldest
truncation and the two-learning-rate parameter split) and a guard on the recipe constants."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "llama_train", Path(__file__).resolve().parents[1] / "pipeline" / "14_llama_train.py")
lt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lt)


# -- shared metric code (identical to BCM; same asserts) ----------------------
def test_build_class_to_loinc_array_real():
    """(1623,) class->LOINC array from the real code files. About 828 classes map to LOINC, and the
    rest get unique '?{i}' sentinels."""
    arr = lt.build_class_to_loinc_array()
    assert arr.shape == (lt.N_LAB_CLASSES,) == (1623,)
    mapped = sum(1 for x in arr if not str(x).startswith("?"))
    assert 700 < mapped < 900                      # ~828 LOINC-mappable classes
    unmapped = [str(x) for x in arr if str(x).startswith("?")]
    assert len(unmapped) == len(set(unmapped))     # unique sentinels, so no spurious cross-class match


def test_preprocess_logits_topk_over_class_dim():
    """Top-K reduction returns the K largest class indices directly (logits already 1623-wide)."""
    logits = torch.full((1, 1, 20), -5.0)
    logits[0, 0, 12] = 9.0     # largest class
    logits[0, 0, 4] = 4.0      # 2nd
    idx = lt.preprocess_logits_for_metrics(logits, torch.zeros(1, 1, dtype=torch.long))
    assert idx.shape == (1, 1, lt.TOP_K_MAX)
    assert idx[0, 0, 0].item() == 12
    assert idx[0, 0, 1].item() == 4


def test_compute_metrics_topk_accuracy():
    """Top-K LOINC accuracy: one top-1 hit, one top-3-only hit, one ignored (-100)."""
    arr = np.array(["A", "B", "C", "?3", "?4"], dtype=object)
    preds = np.zeros((1, 3, lt.TOP_K_MAX), dtype=np.int64)      # default class 0 ("A")
    preds[0, 1, 1] = 1                                          # pos1 has class 1 ("B") at rank 2
    labels = np.array([[0, 1, -100]], dtype=np.int64)          # pos0 "A", pos1 "B", pos2 ignored
    m = lt.compute_metrics((preds, labels), arr)
    assert m["n_active_positions"] == 2
    assert m["loinc_top1"] == 0.5    # pos0 A@1 hit; pos1 A@1 != B miss
    assert m["loinc_top3"] == 1.0    # pos1 B@2 within top3


def test_compute_metrics_unmapped_and_empty():
    arr = np.array(["A", "B", "?2", "?3"], dtype=object)
    p_hit = np.full((1, 1, lt.TOP_K_MAX), 2, dtype=np.int64)    # predict class 2 ("?2")
    m_hit = lt.compute_metrics((p_hit, np.array([[2]], dtype=np.int64)), arr)
    assert m_hit["loinc_top1"] == 1.0                           # same unmapped class = correct
    p_miss = np.full((1, 1, lt.TOP_K_MAX), 3, dtype=np.int64)   # predict class 3 ("?3")
    m_miss = lt.compute_metrics((p_miss, np.array([[2]], dtype=np.int64)), arr)
    assert m_miss["loinc_top1"] == 0.0                          # "?3" != "?2"
    m0 = lt.compute_metrics((p_hit, np.array([[-100]], dtype=np.int64)), arr)  # all ignored
    assert m0["n_active_positions"] == 0 and m0["loinc_top1"] == 0.0


# -- Llama-specific: keep-oldest truncation -----------------------------------
def test_truncate_keep_oldest_drops_newest():
    """Above max_len, keep the first max_len tokens and drop the newest, in all three aligned fields."""
    feats = [{"input_ids": list(range(10)), "attention_mask": [1] * 10, "labels": list(range(100, 110))}]
    out = lt.truncate_features_keep_oldest(feats, max_len=4)
    assert out[0]["input_ids"] == [0, 1, 2, 3]            # the four oldest are kept
    assert out[0]["attention_mask"] == [1, 1, 1, 1]
    assert out[0]["labels"] == [100, 101, 102, 103]      # labels truncated in lockstep


def test_truncate_noop_when_within_limit_and_preserves_keys():
    """At or below max_len nothing changes. Unrelated keys survive and the original feature is not
    mutated."""
    feats = [{"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1], "labels": [-100, 5, -100], "uuid": "x"}]
    out = lt.truncate_features_keep_oldest(feats, max_len=8192)
    assert out[0]["input_ids"] == [0, 1, 2] and out[0]["uuid"] == "x"
    # a long one alongside a short one: only the long is cut, short untouched
    feats2 = [{"input_ids": list(range(3)), "attention_mask": [1] * 3, "labels": [1, 2, 3]},
              {"input_ids": list(range(5)), "attention_mask": [1] * 5, "labels": [1, 2, 3, 4, 5]}]
    out2 = lt.truncate_features_keep_oldest(feats2, max_len=4)
    assert out2[0]["input_ids"] == [0, 1, 2]             # short: unchanged
    assert out2[1]["input_ids"] == [0, 1, 2, 3]          # long: cut to 4
    assert feats2[1]["input_ids"] == list(range(5))      # non-destructive: original intact


# -- Llama-specific: two-LR param split ---------------------------------------
def test_split_param_groups_by_name_and_requires_grad():
    """Parameters with 'score' in the name go to the head group and the rest to the LoRA group.
    Parameters with requires_grad=False are skipped."""
    def P(rg):
        return torch.nn.Parameter(torch.zeros(1), requires_grad=rg)
    named = [
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", P(True)),   # LoRA
        ("base_model.model.layers.0.mlp.down_proj.lora_B.default.weight", P(True)),       # LoRA
        ("base_model.model.score.modules_to_save.default.weight", P(True)),               # head
        ("base_model.model.score.original_module.weight", P(False)),                      # frozen head copy -> skip
        ("base_model.model.layers.0.self_attn.q_proj.base_layer.weight", P(False)),       # frozen 4-bit base -> skip
    ]
    lora, head = lt.split_param_groups(named)
    assert len(lora) == 2 and len(head) == 1     # 2 adapters trainable, 1 head trainable, 2 frozen dropped


# -- recipe-fidelity guard: constants must not drift from the reference recipe --
def test_recipe_constants_match_reference():
    """Locks the reference QLoRA recipe and the two changes made to it (max_len 8192, save==eval 400)."""
    assert lt.LORA_R == 16 and lt.LORA_ALPHA == 32 and lt.LORA_DROPOUT == 0.05
    assert lt.LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert lt.LORA_MODULES_TO_SAVE == ["score"]
    assert lt.LEARNING_RATE_LORA == 2e-4 and lt.LEARNING_RATE_HEAD == 1e-3
    assert lt.ADAM_BETA1 == 0.9 and lt.ADAM_BETA2 == 0.95 and lt.WEIGHT_DECAY == 0.1
    assert lt.GPU_BATCH_SIZE == 1 and lt.TOTAL_BATCH_SIZE == 16 and lt.MAX_STEPS == 8000
    assert lt.NUM_LABELS == 1623
    assert lt.MAX_LEN == 8192                     # as FHIR-MLM and BCM; the reference uses 4096
    assert lt.EVAL_STEPS == lt.SAVE_STEPS == 400  # harness invariant (best step is a saved checkpoint)
    assert lt.MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct"
