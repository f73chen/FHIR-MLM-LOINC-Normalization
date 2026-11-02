"""Pad ragged single-observation rows to the batch maximum, then hand the batch to the FHIR-MLM
collator. Used by the single-observation FHIR-MLM training and evaluation (thesis Chapter 5).

`DataCollatorForLabCodeMLM.torch_call` calls `torch.tensor(v)` on the batch, which needs every row to
be the same length (its sibling collator says `# Assume already padded`). The full-history datasets
are padded to a fixed 8192 tokens, so they arrive rectangular. The single-observation datasets are
trimmed to their true length instead, because fixed 320-token padding costs 8.54x as many tokens per
step as the full-history run, against 0.62x for trimmed, length-grouped rows. This wrapper pads each
batch and passes it to the unmodified collator, so the masking and label logic stay where they are.

Pad values and side follow the tokenizer (`fhir_tokenizer._pad`):
    input_ids -> pad_token_id | token_type_ids -> pad_token_type_id
    attention_mask -> 0       | special_tokens_mask -> 1
Padding is on the left. `NdjsonTokenizer.padding_side == "left"`, so every dataset is left-padded and
the model was fine-tuned that way. Padding on the right would put real tokens at different absolute
positions than in training. The BertTokenizerFast inside the collator reports padding_side "right",
but it does not do the padding. Matching the tokenizer makes trimming reversible, because a trimmed
row re-padded here is identical to the original fixed-length row.

Only the four model arrays are forwarded. `uuid` is a string and `length`/`obs_offset` are scalars,
and `torch.tensor` on any of them would fail inside the inner collator. The Trainer's
`remove_unused_columns` could strip them, but then correctness would depend on a flag set elsewhere.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ARRAYS = ("input_ids", "token_type_ids", "attention_mask", "special_tokens_mask")


@dataclass
class PadToBatchMax:
    inner: Any                                  # DataCollatorForLabCodeMLM
    pad_token_id: int
    pad_token_type_id: int = 0
    pad_to_multiple_of: Optional[int] = 8       # tensor-core friendly; 8 costs <1% extra tokens
    _pad_value: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._pad_value = {"input_ids": self.pad_token_id,
                           "token_type_ids": self.pad_token_type_id,
                           "attention_mask": 0,
                           "special_tokens_mask": 1}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_max = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and n_max % self.pad_to_multiple_of:
            n_max = ((n_max // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        padded = []
        for f in features:
            n = len(f["input_ids"])
            row = {}
            for k in ARRAYS:
                if k not in f:
                    continue
                v = list(f[k])
                if len(v) != n:
                    raise ValueError(f"{k} length {len(v)} != input_ids length {n} (misaligned row)")
                row[k] = [self._pad_value[k]] * (n_max - n) + v      # left, as the tokenizer pads
            padded.append(row)
        return self.inner(padded)
