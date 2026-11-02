#!/usr/bin/env python
"""Write T-08, the model sizes: total and trained parameters per model (thesis Chapter 5).

Where every number comes from (nothing that can be read or derived is typed in):
    FHIR-MLM, BCM   `n_params` = sum(p.numel()) in the v3 seed-0 eval manifests (pipeline/15_eval_runner.py).
                    Every parameter is trained.
    Llama           derived from the HF config of the model (vocab 128,256; hidden 4,096; 32 layers;
                    intermediate 14,336; 8 KV heads; no biases; untied embeddings) and the LoRA setup
                    in 14 (r=16 on q/k/v/o/gate/up/down; the 1,623-way `score` head trained in full;
                    pipeline/14_llama_train.py). The eval manifest cannot be used, because it counts
                    the 4-bit weights packed. The gates check that the derivation gives the known
                    full-model (8,030,261,248) and base + head (7,511,574,103) counts, and both numbers
                    in the PEFT printout in results/14_llama_train/train_para_seed0.log.
    TF-IDF          679 one-vs-rest classifiers x (V vocabulary weights + 1 bias). V is read from the
                    pipeline/07 summary (para_xsite clean, seed 0), but only if that summary records
                    the current recipe. Otherwise the row reads "pending". The V IDF weights are
                    not counted, because they are fitted statistics.

The Llama total is not the PEFT "all params" count (7,560,166,574). That count is base + head plus all
48,592,471 trainable parameters, which already include the head, so it counts the head twice. PEFT's
modules_to_save keeps a frozen copy of the original head beside the trained one, and that copy is never
used in the forward pass. The size of the model is base + head + adapters.

Reads the eval manifests in WORK_DIR/eval and the Llama-3.1-8B-Instruct config from the Hugging Face
cache.
Writes results/analysis/a1/tables/ch5_model_sizes.tex (booktabs LaTeX) and the same table as an image,
results/analysis/a1/figures/ch5_model_sizes_table.{png,pdf}.

  python analysis/ch5_model_sizes_table.py
"""
from __future__ import annotations
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_F = importlib.util.spec_from_file_location("figstyle", HERE / "lib" / "figstyle.py")
FS = importlib.util.module_from_spec(_F); _F.loader.exec_module(FS)
_B = importlib.util.spec_from_file_location("booktabs", HERE / "lib" / "booktabs.py")
BT = importlib.util.module_from_spec(_B); _B.loader.exec_module(BT)
sys.path.insert(0, str(REPO / "pipeline"))
import tfidf_recipe  # noqa: E402

EVAL = locations.WORK / "eval"
LLAMA_CONFIG = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
                                   / "snapshots/*/config.json")))
PEFT_LOG = REPO / "results" / "14_llama_train" / "train_para_seed0.log"
TFIDF_07 = REPO / "results" / "07_tfidf_train_eval" / "tfidf_para_xsite__clean.json"
FIG = REPO / "results" / "analysis" / "a1" / "figures"
TAB = REPO / "results" / "analysis" / "a1" / "tables"

N_CLASSES_DEEP = 1623        # the deep models' MIMIC-code head (T-02)
N_CLASSES_TFIDF = 679        # the LOINC label space TF-IDF classifies directly
LORA_R = 16                  # the LoRA rank in pipeline/14_llama_train.py
HEADER = ["Model", "Total parameters", "Trained parameters", "Trained (%)"]
RIGHT = {1, 2, 3}


def manifest_params(run):
    return int(json.loads((EVAL / run / "manifest.json").read_text())["n_params"])


def llama_counts():
    c = json.loads(Path(LLAMA_CONFIG[-1]).read_text())
    h, L, inter, V = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], c["vocab_size"]
    kv = c["num_key_value_heads"] * (c.get("head_dim") or h // c["num_attention_heads"])
    assert not c.get("attention_bias") and not c.get("mlp_bias") and not c.get("tie_word_embeddings")
    layer = 2 * h * h + 2 * h * kv + 3 * h * inter + 2 * h          # q,o + k,v + gate,up,down + 2 norms
    base = V * h + L * layer + h                                     # embeddings + layers + final norm
    head = h * N_CLASSES_DEEP + N_CLASSES_DEEP                       # `score`, with bias
    adapters = L * LORA_R * ((h + h) * 2 + (h + kv) * 2 + (h + inter) * 3)
    out = {"full_lm": base + V * h, "base_plus_head": base + head, "adapters": adapters, "head": head,
           "trained": adapters + head, "total": base + head + adapters}
    # gates: the known full-model and base + head counts, and the PEFT printout
    assert out["full_lm"] == 8_030_261_248 and out["base_plus_head"] == 7_511_574_103, out
    m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)", PEFT_LOG.read_text())
    peft_trained, peft_all = (int(x.replace(",", "")) for x in m.groups())
    assert out["trained"] == peft_trained == 48_592_471, (out["trained"], peft_trained)
    assert out["total"] + head == peft_all == 7_560_166_574, "PEFT counts the frozen head copy once more"
    return out


def tfidf_params():
    """679 x (V + 1) for the para_xsite clean seed-0 model, or None when the 07 summary does not
    record the current recipe."""
    if not tfidf_recipe.records_current_recipe(TFIDF_07, "pinned_params"):
        return None, None
    j = json.loads(TFIDF_07.read_text())
    v = next(r["vocab_size"] for r in j["per_seed"] if r["seed"] == 0)
    return N_CLASSES_TFIDF * (v + 1), v


def build(write=True):
    mlm = manifest_params("mlm__para_xsite__clean__seed0__mimictest")
    bcm = manifest_params("bcm__para_xsite__clean__seed0__mimictest")
    assert (mlm, bcm) == (199_808_256, 397_494_871)                 # the manifests' known counts
    ll = llama_counts()
    tf, vocab = tfidf_params()
    pct = lambda t, n: f"{100 * t / n:.2f}"
    body = [
        ["TF-IDF + LR", *((f"{tf:,}", f"{tf:,}", "100.00") if tf else ("pending",) * 3)],
        ["FHIR-MLM", f"{mlm:,}", f"{mlm:,}", "100.00"],
        ["BioClinical ModernBERT", f"{bcm:,}", f"{bcm:,}", "100.00"],
        ["Llama-3.1-8B (QLoRA)", f"{ll['total']:,}", f"{ll['trained']:,}", pct(ll["trained"], ll["total"])],
    ]
    for r in body:
        print("  " + " | ".join(r))
    print(f"  TF-IDF vocabulary (para_xsite clean, seed 0): {vocab if vocab else 'not available'}")
    print(f"  Llama: base + head {ll['base_plus_head']:,} + adapters {ll['adapters']:,} = {ll['total']:,}; "
          f"trained = adapters + head {ll['trained']:,}")
    if write:
        TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
        tex = TAB / "ch5_model_sizes.tex"
        tex.write_text(BT.tex(HEADER, body, right=RIGHT, total=False,
                              caption="Model sizes",
                              label="tab:model-sizes",
                              comment="T-08 -- generated by analysis/ch5_model_sizes_table.py"))
        print("wrote", tex)
        png, w = BT.draw(FS, HEADER, body, str(FIG / "ch5_model_sizes_table"), right=RIGHT, total=False)
        print(f"wrote {png} (natural width {w:.2f} in; text width {FS.TEXTWIDTH} in)")


if __name__ == "__main__":
    build()
