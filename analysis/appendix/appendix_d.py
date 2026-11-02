"""The values and tables of thesis Appendix D (training and evaluation settings).

Sources: pipeline/tfidf_recipe.py (the TF-IDF tokenizer, vocabulary rule and classifier),
pipeline/06_tfidf_dataset.py (the input text, rendered for a made-up observation), 07 (the downsample),
08 (the search), the three training scripts 12-14 (read with `ast`), the analysis library
(calibration, temperature, bootstrap, McNemar, power) and pipeline/23_sliding_dataset.py. Asserts
check the appendix text against the code, and a failed assert stops the render.
analysis/appendix/tables.py renders the output:

  python analysis/appendix/tables.py
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _consts(path):
    """Module-level literal constants of a script, without running it."""
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Tuple) and len(node.targets[0].elts) == len(node.value.elts)):
            for t, val in zip(node.targets[0].elts, node.value.elts):       # A, B = 1, 2
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(val)
                    except ValueError:
                        pass
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        # an expression over constants already read (e.g. SPACE uses NO_CAP), with no builtins
                        try:
                            out[t.id] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                             {"__builtins__": {}}, dict(out))
                        except Exception:
                            pass
    return out


def _has(path, *frags):
    src = open(path, encoding="utf-8").read()
    for f in frags:
        assert f in src, f"{path.name}: {f!r} not found"


#: each table as the appendix prints it: key -> (header, rows, align). tables.py draws the PNG and .tex from here.
TABLES = {}


def _table(key, header, rows, align=""):
    TABLES[key] = (list(header), [[str(c) for c in r] for r in rows], align)


def values():
    import tfidf_recipe as R
    import unit_standardisation as us
    from input_preprocessing_utils import (load_qual_normalizer, load_qual_passthrough,
                                           load_unit_alias_map, normalize_qual_value, normalize_unit)

    # -- the input text, for a made-up observation (the formatter in 06 and the extractor 07 uses) --
    m06 = _load("m06", C.SCRIPTS / "06_tfidf_dataset.py")
    from utils import extract_structured_features_single
    amap = load_unit_alias_map(C.UNIT_ALIAS_JSON)
    v2c = load_qual_normalizer(m06.QUAL_NORMALIZE_PATH)
    lit, rx = load_qual_passthrough(m06.QUAL_PASSTHROUGH_PATH)
    unit_norm = lambda raw: normalize_unit(raw, amap)[0] if raw else ""        # noqa: E731
    def qual_norm(raw):
        outcome, val = normalize_qual_value(raw, v2c, lit, rx)
        return None if outcome == "blank" else val
    qfmt = lambda value, raw_unit: us.standardise_quantity(value, raw_unit, amap)        # noqa: E731
    rfmt = lambda rrs, raw_unit: us.standardise_reference_range(rrs, raw_unit, amap)      # noqa: E731
    obs = {"resourceType": "Observation",
           "valueQuantity": {"value": 4.2, "unit": "mEq/L"},
           "referenceRange": [{"low": {"value": 3.3, "unit": "mEq/L"}, "high": {"value": 5.1, "unit": "mEq/L"}}],
           "extension": [{"valueString": "STAT"}]}
    prompt = m06.format_observation_as_prompt(obs, "K, serum", unit_norm, qual_norm, quantity_fmt=qfmt,
                                              include_reference_range=True, range_fmt=rfmt)
    line = extract_structured_features_single(prompt)
    qual = {"valueString": "Neg."}
    prompt_q = m06.format_observation_as_prompt(dict(qual, resourceType="Observation"), "Urine protein, dipstick",
                                                unit_norm, qual_norm, quantity_fmt=qfmt,
                                                include_reference_range=True, range_fmt=rfmt)

    # -- TF-IDF settings --
    P = R.PINNED
    assert (P["max_features"], P["ngram_max"], P["sublinear_tf"], P["min_df"], P["max_df"]) == (None, 1, False, 5, 1.0)
    assert P["solver"] == "liblinear" and P["stop_words"] is None and P["lowercase"]
    m07 = _consts(C.SCRIPTS / "07_tfidf_train_eval.py")
    c08 = _consts(C.SCRIPTS / "08_tfidf_optuna_b0.py")
    space = c08["SPACE"]
    fmt = lambda vals: ", ".join("no cap" if v == c08["NO_CAP"] else (f"{v:,}" if isinstance(v, int) and v > 999 else str(v)) for v in vals)  # noqa: E731
    search_rows = [                     # settings in words. The scikit-learn parameter names are in the code.
        ["Vocabulary size", fmt(space["max_features"])],
        ["Largest n-gram", fmt(space["ngram_max"])],
        ["Sublinear term frequency", "yes, no"],
        ["Minimum document frequency", fmt(space["min_df"])],
        ["Maximum document frequency", fmt(space["max_df"])],
        ["Solver", ", ".join(space["solver"])],
        ["Inverse regularization strength C", f"{c08['C_RANGE'][0]}–{c08['C_RANGE'][1]:g}, log-uniform"],
        ["Maximum iterations", f"{c08['MAX_ITER'][0]:,}–{c08['MAX_ITER'][1]:,} in steps of {c08['MAX_ITER'][2]}"],
    ]
    assert space["sublinear_tf"] == [True, False]
    _table("D.2", ["Setting", "Values searched"], search_rows)
    n_classes = sum(1 for r in C.read_csv(C.RERUN / "data" / "labelspace" / "labelspace.csv") if r["in_primary_ge10occ"] == "1")
    # one setting per row, a few words each, in three groups (training data, text features, classifier). The
    # settings that Optuna searched use the names in Table D.2. The vectorizer keeps the scikit-learn default
    # L2 norm (unit-length vectors), and the classifier keeps its default L2 penalty.
    recipe_src = open(C.SCRIPTS / "tfidf_recipe.py", encoding="utf-8").read()
    assert "norm=" not in recipe_src and "penalty=" not in recipe_src and "OneVsRestClassifier(base" in recipe_src
    tfidf_rows = [
        ["Training rows", f"~{m07['SUBSET_SIZE']:,}, stratified"],
        ["Minimum rows per class", f"{m07['DOWNSAMPLE_MIN']:,}, or all if fewer"],
        ["Seeds", f"{min(m07['SEEDS'])}–{max(m07['SEEDS'])}, one downsample each"],
        ["Stop words", "none"],
        ["Largest n-gram", str(P["ngram_max"])],
        ["Minimum document frequency", str(P["min_df"])],
        ["Maximum document frequency", str(P["max_df"])],
        ["Vocabulary size", "no cap"],
        ["Sublinear term frequency", "no"],
        ["Vector norm", "L2"],
        ["Classifier", "one-vs-rest logistic regression"],
        ["Classes", str(n_classes)],
        ["Solver", P["solver"]],
        ["Penalty", "L2"],
        ["Inverse regularization strength C", f"{P['C']:.4f}"],
        ["Maximum iterations", f"{P['max_iter']:,}"],
    ]
    _table("D.1", ["Setting", "Value"], tfidf_rows)
    _has(C.SCRIPTS / "tfidf_recipe.py", "random_state=42", 'max_df=p["max_df"]')

    # -- deep models --
    c12, c13, c14 = (_consts(C.SCRIPTS / f) for f in ("12_mlm_train.py", "13_bcm_train.py", "14_llama_train.py"))
    _has(C.SCRIPTS / "12_mlm_train.py", 'adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.1, lr_scheduler_type="cosine"',
         'config._attn_implementation = "flash_attention_2"', 'fsdp=["shard_grad_op"]', "bf16=True")
    _has(C.SCRIPTS / "13_bcm_train.py", "gradient_checkpointing=True", "bf16=True", 'lr_scheduler_type="cosine"')
    _has(C.SCRIPTS / "14_llama_train.py", 'bnb_4bit_quant_type="nf4"', "bnb_4bit_use_double_quant=True",
         "bnb_4bit_compute_dtype=torch.bfloat16", "use_gradient_checkpointing=True",
         "bnb.optim.PagedAdamW8bit", 'lr_scheduler_type="cosine"', "bf16=True")
    for c in (c12, c13, c14):
        assert (c["TOTAL_BATCH_SIZE"], c["MAX_STEPS"], c["EVAL_STEPS"], c["SAVE_STEPS"], c["EARLY_STOP_PATIENCE"],
                c["BEST_METRIC"], c["WARMUP_RATIO"]) == (16, 8000, 400, 400, 3, "eval_loinc_top1", 0.05)
    assert c13["MODEL_NAME"] == "thomas-sounack/BioClinical-ModernBERT-large"
    assert c14["MODEL_NAME"] == "meta-llama/Llama-3.1-8B-Instruct"
    import appendix_f      # the caption of D.3 sends readers to Table F.2 for the IDs and revisions
    assert {c13["MODEL_NAME"], c14["MODEL_NAME"]} <= {r for *_, r in appendix_f.MODELS}
    m09 = _consts(C.SCRIPTS / "09_mlm_dataset.py")
    assert m09["MAX_LEN"] == 8192 and c14["MAX_LEN"] == 8192 and c13["MAX_LEN"] == 8192
    def lr(x):
        """5e-05 -> '5 × 10⁻⁵'"""
        mant, exp = f"{x:.0e}".split("e")
        return f"{mant} × 10" + str(int(exp)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    # the schedule (cosine, 5% warmup) and the sequence length (8,192) are the same for all three, so the
    # caption states them once. The Hugging Face IDs and revisions are in Table F.2. The quantization is
    # listed under precision. The table leaves out two details that the code keeps, double
    # quantization and SDPA when FlashAttention 2 is unavailable.
    deep_rows = [
        ["Starting weights", "Qwen2 MLM pretrained on MIMIC-IV FHIR [Yu 2025 (MASc thesis)]",
         "BioClinical-ModernBERT-large [Sounack 2025]", "Llama-3.1-8B-Instruct [Grattafiori 2024]"],
        ["Trained parameters", "All", "All", f"LoRA on all {len(c14['LORA_TARGET_MODULES'])} projections (rank {c14['LORA_R']}, alpha {c14['LORA_ALPHA']}, dropout {c14['LORA_DROPOUT']}), plus the head"],
        ["Learning rate", lr(c12["LEARNING_RATE"]), lr(c13["LEARNING_RATE"]), f"{lr(c14['LEARNING_RATE_LORA'])} (adapters), {lr(c14['LEARNING_RATE_HEAD'])} (head)"],
        ["Optimizer", "AdamW, β = (0.9, 0.95), weight decay 0.1", f"AdamW, β = ({c13['ADAM_BETA1']}, {c13['ADAM_BETA2']}), weight decay {c13['WEIGHT_DECAY']}",
         f"8-bit AdamW, β = ({c14['ADAM_BETA1']}, {c14['ADAM_BETA2']}), weight decay {c14['WEIGHT_DECAY']}"],
        ["Batch", f"{c12['GPU_BATCH_SIZE']} × 2 GPUs × {16 // (2 * c12['GPU_BATCH_SIZE'])} accumulation = 16",
         f"{c13['GPU_BATCH_SIZE']} × 2 GPUs = 16", f"{c14['GPU_BATCH_SIZE']} × 2 GPUs × {16 // (2 * c14['GPU_BATCH_SIZE'])} accumulation = 16"],
        ["Precision and memory", "bf16; FlashAttention 2; FSDP gradient sharding", "bf16; FlashAttention 2; gradient checkpointing",
         "4-bit NF4 weights; bf16; FlashAttention 2; gradient checkpointing"],
        ["Seeds", "0–4", "0–4", "0"],
    ]
    _table("D.3", ["Setting", "FHIR-MLM", "BCM", "Llama"], deep_rows)

    # -- evaluation settings --
    a10 = _consts(C.ANALYSIS / "a10_catalogue_seedjoint.py")
    assert a10["N_BINS"] == 15 and a10["N_BOOT"] == 4000
    _has(C.ANALYSIS / "v3_calibration.py", 'for arm in ("pert", "clean"):', 'split="val"',
         '"arm": arm, "seed": c.seed', "fast_temperature_objective")
    _has(C.ANALYSIS / "lib" / "calib.py", 'def calibration_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15',
         "lo: float = 0.05, hi: float = 20.0", "coarse: int = 40, refine_iters: int = 40")
    _has(C.ANALYSIS / "lib" / "dumps.py", "binomtest(x, x + y, 0.5)")
    _has(C.ANALYSIS / "lib" / "power.py", "power: float = 0.80, alpha: float = 0.05")
    c23 = _consts(C.SCRIPTS / "23_sliding_dataset.py")
    _has(C.SCRIPTS / "23_sliding_dataset.py", "NON-OVERLAPPING windows", "Patient + most-recent Encounter prepended")
    n_val = sum(1 for l in open(C.RERUN / "data" / "splits" / "val_patients.txt") if l.strip())

    tune = json.load(open(C.RESULTS / "08_tfidf_optuna_b0" / "tfidf_tune_v3.json"))
    assert tune["condition"] == "para_xsite" and tune["train_arm"] == "clean" and tune["tpe_seed"] == c08["TPE_SEED"]
    assert tune["n_complete"] == c08["N_TRIALS"] + 1 and tune["n_failed"] == 0 and tune["default_trial"] == 0
    # the qualitative-value step: list sizes, and its outcome shares in the v3 TF-IDF build
    man = json.load(open(str(locations.WORK / "tfidf/dataset_v3__pert/manifest.json")))
    q = {k: v for k, v in man["normalization"].items() if k.startswith("qual_")}
    qt = sum(q.values())

    v = {
        "prompt_example": prompt, "tfidf_line": line, "prompt_qual": prompt_q,
        "n_canon": str(len(set(v2c.values()))), "n_lit": str(len(lit)), "n_rx": str(len(rx)),
        "q_norm": C.pct(q["qual_normalize"], qt), "q_pass": C.pct(q["qual_passthrough"], qt),
        "q_blank": C.pct(q["qual_blank"], qt),
        "search_obs": f"{tune['search_obs']:,}", "val_obs": f"{tune['val_obs']:,}",
        "n_lab": C.num(n_classes),
        "n_trials": f"{c08['N_TRIALS']:,}", "tpe_seed": str(c08["TPE_SEED"]),
        "search_budget": f"{c08['SUBSET']:,}",
        "n_val": f"{n_val:,}",
        "n_bins": str(a10["N_BINS"]), "n_boot_code": f"{a10['N_BOOT']:,}",
        "window_cap": str(c23["WINDOW_CAP"]),
        "so_batch": f"{c12['SINGLEOBS_TOTAL_BATCH']:,}",
    }
    return v
