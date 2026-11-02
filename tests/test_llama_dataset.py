"""Tests for pipeline/11_llama_dataset.py: the Llama-specific constants and the paraphrase split it
shares with 09_mlm_dataset."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "llama", Path(__file__).resolve().parents[1] / "pipeline" / "11_llama_dataset.py")
llama = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(llama)


def test_uses_09_split_function():
    """Cross-model control. The paraphrase split and UUID reader in 11 are the same objects as in 09.
    The 70/15/15 pools are therefore identical, and every (patient, LOINC) draws the same paraphrase
    for FHIR-MLM, BCM and Llama."""
    assert llama.load_paraphrases_3way is llama.md09.load_paraphrases_3way
    assert llama.read_uuid_list is llama.md09.read_uuid_list


def test_llama_mask_constants():
    """Llama 3.1 has no native [MASK]. The mask is the reserved token id 128002, which the build checks
    for drift."""
    assert llama.LLAMA_MASK_TOKEN == "<|reserved_special_token_0|>"
    assert llama.LLAMA_MASK_TOKEN_ID == 128002


def test_new_stats_keys():
    s = llama._new_stats(5)
    assert set(s) == {"n_input", "n_missing_file", "n_format_error", "n_no_codes",
                      "n_other_error", "n_no_labels", "n_ok", "total_obs_patched",
                      # _new_stats declares the perturbation counts, because _merge_stats sums
                      # over the keys of _new_stats. A key created on first use by one parallel
                      # worker would be dropped when the shards are merged
                      "total_fields_perturbed", "total_units_randomised"}
    assert s["n_input"] == 5 and all(v == 0 for k, v in s.items() if k != "n_input")


def test_merge_stats():
    a, b = llama._new_stats(10), llama._new_stats(10)
    a["n_ok"], a["n_no_codes"] = 3, 1
    b["n_ok"], b["n_format_error"] = 4, 2
    m = llama._merge_stats([a, b], 20)
    assert m["n_input"] == 20 and m["n_ok"] == 7 and m["n_no_codes"] == 1 and m["n_format_error"] == 2
