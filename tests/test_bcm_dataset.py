"""Tests for pipeline/10_bcm_dataset.py: its BCM-specific helpers and the paraphrase split it shares
with 09_mlm_dataset."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "bcm", Path(__file__).resolve().parents[1] / "pipeline" / "10_bcm_dataset.py")
bcm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcm)


def test_uses_09_split_function():
    """Cross-model control. The paraphrase split and UUID reader in 10 are the same objects as in 09.
    The 70/15/15 pools are therefore identical, and every (patient, LOINC) draws the same paraphrase
    for FHIR-MLM and BCM."""
    assert bcm.load_paraphrases_3way is bcm.md09.load_paraphrases_3way
    assert bcm.read_uuid_list is bcm.md09.read_uuid_list


def test_new_stats_bcm_keys():
    s = bcm._new_stats(5)
    assert set(s) == {"n_input", "n_missing_file", "n_format_error", "n_no_codes",
                      "n_other_error", "n_no_labels", "n_ok", "total_obs_patched",
                      # the perturbation counts are declared up front in _new_stats, because
                      # _merge_stats sums over these keys. A key that one parallel worker created
                      # on first use would be dropped when the shards are merged
                      "total_fields_perturbed", "total_units_randomised"}
    assert s["n_input"] == 5 and all(v == 0 for k, v in s.items() if k != "n_input")


def test_merge_stats():
    a, b = bcm._new_stats(10), bcm._new_stats(10)
    a["n_ok"], a["n_no_codes"] = 3, 1
    b["n_ok"], b["n_format_error"] = 4, 2
    m = bcm._merge_stats([a, b], 20)
    assert m["n_input"] == 20 and m["n_ok"] == 7 and m["n_no_codes"] == 1 and m["n_format_error"] == 2
