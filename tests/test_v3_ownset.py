"""
Tests for analysis/v3_ownset.py: a model on all the observations it scores against the evaluated
set. The logic is checked on small made-up tables, so nothing here reads a result file.
"""
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "analysis"
_spec = importlib.util.spec_from_file_location("v3_ownset_t", HERE / "v3_ownset.py")
OWN = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(OWN)

# the label space of this toy example: three LOINC codes, one of them rare (fewer than 1,000 training observations)
COUNTS = {"L-frequent": 500_000, "L-common": 5_000, "L-rare": 40}


def percode(rows):
    """rows: (code, gold, {seed: right?}, n_obs) -> one row per local code and model seed."""
    return pd.DataFrame([{"code": c, "gold": g, "seed": s, "sm_correct": ok, "n_obs": n}
                         for c, g, by_seed, n in rows for s, ok in by_seed.items()])


def obs(model, top1, n):
    return pd.DataFrame([{"model": model, "top1_D_mean": top1, "n_paired": n}])


@pytest.fixture
def toy():
    # the evaluated set: two local codes, both right in seed 0, one wrong in seed 1
    ev = percode([(1, "L-frequent", {0: True, 1: True}, 10), (2, "L-common", {0: True, 1: False}, 4)])
    # the model's own observations: the same two (code 2 now right in both seeds, with more observations), two added
    # local codes of a rare LOINC code (one right in one seed), and one local code whose LOINC code is outside the space
    own = percode([(1, "L-frequent", {0: True, 1: True}, 30), (2, "L-common", {0: True, 1: True}, 9),
                   (3, "L-rare", {0: False, 1: True}, 1), (4, "L-rare", {0: False, 1: False}, 3),
                   (5, "L-outside", {0: False, 1: False}, 1)])
    return OWN.compare(obs("M", 0.9950, 44), obs("M", 0.9940, 14), {"M": own}, {"M": ev}, COUNTS).set_index("model").loc["M"]


def test_local_codes_outside_the_label_space_are_left_out_and_counted(toy):
    assert toy.codes_outside_space == 1
    assert toy.codes_own == 4 and toy.codes_eval == 2          # code 5 is not one of the model's local codes here


def test_per_code_top1_is_the_mean_over_model_seeds_of_each_seeds_share(toy):
    # own, inside the label space: seed 0 has 2 of 4 right, seed 1 has 3 of 4; the evaluated set: 2 of 2 and 1 of 2
    assert toy.top1_code_own == pytest.approx((0.5 + 0.75) / 2)
    assert toy.top1_code_eval == pytest.approx((1.0 + 0.5) / 2)
    assert toy.code_diff_pp == pytest.approx(100 * (0.625 - 0.75))


def test_the_same_local_codes_with_all_of_the_models_observations(toy):
    # on the evaluated set's two local codes the own observations get 2 of 2 in both seeds
    assert toy.code_diff_same_codes_pp == pytest.approx(100 * (1.0 - 0.75))


def test_added_local_codes_are_counted_once_each(toy):
    assert toy.codes_added == 2 and toy.added_rare == 2
    assert toy.added_one_obs == 1 and toy.added_max_obs == 3 and toy.added_obs == 4
    assert toy.top1_code_added == pytest.approx((0.0 + 0.5) / 2)


def test_per_observation_difference_is_in_percentage_points(toy):
    assert toy.obs_diff_pp == pytest.approx(0.10)
    assert toy.obs_own == 44 and toy.obs_eval == 14


def test_an_evaluated_set_with_a_local_code_outside_the_label_space_is_refused():
    ev = percode([(1, "L-outside", {0: True}, 2)])
    own = percode([(1, "L-outside", {0: True}, 2)])
    with pytest.raises(AssertionError, match="outside the label space"):
        OWN.compare(obs("M", 1.0, 2), obs("M", 1.0, 2), {"M": own}, {"M": ev}, COUNTS)


def test_no_added_local_codes_gives_zero_counts_and_no_accuracy():
    ev = percode([(1, "L-frequent", {0: True}, 2)])
    r = OWN.compare(obs("M", 1.0, 5), obs("M", 1.0, 2), {"M": percode([(1, "L-frequent", {0: True}, 5)])}, {"M": ev},
                    COUNTS).iloc[0]
    assert r.codes_added == 0 and r.added_max_obs == 0 and math.isnan(r.top1_code_added)


def test_rows_name_the_two_models_that_keep_every_test_patient():
    t = pd.DataFrame([dict(model=m, n_seeds=1, obs_own=10, obs_eval=5, top1_obs_own=0.99, top1_obs_eval=0.98, obs_diff_pp=1.0,
                           codes_own=4, codes_eval=3, top1_code_own=0.5, top1_code_eval=0.75, code_diff_pp=-25.0,
                           code_diff_same_codes_pp=1.0, codes_added=1, added_rare=1, added_one_obs=1, added_max_obs=1,
                           added_obs=1, top1_code_added=0.0, codes_outside_space=2)
                      for m in ("FHIR-MLM", "BCM", "Llama-3.1-8B", "TF-IDF")])
    rows = OWN.rows(t)
    assert len(rows) == 3 and all(r[0] == "7.7" for r in rows)
    for r in rows:
        assert "BCM" in r[3] and "Llama" in r[3] and "FHIR-MLM" not in r[3] and "TF-IDF" not in r[3]
    assert "−" not in rows[1][3] and "-25.00 pp" in rows[1][3]
