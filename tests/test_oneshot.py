"""Tests for the one-shot baseline helpers in analysis/lib/oneshot.py.

The LOINC extractor must not be so brittle that it turns model successes into apparent errors, and
`unparseable` must stay separate from `wrong`, so that a regex failure is not scored as a knowledge
failure. Prompt building, the patient-clustered sample and the description swap are tested too.
"""
import importlib.util
from pathlib import Path

import pytest

_S = importlib.util.spec_from_file_location(
    "oneshot", Path(__file__).resolve().parents[1] / "analysis" / "lib" / "oneshot.py")
O = importlib.util.module_from_spec(_S)
_S.loader.exec_module(O)

# 2160-0 Creat SerPl-mCnc; 2345-7 Glucose SerPl-mCnc; 718-7 Hgb Bld-mCnc; 6301-6 INR PPP;
# 5902-2 Prothrombin time
ALL = {"2160-0", "2345-7", "718-7", "6301-6", "5902-2"}


# -- extraction ---------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("2160-0", "2160-0"),
    ("LOINC code: 2160-0", "2160-0"),
    ("The answer is 2160-0.", "2160-0"),
    ("  2160-0\n", "2160-0"),
    ("**2160-0**", "2160-0"),
    ("718-7", "718-7"),
    # 38483-4 Creat Bld-mCnc
    ("The LOINC code is 2160-0 (Creatinine); a related code is 38483-4.", "2160-0"),
])
def test_extracts_the_first_loinc_shaped_token(text, want):
    assert O.extract_loinc(text) == want


@pytest.mark.parametrize("text", ["", "I cannot determine this.", "N/A", "unknown",
                                  "Creatinine in serum"])
def test_returns_none_when_no_code_present(text):
    assert O.extract_loinc(text) is None


def test_does_not_match_a_bare_number_or_a_year():
    assert O.extract_loinc("the value is 42") is None
    assert O.extract_loinc("published in 2019") is None


def test_matches_a_code_with_five_leading_digits():
    assert O.extract_loinc("77147-7") == "77147-7"  # GFRSerPlBld MDRD-ArVRat


# -- classification -----------------------------------------------------------

def test_four_way_outcomes():
    assert O.classify("2160-0", "2160-0", ALL) == "correct"
    assert O.classify("2345-7", "2160-0", ALL) == "valid_wrong"
    assert O.classify("99999-9", "2160-0", ALL) == "not_a_loinc"
    assert O.classify(None, "2160-0", ALL) == "unparseable"


def test_unparseable_is_not_folded_into_wrong():
    """A parser failure must not be counted as a model error."""
    outs = [O.classify(O.extract_loinc(t), "2160-0", ALL)
            for t in ("I cannot determine this.", "2345-7")]
    assert outs == ["unparseable", "valid_wrong"]


def test_a_real_loinc_outside_our_label_space_is_valid_wrong_not_hallucination():
    """Naming a real code outside the label space is an error but not a hallucination. The four-way
    outcome keeps the two apart."""
    all_loinc = ALL | {"33914-3"}  # GFR SerPl Creatinine-bsd frml MDRD
    assert O.classify("33914-3", "2160-0", all_loinc) == "valid_wrong"


# -- prompts ------------------------------------------------------------------

def test_oneshot_messages_have_the_exemplar_before_the_query():
    m = O.build_oneshot_messages("QUERY", "EXAMPLE", "2160-0")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert "EXAMPLE" in m[1]["content"] and m[2]["content"] == "2160-0"
    assert "QUERY" in m[3]["content"]


def test_plain_oneshot_contains_both_turns_for_base_models():
    s = O.build_plain_oneshot("QUERY", "EXAMPLE", "2160-0")
    assert "EXAMPLE" in s and "2160-0" in s and s.rstrip().endswith("LOINC code:")


# -- sampling -----------------------------------------------------------------

def test_sampling_is_patient_clustered_and_deterministic():
    per = {f"p{i}": [{"patient": f"p{i}", "j": j} for j in range(10)] for i in range(50)}
    a = O.sample_observations(per, n_patients=5, n_obs=50, seed=0)
    b = O.sample_observations(per, n_patients=5, n_obs=50, seed=0)
    assert [x["j"] for x in a] == [x["j"] for x in b]
    assert len({x["patient"] for x in a}) <= 5


def test_sampling_spreads_across_patients_when_some_are_huge():
    """One ICU stay can carry over 1000 labs. Taking whole patients until the budget fills can then
    draw 5,000 observations from five patients, with almost no cluster diversity."""
    per = {"big": [{"patient": "big", "j": j} for j in range(5000)]}
    per.update({f"p{i}": [{"patient": f"p{i}", "j": j} for j in range(20)] for i in range(200)})
    s = O.sample_observations(per, n_patients=100, n_obs=1000, seed=0)
    assert len(s) == 1000
    assert len({x["patient"] for x in s}) >= 80          # spread over many patients
    from collections import Counter
    assert Counter(x["patient"] for x in s).most_common(1)[0][1] <= 10   # cap respected


def test_within_patient_draw_is_not_just_the_first_k():
    """obs_ordinal is chronological, so taking the front would bias the draw toward admission labs."""
    per = {"p": [{"patient": "p", "j": j} for j in range(100)]}
    s = O.sample_observations(per, n_patients=1, n_obs=10, seed=0, per_patient_cap=10)
    assert [x["j"] for x in s] != list(range(10))


def test_sampling_respects_the_observation_cap():
    """Guards against a floor-vs-ceil bug. With floor, n_obs=37 over 20 patients gives cap=1 and
    returns 20."""
    per = {f"p{i}": [{"patient": f"p{i}", "j": j} for j in range(10)] for i in range(50)}
    assert len(O.sample_observations(per, n_patients=20, n_obs=37, seed=1)) == 37


def test_different_seeds_draw_different_patients():
    per = {f"p{i}": [{"patient": f"p{i}", "j": 0}] for i in range(200)}
    a = {x["patient"] for x in O.sample_observations(per, 20, 20, seed=0)}
    b = {x["patient"] for x in O.sample_observations(per, 20, 20, seed=1)}
    assert a != b


# -- RELMA description swap (paired comparison) -------------------------------

def test_swap_description_replaces_only_that_line():
    p = "Description: POC alanine aminotransferase\nValue: 14 [IU]/L\nPriority: ROUTINE"
    out = O.swap_description(p, "MBI_GPT/ALT_U/L")
    assert out.split("\n")[0] == "Description: MBI_GPT/ALT_U/L"
    assert "Value: 14 [IU]/L" in out and "Priority: ROUTINE" in out
    assert len(out.split("\n")) == len(p.split("\n"))


def test_swap_description_handles_a_multiline_prompt_with_reference_range():
    p = ("Description: Calcium, Serum Calcium\nValue: 9.5 mg/dL\n"
         "ReferenceRange: Low: 8.4 mg/dL, High: 10.3 mg/dL\nPriority: STAT")
    out = O.swap_description(p, "CALCIUM")
    assert out.startswith("Description: CALCIUM\n")
    assert "ReferenceRange: Low: 8.4 mg/dL, High: 10.3 mg/dL" in out


def test_swap_description_prepends_when_absent():
    out = O.swap_description("Value: 9.5 mg/dL", "CALCIUM")
    assert out.startswith("Description: CALCIUM\n") and "Value: 9.5 mg/dL" in out


def test_swap_does_not_touch_a_value_that_contains_the_word_description():
    p = "Description: A\nValue: Description of sample"
    out = O.swap_description(p, "B")
    assert out == "Description: B\nValue: Description of sample"
