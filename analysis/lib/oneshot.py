"""Pure logic for the off-the-shelf open-model baselines (analysis/b4_oneshot_baselines.py; thesis
Chapters 5 and 6, Appendix E): prompt building, LOINC extraction, scoring.

No model is loaded here, so every function can be unit-tested on CPU without a GPU (tests/test_oneshot.py).

The outcome has four classes. The trained models choose from a fixed output head (the 1,623 lab items
for the three deep models, 679 LOINC codes for TF-IDF), so they cannot name a code that does not
exist. The hallucination failure mode of Yu et al. (2026) cannot happen for them. A free-generating
LLM has no such guarantee, so accuracy alone hides the interesting difference. Every response is
therefore classified as:

    correct        exact match on the gold LOINC
    valid_wrong    a real LOINC, but the wrong one  (directly comparable to the trained models' errors)
    not_a_loinc    well-formed-looking code that does not exist in the LOINC release
    unparseable    no LOINC-shaped string in the response

`unparseable` is counted separately and is not scored as wrong. It measures format compliance, which is
separate from terminology knowledge. Merging the two would make a brittle parser look like model error.
"""
from __future__ import annotations

import re

# LOINC codes are NNNNN-N: 1-5 digits, hyphen, single check digit.
LOINC_RE = re.compile(r"\b(\d{1,5}-\d)\b")

SYSTEM = ("You are a clinical terminology expert. Map the laboratory observation to its LOINC code. "
          "Answer with the LOINC code only, in the form NNNNN-N. Do not explain.")


def build_user_turn(prompt: str) -> str:
    return f"{prompt}\n\nLOINC code:"


def build_fewshot_messages(prompt: str, examples) -> list[dict]:
    """Chat-format few-shot. `examples` is an ordered list of (example_prompt, example_code).

    Every exemplar must come from the training split. An exemplar drawn from test would leak, and with
    only a handful of examples the leak would be obvious.

    More than one exemplar is used because the exemplars also show what an observation looks like, as
    well as teaching the output format. A single exemplar can only show one shape. Training prompts come
    in three main shapes: numeric with a reference range (by far the most common), a qualitative word
    value, and numeric with no range. A lone qualitative exemplar, for instance, leaves out the dominant
    numeric-with-range shape.
    """
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_prompt, ex_code in examples:
        msgs.append({"role": "user", "content": build_user_turn(ex_prompt)})
        msgs.append({"role": "assistant", "content": ex_code})
    msgs.append({"role": "user", "content": build_user_turn(prompt)})
    return msgs


def build_plain_fewshot(prompt: str, examples) -> str:
    """Completion-style fallback for base models with no chat template (Meditron is one). A chat template
    that a model was never trained on degrades its output badly, so the two formats are not interchangeable."""
    shots = "".join(f"{build_user_turn(p)} {c}\n\n" for p, c in examples)
    return f"{SYSTEM}\n\n{shots}{build_user_turn(prompt)}"


def build_oneshot_messages(prompt: str, example_prompt: str, example_code: str) -> list[dict]:
    """One-shot special case of `build_fewshot_messages`. Only `test_oneshot.py` calls it."""
    return build_fewshot_messages(prompt, [(example_prompt, example_code)])


def build_plain_oneshot(prompt: str, example_prompt: str, example_code: str) -> str:
    """One-shot special case of `build_plain_fewshot`."""
    return build_plain_fewshot(prompt, [(example_prompt, example_code)])


def extract_loinc(text: str) -> str | None:
    """First LOINC-shaped token in the response, or None.

    Takes the first match, because instruction-tuned models often continue with an explanation that
    names other codes. The first code is the answer.
    """
    if not text:
        return None
    m = LOINC_RE.search(text)
    return m.group(1) if m else None


def classify(pred: str | None, gold: str, all_loinc: set) -> str:
    """Four-way outcome. `all_loinc` is every code in the LOINC release, including codes outside the 679
    of the label space. A model that names a real code outside the label space is wrong, but it is not
    hallucinating."""
    if pred is None:
        return "unparseable"
    if pred == gold:
        return "correct"
    return "valid_wrong" if pred in all_loinc else "not_a_loinc"


def sample_observations(per_patient: dict[str, list], n_patients: int, n_obs: int, seed: int = 0,
                        per_patient_cap: int | None = None):
    """Patient-clustered sample, spread across patients.

    The sample is clustered because observations within a patient share a history and description
    assignment. Sampling observations independently would understate every downstream interval. The
    calibration bootstrap resamples patients for the same reason.

    `per_patient_cap` makes this usable. Without it, taking whole patients until the observation budget
    fills can draw thousands of observations from a handful of patients, since some ICU stays carry over
    a thousand labs. That is a valid clustered sample, but it has almost no patient diversity and a
    correspondingly wide clustered interval. With the cap, e.g. 500 patients x 10 observations, the same
    budget spreads across the intended number of clusters.

    Observations are drawn at random within a patient. `obs_ordinal` is chronological, so taking the
    first k would bias the sample toward admission labs.
    """
    import random
    rng = random.Random(seed)
    pats = sorted(per_patient)                       # sorted first, so the draw is reproducible
    rng.shuffle(pats)
    # ceil division. With floor, n_obs=37 over 20 patients would give cap=1 and stop at 20
    # observations, fewer than asked for.
    cap = per_patient_cap or max(1, -(-n_obs // max(n_patients, 1)))
    out = []
    for p in pats[:n_patients]:
        recs = per_patient[p]
        take = recs if len(recs) <= cap else rng.sample(recs, cap)
        for rec in take:
            out.append(rec)
            if len(out) >= n_obs:
                return out
    return out


def swap_description(prompt: str, new_desc: str) -> str:
    """Replace only the `Description:` line, leaving Value/ReferenceRange/Priority untouched.

    This keeps the RELMA comparison paired. The observation and its numbers stay the same, and only the
    name changes. Changing anything else would confound the name effect with a difference in the
    observation.
    """
    lines = prompt.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("Description:"):
            lines[i] = f"Description: {new_desc}"
            return "\n".join(lines)
    return f"Description: {new_desc}\n" + prompt        # no Description line: prepend one
