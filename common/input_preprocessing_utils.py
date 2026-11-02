"""Preprocessing of unit strings and qualitative values in model inputs.

The dataset builders (pipeline/), the unit standardisation (pipeline/unit_standardisation.py) and
the qualitative-value map builder (prep/4.4_pcornet_qual_normalize.py) share this module. Thesis
Appendices A and D describe the steps.

Two preprocessing functions, one per side of the value:

  - preprocess_qual_string(s) is for qualitative values (valueString,
    valueCodeableConcept). It lowercases the string and turns each run of
    formatting punctuation into a space. It keeps +, -, / and \\ because
    they carry meaning in qual strings.

  - normalize_unit_key(s) is for quant units (valueQuantity.unit/code,
    referenceRange.{low,high}.unit/code). It ignores case and spaces, strips
    annotation `{...}` and bracket `[...]` decoration, and rewrites `10^N` as
    `10*N`. It matches the build-time key in prep/4.3_collect_value_units.py,
    which builds unit_synonym_map.json, so lookups land on the same key.

The dataset builders also use the runtime normalization API:

  - load_qual_normalizer and load_qual_passthrough read the qual asset files
    and return the structures that the lookup function uses.
  - load_unit_alias_map reads unit_synonym_map.json and returns its
    `alias_to_canonical` dict. The dict already includes the manual aliases,
    which prep/4.3_collect_value_units.py merges in.
  - normalize_qual_value makes a three-way decision (normalize, passthrough
    or blank).
  - normalize_unit looks up a surface-form unit alias and falls back to the
    raw string.

This module does not load the manual-alias files (pcornet_qual_manual_aliases.json,
unit_manual_aliases.json). They are inputs to prep/4.4_pcornet_qual_normalize.py and
prep/4.3_collect_value_units.py.
"""
import json
import pathlib
import re
from typing import Literal


# =============================================================================
# Qualitative-value preprocessing
# =============================================================================

# Punctuation treated as formatting or annotation. Each run of it is replaced by a space.
# +, -, / and \ are kept because they carry meaning in qual strings (ordinals, negations, ratios).
_QUAL_DROP_PUNCT = re.compile(r"[.,;:!?*\"'`()\[\]{}]+")
_QUAL_WS = re.compile(r"\s+")


def preprocess_qual_string(s: str) -> str:
    """Canonicalize a qualitative value string for map lookup or map seeding.

    Steps:
      1. Lowercase.
      2. Replace each run of formatting punctuation (. , ; : ! ? * " ' ` ( ) [ ] { }) with a
         space, so "1.5" becomes "1 5".
         Keep +, -, / and \\ because they carry meaning in qual strings
         (e.g. "1+", "non-reactive", "pos/neg").
      3. Collapse any run of whitespace to a single space.
      4. Strip leading/trailing whitespace.

    Returns "" for inputs that are empty or collapse to empty after cleanup.
    """
    if not s:
        return ""
    s = s.lower()
    s = _QUAL_DROP_PUNCT.sub(" ", s)
    s = _QUAL_WS.sub(" ", s).strip()
    return s


# =============================================================================
# Unit-string preprocessing (matches prep/4.3_collect_value_units.py)
# =============================================================================

_RE_ANNOT       = re.compile(r"\{[^}]*\}")
_RE_BRACKET     = re.compile(r"\[([^\]]*)\]")
_RE_POW         = re.compile(r"(\d)\^(\d)")
_RE_SPACE_UNDER = re.compile(r"[\s_]+")


def normalize_unit_key(s: str) -> str:
    """Case/space-insensitive merge key for unit surface forms.

    The steps must match the build-time key in prep/4.3_collect_value_units.py
    so that lookups hit the same alias_to_canonical entries.
      1. Drop annotations like `{cells}` and `{INR}`. A string that is only an
         annotation (e.g. `{INR}`) is kept as it is, so that different
         annotation-only units don't all collapse to "".
      2. Strip `[...]` brackets and keep their contents (`[IU]` becomes `IU`).
      3. Rewrite `10^3` as `10*3` (UCUM accepts both forms).
      4. Lowercase, and drop whitespace and underscores.
    """
    if not s:
        return ""
    stripped = _RE_ANNOT.sub("", s)
    if not stripped.strip():
        stripped = s
    stripped = _RE_BRACKET.sub(r"\1", stripped)
    stripped = _RE_POW.sub(r"\1*\2", stripped)
    return _RE_SPACE_UNDER.sub("", stripped.lower())


# =============================================================================
# Asset loaders
# =============================================================================

def load_qual_normalizer(path) -> dict:
    """Read pcornet_qual_normalize.json and return the flat
    variation_to_canonical dict used by normalize_qual_value.

    JSON layout: {canonical: [variation, ...]}.
    Both keys and variations are already preprocessed (see
    build_normalize_map in prep/4.4_pcornet_qual_normalize.py). The canonical
    maps to itself for idempotent lookup.
    """
    payload = json.loads(pathlib.Path(path).read_text())
    v2c = {}
    for canon, vars_ in payload.items():
        v2c[canon] = canon
        for v in vars_:
            v2c[v] = canon
    return v2c


def load_qual_passthrough(path) -> tuple[set, list]:
    """Read pcornet_qual_passthrough.json and return (literals, patterns).

    literals: set of preprocessed strings that pass through unchanged.
    patterns: list of (name, compiled_regex) tuples; first match wins.

    Both literal keys and the regexes operate on preprocessed qual strings.
    """
    data = json.loads(pathlib.Path(path).read_text())
    literals = set(data.get("literals", {}).keys())
    patterns = [(name, re.compile(spec["regex"]))
                for name, spec in data.get("numeric_patterns", {}).items()]
    return literals, patterns


def load_unit_alias_map(path) -> dict[str, str]:
    """Read unit_synonym_map.json and return its alias_to_canonical dict.

    Keys are surface forms preprocessed by normalize_unit_key(). Values are the
    canonical UCUM (or UCUM-style) code in its original casing. The manual
    aliases from unit_manual_aliases.json are merged in at build time
    (prep/4.3_collect_value_units.py), so they are not loaded again here.
    """
    payload = json.loads(pathlib.Path(path).read_text())
    return dict(payload["alias_to_canonical"])


# =============================================================================
# Runtime normalization API
# =============================================================================

QualOutcome = Literal["normalize", "passthrough", "blank"]


def normalize_qual_value(
    raw: str,
    v2c: dict,
    literals: set,
    patterns: list,
) -> tuple[QualOutcome, str | None]:
    """Three-way decision for a raw qualitative value string.

    Order (first match wins):
      1. preprocess_qual_string(raw) hits the normalize map → ("normalize", canonical)
      2. preprocessed string is in passthrough literals       → ("passthrough", preprocessed)
      3. any passthrough numeric_pattern matches              → ("passthrough", preprocessed)
      4. otherwise (paraphrase candidate)                     → ("blank", None)

    "blank" means the caller drops the Value: line and writes no empty
    placeholder. These strings are a memorization risk, and they are removed
    from both training and evaluation inputs.
    """
    if not raw:
        return ("blank", None)
    pk = preprocess_qual_string(raw)
    if not pk:
        return ("blank", None)
    if pk in v2c:
        return ("normalize", v2c[pk])
    if pk in literals:
        return ("passthrough", pk)
    for _name, rx in patterns:
        if rx.match(pk):
            return ("passthrough", pk)
    return ("blank", None)


def normalize_unit(raw: str, alias_map: dict[str, str]) -> tuple[str, bool]:
    """Surface-form unit lookup. It replaces the unit string with its canonical
    UCUM (or UCUM-style) form and does not scale the value.

    Returns (canonical_or_raw, was_mapped). On a miss it returns the raw string
    unchanged so the caller can keep the prompt readable. was_mapped lets the
    caller log or count unmapped surface forms.
    """
    if not raw:
        return ("", False)
    key = normalize_unit_key(raw)
    if not key:
        return (raw, False)
    canon = alias_map.get(key)
    if canon is None:
        return (raw, False)
    return (canon, True)
