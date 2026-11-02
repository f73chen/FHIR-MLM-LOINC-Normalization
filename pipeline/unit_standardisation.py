"""
The v2 input representation: code-blind unit standardisation and fixed-width numeric rendering,
plus the standardised reference range that v3 adds (thesis Chapter 4, Appendix A). Shared by
TF-IDF (pipeline/06_tfidf_dataset.py, pipeline/16_tfidf_eval_dump.py) and the text models'
formatter (common/fhir_formatter.py).

TF-IDF formats prompts in `06_tfidf_dataset.format_observation_as_prompt`, and the text models format
them in `fhir_formatter.FHIRFormatter`. Both take their value and range formatting from this module,
so the model families get the same transform.

Standardisation converts a unit only to another unit of the same kind, as defined below.

A purely dimensional rule is not safe. UCUM parses International Units as dimensionless, so `[IU]/L`
would become `/L` at factor 1.0. That would delete the only token that says what was measured. `kind`
is therefore the triple (dimension vector, mole exponent, analyte-dependent units such as `[IU]`), as
in `unit_kind` in analysis/a19_scale_only_standardisation.py. All three are read from the unit string
alone, so a deploying site needs no LOINC code and no property table.

Deployment: a receiving site needs only STANDARD_UNIT below. It looks up the canonical form of its
unit string, multiplies the value by the factor and emits the target unit. The table is fixed here,
because a table derived from whatever data happens to be present is not something a second site can
reproduce.

Pure functions over unit strings and numbers, with no data access and no I/O.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

_FHIR = str(locations.COMMON)
if _FHIR not in sys.path:
    sys.path.insert(0, _FHIR)
from input_preprocessing_utils import normalize_unit, normalize_unit_key  # noqa: E402

#: The pipeline's key. `analysis/lib/units._norm_key` only lowercases and strips whitespace. This one
#: also drops `{annotations}`, strips `[...]` but keeps the contents, rewrites `10^N` as `10*N` and
#: drops underscores. The two agree on all 54 distinct raw unit surface forms in the corpus, but they
#: are different functions. The pipeline's key is used here, and the analysis one must not be
#: substituted for it.
NORMALISE_KEY = normalize_unit_key

#: Below this many significant figures the representation starts to lose real precision. 3 s.f.
#: collapses urine specific gravity (5811-5, Sp Gr Ur Strip) from 55 distinct values to 8, and
#: 88.5% of that code's observations need more than 3. 4 s.f. rounds only 0.03% of all observations
#: and costs the same tokens and the same format uniformity as 3.
SIG_FIGS = 4

#: No cap on the rescale magnitude. `None` disables the check (tests/test_unit_standardisation.py).
#: Setting a number turns the check on, if another corpus needs one.
#:
#: Without a cap, converting `mL` to `fL` (x1e12) moves the urine-volume observations of LOINC
#: 28009-9 (Specimen vol Ur) to exponents +13..+16, which no other code has. A shortcut could then
#: seem to move from the unit token into the value's exponent. A cap of 6 is still not used, for two
#: reasons.
#:
#:   1. The merge costs nothing measurable. Over the full corpus, with caps of {6, 7, 9, 12, none},
#:      unit-alone and value+unit identifiability are equal or better without the cap.
#:   2. The "absurd number" objection comes from thinking in decimal, where 500 mL becomes
#:      500000000000000. Under v2's fixed-width 4 s.f. scientific rendering it is `5.000e+14`,
#:      the same token shape as `5.000e+00`. Only the exponent digits differ.
#:
#: A cap would give a narrower exponent range (max |exp| 7 vs 16), but it would add two exceptions
#: to a rule that otherwise has none. A deployment rule with no special cases is easier to publish
#: and to implement.
MAX_LOG10_FACTOR = None

#: canonical UCUM unit -> (standard target, multiply the value by this), fixed here.
#: tests/test_unit_standardisation.py::test_frozen_table_matches_the_kind_rule re-derives it from
#: the kind rule and fails if the two disagree. The trailing comment on each entry is the unit's
#: observation count. Each kind's target is its most frequent unit.
STANDARD_UNIT = {
    'mg/dL'       : ('mg/dL'       , 1.0),                    #  7,374,806
    '%'           : ('%'           , 1.0),                    #  6,432,244
    'meq/L'       : ('meq/L'       , 1.0),                    #  6,184,521
    '10*3/uL'     : ('10*3/uL'     , 1.0),                    #  3,491,777
    'g/dL'        : ('mg/dL'       , 1000.0),                 #  2,181,364
    '[IU]/L'      : ('[IU]/L'      , 1.0),                    #  1,860,524
    'fL'          : ('fL'          , 1.0),                    #  1,728,318
    '10*12/L'     : ('10*3/uL'     , 1000.0),                 #  1,214,275
    'pg'          : ('pg'          , 1.0),                    #  1,205,183
    's'           : ('s'           , 1.0),                    #    921,788
    '{#}/[HPF]'   : ('{#}/[HPF]'   , 1.0),                    #    442,592
    "[arb'U]"     : ("[arb'U]"     , 1.0),                    #    433,001
    'ng/mL'       : ('mg/dL'       , 0.0001),                 #    333,126
    'mm[Hg]'      : ('mm[Hg]'      , 1.0),                    #    259,634
    'mmol/L'      : ('mmol/L'      , 1.0),                    #    231,232
    'u[IU]/mL'    : ('[IU]/L'      , 0.001),                  #    136,119
    '/uL'         : ('10*3/uL'     , 0.001),                  #    117,907
    'ug/dL'       : ('mg/dL'       , 0.001),                  #    115,958
    'pg/mL'       : ('mg/dL'       , 1e-07),                  #     95,228
    'mg/L'        : ('mg/dL'       , 0.1),                    #     63,436
    'ng/dL'       : ('mg/dL'       , 1e-06),                  #     60,081
    'ug/mL'       : ('mg/dL'       , 0.1),                    #     46,016
    '{#}/[LPF]'   : ('{#}/[LPF]'   , 1.0),                    #     44,227
    'mosm/kg'     : ('mosm/kg'     , 1.0),                    #     29,926
    'mm/h'        : ('mm/h'        , 1.0),                    #     26,734
    'mg/g'        : ('%'           , 0.1),                    #     23,870
    'mg/mg'       : ('%'           , 100.0),                  #     18,870
    'U/mL'        : ('U/mL'        , 1.0),                    #     13,597
    'm[IU]/mL'    : ('[IU]/L'      , 1.0),                    #     11,130
    '[IU]/mL'     : ('[IU]/L'      , 1000.0),                 #      5,449
    'mg/(24.h)'   : ('mg/(24.h)'   , 1.0),                    #      5,351
    'umol/L'      : ('mmol/L'      , 0.001),                  #      5,091
    'mL'          : ('fL'          , 1000000000000.0),        #      4,144
    'nmol/L'      : ('mmol/L'      , 1e-06),                  #      3,179
    'L/min'       : ('L/min'       , 1.0),                    #      2,810
    'U/g'         : ('U/g'         , 1.0),                    #      1,003
    "[GPL'U]"     : ("[GPL'U]"     , 1.0),                    #        814
    "[MPL'U]"     : ("[MPL'U]"     , 1.0),                    #        813
    'h'           : ('s'           , 3600.0),                 #        262
    'mL/min'      : ('L/min'       , 0.001),                  #        261
    '/mL'         : ('10*3/uL'     , 1e-06),                  #        153
    'U/L'         : ('U/mL'        , 0.001),                  #        103
    'uU/mL'       : ('U/mL'        , 1e-06),                  #         44
    '{score}'     : ('{score}'     , 1.0),                    #         26
    'min'         : ('s'           , 60.0),                   #         24
    '/[HPF]'      : ('{#}/[HPF]'   , 1.0),                    #          8
}

#: Units whose value is already on a logarithmic scale. Rescaling one by a magnitude factor is wrong
#: arithmetically, because 10x a log10 value is not 10x the quantity. None of these reach
#: STANDARD_UNIT because the alias map cannot parse them, so the guard does not fire on this corpus.
#: It is there so that extending the alias map cannot start rescaling these values.
LOG_SCALE_MARKERS = re.compile(r"\blog\s*_?1?0?\b|\bln\b", re.I)


def is_log_scale(raw_unit: str) -> bool:
    return bool(LOG_SCALE_MARKERS.search(str(raw_unit or "")))


#: Known defects in the shared alias map, corrected here.
#:
#: `unit_synonym_map.json` maps two MIMIC surface forms onto UCUM codes that mean something else
#: entirely. The cross-terminology comparison in analysis/a20_unit_code_concentration.py found both.
#: Both affect many observations (see the counts in STANDARD_UNIT), and in both cases what the lab
#: meant is unambiguous.
#:
#:   [S]  MIMIC writes `sec` for PT / aPTT / thrombin clotting times. The alias map sends `sec` and
#:        `second` to `[S]`, which in UCUM is the Svedberg (1e-13 s, a sedimentation coefficient).
#:        The correct code for the second is `s`. LOINC, UCUM and CDISC all attest `s`/`sec` for
#:        these three codes, so the intent is not in doubt.
#:        The map is also inconsistent with itself. It maps `seconds` to `s` but `sec` to `[S]`.
#:        It also breaks deployment. A site sending the correct `s` canonicalises to `S`, the
#:        siemens, with dimension (-2,1,-1,0,0,2,0), three dimensions away from time.
#:
#:   AU   MIMIC writes `AU` for arbitrary units. UCUM has no bare `AU`, and lab2clean parses it as
#:        the astronomical unit (1.496e11 m, a length). The UCUM code for an arbitrary unit is
#:        `[arb'U]`, which is correctly dimensionless.
#:
#: The shared map is left as it is, because that file is an input to other work.
#: The correction is applied inside canonical_unit(), so every consumer sees the corrected form.
ALIAS_MAP_CORRECTIONS = {
    "[S]": "s",          # Svedberg      -> second
    "AU": "[arb'U]",     # astronomical  -> arbitrary unit
}


def correct_canonical(canon: "str | None") -> "str | None":
    """Apply the known alias-map corrections to an already-canonicalised unit."""
    return ALIAS_MAP_CORRECTIONS.get(canon, canon)


def canonical_unit(raw_unit: str, alias_map: dict) -> "str | None":
    """Map a raw surface form to canonical UCUM, or to None if the alias map does not recognise it.

    Includes ALIAS_MAP_CORRECTIONS, so downstream code never sees `[S]` or `AU`."""
    if not raw_unit:
        return None
    canon, mapped = normalize_unit(raw_unit, alias_map)
    return correct_canonical(canon) if mapped else None


def standardise_unit(raw_unit: str, alias_map: dict):
    """(target_unit, factor) for a raw surface form. The factor multiplies the value.

    Returns factor 1.0 whenever the unit must not be converted. An unrecognised surface form or a
    log-scale unit comes back as written, and a canonical form outside the table comes back in its
    canonical spelling. The function never raises. An unknown unit passes through, because refusing
    to emit an observation would change the dataset's composition."""
    if not raw_unit:
        return "", 1.0
    if is_log_scale(raw_unit):
        return raw_unit, 1.0
    canon = canonical_unit(raw_unit, alias_map)
    if canon is None:
        return raw_unit, 1.0
    hit = STANDARD_UNIT.get(canon)
    if hit is None:
        return canon, 1.0
    return hit[0], hit[1]


def render_value(value, nsf: int = SIG_FIGS) -> str:
    """Fixed-width scientific: always one digit, a point, nsf-1 digits, `e`, sign, two-digit
    exponent. Every value has the same shape, so the model sees one numeric grammar and only the
    digits vary.

    Returns "" for anything non-numeric, so callers can fall back to emitting the raw text."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{nsf - 1}e}"


def standardise_quantity(value, raw_unit: str, alias_map: dict, nsf: int = SIG_FIGS):
    """(rendered_value_string, unit_string) for one valueQuantity. This is the whole v2 transform.

    The value and the unit must be handled together, because the unit determines the factor and the
    factor changes the value. Handled separately, a rescaled number can end up beside its old unit.
    Falls back to the raw value text when the value is not numeric, so qualitative content passes
    through untouched."""
    target, factor = standardise_unit(raw_unit, alias_map)
    rendered = render_value(_scale(value, factor), nsf)
    if rendered == "":
        return ("" if value is None else str(value)), target
    return rendered, target


def _scale(value, factor):
    try:
        return float(value) * factor
    except (TypeError, ValueError):
        return value


def round_sig(value, nsf: int = SIG_FIGS):
    """The value as the rendering will represent it, for round-trip checks and gates."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v == 0:
        return 0.0 if v == 0 else None
    f = 10.0 ** (nsf - 1 - math.floor(math.log10(abs(v))))
    return round(v * f) / f


def standardise_reference_range(rrs, value_raw_unit: str, alias_map: dict,
                               nsf: int = SIG_FIGS):
    """Turn a FHIR `referenceRange` list into [(label, value_text, unit_text), ...].

    v2 has no reference range, so only the value goes through `standardise_quantity`. v3 has the
    range, and rendering it separately from the value would print a rescaled number beside its
    old unit. Almost every observation whose unit has a factor other than 1 also carries a
    reference range. An unstandardised bound would print haemoglobin's value at 1000x its range.

    A bound with no unit inherits the value's. In MIMIC, a bound that states a unit always states
    the value's unit, and a small share of bounds state none. Without the fallback those would
    take factor 1.0 and be emitted unscaled beside a scaled value.

    Each bound is still standardised through `standardise_quantity` on its own (or inherited) unit.
    A bound that is expressed in a different unit would therefore be converted by that unit's factor.

    >>> amap = {"alias_to_canonical": {}}
    >>> standardise_reference_range([], "mg/dL", amap)
    []
    """
    out = []
    for rr in (rrs or []):
        for label, key in (("Low", "low"), ("High", "high")):
            q = rr.get(key)
            if not q:
                continue
            raw = q.get("unit") or q.get("code") or ""
            if not str(raw).strip():
                raw = value_raw_unit            # no unit of its own: inherit the value's
            text, unit = standardise_quantity(q.get("value", ""), raw, alias_map, nsf)
            out.append((label, text, unit))
    return out


def range_is_consistent(value_unit: str, bounds) -> bool:
    """True when every emitted bound carries the same unit string as the emitted value.

    The tests assert this. It is the property that makes a value and its reference range
    comparable."""
    return all(u == value_unit for _l, _t, u in bounds)


#: What a v2 value must look like. Used by the tests.
VALUE_RE = re.compile(r"^-?\d\.\d{%d}e[+-]\d{2,3}$" % (SIG_FIGS - 1))

#: The set of unit strings a v2 dataset may emit for a recognised unit. Anything else must be a
#: documented pass-through (no unit, unparseable surface form, or log-scale).
STANDARD_TARGETS = sorted({t for t, _ in STANDARD_UNIT.values()})


def deep_model_fmts(rep: str, alias_map: dict):
    """(quantity_fmt, range_fmt) for FHIRFormatter under representation `rep`.

    Both deep-model builders get their formatters from here, built on the functions 06 uses.
    Returns (None, None) for v1, which keeps the formatter's plain v1 output.

    The units returned are the standardised targets, verbatim. They are not remapped to a MIMIC
    surface form. 10 and 11 feed BPE text tokenizers, where any string tokenises, and remapping
    would only make the deep models' text disagree with 06's.

    They are built inside each worker process, because they are closures and the parallel path
    pickles its arguments.
    """
    if rep == "v1":
        return None, None

    def quantity_fmt(value, raw_unit):
        return standardise_quantity(value, raw_unit, alias_map)

    def range_fmt(rrs, value_raw_unit):
        return standardise_reference_range(rrs, value_raw_unit, alias_map)

    # v2 writes no reference range, so it needs no range formatter
    return quantity_fmt, (range_fmt if rep == "v3" else None)


def is_legitimate_output(unit: str, alias_map: dict) -> bool:
    """Could a correct v2 build have emitted this unit string?

    This is not an idempotence check, because the alias map is not idempotent on its outputs.
    `[S]` canonicalises to `S`, `h` to `H` and `{#}/[HPF]` to `/[HPF]`. Re-feeding an emitted unit
    through the transform therefore changes it even when the build was correct, and an
    idempotence-based check would wrongly fail many correct observations.

    The rule here asks whether the table has an opinion about this unit that the build ignored:
      * a standard target                      -> emitted by a conversion, fine
      * a table entry that maps to itself      -> nothing to convert, fine
      * log-scale                              -> the guard refused, fine
      * the table has no entry for it          -> pass-through, fine
      * anything else                          -> the table said convert and the build did not
    """
    if unit in STANDARD_TARGETS or is_log_scale(unit):
        return True
    hit = STANDARD_UNIT.get(unit)
    if hit is not None:
        return hit[0] == unit
    canon = canonical_unit(unit, alias_map)
    return canon is None or canon not in STANDARD_UNIT
