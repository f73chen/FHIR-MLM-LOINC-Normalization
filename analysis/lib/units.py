"""Label-preserving unit alternatives and conversion factors, for the unit analyses behind the unit
randomisation (analysis/a16_unit_collision.py, and a19 and a20 through it; thesis Chapter 4).

The value and unit pair is a strong code-identifying key in the data (analysis/a15). The question is
whether re-expressing observations in equivalent units, as a second site would, breaks that key by
making different analytes collide.

The hard constraint is that a conversion must not change the correct LOINC code. LOINC's PROPERTY axis
separates mass concentration (MCnc) from substance concentration (SCnc), and those are different
codes. So a conversion between mass and molar units is forbidden. This module enforces it in three ways:
  1. dimension vector and mole exponent must match  (mg/dL is dim mass/volume, mmol/L is mole/volume)
  2. the pair must not appear in Hauser et al. JAMIA 2018's table of LOINC-code-changing conversions
  3. the pair must not appear in the PrecisionHealthIntelligence harmonisation map

Sources (all read from UPSTREAM_DIR):
  * unit strings          -> unit_synonym_map.json  (5,942 aliases -> canonical UCUM)
  * magnitude + dimension -> lab2clean v2.0.0 `parsed_units_df` (GPL-3), 1,540 pre-parsed UCUM codes
  * attested units/code   -> loinc_measurements_enriched.jsonl (MIMIC/LOINC/UCUM/CDISC/CIEL)
  * LOINC property/unit   -> ucum.csv (NLM ucum-lhc)
  * magnitude source file -> lab2clean_parsed_units_df.rda (extracted from the lab2clean v2.0.0 R
                             package; GPL-3, and not included in this repository)
  * blocklists            -> loinc_unit_conversions.csv (Hauser 2018, 285 rows)
                             loinc_unit_harmonization/loinc_conversion_map.csv (814 rows)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

FHIR = locations.UPSTREAM
SYNONYM_MAP = FHIR / "unit_synonym_map.json"
MEASUREMENTS = FHIR / "loinc_measurements_enriched.jsonl"
UCUM_CSV = FHIR / "ucum.csv"
HAUSER = FHIR / "loinc_unit_conversions.csv"
HARMONISATION = FHIR / "loinc_unit_harmonization" / "loinc_conversion_map.csv"
LAB2CLEAN_RDA = FHIR / "lab2clean_parsed_units_df.rda"


# --------------------------- unit string normalisation ---------------------------
def _norm_key(u: str) -> str:
    """Lookup key for the alias map: lowercase, whitespace removed.

    MIMIC writes `mm hg`, and the alias map keys on `mmhg`. Stripping whitespace recovers observations
    that would otherwise be unresolvable."""
    return re.sub(r"\s+", "", str(u)).lower()


class UnitTable:
    """Canonical UCUM code -> (magnitude, dimension, mole exponent), plus alias resolution."""

    def __init__(self):
        self.alias = json.loads(SYNONYM_MAP.read_text())["alias_to_canonical"]
        self._alias_norm = {_norm_key(k): v for k, v in self.alias.items()}
        self.mag, self.dim, self.mole = {}, {}, {}
        self._load_parsed_units()

    def _load_parsed_units(self):
        import rdata
        obj = rdata.read_rda(str(LAB2CLEAN_RDA))
        df = obj[list(obj)[0]]
        cols = {str(c): c for c in df.columns}
        for cs, m, d, me in zip(df[cols["csCode_"]], df[cols["magnitude_"]],
                                df[cols["dim_"]], df[cols["moleExp_"]]):
            cs = str(cs)
            if cs in self.mag:
                continue
            try:
                self.mag[cs] = float(m)
            except (TypeError, ValueError):
                continue
            self.dim[cs] = _parse_dim(d)
            self.mole[cs] = float(me) if me == me else 0.0

    def canonical(self, raw: str | None) -> str | None:
        """MIMIC/free-text unit string -> canonical UCUM code, or None if unresolvable."""
        if raw is None or raw != raw:
            return None
        k = _norm_key(raw)
        return self._alias_norm.get(k)

    def known(self, ucum: str) -> bool:
        return ucum in self.mag

    def commensurable(self, a: str, b: str) -> bool:
        """Same physical dimension and same mole exponent.

        The mole exponent keeps mass and molar apart. mg/dL has moleExp 0 and mmol/L has 1, and both
        have dimension (-3,0,...) once mass is folded in. So dimension alone is not enough."""
        if a not in self.mag or b not in self.mag:
            return False
        return self.dim.get(a) == self.dim.get(b) and self.mole.get(a) == self.mole.get(b)

    def factor(self, src: str, dst: str) -> float | None:
        """Multiply a value in `src` by this to express it in `dst`. None if not commensurable."""
        if not self.commensurable(src, dst):
            return None
        d = self.mag[dst]
        return None if d == 0 else self.mag[src] / d


def _parse_dim(d) -> tuple | None:
    """lab2clean stores the dimension vector as the R literal string 'c(-3, 0, 1, 0, 0, 0, 0)'."""
    if d is None or (isinstance(d, float) and d != d):
        return None
    s = str(d)
    nums = re.findall(r"-?\d+", s)
    return tuple(int(x) for x in nums) if nums else None


# --------------------------- blocklists ---------------------------
def load_blocked_code_pairs() -> set:
    """LOINC code pairs related by a unit conversion, i.e. converting between them changes the code.

    Hauser et al. JAMIA 2018;25(2):192-6 screened 2,398 LOINC codes and found 285 such pairs, mostly
    between mass and molar units. Returned unordered so membership can be tested either way."""
    import csv
    out = set()
    with open(HAUSER) as f:
        for r in csv.DictReader(f):
            a, b = r.get("loinc_a"), r.get("loinc_b")
            if a and b:
                out.add(frozenset((a.strip(), b.strip())))
    return out


def load_blocked_code_unit() -> set:
    """(loinc_code, unit) combinations that the harmonisation map says belong to a different code.

    e.g. 2339-0 (Glucose Bld-mCnc) Glucose [Mass/volume] + `mmol/L` -> 15074-8 (Glucose Bld-sCnc)
    Glucose [Moles/volume]. Any generated alternative that lands on one of these is rejected."""
    import csv
    out = set()
    with open(HARMONISATION) as f:
        for r in csv.DictReader(f):
            code, unit = r.get("old_loinc_code"), r.get("unit_display")
            if code and unit:
                out.add((code.strip(), _norm_key(unit)))
    return out


# --------------------------- attested alternatives ---------------------------
def load_attested_units() -> dict:
    """LOINC code -> {source: [unit strings]} for the quantitative bucket only.

    Sources are systems that use the unit for that code: MIMIC train split, LOINC
    EXAMPLE_UNITS, LOINC EXAMPLE_UCUM_UNITS, CDISC LBORRESU, CIEL. Built by
    prep/4.3_collect_value_units.py."""
    out = {}
    with open(MEASUREMENTS) as f:
        for line in f:
            r = json.loads(line)
            m = r.get("measurements") or {}
            per = {}
            for src, d in m.items():
                v = (d or {}).get("quantitative") or []
                if v:
                    per[src] = list(v)
            if per:
                out[r["loinc_code"]] = per
    return out


def load_ucum_property() -> dict:
    """UCUM code -> LOINC PROPERTY string, from the NLM ucum-lhc table.

    Used only by the property-grouped (noisier) alternative tier. The attested tier does not need it."""
    import csv
    out = {}
    with open(UCUM_CSV) as f:
        for r in csv.DictReader(f):
            code = (r.get("case-sensitive code") or r.get("csCode_") or "").strip()
            prop = (r.get("LOINC property") or r.get("loincProperty_") or "").strip()
            if code and prop:
                out[code] = prop
    return out
