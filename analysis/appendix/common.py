"""Shared helpers for the appendix scripts (thesis Appendices A–F).

They read the pipeline's code, constants and outputs and never write to them.
"""
import csv
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import locations  # noqa: E402

REPO = locations.REPO
RERUN = REPO
SCRIPTS = RERUN / "pipeline"
ANALYSIS = RERUN / "analysis"
RESULTS = RERUN / "results"
FHIR = locations.COMMON
#: LoincTable/Loinc.csv of the LOINC 2.82 release
LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
UNIT_ALIAS_JSON = locations.UPSTREAM / "unit_synonym_map.json"

for _p in (str(SCRIPTS), str(FHIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@lru_cache(maxsize=1)
def loinc_table() -> dict:
    """LOINC_NUM -> its row in the LOINC 2.82 table."""
    with open(LOINC_CSV, encoding="utf-8") as f:
        return {r["LOINC_NUM"]: r for r in csv.DictReader(f)}


def long_name(code: str) -> str:
    return loinc_table()[code]["LONG_COMMON_NAME"]


def read_csv(path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(n, d, nd: int = 1) -> str:
    """'88.0%'. Shares below the last shown digit print as '<0.01%' rather than '0.00%'."""
    v = 100 * n / d
    if 0 < v < 10 ** -nd:
        return f"<{10 ** -nd:.{nd}f}%"
    return f"{v:.{nd}f}%"


def num(n) -> str:
    """Integers with thousands separators."""
    return f"{int(n):,}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def factor(f: float) -> str:
    """A conversion factor as a table column prints it. Powers of ten print as 10³, 10⁻¹ or 10¹², and
    other factors as 1, 60 or 3,600."""
    import math
    e = round(math.log10(f))
    if e not in (0, 1) and abs(f - 10.0 ** e) <= 1e-9 * 10.0 ** e:
        return "10" + str(e).translate(_SUP)
    return f"{f:,.0f}" if f == int(f) else f"{f:,}"
