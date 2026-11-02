"""Build the qualitative-value normalisation map (thesis Appendix D).

The map lists each PCORnet CDM v1.14 QUAL value with its spelling variants. The TF-IDF dataset
builders (pipeline/06, pipeline/16) use it to standardise qualitative results.

Reads UPSTREAM_DIR/PCORnet_CDM_ValueSet_ReferenceFile_v1.14_QUAL.csv and
pcornet_qual_manual_aliases.json (installed by prep/0_install_resources.py). Writes
UPSTREAM_DIR/pcornet_qual_normalize.json as {canonical value: [variants]}. Every string in it is
passed through preprocess_qual_string.

  python prep/4.4_pcornet_qual_normalize.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
from input_preprocessing_utils import preprocess_qual_string

PCORNET_QUAL_CSV = locations.UPSTREAM / "PCORnet_CDM_ValueSet_ReferenceFile_v1.14_QUAL.csv"
MANUAL_ALIASES_PATH = locations.UPSTREAM / "pcornet_qual_manual_aliases.json"
NORMALIZE_PATH      = locations.UPSTREAM / "pcornet_qual_normalize.json"

def _read_pcornet_qual_csv():
    with open(PCORNET_QUAL_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = (row.get("CODE") or "").strip()
            desc = (row.get("DESCRIPTIVE_TEXT") or "").strip()
            if code:
                yield code, desc


def build_normalize_map():
    """Rebuild pcornet_qual_normalize.json from the PCORnet QUAL value set and the manual aliases.
    The JSON is a derived file. To change it, edit the aliases."""
    canonical_to_vars = {}
    for code, desc in _read_pcornet_qual_csv():
        canon = preprocess_qual_string(code)
        if not canon:
            continue
        vars_ = {canon}
        desc_p = preprocess_qual_string(desc)
        if desc_p and desc_p != canon:
            vars_.add(desc_p)
        canonical_to_vars[canon] = vars_

    manual_aliases = json.loads(MANUAL_ALIASES_PATH.read_text())
    unmatched = []
    for cat, syns in manual_aliases.items():
        canon = preprocess_qual_string(cat)
        if canon not in canonical_to_vars:
            unmatched.append(cat)
            continue
        for syn in syns:
            v = preprocess_qual_string(syn)
            if v:
                canonical_to_vars[canon].add(v)

    assert not unmatched, (
        "manual_aliases.json contains categories with no matching PCORnet "
        f"canonical. Fix {MANUAL_ALIASES_PATH.name} (typo?): {unmatched}"
    )

    payload = {c: sorted(v) for c, v in sorted(canonical_to_vars.items())}
    NORMALIZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORMALIZE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {NORMALIZE_PATH}: {len(payload)} canonicals, "
          f"{sum(len(v) for v in payload.values())} variations "
          f"(derived from {PCORNET_QUAL_CSV.name} + {MANUAL_ALIASES_PATH.name}).")


build_normalize_map()
