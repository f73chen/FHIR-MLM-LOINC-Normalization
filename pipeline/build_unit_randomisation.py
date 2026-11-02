"""Freeze the unit-notation randomisation table (thesis Chapter 4 and Appendix A). It lists the codes
whose unit each patient draws in one of two notations, to simulate a second site's spelling.

analysis/a20_unit_code_concentration.py lists the LOINC codes where the MIMIC unit and the terminology
unit disagree. analysis/a31_unit_disagreement_resolution.py sorts them by one test: can the
disagreement be resolved without knowing the code? At inference the code is the label, so a rule that
needs it is circular. A disagreement can merge, as with a dropped UCUM annotation or
`[IU]` == `U` == `[arb'U]`. It can be a code-mapping problem. Otherwise it can only be randomised,
because the conversion factor is real but cannot be derived from the unit string.

The codes kept here alternate between `mEq/L` and `mmol/L`. The factor between equivalents and
moles depends on valence, which no code-blind rule can read off the unit. The standardiser therefore
cannot merge them, and the notation survives as a site artefact. `mEq/L` is used by one family of
codes only, which makes it a family-level key that a second site writing `mmol/L` would break.
Every code kept here is monovalent, so the value is unchanged. The script checks that both spellings
standardise at factor 1.0.

The pH codes, which would alternate `[arb'U]` and `[pH]`, are excluded for the three reasons below.
The third, about the vocabulary, is the weakest.
  1. Randomising swaps one pH-family token for another. Nearly all of `[arb'U]`'s observations are
     pH, so `[pH]` says nothing `[arb'U]` did not. Neither spelling discriminates among the pH codes,
     which is the actual classification problem.
  2. It is speculative. MIMIC only ever writes `units` for pH. `[pH]` is the terminology's preferred
     unit, and no other site has been observed writing it. The eq/mol case, by contrast, is a real
     ambiguity between two spellings that are both in clinical use.
  3. No spelling of pH (`pH`, `[pH]`, `PH`) is in FHIR-MLM's unit vocabulary, so randomising on the
     resource would emit `[UNK]` there. Doing it in the text models only would break the cross-family
     input parity that the paired comparisons rest on.

Both spellings of a pair must be in the FHIR-MLM unit vocabulary, and the script exits if they are
not. Randomisation writes a raw unit onto the resource. That is the only place FHIR-MLM sees the
unit, because its unit token is read from `Quantity.code`. Unlike the standardised targets, an
unknown spelling here would therefore tokenise to [UNK].

Reads results/analysis/a31/disagreement_resolution.csv and FHIR_MLM_TOKENIZER_DIR/codes_to_tokens.pkl.
Writes data/variability/unit_randomisation.csv. Aggregate output only.

  python pipeline/build_unit_randomisation.py
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import unit_standardisation as us  # noqa: E402
from input_preprocessing_utils import load_unit_alias_map  # noqa: E402

A31 = ROOT / "results" / "analysis" / "a31" / "disagreement_resolution.csv"
OUT = ROOT / "data" / "variability" / "unit_randomisation.csv"
UNIT_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-units"
CODES_TO_TOKENS = str(locations.TOKENIZER / "codes_to_tokens.pkl")
UNIT_SYNONYM_PATH = str(locations.UPSTREAM / "unit_synonym_map.json")

#: canonical pair (as a31 writes it) -> the two raw surface forms to alternate between. The forms are
#: raw because they are written onto the resource before anything standardises them.
PAIRS = {("meq/L", "mmol/L"): ("mEq/L", "mmol/L")}
#: a31 pairs that are not randomised, with the reason for each
EXCLUDED = {("[arb'U]", "[pH]"): "no spelling of pH is in the MLM unit vocabulary; see the module docstring"}


def main() -> int:
    with open(CODES_TO_TOKENS, "rb") as fh:
        vocab = set(pickle.load(fh)[UNIT_SYSTEM])
    amap = load_unit_alias_map(UNIT_SYNONYM_PATH)
    rows, skipped = [], []
    for r in csv.DictReader(open(A31)):
        if r["resolution"] != "RANDOMISE":
            continue
        key = (r["mimic_unit"], r["external_unit"])
        if key in EXCLUDED:
            skipped.append((r, EXCLUDED[key]))
            continue
        if key not in PAIRS:
            sys.exit(f"a31 RANDOMISE pair {key} is in neither PAIRS nor EXCLUDED. Add it to one of them.")
        a, b = PAIRS[key]
        for surface in (a, b):
            if surface not in vocab:
                sys.exit(f"{r['loinc']}: surface {surface!r} is not in the {len(vocab)}-entry "
                         "unit vocabulary, so it would tokenise to [UNK]")
        # the value must not move: both spellings have to standardise at factor 1.0
        fa, fb = us.standardise_unit(a, amap)[1], us.standardise_unit(b, amap)[1]
        if (fa, fb) != (1.0, 1.0):
            sys.exit(f"{r['loinc']}: {a!r} factor {fa}, {b!r} factor {fb}. Randomising would "
                     "change the value, which a31 says it must not.")
        rows.append({"loinc": r["loinc"], "component": r["component"], "n_obs": int(r["n_obs"]),
                     "notation_a": a, "notation_b": b,
                     "target_a": us.standardise_unit(a, amap)[0],
                     "target_b": us.standardise_unit(b, amap)[0]})
    if not rows:
        sys.exit("no codes resolved to RANDOMISE. Check a31.")
    rows.sort(key=lambda r: -r["n_obs"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    n = sum(r["n_obs"] for r in rows)
    print(f"{len(rows)} codes randomised, {n:,} observations")
    print(f"  {'loinc':<9}{'component':<26}{'obs':>10}   notation A <-> B   (targets)")
    for r in rows[:10]:
        print(f"  {r['loinc']:<9}{r['component'][:25]:<26}{r['n_obs']:>10,}   "
              f"{r['notation_a']} <-> {r['notation_b']}   ({r['target_a']} / {r['target_b']})")
    if len(rows) > 10:
        print(f"  ... and {len(rows)-10} more")
    print(f"\n  excluded: {len(skipped)} codes, "
          f"{sum(int(r['n_obs']) for r, _ in skipped):,} observations")
    for r, why in skipped[:4]:
        print(f"     {r['loinc']:<9}{r['component'][:25]:<26}{int(r['n_obs']):>10,}   "
              f"{r['mimic_unit']} <-> {r['external_unit']}: {why}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
