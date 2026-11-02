"""Simulate a second clinical site, per patient. `perturb_patient` perturbs each patient's lab values
and reference-range bounds by literature magnitudes of between-site variation. `randomise_units`
picks one of two equally valid unit spellings. Together they are the site simulation of the v3
representation (thesis Chapters 4 and 5; Appendix B for the magnitudes and guards, Appendix A for
the randomised units).

In MIMIC, a lookup keyed on the reference range alone identifies the LOINC code for most
observations that state one. That key does not transfer to a site whose ranges plausibly differ.
Perturbation replaces MIMIC's single stated range per code with a distribution of plausible ones.
For example, a potassium range of 3.5–5.0 might read 3.4–5.2 after perturbation. Ranges should
still tell analytes apart, but an exact match to MIMIC's range for a code should no longer work.

The perturbation runs once per patient, over the whole resource list, before formatting or
tokenising. It works per patient because two guards need panel context. MCHC is recomputed from Hb
and Hct, and the differential is renormalised across all five components.

The draws are deterministic. Every draw comes from SHA1(salt:uuid:loinc:kind), as in
`paraphrase_utils.patient_rng`, so it is stable across processes whatever PYTHONHASHSEED is. A
patient gets one draw per analyte and kind (value or range), so the patient's repeated measurements
of the same analyte all move together.

The magnitudes are not computed here. They come from the fixed table
data/variability/perturbation_spec.csv, which pipeline/build_perturbation_spec.py builds. The
dataset builders read it through `load_spec`, and the magnitude tables in Chapter 4 and Appendix B
read the same file.

Operates on resource dicts in memory. The only files read are the spec and unit-randomisation
tables.
"""
from __future__ import annotations

import csv
import hashlib
import math
import struct
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parents[1] / "data" / "variability" / "perturbation_spec.csv"
DEFAULT_SALT = "xsite-v3"

#: the code system of the model head. Perturbation is scoped to it.
LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"


def lab_loinc_of(obs: dict, mimic_to_loinc: dict):
    """The LOINC this observation is perturbed under, or None. Every builder calls this function.

    It covers the lab system only. The MLM path's notion of "lab" also includes chartevents
    (paraphrase_utils.DEFAULT_LAB_SYSTEMS). 09's mimic-to-LOINC map is unfiltered, while 06's keeps
    only `observation_type == "lab"`. If each builder used its own map, some chartevents codes would
    be perturbed for FHIR-MLM but left as they are for TF-IDF, and the paired comparison between the
    models would compare different inputs.
    """
    codings = (obs.get("code") or {}).get("coding") or []
    if not codings or codings[0].get("system") != LAB_CODE_SYSTEM:
        return None
    return mimic_to_loinc.get(codings[0].get("code"))

#: G7 never emits more decimals than this, whatever the input looks like. It only guards against a
#: float-repr artefact (`1.1000000000000001`), and a real MIMIC value never reaches it. The
#: `decimals()` helper in analysis/a30_measurement_scales.py uses the same ceiling.
MAX_EMIT_DECIMALS = 6


def value_decimals(value, cap: int = MAX_EMIT_DECIMALS) -> int:
    """How many decimal places this observation was written with. Same rule as
    analysis/a30_measurement_scales.py."""
    try:
        t = repr(float(value))
    except (TypeError, ValueError):
        return cap
    if "e" in t or "E" in t:
        return cap
    return min(len(t.split(".")[1].rstrip("0")) if "." in t else 0, cap)


def emit_decimals(value, spec: dict, cap: int = MAX_EMIT_DECIMALS) -> int:
    """G7's precision: the greater of the observation's own decimal count and the code's modal one,
    capped at `cap`.

    `native_decimals` is the modal decimal count over the code's observations
    (analysis/a30_measurement_scales.py). For a code whose site varies its precision, the modal count
    alone would round small nucleated-erythrocyte ratios and TSH results to 0.0. The observation's
    own count alone reads 12.0 as zero decimals (`repr(12.0)` strips to nothing after the point), so
    a haemoglobin reference low of 12.0 or a creatinine of 1.0 could never move by 0.1. The maximum
    keeps TSH's 0.012 at three decimals and still lets creatinine's 1.0 move to 1.1.
    """
    return min(max(value_decimals(value, cap), int(spec.get("decimals") or 0)), cap)


#: the roles G3 recomputes, and the formula each uses. Mirrors DERIVED in
#: pipeline/build_perturbation_spec.py.
DERIVED_FORMULA = {
    "MCV":  (("Hct", "RBC"), lambda hct, rbc: hct / rbc * 10.0),
    "MCH":  (("Hb", "RBC"),  lambda hb, rbc: hb / rbc * 10.0),
    "MCHC": (("Hb", "Hct"),  lambda hb, hct: hb / hct * 100.0),
}


# ------------------------------- the spec table -------------------------------
def load_spec(path: Path = SPEC_PATH) -> dict:
    """loinc -> spec dict, read once and passed in. Nothing recomputes it at build time."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["loinc"]] = {
            "tier0": r["tier0_no_perturb"] == "1",
            "value_cv": float(r["value_cv_pct"]) if r["value_cv_pct"] else None,
            "range_pctV": float(r["range_pctV"]) if r["range_pctV"] else None,
            "scale": r["scale_class"] or "ratio",
            "kind": r["magnitude_kind"] or "relative_pct",
            "decimals": int(r["native_decimals"] or 2),
            "allow_negative": r["allow_negative"] == "1",
            "clamp_0_100": r["clamp_0_100"] == "1",
            "derived_role": r["derived_role"] or None,
            "parent_role": r["parent_role"] or None,
            "fraction_panel": r["fraction_panel"] == "1",
            "ladder": [float(x) for x in r["ladder_values"].split()] if r["ladder_values"] else None,
        }
    return out


# ------------------------------- deterministic draws -------------------------------
def site_z(uuid: str, loinc: str, kind: str, salt: str = DEFAULT_SALT) -> float:
    """A standard-normal draw, deterministic in (patient, analyte, kind).

    SHA1 keeps the value stable across processes and machines (see paraphrase_utils.patient_rng).
    Box-Muller runs over two 32-bit words of the digest, with u1 clamped away from 0.

    `kind` separates the value draw from the range draw, so the two are independent. Across
    manufacturer groups, which a26 uses as a proxy for sites, value error and reference-range shift
    are uncorrelated (analysis/a26_value_range_coupling.py).
    """
    d = hashlib.sha1(f"{salt}:{uuid}:{loinc}:{kind}".encode()).digest()
    a, b = struct.unpack(">II", d[:8])
    u1 = max((a + 0.5) / 4294967296.0, 1e-12)
    u2 = (b + 0.5) / 4294967296.0
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ------------------------------- one number -------------------------------
def _nearest_rung(value: float, ladder, steps: int):
    """Move `steps` rungs along the observed reporting ladder, clamped at its ends.

    Returns None when `value` is not on the ladder, and the caller then leaves it alone.

    The ladder is built from a code's observed values, and a reference bound need not lie on the
    same grid. Urine protein's ladder starts at 10 mg/dL (10, 15, 20, 25, 30, 50, ...), while its
    stated reference interval is 0–0, i.e. "negative". Snapping to the nearest rung would turn a
    reference range of 0–0 into 15–15 mg/dL. Several rung codes state such an off-ladder "none
    detected" zero: urine protein, glucose and ketones, and log viral loads.

    "On the ladder" is judged against the local spacing. A value that would round onto a rung
    still steps (5.2 on a half-step pH ladder). A value that is plainly not a rung does not (0
    against a ladder starting at 10).
    """
    i = min(range(len(ladder)), key=lambda j: abs(ladder[j] - value))
    if len(ladder) > 1:
        nbr = ladder[i + 1] if i == 0 else ladder[i - 1]
        if abs(value - ladder[i]) > abs(nbr - ladder[i]) / 2.0:
            return None
    elif value != ladder[0]:
        return None
    return ladder[max(0, min(len(ladder) - 1, i + steps))]


def perturb_number(value, z: float, spec: dict, magnitude: float, severity: float = 1.0):
    """One number, under the scale rule its code lives on (G6), re-rounded (G7).

    Returns the value unchanged when it cannot be perturbed meaningfully. That happens for a
    non-number and for an exact zero under a relative rule. The zero case is a no-op by construction.
    A second site would also report zero eosinophils as zero.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(v):
        return value
    kind, mag = spec["kind"], magnitude * severity
    nd = emit_decimals(value, spec)                   # G7

    # Dispatch on the declared kind. Inferring it from the scale class would give log codes a
    # relative CV that this function applies as an absolute offset, which turns a blood pH of 7.40
    # into 12.5.
    if kind == "absolute":
        out = v + z * mag                       # mag is in the value's units (pH units)
    elif kind == "excess_over_1":
        out = 1.0 + (v - 1.0) * (1.0 + z * mag / 100.0)
    elif kind == "rung":
        if not spec["ladder"]:
            return value                        # nothing to step along; leave it alone
        stepped = _nearest_rung(v, spec["ladder"], 1 if z > 0 else -1)
        return value if stepped is None else stepped
    elif kind == "poisson":
        out = v + z * math.sqrt(max(v, 0.0))
    else:                                       # relative_pct
        out = v * (1.0 + z * mag / 100.0)

    if not spec["allow_negative"] and out < 0:          # G1
        out = 0.0
    if spec["clamp_0_100"]:                             # G2
        out = min(100.0, max(0.0, out))
    return round(out, nd)                               # G7


# ----------------------- unit-notation randomisation -----------------------
RANDOMISE_PATH = (Path(__file__).resolve().parents[1] / "data" / "variability"
                  / "unit_randomisation.csv")


def load_randomisation(path: Path = RANDOMISE_PATH) -> dict:
    """loinc -> (notation_a, notation_b). Fixed by pipeline/build_unit_randomisation.py."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["loinc"]] = (r["notation_a"], r["notation_b"])
    return out


def randomise_units(resources, uuid, table, code_of, *, salt=DEFAULT_SALT):
    """Pick one of the two equally valid unit spellings per (patient, notation family), and write it
    onto the resource. The value is untouched.

    It acts on the resource because the resource is the only place FHIR-MLM sees the unit.
    FHIR-MLM's unit is a code token looked up from `Quantity.code`, so this is the only part of the
    v2/v3 unit handling that reaches it. Done in the renderers, the randomisation would never reach
    FHIR-MLM's input.

    The reference-range bounds are rewritten too. Randomising the value to `mmol/L` while leaving
    its range in `mEq/L` would give a mix of units that no real site reports.

    Unit randomisation is not part of the perturbation. It is part of the v3 representation and is
    on in every arm, and the builders turn it off only with --no-randomise-units.
    `perturb_values=False` and `perturb_ranges=False` do not disable it. With both switches off,
    `perturb_patient` changes nothing, but the units are still randomised.

    There is one draw per patient per notation family. A draw per (patient, LOINC), as the
    perturbation magnitudes use, would give many patients a mixed electrolyte panel, with chloride
    in `mmol/L` beside potassium in `mEq/L` on the same specimen at the same site. No laboratory
    information system does that, because the LIS sets the convention for every assay it reports.
    Keying the draw on the pair makes all the family's codes move together.

    Another family gets an independent draw, since a laboratory's choice between eq and mol says
    nothing about any other notation choice it makes.

    The draw uses the kind `unit`, so it is independent of the value and range draws, and adding it
    does not shift them.
    """
    n = 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        loinc = code_of(r)
        pair = table.get(loinc) if loinc else None
        if not pair:
            continue
        family = f"{pair[0]}|{pair[1]}"
        chosen = pair[0] if site_z(uuid, family, "unit", salt) < 0 else pair[1]
        qs = [r.get("valueQuantity")]
        qs += [rr.get(k) for rr in (r.get("referenceRange") or []) for k in ("low", "high")]
        for q in qs:
            if isinstance(q, dict) and (q.get("unit") is not None or q.get("code") is not None):
                q["unit"] = q["code"] = chosen
                n += 1
    return n


# ------------------------------- one patient -------------------------------
def _panel_key(obs: dict) -> str:
    """Observations drawn together, keyed on `effectiveDateTime`. Grouping by position in the file
    would do for the red-cell indices, which are always drawn in one CBC, but it mis-groups the
    sparser panels."""
    return str(obs.get("effectiveDateTime") or obs.get("issued") or "")


def _qty(obs):
    return obs.get("valueQuantity") if isinstance(obs.get("valueQuantity"), dict) else None


def perturb_patient(resources, uuid, spec, code_of, *, severity=1.0,
                    perturb_values=True, perturb_ranges=True, salt=DEFAULT_SALT):
    """Perturb one patient's observations, in place.

      resources   the patient's FHIR resource list
      uuid        patient id and site identity, with one draw per (uuid, analyte, kind)
      spec        load_spec()
      code_of     resource -> LOINC code or None (the builders pass lab_loinc_of)

    The order matters, and it is the reason this works per patient:
      1. perturb every ordinary value
      2. recompute the derived indices from their perturbed parents (G3), without perturbing them
      3. renormalise each fraction panel to the sum it had before (G4)
      4. perturb the reference-range bounds with one factor shared by both. Lower and upper limits
         are strongly correlated across sites, so a single multiplier is justified, and it cannot
         invert the interval.

    Returns the number of fields touched, which the builders record and the tests check.
    """
    obs_list, touched = [], 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        loinc = code_of(r)
        if not loinc or loinc not in spec:
            continue
        s = spec[loinc]
        if s["tier0"]:                                   # G0: not a specimen measurement
            continue
        obs_list.append((r, loinc, s))

    # Capture each fraction panel's original sum before anything is perturbed. Computed
    # afterwards, the renormalisation would be a no-op with factor 1.0.
    panel_sum_before, orig_decimals = {}, {}
    for r, loinc, s in obs_list:
        if (q := _qty(r)) and q.get("value") is not None:
            # captured before step 1 overwrites the value, because G4 renormalises an
            # already-perturbed number and by then the original precision is gone
            orig_decimals[id(r)] = emit_decimals(q["value"], s)
        if s["fraction_panel"] and (q := _qty(r)) and q.get("value") is not None:
            k = _panel_key(r)
            panel_sum_before[k] = panel_sum_before.get(k, 0.0) + float(q["value"])

    # ---- 1. ordinary values, skipping the derived ones -------------------------
    if perturb_values:
        for r, loinc, s in obs_list:
            if s["derived_role"]:
                continue                                  # step 2 owns these
            q = _qty(r)
            if q is None or q.get("value") is None or s["value_cv"] is None:
                continue
            new = perturb_number(q["value"], site_z(uuid, loinc, "value", salt), s,
                                 s["value_cv"], severity)
            if new != q["value"]:
                touched += 1
            q["value"] = new

        # ---- 2. G3: recompute, never perturb ------------------------------------
        parents = {}
        for r, loinc, s in obs_list:
            if s["parent_role"] and (q := _qty(r)) and q.get("value") is not None:
                parents.setdefault(_panel_key(r), {})[s["parent_role"]] = float(q["value"])
        for r, loinc, s in obs_list:
            role = s["derived_role"]
            if not role:
                continue
            q = _qty(r)
            if q is None or q.get("value") is None:
                continue
            need, fn = DERIVED_FORMULA[role]
            have = parents.get(_panel_key(r), {})
            if not all(k in have for k in need) or any(have[k] == 0 for k in need):
                continue          # incomplete panel: leave it alone rather than invent a value
            # the derived value is untouched at this point (step 1 skips derived roles), so
            # q["value"] is still the original and carries the precision it was reported with
            q["value"] = round(fn(*[have[k] for k in need]), emit_decimals(q["value"], s))
            touched += 1

        # ---- 3. G4: a differential still sums to what it summed to ---------------
        panels = {}
        for r, loinc, s in obs_list:
            if s["fraction_panel"] and (q := _qty(r)) and q.get("value") is not None:
                panels.setdefault(_panel_key(r), []).append((r, loinc, s))
        for pkey, members in panels.items():
            if len(members) < 2:
                continue
            before = panel_sum_before.get(pkey, 0.0)
            after = sum(float(_qty(r)["value"]) for r, _l, _s in members)
            if after <= 0 or before <= 0:
                continue
            k = before / after
            for r, _l, s in members:
                q = _qty(r)
                q["value"] = round(float(q["value"]) * k,
                                   orig_decimals.get(id(r), emit_decimals(q["value"], s)))

    # ---- 4. reference-range bounds -------------------------------------------
    if perturb_ranges:
        for r, loinc, s in obs_list:
            if s["range_pctV"] is None:
                continue
            z = site_z(uuid, loinc, "range", salt)        # independent of the value draw
            for rr in (r.get("referenceRange") or []):
                for key in ("low", "high"):
                    q = rr.get(key)
                    if not isinstance(q, dict) or q.get("value") is None:
                        continue
                    q["value"] = perturb_number(q["value"], z, s, s["range_pctV"], severity)
                    touched += 1
    return touched
