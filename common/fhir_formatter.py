"""
FHIR-Bundle-to-text formatter for lab-code prediction (thesis Chapters 2 and 5). It renders a
patient's resources as plain text for the text models (BCM and Llama), with a mask token at each
lab-code position. It is tokenizer-agnostic, and the caller passes the mask token that its
tokenizer treats as a single token.

Follows the field-extraction logic of the FHIR-MLM tokenizer (NdjsonTokenizer,
third_party/fhir_mlm/fhir_tokenizer.py) but emits natural-language text instead of token IDs.

The formatter also emits the medication and dosage of a MedicationRequest, which the FHIR-MLM Arrow
dataset lacks because of a missing comma in its original relevant_keys.

Where the FHIR-MLM tokenizer emits nothing, the formatter emits nothing too:
  - Specimen and Observation.collection: Specimen has no relevant_keys entry, and Observation
    has no .collection attribute.
  - a valueQuantity whose item code has no bins in labevent_bins.pkl,
    medication_dose_bins.pkl or medication_dispense_max_dose_bins.pkl.
  - Procedure location: encounter:reference:location:name reaches a .name attribute that
    EncounterLocation does not have.
Only mimic-d-labitems Observations get `Code: <mask>`. Other Observations
(e.g. mimic-microbiology-test) emit `Code: <code_text>`.
"""

from __future__ import annotations
import pickle
import sys
from pathlib import Path
from typing import Optional

import ndjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locations  # noqa: E402


LAB_CODE_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"

# The path to the date that orders each resource type, from the original FHIR-MLM
# tokenization code (its `sort_key_function`).
DATE_KEYS = {
    "Patient": ["birthDate"],
    "Encounter": ["period", "start"],
    "Condition": ["encounter", "end"],          # via encounter resolution
    "Procedure": ["performedDateTime"],
    "Observation": ["effectiveDateTime"],
    "MedicationRequest": ["authoredOn"],
    "MedicationDispense": ["context", "end"],   # via encounter resolution
    "MedicationAdministration": ["effectiveDateTime"],
    "Specimen": ["collection", "collectedDateTime"],
}

# Patient.extension URLs that NdjsonTokenizer's Extension handler recognizes
# (third_party/fhir_mlm/fhir_tokenizer.py). Other URLs are dropped.
EXTENSION_HANDLERS = {
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race": "Race",
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity": "Ethnicity",
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex": "BirthSex",
}


def get_encounters(resources: list[dict]) -> dict[str, dict]:
    """Build {encounter_id: encounter.period} for sort-key resolution, as the
    original FHIR-MLM tokenization code does."""
    return {
        r["id"]: r.get("period", {})
        for r in resources
        if r.get("resourceType") == "Encounter"
    }


def sort_key(resource: dict, encounters: dict[str, dict]) -> str:
    """Lexicographic ISO-date sort key. It is sort_key_function from the
    original FHIR-MLM tokenization code, unchanged, so the FHIR-MLM and text
    encodings order resources the same way."""
    rtype = resource.get("resourceType", "")
    keys = DATE_KEYS.get(rtype, [])
    date = resource
    for k in keys:
        if k in ("encounter", "context"):
            ref = date.get(k, {}).get("reference", "/").split("/")[-1]
            date = encounters.get(ref, {})
        else:
            date = date.get(k, {})
    if date:
        return date if isinstance(date, str) else "z"
    # Fallback: linked encounter end-time
    if (ref := resource.get("encounter", {}).get("reference", "/").split("/")[-1]):
        date = encounters.get(ref, {}).get("end")
        if date:
            return date
    if (ref := resource.get("context", {}).get("reference", "/").split("/")[-1]):
        date = encounters.get(ref, {}).get("end")
        if date:
            return date
    return "z"


class FHIRFormatter:
    """Convert a per-patient FHIR ndjson into a prompt + label list. The
    `mask_token` is whatever string the downstream tokenizer treats as a single
    token at lab-code positions, and the caller must pass it explicitly."""

    def __init__(
        self,
        static_refs: dict[str, dict],
        numeric_bins: dict[str, list[float]],
        mask_token: str,
        resource_separator: str = "\n\n",
        include_description: bool = False,
        include_reference_range: bool = True,
        quantity_fmt=None,
        range_fmt=None,
        code_last: bool = False,
    ):
        self.static_refs = static_refs        # global Medications + Locations
        self.numeric_bins = numeric_bins      # item_code -> percentile bins
        self.mask_token = mask_token
        self.sep = resource_separator
        # Opt-in: when True, _format_observation emits a `Description: <display>`
        # line right after `Resource: Observation` and before `Code:` (the `para`
        # conditions). Off by default.
        self.include_description = include_description
        # Opt-out: when False, _format_observation omits the `ReferenceRange:` line entirely (the
        # `norr` conditions). On by default.
        self.include_reference_range = include_reference_range

        # v2 / v3 numeric representation (see pipeline/unit_standardisation.py).
        #
        #   quantity_fmt(value, raw_unit) -> (value_text, unit_text)
        #   range_fmt(referenceRange_list, value_raw_unit) -> [(label, text, unit), ...]
        #
        # Both are applied at the lab observation call sites only, never inside _quantity_text,
        # which also renders the medication quantities and doses. Applying the v2 transform
        # inside it would rescale medication doses into laboratory units.
        #
        # None (the default) keeps the v1 output.
        self.quantity_fmt = quantity_fmt
        self.range_fmt = range_fmt

        # Opt-in: when True, _format_observation writes the `Code:` line last among the
        # Observation's own lines (after time, values, range, interpretation and notes), but still
        # before any inlined hasMember children. A causal decoder (Llama) predicts the code at its
        # mask. With the code in the middle of the block, it would not see the observation's time,
        # value, unit or range when predicting it. The children stay after the parent's
        # code, so the order of masks in the text still matches the order of the returned codes.
        # Off by default. The Llama datasets turn it on.
        self.code_last = code_last

    @classmethod
    def from_data_dirs(
        cls,
        mask_token: str,
        canonical_dir: str | Path = str(locations.MIMIC_FHIR),
        bins_dir: str | Path = str(locations.TOKENIZER),
        include_description: bool = False,
        include_reference_range: bool = True,
        quantity_fmt=None,
        range_fmt=None,
        code_last: bool = False,
    ) -> "FHIRFormatter":
        canonical_dir = Path(canonical_dir)
        bins_dir = Path(bins_dir)

        static_refs: dict[str, dict] = {}
        for fname in ("MimicMedication.ndjson", "MimicMedicationMix.ndjson", "MimicLocation.ndjson"):
            with open(canonical_dir / fname) as f:
                for r in ndjson.load(f):
                    static_refs[r["id"]] = r

        numeric_bins: dict[str, list[float]] = {}
        for fname in ("labevent_bins.pkl", "medication_dose_bins.pkl", "medication_dispense_max_dose_bins.pkl"):
            with open(bins_dir / fname, "rb") as f:
                numeric_bins.update(pickle.load(f))

        return cls(static_refs, numeric_bins, mask_token=mask_token,
                   include_description=include_description,
                   include_reference_range=include_reference_range,
                   quantity_fmt=quantity_fmt, range_fmt=range_fmt,
                   code_last=code_last)

    # ----- public API -----
    def format_patient_bundle(self, ndjson_path: str | Path) -> tuple[str, list[str]]:
        """Read patient ndjson and format it. Thin wrapper around format_resources."""
        with open(ndjson_path) as f:
            resources = ndjson.load(f)
        return self.format_resources(resources)

    def format_resources(self, resources: list[dict]) -> tuple[str, list[str]]:
        """Format a pre-loaded list of FHIR resource dicts.

        Returns (prompt_text, list_of_mimic_codes_at_each_mask).

        Iterates resources in chronological order (per sort_key). Each resource
        becomes one text block. Sub-Observations referenced via hasMember are
        inlined within the parent's block. They are tracked as seen so that
        they are not emitted again at top level.

        Use this entry point when resources come from somewhere other than a
        per-patient ndjson file, e.g. a Synthea bundle that has already been
        parsed and remapped via SyntheaConverter.convert_bundle().
        """
        encounters_period = get_encounters(resources)
        # Index for reference resolution (in-bundle resources)
        bundle_table = {r["id"]: r for r in resources if "id" in r}

        sorted_resources = sorted(resources, key=lambda r: sort_key(r, encounters_period))
        seen_ids: set[str] = set()
        blocks: list[str] = []
        codes: list[str] = []

        for r in sorted_resources:
            rid = r.get("id")
            if rid in seen_ids:
                continue
            block, block_codes = self._format_resource(
                r, bundle_table=bundle_table, seen_ids=seen_ids,
                currently_inlining=set(),
            )
            if rid:
                seen_ids.add(rid)
            if block:
                blocks.append(block)
                codes.extend(block_codes)

        return self.sep.join(blocks), codes

    # ----- per-resource dispatch -----
    def _format_resource(
        self,
        r: dict,
        bundle_table: dict[str, dict],
        seen_ids: set[str],
        currently_inlining: set[str],
    ) -> tuple[str, list[str]]:
        rtype = r.get("resourceType")
        # cycle protection: skip a resource that is already being inlined (a hasMember cycle)
        rid = r.get("id")
        if rid and rid in currently_inlining:
            return "", []

        match rtype:
            case "Patient":              return self._format_patient(r), []
            case "Encounter":            return self._format_encounter(r), []
            case "Condition":            return self._format_condition(r, bundle_table), []
            case "Procedure":            return self._format_procedure(r), []
            case "Observation":          return self._format_observation(
                r, bundle_table, seen_ids, currently_inlining
            )
            case "MedicationRequest":    return self._format_medication_request(r), []
            case "MedicationDispense":   return self._format_medication_dispense(r), []
            case "MedicationAdministration": return self._format_medication_administration(r), []
            case _:
                # No relevant_keys entry, so MLM emits nothing and neither does this formatter
                return "", []

    # ----- Patient -----
    def _format_patient(self, p: dict) -> str:
        """relevant_keys: gender, maritalStatus, communication:language,
        extension, birthDate, deceasedDateTime."""
        lines = ["Resource: Patient"]

        if (g := p.get("gender")):
            lines.append(f"Gender: {g}")

        if (ms := p.get("maritalStatus")):
            code = self._codeable_concept_code(ms)
            if code:
                lines.append(f"MaritalStatus: {code}")

        # communication:language, where communication is a list of {language: CodeableConcept, ...}
        if (comm := p.get("communication")):
            langs = []
            for c in comm:
                lang_code = self._codeable_concept_code(c.get("language", {}))
                if lang_code:
                    langs.append(lang_code)
            if langs:
                lines.append(f"Languages: {', '.join(langs)}")

        # extension: only us-core-race / us-core-ethnicity / us-core-birthsex are emitted
        for ext in p.get("extension", []):
            url = ext.get("url", "")
            label = EXTENSION_HANDLERS.get(url)
            if not label:
                continue
            if url.endswith("birthsex"):
                if (val := ext.get("valueCode")):
                    lines.append(f"{label}: {val}")
            else:
                # race/ethnicity have nested ombCategory extensions with valueCoding
                for sub in ext.get("extension", []):
                    if sub.get("url") == "ombCategory":
                        coding = sub.get("valueCoding", {})
                        if (code := coding.get("code")):
                            lines.append(f"{label}: {code}")
                            break

        if (bd := p.get("birthDate")):
            lines.append(f"BirthDate: {bd}")
        if (dd := p.get("deceasedDateTime")):
            lines.append(f"DeceasedDateTime: {dd}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- Encounter -----
    def _format_encounter(self, e: dict) -> str:
        """relevant_keys: class_fhir, type, serviceType, priority,
        hospitalization:admitSource, hospitalization:dischargeDisposition,
        period:start, period:end. Note: 'class_fhir' is fhirclient's name for
        FHIR's 'class' field."""
        lines = ["Resource: Encounter"]

        if (cls := e.get("class")):  # raw FHIR uses "class"
            # class is a single Coding (not CodeableConcept) per FHIR R4
            if isinstance(cls, dict) and (code := cls.get("code")):
                lines.append(f"Class: {code}")

        # type is a list of CodeableConcept
        for t in e.get("type", []):
            code = self._codeable_concept_code(t)
            if code:
                lines.append(f"Type: {code}")

        if (st := e.get("serviceType")):
            code = self._codeable_concept_code(st)
            if code:
                lines.append(f"ServiceType: {code}")

        if (pr := e.get("priority")):
            code = self._codeable_concept_code(pr)
            if code:
                lines.append(f"Priority: {code}")

        # hospitalization sub-element
        if (hosp := e.get("hospitalization")):
            for hosp_field, label in [("admitSource", "AdmitSource"),
                                       ("dischargeDisposition", "DischargeDisposition")]:
                if (cc := hosp.get(hosp_field)):
                    code = self._codeable_concept_code(cc)
                    if code:
                        lines.append(f"{label}: {code}")

        period = e.get("period", {})
        if (s := period.get("start")):
            lines.append(f"PeriodStart: {s}")
        if (en := period.get("end")):
            lines.append(f"PeriodEnd: {en}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- Condition -----
    def _format_condition(self, c: dict, bundle_table: dict[str, dict]) -> str:
        """relevant_keys: code, encounter:reference:period:end."""
        lines = ["Resource: Condition"]

        if (code_cc := c.get("code")):
            code = self._codeable_concept_code(code_cc)
            if code:
                lines.append(f"Code: {code}")

        # encounter:reference:period:end, read from the referenced encounter's period.end
        if (enc_ref := c.get("encounter", {}).get("reference")):
            enc_id = enc_ref.split("/")[-1]
            enc = bundle_table.get(enc_id)
            if enc and (end := enc.get("period", {}).get("end")):
                lines.append(f"EncounterEnd: {end}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- Procedure -----
    def _format_procedure(self, p: dict) -> str:
        """relevant_keys: code, encounter:reference:location:name (emits nothing:
        EncounterLocation has no .name attribute in fhirclient/FHIR),
        performedDateTime."""
        lines = ["Resource: Procedure"]

        if (code_cc := p.get("code")):
            code = self._codeable_concept_code(code_cc)
            if code:
                lines.append(f"Code: {code}")

        # encounter:reference:location:name emits nothing in FHIR-MLM. Skip.

        if (pdt := p.get("performedDateTime")):
            lines.append(f"PerformedDateTime: {pdt}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- Observation -----
    def _format_observation(
        self,
        o: dict,
        bundle_table: dict[str, dict],
        seen_ids: set[str],
        currently_inlining: set[str],
        nesting_level: int = 0,
    ) -> tuple[str, list[str]]:
        """relevant_keys: code, effectiveDateTime, valueString,
        specimen:reference (emits nothing), collection:collectedDateTime
        (emits nothing), hasMember:reference, valueCodeableConcept,
        valueQuantity, referenceRange, interpretation, note:text.

        Returns (text_block, list_of_codes_at_mask_positions).
        """
        oid = o.get("id")
        if oid:
            currently_inlining = currently_inlining | {oid}

        # Determine if this is a lab obs that should be masked
        codings = o.get("code", {}).get("coding", [])
        primary_code = None
        primary_system = None
        if codings:
            primary_code = codings[0].get("code")
            primary_system = codings[0].get("system")
        is_lab = primary_system == LAB_CODE_SYSTEM

        indent = "  " * nesting_level
        lines = [f"{indent}Resource: Observation"]
        codes_for_block: list[str] = []

        # Optional Description line, emitted directly after `Resource: Observation`
        # and before `Code:`. It reads `coding[0].display`. paraphrase_utils patches
        # this field in the MIMIC datasets and SyntheaConverter leaves it untouched,
        # so a Synthea bundle keeps its native LOINC display name.
        # Whitespace is collapsed so the prompt stays line-based.
        if self.include_description and codings:
            display = codings[0].get("display")
            if display and (cleaned := " ".join(display.split())):
                lines.append(f"{indent}Description: {cleaned}")

        # Code field, masked if lab and plain if not. With code_last the line is held back and
        # written after the notes (below). The code is recorded here in both modes, which keeps it
        # ahead of the codes of any hasMember children, in the same way that its mask stays ahead
        # in the text.
        code_line = None
        if is_lab and primary_code:
            code_line = f"{indent}Code: {self.mask_token}"
            codes_for_block.append(str(primary_code))
        elif primary_code:
            code_line = f"{indent}Code: {primary_code}"
        if code_line is not None and not self.code_last:
            lines.append(code_line)

        if (edt := o.get("effectiveDateTime")):
            lines.append(f"{indent}EffectiveDateTime: {edt}")

        # valueString
        if (vs := o.get("valueString")):
            lines.append(f"{indent}ValueString: {vs}")

        # specimen:reference emits nothing (Specimen has no relevant_keys). Skip.
        # collection:collectedDateTime emits nothing (Observation has no
        #   .collection attribute). Skip.

        # valueCodeableConcept
        if (vcc := o.get("valueCodeableConcept")):
            code = self._codeable_concept_code(vcc)
            if code:
                lines.append(f"{indent}ValueCodeableConcept: {code}")

        # valueQuantity is only emitted if the item code has bins, as in FHIR-MLM
        value_raw_unit = ""
        if (vq := o.get("valueQuantity")) and primary_code:
            # read even when the line is suppressed: the reference range below needs it for
            # bounds that state no unit of their own
            value_raw_unit = vq.get("unit") or vq.get("code") or ""
            if str(primary_code) in self.numeric_bins:
                if is_lab and self.quantity_fmt is not None:
                    val, unit = self.quantity_fmt(vq.get("value"), value_raw_unit)
                    # the FHIR comparator, which _quantity_text drops. A censored result is not
                    # the same measurement as an uncensored one: ">37 epithelial cells/HPF" would
                    # read "37". The TF-IDF prompts (pipeline/06_tfidf_dataset.py) emit it too.
                    #
                    # Only the v2/v3 branch emits it, so the v1 rendering stays the one its models
                    # were trained on. Reference-range bounds in MIMIC carry no comparator, so only
                    # the value needs it.
                    comparator = vq.get("comparator", "")
                    qty_text = f"{comparator}{val} {unit}".strip() or None
                else:
                    qty_text = self._quantity_text(vq)
                if qty_text:
                    lines.append(f"{indent}ValueQuantity: {qty_text}")

        # referenceRange (a list of {low, high, ...})
        # Not gated on numeric_bins, unlike ValueQuantity above, so an observation can show a
        # ReferenceRange whose ValueQuantity was suppressed. The asymmetry is part of the v1
        # rendering and is kept.
        if self.include_reference_range and (rrs := o.get("referenceRange")):
            if is_lab and self.range_fmt is not None:
                parts = [f"{label}: {text} {unit}".rstrip()
                         for label, text, unit in self.range_fmt(rrs, value_raw_unit)]
                rr_text = ", ".join(parts) if parts else None
            else:
                rr_text = self._reference_range_text(rrs)
            if rr_text:
                lines.append(f"{indent}ReferenceRange: {rr_text}")

        # interpretation (a list of CodeableConcept)
        if (interps := o.get("interpretation")):
            i_codes = []
            for i in interps:
                code = self._codeable_concept_code(i)
                if code:
                    i_codes.append(code)
            if i_codes:
                lines.append(f"{indent}Interpretation: {', '.join(i_codes)}")

        # note:text (note is a list of Annotation)
        if (notes := o.get("note")):
            for n in notes:
                if (txt := n.get("text")):
                    lines.append(f"{indent}Note: {txt}")

        # code_last: the Code line closes the Observation's own lines, before any inlined child
        if code_line is not None and self.code_last:
            lines.append(code_line)

        # hasMember:reference, with sub-Observations inlined as nested blocks
        if (hms := o.get("hasMember")):
            for ref_obj in hms:
                ref_str = ref_obj.get("reference", "")
                sub_id = ref_str.split("/")[-1] if ref_str else None
                if not sub_id:
                    continue
                sub = bundle_table.get(sub_id)
                if sub is None or sub_id in currently_inlining:
                    continue
                sub_text, sub_codes = self._format_observation(
                    sub, bundle_table, seen_ids, currently_inlining,
                    nesting_level=nesting_level + 1,
                )
                if sub_text:
                    lines.append(sub_text)
                    codes_for_block.extend(sub_codes)
                    seen_ids.add(sub_id)  # don't re-emit at top level

        block = "\n".join(lines) if len(lines) > 1 else ""
        return block, codes_for_block

    # ----- MedicationRequest -----
    def _format_medication_request(self, m: dict) -> str:
        """relevant_keys (as intended, without the missing comma):
        medicationReference:reference, dosageInstruction, authoredOn, status.

        Unlike FHIR-MLM, this emits the medication identity and dosage. The
        FHIR-MLM Arrow dataset is missing these because of a missing comma in
        the original relevant_keys.

        Note: MIMIC Medications use `identifier[0].value` (a readable string like
        'Sodium Phosphate--NAPH15I--63323017015') instead of a top-level
        `.code` CodeableConcept. Mix Medications walk to component Medications
        via `ingredient[*].itemReference`."""
        lines = ["Resource: MedicationRequest"]

        # medicationReference:reference, resolved through static_refs
        if (med_ref := m.get("medicationReference", {}).get("reference")):
            med_id = med_ref.split("/")[-1]
            med_text = self._medication_text(med_id)
            if med_text:
                lines.append(f"Medication: {med_text}")

        # dosageInstruction (a list of Dosage)
        if (dis := m.get("dosageInstruction")):
            for d in dis:
                dose_text = self._format_dosage(d, indent="  ")
                if dose_text:
                    lines.append(dose_text)

        if (a := m.get("authoredOn")):
            lines.append(f"AuthoredOn: {a}")

        if (s := m.get("status")):
            lines.append(f"Status: {s}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- MedicationDispense -----
    def _format_medication_dispense(self, m: dict) -> str:
        """relevant_keys: medicationCodeableConcept, quantity (binned),
        dosageInstruction, effectiveDateTime."""
        lines = ["Resource: MedicationDispense"]

        if (mcc := m.get("medicationCodeableConcept")):
            code = self._codeable_concept_code(mcc)
            if code:
                lines.append(f"Medication: {code}")

        # quantity: Quantity dispatch with item_code="dose"
        if (q := m.get("quantity")) and "dose" in self.numeric_bins:
            qty_text = self._quantity_text(q)
            if qty_text:
                lines.append(f"Quantity: {qty_text}")

        if (dis := m.get("dosageInstruction")):
            for d in dis:
                dose_text = self._format_dosage(d, indent="  ")
                if dose_text:
                    lines.append(dose_text)

        if (edt := m.get("effectiveDateTime")):
            lines.append(f"EffectiveDateTime: {edt}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- MedicationAdministration -----
    def _format_medication_administration(self, m: dict) -> str:
        """relevant_keys: medicationCodeableConcept, dosage:text, dosage:method,
        dosage:dose, effectiveDateTime."""
        lines = ["Resource: MedicationAdministration"]

        if (mcc := m.get("medicationCodeableConcept")):
            code = self._codeable_concept_code(mcc)
            if code:
                lines.append(f"Medication: {code}")

        # dosage is a single object (not list) per FHIR MedicationAdministration spec
        if (d := m.get("dosage")):
            if (txt := d.get("text")):
                lines.append(f"DosageText: {txt}")
            if (method_cc := d.get("method")):
                code = self._codeable_concept_code(method_cc)
                if code:
                    lines.append(f"DosageMethod: {code}")
            if (dose_q := d.get("dose")) and "dose" in self.numeric_bins:
                qty_text = self._quantity_text(dose_q)
                if qty_text:
                    lines.append(f"Dose: {qty_text}")

        if (edt := m.get("effectiveDateTime")):
            lines.append(f"EffectiveDateTime: {edt}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ----- Dosage helper -----
    def _format_dosage(self, d: dict, indent: str = "") -> str:
        """relevant_keys["Dosage"]: text, route, doseQuantity, timing:code,
        maxDosePerPeriod."""
        parts = []
        if (txt := d.get("text")):
            parts.append(f"{indent}DosageText: {txt}")
        if (route_cc := d.get("route")):
            code = self._codeable_concept_code(route_cc)
            if code:
                parts.append(f"{indent}Route: {code}")
        if (dq := d.get("doseAndRate")):
            # FHIR R4: dose is under doseAndRate[].doseQuantity
            for dr in dq:
                if (doseQ := dr.get("doseQuantity")) and "dose" in self.numeric_bins:
                    qty_text = self._quantity_text(doseQ)
                    if qty_text:
                        parts.append(f"{indent}DoseQuantity: {qty_text}")
        if (tcc := d.get("timing", {}).get("code")):
            code = self._codeable_concept_code(tcc)
            if code:
                parts.append(f"{indent}TimingCode: {code}")
        if (mdpp := d.get("maxDosePerPeriod")) and "maxDosePerPeriod" in self.numeric_bins:
            qty_text = self._ratio_text(mdpp)
            if qty_text:
                parts.append(f"{indent}MaxDosePerPeriod: {qty_text}")
        return "\n".join(parts)

    # ----- Medication helper -----
    def _medication_text(self, med_id: str, depth: int = 0) -> Optional[str]:
        """Resolve a Medication ID to a readable string.

        - For leaf Medications (have `code`): emit the code.
        - For mix Medications (have `ingredient` but no `code`): emit
          identifier[0].value if present (it's a human-readable concatenated
          name); otherwise walk ingredients and join their resolved names.
        - For unresolvable refs: None.
        Cycle protection: depth-limited at 3."""
        if depth > 3:
            return None
        med = self.static_refs.get(med_id)
        if not med:
            return None

        if (code := self._codeable_concept_code(med.get("code", {}))):
            return code

        # Prefer the human-readable identifier value if present
        for ident in med.get("identifier", []):
            if (val := ident.get("value")):
                return str(val)

        # Last resort: walk ingredients
        parts = []
        for ing in med.get("ingredient", []):
            if (inner_ref := ing.get("itemReference", {}).get("reference")):
                inner_id = inner_ref.split("/")[-1]
                inner_text = self._medication_text(inner_id, depth + 1)
                if inner_text:
                    parts.append(inner_text)
        return " + ".join(parts) if parts else None

    # ----- value helpers -----
    @staticmethod
    def _codeable_concept_code(cc: dict) -> Optional[str]:
        """Mirror NdjsonTokenizer's CodeableConcept→Coding→code path. Returns
        the first coding's code (no display text, as in FHIR-MLM)."""
        codings = cc.get("coding", [])
        if codings and (code := codings[0].get("code")):
            return str(code)
        return None

    @staticmethod
    def _quantity_text(q: dict) -> Optional[str]:
        """'value unit' as text. Returns None if value missing.
        The caller checks that the item code is in numeric_bins, as FHIR-MLM
        emits a quantity only when its item code has bins."""
        val = q.get("value")
        if val is None:
            return None
        unit_code = q.get("code") or q.get("unit") or ""
        return f"{val} {unit_code}".strip()

    @staticmethod
    def _ratio_text(r: dict) -> Optional[str]:
        """'num_value num_unit / den_value den_unit' for a Ratio."""
        num = r.get("numerator", {})
        den = r.get("denominator", {})
        nv, du = num.get("value"), num.get("code") or num.get("unit") or ""
        dv, dnu = den.get("value"), den.get("code") or den.get("unit") or ""
        if nv is None or dv is None:
            return None
        return f"{nv} {du} / {dv} {dnu}".strip()

    def _reference_range_text(self, rrs: list[dict]) -> Optional[str]:
        """Compose 'Low: x unit, High: y unit' from a list of reference ranges."""
        parts = []
        for rr in rrs:
            for label, key in [("Low", "low"), ("High", "high")]:
                q = rr.get(key)
                if q:
                    text = self._quantity_text(q)
                    if text:
                        parts.append(f"{label}: {text}")
        return ", ".join(parts) if parts else None
