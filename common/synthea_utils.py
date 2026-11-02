"""Synthea FHIR bundle conversion utilities.

Converts Synthea-generated FHIR patient bundles into formats compatible with
MIMIC-trained models (MLM tokenization, TFIDF feature extraction). The thesis
experiments use only this module's MIMIC code-system constants, which
common/paraphrase_utils.py imports.

Usage:
    from synthea_utils import SyntheaConverter, fhir_observation_to_features

    converter = SyntheaConverter(
        seen_codes_path=...,
        le_code_counter_path=...,
        unit_crosswalk_path=...,
        snomed_condition_map_path=...,
        snomed_procedure_map_path=...,
        rxnorm_cache_path=...,
    )
    resources = converter.convert_bundle("patient_001.json")
"""

import csv
import json
import pathlib
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Literal

# -- MIMIC code system URIs --------------------------------------------
MIMIC_LAB_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems"
MIMIC_CHARTEVENTS_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items"
MIMIC_UNIT_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-units"
MIMIC_ADMISSION_CLASS_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-admission-class"
MIMIC_MEDICATION_NAME_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication-name"


# -- Resource ordering constants ---------------------------------------

_TYPE_PRIORITY = {
    "Patient": 0,
    "Encounter": 1,
    "Condition": 2,
    "Procedure": 3,
    "Observation": 4,
    "Specimen": 5,
    "MedicationDispense": 6,
    "MedicationAdministration": 7,
    "MedicationRequest": 8,
}

_DATE_KEYS = {
    "Patient":                  ["birthDate"],
    "Encounter":                ["period", "start"],
    "Condition":                ["encounter"],
    "Procedure":                ["performedDateTime"],
    "Observation":              ["effectiveDateTime"],
    "MedicationRequest":        ["authoredOn"],
    "MedicationDispense":       ["context"],
    "MedicationAdministration": ["effectiveDateTime"],
    "Specimen":                 ["collection", "collectedDateTime"],
}


class SyntheaConverter:
    """Converts Synthea FHIR bundles to MIMIC-compatible format.

    Holds all mapping data (LOINC-MIMIC, SNOMED-ICD, unit crosswalk, RxNorm cache)
    and provides methods for each conversion step. The main entry point is
    ``convert_bundle(filepath)`` which runs the full pipeline.

    Attributes:
        loinc_to_mimics: dict mapping LOINC code -> [(mimic_code, obs_type), ...]
        mimic_to_loinc: dict mapping MIMIC code -> LOINC code
        loinc_to_best_mimic: dict mapping LOINC code -> (best_mimic_code, obs_type)
        unit_crosswalk: dict mapping source_unit -> (target_unit, multiplier)
        snomed_to_icd_condition: dict mapping SNOMED code -> (icd_code, mimic_system)
        snomed_to_icd_procedure: dict mapping SNOMED code -> (icd_code, mimic_system)
        rxnorm_to_mimic: dict mapping RxCUI -> ingredient name (or None)
    """

    def __init__(
        self,
        *,
        seen_codes_path,
        le_code_counter_path,
        unit_crosswalk_path,
        snomed_condition_map_path,
        snomed_procedure_map_path,
        rxnorm_cache_path,
        reverse_mapping="highest_frequency",
        resource_ordering="time_sorted",
        conversions=None,
    ):
        self.resource_ordering = resource_ordering
        self.conversions = conversions or {
            "observation_units": True,
            "condition": True,
            "procedure": True,
            "medication": True,
            "encounter": True,
        }
        self.rxnorm_cache_path = Path(rxnorm_cache_path)

        # -- LOINC <-> MIMIC mappings ----------------------------------
        self.loinc_to_mimics = {}
        self.mimic_to_loinc = {}
        with open(seen_codes_path) as f:
            for row in csv.DictReader(f):
                mc, lc = row["mimic_code"], row["loinc_code"]
                obs_type = row.get("observation_type", "lab")
                if mc and lc:
                    self.loinc_to_mimics.setdefault(lc, []).append((mc, obs_type))
                    self.mimic_to_loinc[mc] = lc

        print(f"LOINC->MIMIC mappings: {len(self.loinc_to_mimics)} LOINC codes")
        print(f"MIMIC->LOINC mappings: {len(self.mimic_to_loinc)} MIMIC codes")
        print(f"One-to-many LOINC codes: {sum(1 for v in self.loinc_to_mimics.values() if len(v) > 1)}")

        # -- LOINC -> best MIMIC resolver ------------------------------
        with open(le_code_counter_path, "rb") as f:
            self.le_code_counter = pickle.load(f)
        le_code_counter = self.le_code_counter

        self.loinc_to_best_mimic = {}
        for loinc_code, mimic_codes_with_type in self.loinc_to_mimics.items():
            if len(mimic_codes_with_type) == 1:
                self.loinc_to_best_mimic[loinc_code] = mimic_codes_with_type[0]
            elif reverse_mapping == "highest_frequency":
                best = max(mimic_codes_with_type, key=lambda x: le_code_counter.get(x[0], 0))
                self.loinc_to_best_mimic[loinc_code] = best
            else:
                self.loinc_to_best_mimic[loinc_code] = mimic_codes_with_type[0]

        print(f"LOINC -> MIMIC resolver: {len(self.loinc_to_best_mimic)} mappings (strategy: {reverse_mapping})")

        # -- Unit crosswalk --------------------------------------------
        self.unit_crosswalk = {}
        with open(unit_crosswalk_path) as f:
            for row in csv.DictReader(f):
                self.unit_crosswalk[row["source_unit"]] = (row["target_unit"], float(row["multiplier"]))
        print(f"Unit crosswalk: {len(self.unit_crosswalk)} entries")

        # -- SNOMED -> ICD mappings ------------------------------------
        self.snomed_to_icd_condition = {}
        with open(snomed_condition_map_path) as f:
            for row in csv.DictReader(f):
                self.snomed_to_icd_condition[row["snomed_code"]] = (row["icd_code"], row["mimic_system"])

        self.snomed_to_icd_procedure = {}
        with open(snomed_procedure_map_path) as f:
            for row in csv.DictReader(f):
                self.snomed_to_icd_procedure[row["snomed_code"]] = (row["icd_code"], row["mimic_system"])

        print(f"SNOMED->ICD conditions: {len(self.snomed_to_icd_condition)} mappings")
        print(f"SNOMED->ICD procedures: {len(self.snomed_to_icd_procedure)} mappings")

        # -- RxNorm -> MIMIC cache -------------------------------------
        if self.rxnorm_cache_path.exists():
            with open(self.rxnorm_cache_path) as f:
                self.rxnorm_to_mimic = json.load(f)
        else:
            self.rxnorm_to_mimic = {}
        print(f"RxNorm cache: {len(self.rxnorm_to_mimic)} entries")

    # -- Bundle loading (static, no mapping data needed) ---------------

    @staticmethod
    def load_bundle(filepath):
        """Load a Synthea FHIR Bundle JSON and return (resources, uuid_map).

        Returns:
            resources: list of resource dicts
            uuid_map: dict mapping urn:uuid:xxx -> (resourceType, resource_id)
        """
        with open(filepath) as f:
            bundle = json.load(f)

        resources = []
        uuid_map = {}

        for entry in bundle.get("entry", []):
            resource = entry["resource"]
            full_url = entry.get("fullUrl", "")
            if full_url.startswith("urn:uuid:"):
                uuid_map[full_url] = (resource["resourceType"], resource["id"])
            resources.append(resource)

        return resources, uuid_map

    @staticmethod
    def convert_references(obj, uuid_map):
        """Recursively convert urn:uuid: references to ResourceType/id format."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "reference" and isinstance(value, str) and value.startswith("urn:uuid:"):
                    if value in uuid_map:
                        rtype, rid = uuid_map[value]
                        obj[key] = f"{rtype}/{rid}"
                else:
                    SyntheaConverter.convert_references(value, uuid_map)
        elif isinstance(obj, list):
            for item in obj:
                SyntheaConverter.convert_references(item, uuid_map)

    # -- Code remapping ------------------------------------------------

    def remap_observation_code(self, obs):
        """Rewrite a Synthea Observation's code from LOINC to a MIMIC item ID, in place.

        Returns True if remapped, False if no mapping found.
        """
        codings = obs.get("code", {}).get("coding", [])
        if not codings:
            return False

        primary = codings[0]
        loinc_code = primary.get("code", "")

        mapping = self.loinc_to_best_mimic.get(loinc_code)
        if mapping is None:
            return False

        mimic_code, obs_type = mapping

        if obs_type == "lab":
            primary["system"] = MIMIC_LAB_SYSTEM
        else:
            primary["system"] = MIMIC_CHARTEVENTS_SYSTEM

        primary["code"] = mimic_code
        return True

    def remap_condition_code(self, condition):
        """Remap a Synthea Condition code (SNOMED) to a MIMIC ICD code, in place."""
        codings = condition.get("code", {}).get("coding", [])
        if not codings:
            return False

        primary = codings[0]
        snomed_code = primary.get("code", "")

        mapping = self.snomed_to_icd_condition.get(snomed_code)
        if mapping is None:
            return False

        icd_code, mimic_system = mapping
        primary["system"] = mimic_system
        primary["code"] = icd_code
        return True

    def remap_procedure_code(self, procedure):
        """Remap a Synthea Procedure code (SNOMED) to a MIMIC ICD procedure code, in place."""
        codings = procedure.get("code", {}).get("coding", [])
        if not codings:
            return False

        primary = codings[0]
        snomed_code = primary.get("code", "")

        mapping = self.snomed_to_icd_procedure.get(snomed_code)
        if mapping is None:
            return False

        icd_code, mimic_system = mapping
        primary["system"] = mimic_system
        primary["code"] = icd_code
        return True

    def _get_ingredient_from_api(self, cui):
        """Fetch generic ingredient name for an RxCUI via NIH RxNav API, with caching."""
        if cui in self.rxnorm_to_mimic:
            return self.rxnorm_to_mimic[cui]

        try:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{cui}/related.json?tty=IN"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())

            for group in data.get("relatedGroup", {}).get("conceptGroup", []):
                if group.get("tty") == "IN":
                    props = group.get("conceptProperties", [])
                    if props:
                        name = props[0].get("name", "").lower()
                        self.rxnorm_to_mimic[cui] = name
                        return name
        except Exception:
            pass

        self.rxnorm_to_mimic[cui] = None
        return None

    def remap_medication_code(self, med_request):
        """Remap a Synthea MedicationRequest (RxNorm) to a MIMIC generic string, in place."""
        concept = med_request.get("medicationCodeableConcept", {})
        for coding in concept.get("coding", []):
            if coding.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm":
                cui = coding.get("code")
                if not cui:
                    continue

                ingredient = self._get_ingredient_from_api(cui)
                if ingredient:
                    coding["system"] = MIMIC_MEDICATION_NAME_SYSTEM
                    coding["code"] = ingredient
                    coding["display"] = ingredient
                    return True
        return False

    def save_rxnorm_cache(self):
        """Write the RxNorm-to-MIMIC cache to disk."""
        with open(self.rxnorm_cache_path, "w") as f:
            json.dump(self.rxnorm_to_mimic, f, indent=2)

    def remap_encounter_code(self, encounter):
        """Remap generic Synthea Encounter classes to MIMIC admission classes in-place.

        Mapping:
            AMB / WELLNESS -> AMBULATORY OBSERVATION
            EMER           -> EW EMER.
            IMP / URGENT   -> URGENT
        """
        cls = encounter.get("class", {})
        code = cls.get("code", "")
        if not code:
            return False

        if code in ("AMB", "WELLNESS"):
            cls["system"] = MIMIC_ADMISSION_CLASS_SYSTEM
            cls["code"] = "AMBULATORY OBSERVATION"
            return True
        elif code == "EMER":
            cls["system"] = MIMIC_ADMISSION_CLASS_SYSTEM
            cls["code"] = "EW EMER."
            return True
        elif code in ("IMP", "URGENT"):
            cls["system"] = MIMIC_ADMISSION_CLASS_SYSTEM
            cls["code"] = "URGENT"
            return True

        return False

    # -- Unit conversion -----------------------------------------------

    def rescale_quantity(self, vq):
        """Convert a single valueQuantity/referenceRange bound from UCUM to MIMIC units, in place."""
        if not vq or "code" not in vq or "value" not in vq:
            return False

        source_unit = vq["code"]
        if source_unit not in self.unit_crosswalk:
            return False

        target_unit, multiplier = self.unit_crosswalk[source_unit]

        vq["system"] = MIMIC_UNIT_SYSTEM
        vq["code"] = target_unit
        if "unit" in vq:
            vq["unit"] = target_unit

        try:
            val = float(vq["value"])
            if multiplier != 1.0:
                vq["value"] = round(val * multiplier, 4)
        except (ValueError, TypeError):
            pass

        return True

    def remap_observation_units(self, obs):
        """Rewrite valueQuantity and referenceRange units from UCUM to MIMIC units, in place."""
        remapped = False

        if self.rescale_quantity(obs.get("valueQuantity")):
            remapped = True

        for rr in obs.get("referenceRange", []):
            for bound in ("low", "high"):
                if bound in rr and self.rescale_quantity(rr[bound]):
                    remapped = True

        return remapped

    # -- Resource ordering ---------------------------------------------

    @staticmethod
    def _get_encounter_dates(resources):
        """Build a map of encounter_id -> {'start': ..., 'end': ...}."""
        encounters = {}
        for r in resources:
            if r.get("resourceType") == "Encounter":
                period = r.get("period", {})
                enc_id = r.get("id", "")
                encounters[enc_id] = {
                    "start": period.get("start", ""),
                    "end": period.get("end", ""),
                }
        return encounters

    @staticmethod
    def _resource_date(resource, encounters):
        """Extract a sortable date string from a resource, matching MIMIC training sort."""
        rtype = resource.get("resourceType", "")
        keys = _DATE_KEYS.get(rtype, [])

        val = resource
        for k in keys:
            if k in ("encounter", "context"):
                ref = val.get(k, {}).get("reference", "")
                enc_id = ref.split("/")[-1] if "/" in ref else ref
                val = encounters.get(enc_id, {})
            else:
                val = val.get(k, {}) if isinstance(val, dict) else {}

        if isinstance(val, str) and val:
            return val

        # Fallback: the period.end of the referenced encounter
        ref = resource.get("encounter", resource.get("context", {}))
        if isinstance(ref, dict):
            ref = ref.get("reference", "")
        enc_id = ref.split("/")[-1] if isinstance(ref, str) and "/" in ref else ""
        if enc_id and enc_id in encounters:
            end = encounters[enc_id].get("end", "")
            if end:
                return end

        return "z"

    def reorder_resources(self, resources):
        """Reorder resources according to the configured resource_ordering strategy."""
        if self.resource_ordering == "time_sorted":
            encounters = self._get_encounter_dates(resources)
            return sorted(resources, key=lambda r: self._resource_date(r, encounters))

        elif self.resource_ordering == "group_by_type":
            default_priority = max(_TYPE_PRIORITY.values()) + 1
            return sorted(
                resources,
                key=lambda r: _TYPE_PRIORITY.get(r.get("resourceType", ""), default_priority),
            )

        elif self.resource_ordering == "encounter_time":
            observations = []
            non_observations = []
            for r in resources:
                if r.get("resourceType") == "Observation":
                    observations.append(r)
                else:
                    non_observations.append(r)

            def obs_sort_key(obs):
                enc_ref = obs.get("encounter", {}).get("reference", "")
                dt = obs.get("effectiveDateTime", "")
                return (enc_ref, dt)

            observations.sort(key=obs_sort_key)

            default_priority = max(_TYPE_PRIORITY.values()) + 1
            non_observations.sort(
                key=lambda r: _TYPE_PRIORITY.get(r.get("resourceType", ""), default_priority)
            )

            obs_priority = _TYPE_PRIORITY["Observation"]
            before = [
                r for r in non_observations
                if _TYPE_PRIORITY.get(r.get("resourceType", ""), default_priority) < obs_priority
            ]
            after = [
                r for r in non_observations
                if _TYPE_PRIORITY.get(r.get("resourceType", ""), default_priority) >= obs_priority
            ]
            return before + observations + after

        return resources

    # -- Main pipeline -------------------------------------------------

    def convert_bundle(self, filepath):
        """Load, convert references, remap codes/units, and reorder a Synthea bundle.

        Returns the list of converted resource dicts, ready for tokenization
        or feature extraction.
        """
        resources, uuid_map = self.load_bundle(filepath)
        for r in resources:
            self.convert_references(r, uuid_map)

        for r in resources:
            rt = r.get("resourceType")
            if rt == "Observation":
                self.remap_observation_code(r)
                if self.conversions["observation_units"]:
                    self.remap_observation_units(r)
            elif rt == "Condition":
                if self.conversions["condition"]:
                    self.remap_condition_code(r)
            elif rt == "Procedure":
                if self.conversions["procedure"]:
                    self.remap_procedure_code(r)
            elif rt == "MedicationRequest":
                if self.conversions["medication"]:
                    self.remap_medication_code(r)
                mcc = r.get("medicationCodeableConcept")
                if mcc and not isinstance(mcc, list):
                    r["medicationCodeableConcept"] = [mcc]
            elif rt == "Encounter":
                if self.conversions["encounter"]:
                    self.remap_encounter_code(r)

        if self.resource_ordering != "none":
            resources = self.reorder_resources(resources)

        return resources


def fhir_observation_to_features(obs_json: dict) -> str:
    """Convert a raw FHIR Observation JSON dict into a compact feature string.

    Extracts the clinically relevant fields that the TF-IDF model was trained on:
    description, value (with comparator), reference range, status, priority,
    data-absent reason, and notes.
    """
    features = []

    # Extract description
    codings = obs_json.get("code", {}).get("coding", [])
    desc = codings[0].get("display", "unknown") if codings else "unknown"
    features.append(desc)

    # Extract value (with comparator and unit)
    value_text = None
    if "valueQuantity" in obs_json:
        vq = obs_json["valueQuantity"]
        comparator = vq.get("comparator", "")
        val = vq.get("value", "")
        unit = vq.get("unit", "") or vq.get("code", "")
        value_text = f"{comparator}{val} {unit}".strip()
        if value_text and value_text.lower() not in ("none", "n/a", "null", ""):
            features.append(value_text)
    elif "valueString" in obs_json:
        value_text = obs_json["valueString"]
        if value_text and value_text.lower() not in ("none", "n/a", "null", ""):
            features.append(value_text)
    elif "valueCodeableConcept" in obs_json:
        val_cc = obs_json["valueCodeableConcept"]
        value_text = val_cc.get("text") or (val_cc.get("coding", [{}])[0].get("display", ""))
        if value_text and value_text.lower() not in ("none", "n/a", "null", ""):
            features.append(value_text)

    # Extract reference range
    if "referenceRange" in obs_json and obs_json["referenceRange"]:
        rr = obs_json["referenceRange"][0]
        range_parts = []
        if "low" in rr and rr["low"]:
            low_val = rr["low"].get("value", "")
            low_unit = rr["low"].get("unit", "") or rr["low"].get("code", "")
            range_parts.append(f"Low: {low_val} {low_unit}".strip())
        if "high" in rr and rr["high"]:
            high_val = rr["high"].get("value", "")
            high_unit = rr["high"].get("unit", "") or rr["high"].get("code", "")
            range_parts.append(f"High: {high_val} {high_unit}".strip())
        if range_parts:
            ref_text = ", ".join(range_parts)
            if ref_text.lower() not in ("none", "n/a", "null", ""):
                features.append(f"reference {ref_text}")

    # Extract status (skip common defaults)
    status = obs_json.get("status", "")
    if status and status.lower() not in ("final", "registered"):
        features.append(f"status-{status}")

    # Extract priority
    if "extension" in obs_json:
        extensions = obs_json["extension"]
        if extensions:
            priority = extensions[0].get("valueString", "")
            if priority and priority.lower() not in ("routine", "none", "n/a"):
                features.append(f"priority-{priority}")

    # Extract dataAbsentReason
    if "dataAbsentReason" in obs_json:
        dar_codings = obs_json["dataAbsentReason"].get("coding", [])
        if dar_codings:
            reason = dar_codings[0].get("code", "")
            if reason:
                features.append(f"absent-{reason}")

    # Extract note (first qualifying note, < 200 chars)
    if "note" in obs_json and obs_json["note"]:
        for note_entry in obs_json["note"]:
            note_text = note_entry.get("text", "")
            if note_text and note_text != value_text:
                if len(note_text) < 200:
                    features.append(note_text)
                break

    compact = " ".join(features)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact if compact else "unknown"


# =============================================================================
# Unit / qualitative-value normalization
# -----------------------------------------------------------------------------
# The block below copies common/input_preprocessing_utils.py (asset loaders,
# preprocessors and the runtime API). It adds a Synthea-side helper,
# `apply_unitnorm_to_observation`, which normalizes a single Observation in place
# before `fhir_observation_to_features` turns it into features. Keep the copies
# in sync with that module.
# =============================================================================

# -- Qualitative-value preprocessing ------------------------------------------

_QUAL_DROP_PUNCT = re.compile(r"[.,;:!?*\"'`()\[\]{}]+")
_QUAL_WS = re.compile(r"\s+")


def preprocess_qual_string(s: str) -> str:
    """Canonicalize a qualitative value string for map lookup or map seeding.

    Lowercase the string and replace each run of formatting punctuation
    (. , ; : ! ? * " ' ` ( ) [ ] { }) with a space. Keep +, -, / and \\ because they carry
    meaning in qual strings. Collapse whitespace. Returns "" on empty input or
    when nothing is left after cleanup.
    """
    if not s:
        return ""
    s = s.lower()
    s = _QUAL_DROP_PUNCT.sub(" ", s)
    s = _QUAL_WS.sub(" ", s).strip()
    return s


# -- Unit-string preprocessing (matches prep/4.3_collect_value_units.py) ------

_RE_ANNOT       = re.compile(r"\{[^}]*\}")
_RE_BRACKET     = re.compile(r"\[([^\]]*)\]")
_RE_POW         = re.compile(r"(\d)\^(\d)")
_RE_SPACE_UNDER = re.compile(r"[\s_]+")


def normalize_unit_key(s: str) -> str:
    """Case/space-insensitive merge key for unit surface forms.

    It must match the build-time key in prep/4.3_collect_value_units.py so that
    lookups hit the same alias_to_canonical entries. It drops `{annotations}`
    unless the whole string is an annotation, strips `[...]` but keeps the
    contents, and rewrites 10^N as 10*N. It then lowercases the string and drops
    whitespace and underscores.
    """
    if not s:
        return ""
    stripped = _RE_ANNOT.sub("", s)
    if not stripped.strip():
        stripped = s
    stripped = _RE_BRACKET.sub(r"\1", stripped)
    stripped = _RE_POW.sub(r"\1*\2", stripped)
    return _RE_SPACE_UNDER.sub("", stripped.lower())


# -- Asset loaders ------------------------------------------------------------

def load_qual_normalizer(path) -> dict:
    """Read pcornet_qual_normalize.json and return the flat
    variation_to_canonical dict used by normalize_qual_value.
    """
    payload = json.loads(pathlib.Path(path).read_text())
    v2c = {}
    for canon, vars_ in payload.items():
        v2c[canon] = canon
        for v in vars_:
            v2c[v] = canon
    return v2c


def load_qual_passthrough(path):
    """Read pcornet_qual_passthrough.json and return (literals, patterns).

    literals: set of preprocessed strings that pass through unchanged.
    patterns: list of (name, compiled_regex) tuples; first match wins.
    """
    data = json.loads(pathlib.Path(path).read_text())
    literals = set(data.get("literals", {}).keys())
    patterns = [(name, re.compile(spec["regex"]))
                for name, spec in data.get("numeric_patterns", {}).items()]
    return literals, patterns


def load_unit_alias_map(path):
    """Read unit_synonym_map.json and return its alias_to_canonical dict.
    Keys are preprocessed by normalize_unit_key(), and values are canonical UCUM.
    """
    payload = json.loads(pathlib.Path(path).read_text())
    return dict(payload["alias_to_canonical"])


# -- Runtime normalization API ------------------------------------------------

QualOutcome = Literal["normalize", "passthrough", "blank"]


def normalize_qual_value(raw, v2c, literals, patterns):
    """Three-way decision for a raw qualitative value string.

    1. preprocessed hits normalize map → ("normalize", canonical)
    2. preprocessed is in passthrough literals → ("passthrough", preprocessed)
    3. any passthrough numeric_pattern matches → ("passthrough", preprocessed)
    4. otherwise → ("blank", None). The caller should drop the Value line.
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


def normalize_unit(raw, alias_map):
    """Surface-form unit lookup. It does not scale the value.

    Returns (canonical_or_raw, was_mapped). On a miss it returns raw unchanged
    so the caller can keep the prompt readable. was_mapped lets the caller log
    or count unmapped surface forms.
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


# -- Synthea-side Observation mutator -----------------------------------------

def apply_unitnorm_to_observation(obs, v2c, literals, patterns, unit_alias):
    """Apply the quantity and qualitative-value normalization of the TF-IDF
    training data to a single Observation, in place. Run it after
    SyntheaConverter.convert_bundle and before fhir_observation_to_features,
    so a Synthea observation's features are normalized the same way.

    Mutations:
      - valueQuantity.unit (and .code fallback) → canonical UCUM surface form.
        The value is not scaled.
      - referenceRange[*].low.unit/.code and .high.unit/.code → canonical.
      - valueString → normalize/passthrough result, or the whole field is
        deleted on a `blank` outcome (the training data omits the Value line).
      - valueCodeableConcept → same three-way logic against .text (falling
        back to .coding[0].display), overwriting .text on hit or deleting the
        field on blank.

    Returns `(stats, unmapped_raws)`:
      - stats: per-observation counters
          {"unit_mapped", "unit_unmapped", "unit_rewrites",
           "qual_normalize", "qual_passthrough", "qual_blank"}
      - unmapped_raws: list of raw unit strings that missed the alias map
        (for diagnostic counts).
    `unit_rewrites` counts only cases where the canonical form differs from
    the raw string. A UCUM unit that maps to itself does not increment it.
    """
    stats = {"unit_mapped": 0, "unit_unmapped": 0, "unit_rewrites": 0,
             "qual_normalize": 0, "qual_passthrough": 0, "qual_blank": 0}
    unmapped_raws = []

    def _norm_unit_inplace(bound):
        raw = bound.get("unit") or bound.get("code") or ""
        if not raw:
            return
        canon, mapped = normalize_unit(raw, unit_alias)
        if mapped:
            stats["unit_mapped"] += 1
            if canon != raw:
                stats["unit_rewrites"] += 1
        else:
            stats["unit_unmapped"] += 1
            unmapped_raws.append(raw)
        # Write the canonical form back to .unit and .code, where present.
        # The mutated obs is a throwaway copy used only for feature extraction.
        # The consumer reads .unit before .code, so writing both is safe. It
        # matches writing the source field when .unit exists, and it is correct
        # when only .code exists.
        if "unit" in bound:
            bound["unit"] = canon
        if "code" in bound:
            bound["code"] = canon

    if "valueQuantity" in obs:
        _norm_unit_inplace(obs["valueQuantity"])

    for rr in obs.get("referenceRange", []):
        for key in ("low", "high"):
            if key in rr and rr[key]:
                _norm_unit_inplace(rr[key])

    if "valueString" in obs:
        outcome, val = normalize_qual_value(obs["valueString"], v2c, literals, patterns)
        stats[f"qual_{outcome}"] += 1
        if outcome == "blank":
            del obs["valueString"]
        else:
            obs["valueString"] = val
    elif "valueCodeableConcept" in obs:
        vcc = obs["valueCodeableConcept"]
        raw = vcc.get("text") or (vcc.get("coding", [{}])[0].get("display", "") if vcc.get("coding") else "")
        outcome, val = normalize_qual_value(raw or "", v2c, literals, patterns)
        stats[f"qual_{outcome}"] += 1
        if outcome == "blank":
            del obs["valueCodeableConcept"]
        else:
            vcc["text"] = val

    return stats, unmapped_raws
