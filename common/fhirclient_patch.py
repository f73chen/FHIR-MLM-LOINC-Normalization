"""Patch fhirclient 4.1.0 so that it accepts MIMIC's MedicationRequest resources.

MIMIC-IV on FHIR writes MedicationRequest.medicationCodeableConcept as a list. FHIR R4 and
fhirclient 4.1.0 define it as a single CodeableConcept, so fhirclient rejects those resources. The
copy of fhirclient used for the thesis accepts a list for that element. Importing this module
makes the same change to the PyPI release.
"""
from fhirclient.models import medicationrequest

_element_properties = medicationrequest.MedicationRequest.elementProperties


def _with_list_medication(self):
    return [(name, js_name, typ, True if name == "medicationCodeableConcept" else is_list, of_many, required)
            for name, js_name, typ, is_list, of_many, required in _element_properties(self)]


medicationrequest.MedicationRequest.elementProperties = _with_list_medication
