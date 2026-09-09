"""
dcis_biomarkers.data
--------------------
Data contract, manifest validation, outcome definitions, and patient-level splits
for the DCIS progression biomarkers research platform.
"""

from .manifest import CaseManifest, FeatureManifest, validate_manifest
from .outcome import OutcomeDefinition, BINARY_SCHEMA, TERNARY_SCHEMA
from .splits import PatientSplitter, assert_no_leakage

__all__ = [
    "CaseManifest",
    "FeatureManifest",
    "validate_manifest",
    "OutcomeDefinition",
    "BINARY_SCHEMA",
    "TERNARY_SCHEMA",
    "PatientSplitter",
    "assert_no_leakage",
]
