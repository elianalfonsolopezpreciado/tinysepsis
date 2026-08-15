"""LOINC-code mapping from FHIR Observation resources to TinySepsis's 34
clinical feature channels.

IMPORTANT: this table is a starting point, not a validated terminology
mapping. Codes marked confidence="verified" were checked against loinc.org
during development; "standard" codes are the LOINC codes conventionally
used for that measurement in US EHRs (Epic/Oracle Health) but were not
individually re-verified here; every code must be confirmed against the
receiving institution's own LOINC dictionary (or a terminology service
such as the NLM's LOINC/RxNorm/SNOMED CT browser or a UMLS-backed mapping
tool) by a clinical informatics reviewer before this is used against real
patient data. Getting a single code wrong silently feeds the wrong
variable into the model -- this is exactly the kind of error a terminology
review, not a unit test, is designed to catch.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LoincEntry:
    code: str
    display: str
    confidence: str  # "verified" | "standard" | "needs_verification"
    unit_hint: str = ""


# tinysepsis feature name -> LOINC entry
LOINC_MAP: dict[str, LoincEntry] = {
    # --- Vitals ---
    "HR": LoincEntry("8867-4", "Heart rate", "verified", "/min"),
    "O2Sat": LoincEntry("59408-5", "Oxygen saturation by Pulse oximetry", "standard", "%"),
    "Temp": LoincEntry("8310-5", "Body temperature", "standard", "Cel"),
    "SBP": LoincEntry("8480-6", "Systolic blood pressure", "verified", "mm[Hg]"),
    "DBP": LoincEntry("8462-4", "Diastolic blood pressure", "standard", "mm[Hg]"),
    "MAP": LoincEntry("8478-0", "Mean blood pressure", "standard", "mm[Hg]"),
    "Resp": LoincEntry("9279-1", "Respiratory rate", "standard", "/min"),
    "EtCO2": LoincEntry("19836-9", "End tidal CO2", "needs_verification", "mm[Hg]"),
    # --- Labs ---
    "BaseExcess": LoincEntry("1925-7", "Base excess", "needs_verification", "mmol/L"),
    "HCO3": LoincEntry("1963-8", "Bicarbonate", "standard", "mmol/L"),
    "FiO2": LoincEntry("3150-0", "Inhaled oxygen concentration", "needs_verification", "%"),
    "pH": LoincEntry("2744-1", "pH of Arterial blood", "standard", "pH"),
    "PaCO2": LoincEntry("2019-8", "Carbon dioxide, partial pressure, Arterial blood", "standard", "mm[Hg]"),
    "SaO2": LoincEntry("2708-6", "Oxygen saturation, Arterial blood", "standard", "%"),
    "AST": LoincEntry("1920-8", "Aspartate aminotransferase", "standard", "U/L"),
    "BUN": LoincEntry("3094-0", "Urea nitrogen", "standard", "mg/dL"),
    "Alkalinephos": LoincEntry("6768-6", "Alkaline phosphatase", "needs_verification", "U/L"),
    "Calcium": LoincEntry("17861-6", "Calcium", "standard", "mg/dL"),
    "Chloride": LoincEntry("2075-0", "Chloride", "standard", "mmol/L"),
    "Creatinine": LoincEntry("2160-0", "Creatinine", "standard", "mg/dL"),
    "Bilirubin_direct": LoincEntry("1968-7", "Bilirubin.direct", "needs_verification", "mg/dL"),
    "Glucose": LoincEntry("2345-7", "Glucose", "standard", "mg/dL"),
    "Lactate": LoincEntry("2524-7", "Lactate", "standard", "mmol/L"),
    "Magnesium": LoincEntry("2601-3", "Magnesium", "standard", "mg/dL"),
    "Phosphate": LoincEntry("2777-1", "Phosphate", "standard", "mg/dL"),
    "Potassium": LoincEntry("2823-3", "Potassium", "verified", "mmol/L"),
    "Bilirubin_total": LoincEntry("1975-2", "Bilirubin.total", "standard", "mg/dL"),
    "TroponinI": LoincEntry("10839-9", "Troponin I.cardiac", "needs_verification", "ng/mL"),
    "Hct": LoincEntry("4544-3", "Hematocrit", "standard", "%"),
    "Hgb": LoincEntry("718-7", "Hemoglobin", "standard", "g/dL"),
    "PTT": LoincEntry("14979-9", "aPTT", "needs_verification", "s"),
    "WBC": LoincEntry("6690-2", "Leukocytes [#/volume] in Blood by Automated count", "verified", "10*3/uL"),
    "Fibrinogen": LoincEntry("3255-7", "Fibrinogen", "needs_verification", "mg/dL"),
    "Platelets": LoincEntry("777-3", "Platelets [#/volume] in Blood by Automated count", "standard", "10*3/uL"),
    # --- Static ---
    "Age": LoincEntry("30525-0", "Age", "standard", "a"),
    "Gender": LoincEntry("46098-0", "Sex", "standard", ""),
}

LOINC_TO_FEATURE = {v.code: k for k, v in LOINC_MAP.items()}

UNVERIFIED_CODES = [k for k, v in LOINC_MAP.items() if v.confidence == "needs_verification"]
