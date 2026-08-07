"""Column schema for the PhysioNet/CinC Challenge 2019 sepsis dataset."""

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

NUMERIC_FEATURES = VITALS + LABS  # time-varying clinical measurements
STATIC_FEATURES = ["Age", "Gender"]  # constant per patient

LABEL_COL = "SepsisLabel"
TIME_COL = "ICULOS"  # ICU length-of-stay in hours, 1-indexed, our time axis

ALL_RAW_COLUMNS = VITALS + LABS + DEMOGRAPHICS + [LABEL_COL]

# Physiologically implausible bounds used to null out sensor/entry errors.
# Ranges are intentionally generous (not clinical alarm thresholds) so we
# only remove values that cannot be real.
PLAUSIBLE_RANGE = {
    "HR": (0, 300),
    "O2Sat": (0, 100),
    "Temp": (25, 45),  # Celsius
    "SBP": (0, 300),
    "MAP": (0, 250),
    "DBP": (0, 200),
    "Resp": (0, 80),
    "pH": (6.5, 8.0),
    "Age": (0, 130),
}
