"""Build a TinySepsisDataset-compatible Parquet from the eICU Collaborative
Research Database Demo v2.0.1 (open access, no PhysioNet credentialing --
see regulatory/model_improvement_roadmap.md for why this is the immediate,
actionable step while full eICU/MIMIC-IV credentialing is pending).

This is a genuinely independent external test set: 2,500 unit stays across
20 different US hospitals, none of which appear in the PhysioNet Challenge
2019 Hospital A/B split this project trains and validates on. Small (the
Challenge 2019 data has ~40,000+ patients; this demo has 2,500 total, most
non-septic), but real, multi-institution, and free.

WHAT THIS SCRIPT DOES, HONESTLY:

1. Maps eICU's schema (lab/vitalPeriodic/vitalAperiodic/treatment/
   microLab/nurseCharting tables) onto the same 34 NUMERIC_FEATURES this
   project already uses, hourly-gridded the same way as
   scripts/ingest.py's Challenge 2019 pipeline.
2. Derives its own Sepsis-3 label, because eICU (unlike the Challenge 2019
   .psv files) does not ship a pre-computed SepsisLabel column. This uses
   a SIMPLIFIED SOFA score -- respiratory (PaO2/FiO2), coagulation
   (platelets), liver (bilirubin), cardiovascular (MAP / vasopressor dose
   tier, read off eICU's own treatment-string dose buckets), CNS (GCS,
   from nurseCharting), renal (creatinine) -- intersected with suspicion
   of infection (antibiotic order + culture draw within a Sepsis-3-style
   window, via treatment.csv's antibiotic strings and microLab.csv's
   culturetakenoffset). Missing SOFA components are scored 0, NOT
   imputed or excluded from the sum -- this is a conservative choice that
   likely UNDER-counts true SOFA (and therefore under-detects sepsis,
   biasing toward missing real cases) whenever a component's source table
   has no data for a given hour, rather than over-detecting. This is an
   approximation for a feasibility/evaluation pipeline, not a clinically
   validated Sepsis-3 adjudication -- do not treat these labels as chart-
   reviewed ground truth.
3. Reuses this project's OWN existing normalization stats
   (data/processed/norm_stats.json, fit on Challenge 2019's train split)
   rather than fitting new ones on eICU -- the whole point is evaluating
   already-trained models on genuinely new data, on the scale those
   models already expect.
4. Writes everything under split="eicu_demo_external", a distinct fourth
   split alongside train/val/test/external_test that
   src/tinysepsis/data/splits.py's Hospital A/B logic never touches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.data.schema import NUMERIC_FEATURES, PLAUSIBLE_RANGE  # noqa: E402
from tinysepsis.data.features import add_missingness_and_dynamics  # noqa: E402
from tinysepsis.data.labels import add_early_warning_labels  # noqa: E402
from tinysepsis.data.normalize import load_stats, apply_stats  # noqa: E402

RAW_DIR = ROOT / "data" / "raw_eicu_demo"
OUT_PATH = ROOT / "data" / "processed" / "eicu_demo_enriched.parquet"
NORM_STATS_PATH = ROOT / "data" / "processed" / "norm_stats.json"

MIN_STAY_HOURS = 4
MAX_STAY_HOURS = 500  # sanity cap; eICU offsets occasionally have data-entry outliers

# eICU labname -> our NUMERIC_FEATURES column. 'paO2' is deliberately
# excluded: it is not one of this project's 34 model features, only used
# internally (below) for the SOFA respiratory component.
LAB_NAME_MAP = {
    "O2 Sat (%)": "SaO2",  # ABG-panel saturation, distinct from vitalPeriodic's continuous pulse-ox sao2 (-> O2Sat)
    "Base Excess": "BaseExcess",
    "HCO3": "HCO3",
    "bicarbonate": "HCO3",
    "FiO2": "FiO2",
    "pH": "pH",
    "paCO2": "PaCO2",
    "AST (SGOT)": "AST",
    "BUN": "BUN",
    "alkaline phos.": "Alkalinephos",
    "calcium": "Calcium",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "direct bilirubin": "Bilirubin_direct",
    "glucose": "Glucose",
    "bedside glucose": "Glucose",
    "lactate": "Lactate",
    "magnesium": "Magnesium",
    "phosphate": "Phosphate",
    "potassium": "Potassium",
    "total bilirubin": "Bilirubin_total",
    "troponin - I": "TroponinI",
    "Hct": "Hct",
    "Hgb": "Hgb",
    "PTT": "PTT",
    "WBC x 1000": "WBC",
    "fibrinogen": "Fibrinogen",
    "platelets x 1000": "Platelets",
    "Temperature": "Temp",           # fallback if vitalPeriodic temperature is missing that hour
    "Respiratory Rate": "Resp",      # fallback if vitalPeriodic respiration is missing that hour
}

# treatment.csv's own "|antibiotics|" category strings are sparse in this
# demo (109 rows, a handful of patients) -- medication.csv's actual
# drugname field is far more complete (1,616 rows / 561 patients, ~22% of
# the cohort, a clinically plausible ICU antibiotic-exposure rate). Named
# generics only, deliberately avoiding short fragments like bare "cin" or
# "azole" that collide with unrelated drugs (heparin, proton-pump
# inhibitors like omeprazole/pantoprazole) -- an earlier looser pattern
# was checked and rejected for exactly that false-positive reason.
ANTIBIOTIC_PATTERN = (
    r"(?i)vancomycin|cefazolin|cefepime|ceftriaxone|ceftazidime|cefuroxime|cefotaxime|"
    r"cefdinir|cephalexin|ampicillin|amoxicillin|piperacillin|nafcillin|oxacillin|"
    r"meropenem|ertapenem|imipenem|doripenem|linezolid|daptomycin|metronidazole|"
    r"clindamycin|ciprofloxacin|levofloxacin|moxifloxacin|gentamicin|tobramycin|amikacin|"
    r"azithromycin|erythromycin|clarithromycin|doxycycline|minocycline|tigecycline|"
    r"sulfamethoxazole|trimethoprim|bactrim|septra|aztreonam|colistin|polymyxin|zosyn|unasyn"
)
VASOPRESSOR_PATTERN = "(?i)vasopressor|inotropic agent"
GCS_LABELS = ["Score (Glasgow Coma Scale)", "Glasgow coma score"]


def _hour(offset_minutes: pl.Expr) -> pl.Expr:
    """eICU offsets are minutes from unit (ICU) admission; convert to this
    project's 1-indexed hourly ICULOS convention, matching scripts/ingest.py."""
    return (offset_minutes / 60.0).floor().cast(pl.Int32) + 1


def load_patients() -> pl.DataFrame:
    p = pl.read_csv(RAW_DIR / "patient.csv.gz", infer_schema_length=None)
    p = p.with_columns(
        pl.col("age").str.replace("> 89", "90").cast(pl.Float64, strict=False).alias("Age"),
        pl.when(pl.col("gender") == "Male").then(1)
        .when(pl.col("gender") == "Female").then(0)
        # A handful of eICU records leave gender blank; Gender is fed to the
        # model raw (not z-scored, per STATIC_COLS in dataset.py), so unlike
        # Age it's never mean-filled downstream -- fill here or it's a NaN
        # that silently corrupts every downstream tensor it touches.
        .otherwise(0).alias("Gender"),
        pl.col("unitdischargeoffset").cast(pl.Float64, strict=False),
    )
    return p.select(["patientunitstayid", "hospitalid", "Age", "Gender", "unitdischargeoffset"])


def load_labs() -> pl.DataFrame:
    lab = pl.read_csv(RAW_DIR / "lab.csv.gz", infer_schema_length=None)
    lab = lab.filter(pl.col("labname").is_in(list(LAB_NAME_MAP.keys()) + ["FiO2", "paO2"]))
    lab = lab.with_columns(
        pl.col("labname").replace(LAB_NAME_MAP).alias("feature"),
        _hour(pl.col("labresultoffset")).alias("ICULOS"),
        pl.col("labresult").cast(pl.Float64, strict=False).alias("value"),
    )
    # FiO2 is sometimes charted as a percentage (e.g. 40) instead of a fraction (0.4).
    lab = lab.with_columns(
        pl.when((pl.col("feature") == "FiO2") & (pl.col("value") > 1.5))
        .then(pl.col("value") / 100.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    pao2 = (
        lab.filter(pl.col("labname") == "paO2")
        .group_by(["patientunitstayid", "ICULOS"])
        .agg(pl.col("value").mean().alias("PaO2"))
    )
    labs_pivot = (
        lab.filter(pl.col("labname") != "paO2")
        .group_by(["patientunitstayid", "ICULOS", "feature"])
        .agg(pl.col("value").mean())
        .pivot(on="feature", index=["patientunitstayid", "ICULOS"], values="value")
    )
    return labs_pivot, pao2


def load_vitals() -> pl.DataFrame:
    vp = pl.read_csv(RAW_DIR / "vitalPeriodic.csv.gz", infer_schema_length=None)
    vp = vp.with_columns(_hour(pl.col("observationoffset")).alias("ICULOS"))
    vp_hourly = vp.group_by(["patientunitstayid", "ICULOS"]).agg(
        pl.col("heartrate").cast(pl.Float64, strict=False).mean().alias("HR"),
        pl.col("temperature").cast(pl.Float64, strict=False).mean().alias("Temp"),
        pl.col("sao2").cast(pl.Float64, strict=False).mean().alias("O2Sat"),
        pl.col("respiration").cast(pl.Float64, strict=False).mean().alias("Resp"),
        pl.col("etco2").cast(pl.Float64, strict=False).mean().alias("EtCO2"),
        pl.col("systemicsystolic").cast(pl.Float64, strict=False).mean().alias("SBP"),
        pl.col("systemicdiastolic").cast(pl.Float64, strict=False).mean().alias("DBP"),
        pl.col("systemicmean").cast(pl.Float64, strict=False).mean().alias("MAP"),
    )

    va = pl.read_csv(RAW_DIR / "vitalAperiodic.csv.gz", infer_schema_length=None)
    va = va.with_columns(_hour(pl.col("observationoffset")).alias("ICULOS"))
    va_hourly = va.group_by(["patientunitstayid", "ICULOS"]).agg(
        pl.col("noninvasivesystolic").cast(pl.Float64, strict=False).mean().alias("SBP_ni"),
        pl.col("noninvasivediastolic").cast(pl.Float64, strict=False).mean().alias("DBP_ni"),
        pl.col("noninvasivemean").cast(pl.Float64, strict=False).mean().alias("MAP_ni"),
    )

    vitals = vp_hourly.join(va_hourly, on=["patientunitstayid", "ICULOS"], how="full", coalesce=True)
    vitals = vitals.with_columns(
        pl.coalesce(["SBP", "SBP_ni"]).alias("SBP"),
        pl.coalesce(["DBP", "DBP_ni"]).alias("DBP"),
        pl.coalesce(["MAP", "MAP_ni"]).alias("MAP"),
    ).drop(["SBP_ni", "DBP_ni", "MAP_ni"])
    return vitals


def load_infection_signals() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    med = pl.read_csv(RAW_DIR / "medication.csv.gz", infer_schema_length=None)
    med = med.with_columns(_hour(pl.col("drugstartoffset")).alias("ICULOS"))
    abx = (
        med.filter(pl.col("drugname").str.contains(ANTIBIOTIC_PATTERN))
        .group_by("patientunitstayid").agg(pl.col("ICULOS").min().alias("first_abx_hour"))
    )

    tr = pl.read_csv(RAW_DIR / "treatment.csv.gz", infer_schema_length=None)
    tr = tr.with_columns(_hour(pl.col("treatmentoffset")).alias("ICULOS"))
    vaso = tr.filter(pl.col("treatmentstring").str.contains(VASOPRESSOR_PATTERN))
    # Rough dose-tier score from eICU's own bucketed dose strings, used for the
    # SOFA cardiovascular component (a simplification -- eICU buckets dose into
    # 2-3 tiers per drug in the string itself, not a continuous mcg/kg/min value).
    vaso = vaso.with_columns(
        pl.when(vaso["treatmentstring"].str.contains("(?i)> 0.1|>15|>0.1"))
        .then(4)
        .otherwise(2)
        .alias("vaso_sofa_points")
    )
    vaso_hourly = vaso.group_by(["patientunitstayid", "ICULOS"]).agg(
        pl.col("vaso_sofa_points").max()
    )

    ml = pl.read_csv(RAW_DIR / "microLab.csv.gz", infer_schema_length=None)
    ml = ml.with_columns(_hour(pl.col("culturetakenoffset")).alias("ICULOS"))
    cultures = ml.group_by("patientunitstayid").agg(pl.col("ICULOS").min().alias("first_culture_hour"))

    return abx, vaso_hourly, cultures


def load_gcs() -> pl.DataFrame:
    nc = pl.read_csv(RAW_DIR / "nurseCharting.csv.gz", infer_schema_length=None, ignore_errors=True)
    gcs = nc.filter(pl.col("nursingchartcelltypevallabel").is_in(GCS_LABELS))
    gcs = gcs.with_columns(
        _hour(pl.col("nursingchartoffset")).alias("ICULOS"),
        pl.col("nursingchartvalue").cast(pl.Float64, strict=False).alias("gcs_value"),
    )
    gcs = gcs.filter(pl.col("gcs_value").is_between(3, 15))
    return gcs.group_by(["patientunitstayid", "ICULOS"]).agg(pl.col("gcs_value").min().alias("GCS"))


def sofa_points(df: pl.DataFrame) -> pl.Expr:
    """Sum of the 6 SOFA components computed from whatever's present at this
    hour; a missing component contributes 0, not a null -- see module
    docstring for why that's a conservative (under-count), not permissive,
    choice."""
    resp = (
        pl.when((pl.col("PaO2").is_not_null()) & (pl.col("FiO2") > 0))
        .then(
            pl.when(pl.col("PaO2") / pl.col("FiO2") < 100).then(4)
            .when(pl.col("PaO2") / pl.col("FiO2") < 200).then(3)
            .when(pl.col("PaO2") / pl.col("FiO2") < 300).then(2)
            .when(pl.col("PaO2") / pl.col("FiO2") < 400).then(1)
            .otherwise(0)
        ).otherwise(0)
    )
    coag = (
        pl.when(pl.col("Platelets") < 20).then(4)
        .when(pl.col("Platelets") < 50).then(3)
        .when(pl.col("Platelets") < 100).then(2)
        .when(pl.col("Platelets") < 150).then(1)
        .otherwise(0)
    )
    liver = (
        pl.when(pl.col("Bilirubin_total") >= 12).then(4)
        .when(pl.col("Bilirubin_total") >= 6).then(3)
        .when(pl.col("Bilirubin_total") >= 2).then(2)
        .when(pl.col("Bilirubin_total") >= 1.2).then(1)
        .otherwise(0)
    )
    cardio = (
        pl.when(pl.col("vaso_sofa_points").is_not_null()).then(pl.col("vaso_sofa_points"))
        .when(pl.col("MAP") < 70).then(1)
        .otherwise(0)
    )
    cns = (
        pl.when(pl.col("GCS") < 6).then(4)
        .when(pl.col("GCS") < 10).then(3)
        .when(pl.col("GCS") < 13).then(2)
        .when(pl.col("GCS") < 15).then(1)
        .otherwise(0)
    )
    renal = (
        pl.when(pl.col("Creatinine") >= 5).then(4)
        .when(pl.col("Creatinine") >= 3.5).then(3)
        .when(pl.col("Creatinine") >= 2.0).then(2)
        .when(pl.col("Creatinine") >= 1.2).then(1)
        .otherwise(0)
    )
    return (resp + coag + liver + cardio + cns + renal).alias("sofa")


def main():
    print("Loading patient demographics...", flush=True)
    patients = load_patients()
    n_total = patients.height

    print("Loading labs...", flush=True)
    labs_pivot, pao2 = load_labs()
    print("Loading vitals...", flush=True)
    vitals = load_vitals()
    print("Loading infection signals (antibiotics, vasopressors, cultures)...", flush=True)
    abx, vaso_hourly, cultures = load_infection_signals()
    print("Loading GCS...", flush=True)
    gcs = load_gcs()

    print("Building hourly grid...", flush=True)
    stay_hours = patients.with_columns(
        (pl.col("unitdischargeoffset") / 60.0).ceil().cast(pl.Int32).clip(MIN_STAY_HOURS, MAX_STAY_HOURS).alias("n_hours")
    ).filter(pl.col("n_hours") >= MIN_STAY_HOURS)

    grid = (
        stay_hours.select(["patientunitstayid", "n_hours"])
        .with_columns(pl.int_ranges(1, pl.col("n_hours") + 1).alias("ICULOS"))
        .explode("ICULOS")
        .select(["patientunitstayid", "ICULOS"])
    )
    print(f"Grid: {grid.height} patient-hours across {stay_hours.height} stays "
          f"(dropped {n_total - stay_hours.height} stays under {MIN_STAY_HOURS}h)", flush=True)

    df = grid.join(labs_pivot, on=["patientunitstayid", "ICULOS"], how="left")
    df = df.join(vitals, on=["patientunitstayid", "ICULOS"], how="left")
    df = df.join(pao2, on=["patientunitstayid", "ICULOS"], how="left")
    df = df.join(vaso_hourly, on=["patientunitstayid", "ICULOS"], how="left")
    df = df.join(gcs, on=["patientunitstayid", "ICULOS"], how="left")

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    print("Applying physiologic plausibility ranges...", flush=True)
    range_exprs = []
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        if col in df.columns and col != "Age":
            range_exprs.append(
                pl.when((pl.col(col) < lo) | (pl.col(col) > hi)).then(None).otherwise(pl.col(col)).alias(col)
            )
    df = df.with_columns(range_exprs)

    print("Deriving simplified Sepsis-3 labels...", flush=True)
    # Forward-fill SOFA inputs within-patient before scoring, matching how a
    # clinician reading a chart would use the last known value, not just
    # this exact hour's fresh measurement.
    ffill_cols = ["PaO2", "FiO2", "Platelets", "Bilirubin_total", "MAP", "GCS", "Creatinine", "vaso_sofa_points"]
    df = df.sort(["patientunitstayid", "ICULOS"])
    df = df.with_columns([pl.col(c).forward_fill().over("patientunitstayid").alias(c) for c in ffill_cols])
    df = df.with_columns(sofa_points(df))

    infection = (
        abx.join(cultures, on="patientunitstayid", how="inner")
        .filter((pl.col("first_abx_hour") - pl.col("first_culture_hour")).abs() <= 72)
        .with_columns(
            pl.min_horizontal("first_abx_hour", "first_culture_hour").alias("infection_window_start")
        )
        .select(["patientunitstayid", "infection_window_start"])
    )
    df = df.join(infection, on="patientunitstayid", how="left")

    onset = (
        df.filter(
            pl.col("infection_window_start").is_not_null()
            & (pl.col("ICULOS") >= pl.col("infection_window_start"))
            & (pl.col("sofa") >= 2)
        )
        .group_by("patientunitstayid")
        .agg(pl.col("ICULOS").min().alias("t_susp_mine"))
    )
    df = df.join(onset, on="patientunitstayid", how="left")
    df = df.with_columns(
        pl.when(
            pl.col("t_susp_mine").is_not_null()
            & (pl.col("ICULOS") >= pl.col("t_susp_mine") - 6)
            & (pl.col("ICULOS") < pl.col("t_susp_mine"))
        ).then(1).otherwise(0).cast(pl.Int8).alias("SepsisLabel")
    )

    n_septic = df.filter(pl.col("t_susp_mine").is_not_null())["patientunitstayid"].n_unique()
    print(f"Patients meeting simplified Sepsis-3 criteria: {n_septic} / {stay_hours.height}", flush=True)

    df = df.join(patients.select(["patientunitstayid", "Age", "Gender", "hospitalid"]), on="patientunitstayid", how="left")
    df = df.with_columns(
        ("eicu_" + pl.col("hospitalid").cast(pl.Utf8) + "_" + pl.col("patientunitstayid").cast(pl.Utf8)).alias("patient_id"),
        pl.lit(0.0).alias("HospAdmTime"),
    )

    df = df.select(["patient_id", "ICULOS", "hospitalid"] + NUMERIC_FEATURES + ["Age", "Gender", "HospAdmTime", "SepsisLabel"])

    print("Computing missingness/dynamics channels...", flush=True)
    df = add_missingness_and_dynamics(df)
    print("Deriving pre-suspicion early-warning labels + censoring...", flush=True)
    df = add_early_warning_labels(df)

    print("Applying existing (Challenge 2019 train-fit) normalization stats...", flush=True)
    stats = load_stats(NORM_STATS_PATH)
    df = apply_stats(df, stats)

    df = df.with_columns(pl.lit("eicu_demo_external").alias("split"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH, compression="zstd")

    n_final_patients = df["patient_id"].n_unique()
    n_final_septic = df.filter(pl.col("t_susp").is_not_null())["patient_id"].n_unique()
    n_hospitals = df["hospitalid"].n_unique()
    print(f"Wrote {df.height} rows, {n_final_patients} patients ({n_final_septic} ever-septic), "
          f"{n_hospitals} distinct hospitals -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
