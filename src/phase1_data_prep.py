
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

# ── CONFIG ───────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MIMIC_DIR   = "data/raw"
OUTPUT_PATH = "data/processed/mimic_modeling_ready.csv"
CHUNK_SIZE  = params["data"]["chunk_size"]

os.makedirs("data/processed", exist_ok=True)

# ── Dictionaries defined once — avoids NameError ─────────────
VITAL_ITEMS = {
    220045: "heart_rate",  220050: "sbp",       220051: "dbp",
    220052: "map",         220179: "sbp_ni",    220180: "dbp_ni",
    220181: "map_ni",      220210: "resp_rate", 223761: "temp_f",
    220277: "spo2",        220739: "gcs_eye",   223900: "gcs_verbal",
    223901: "gcs_motor",
}

LAB_ITEMS = {
    50912: "creatinine",      50971: "potassium",  50983: "sodium",
    50885: "bilirubin_total", 51222: "hemoglobin", 51301: "wbc",
    50820: "ph",              50802: "base_excess", 51006: "bun",
    50893: "calcium",         51265: "platelets",  50960: "magnesium",
}

CHARLSON_ICD10 = {
    "mi":             (["I21","I22","I25.2"], 1),
    "chf":            (["I50"], 1),
    "pvd":            (["I70","I71","I73.9","I77.1","Z95.8","Z95.9"], 1),
    "cerebrovascular":(["G45","G46","I60","I61","I62","I63","I64",
                        "I65","I66","I67","I68","I69"], 1),
    "dementia":       (["F00","F01","F02","F03","G30"], 1),
    "copd":           (["J40","J41","J42","J43","J44","J45","J46",
                        "J47","J60","J61","J62","J63","J64","J65",
                        "J66","J67"], 1),
    "rheumatic":      (["M05","M06","M32","M33","M34","M35.1","M35.3","M36.0"], 1),
    "peptic_ulcer":   (["K25","K26","K27","K28"], 1),
    "liver_mild":     (["B18","K70.0","K70.1","K70.2","K70.3",
                        "K70.9","K71","K73","K74","K76.0"], 1),
    "diabetes_wo":    (["E10","E11","E12","E13","E14"], 1),
    "diabetes_w":     (["E10.2","E10.3","E10.4","E10.5",
                        "E11.2","E11.3","E11.4","E11.5"], 2),
    "hemiplegia":     (["G04.1","G11.4","G80.1","G80.2","G81",
                        "G82","G83.0","G83.1","G83.2","G83.3"], 2),
    "renal":          (["I12.0","I13.1","N03.2","N03.3","N03.4",
                        "N03.5","N03.6","N03.7","N05.2","N05.3",
                        "N05.4","N05.5","N05.6","N05.7","N18",
                        "N19","N25.0","Z49.0","Z49.1","Z49.2",
                        "Z94.0","Z99.2"], 2),
    "cancer":         (["C0","C1","C2","C3","C4","C5","C6",
                        "C70","C71","C72","C73","C74","C75","C76"], 2),
    "liver_severe":   (["I85.0","I85.9","I86.4","I98.2","K70.4",
                        "K71.1","K72.1","K72.9","K76.5","K76.6","K76.7"], 3),
    "metastatic":     (["C77","C78","C79","C80"], 6),
    "aids":           (["B20","B21","B22","B24"], 6),
}


# ─────────────────────────────────────────────────────────────
# STEP 1: Load core tables
# ─────────────────────────────────────────────────────────────
print("Loading core tables...")

admissions = pd.read_csv(
    os.path.join(MIMIC_DIR,"admissions.csv.gz"),
    parse_dates=["admittime","dischtime","deathtime","edregtime"],
    compression="gzip"
)
patients = pd.read_csv(
    os.path.join(MIMIC_DIR,"patients.csv.gz"),
    compression="gzip"
)
icustays = pd.read_csv(
    os.path.join(MIMIC_DIR,"icustays.csv.gz"),
    parse_dates=["intime","outtime"],
    compression="gzip"
)
diagnoses = pd.read_csv(
    os.path.join(MIMIC_DIR,"diagnoses_icd.csv.gz"),
    compression="gzip"
)

print(f"  admissions : {admissions.shape}")
print(f"  patients   : {patients.shape}")
print(f"  icustays   : {icustays.shape}")
print(f"  diagnoses  : {diagnoses.shape}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Build base cohort (first ICU stay per admission)
# ─────────────────────────────────────────────────────────────
print("\nBuilding base cohort...")

icu_first = (
    icustays.sort_values("intime")
    .groupby("hadm_id", as_index=False).first()
)

cohort = icu_first.merge(
    admissions[["subject_id","hadm_id","admittime","dischtime",
                "deathtime","admission_type","insurance",
                "marital_status","race","hospital_expire_flag"]],
    on=["subject_id","hadm_id"], how="inner"
).merge(
    patients[["subject_id","gender","anchor_age","anchor_year","dod"]],
    on="subject_id", how="left"
)

print(f"  Cohort size    : {len(cohort):,} ICU stays")
print(f"  Unique patients: {cohort['subject_id'].nunique():,}")


# ─────────────────────────────────────────────────────────────
# STEP 3: Define 30-day readmission label
# Gap = Next ICU Admit Time - ICU Discharge Time
# Positive class: 0 < Gap <= 30 days
# ─────────────────────────────────────────────────────────────
print("\nDefining 30-day readmission label...")

adm_sorted = (
    admissions[["subject_id","hadm_id","admittime","dischtime"]]
    .sort_values(["subject_id","admittime"]).copy()
)
adm_sorted["next_admittime"] = (
    adm_sorted.groupby("subject_id")["admittime"].shift(-1)
)
adm_sorted["days_to_readmit"] = (
    adm_sorted["next_admittime"] - adm_sorted["dischtime"]
).dt.total_seconds() / 86400

adm_sorted["readmitted_30d"] = (
    (adm_sorted["days_to_readmit"] > 0) &
    (adm_sorted["days_to_readmit"] <= 30)
).astype(int)

cohort = cohort.merge(
    adm_sorted[["hadm_id","days_to_readmit","readmitted_30d"]],
    on="hadm_id", how="left"
)
cohort = cohort[cohort["hospital_expire_flag"] == 0].copy()

print(f"  After excluding in-hospital deaths: {len(cohort):,} stays")
print(f"  Readmission rate: {cohort['readmitted_30d'].mean():.2%}")


# ─────────────────────────────────────────────────────────────
# STEP 4: Demographic features
# ─────────────────────────────────────────────────────────────
print("\nEngineering demographic features...")

cohort["age"]         = cohort["anchor_age"]
cohort["gender_male"] = (cohort["gender"] == "M").astype(int)
cohort["los_hospital_days"] = (
    cohort["dischtime"] - cohort["admittime"]
).dt.total_seconds() / 86400
cohort["los_icu_days"]    = cohort["los"]
cohort["night_admission"] = (
    (cohort["admittime"].dt.hour >= 22) |
    (cohort["admittime"].dt.hour < 6)
).astype(int)

cohort["admission_type"] = cohort["admission_type"].str.upper()
for col in ["insurance","marital_status","race","admission_type","first_careunit"]:
    cohort[col] = cohort[col].astype("category").cat.codes

# Encode any remaining string columns
str_cols = [c for c in cohort.select_dtypes(include=["object","category"]).columns
            if c not in ["readmitted_30d","subject_id","gender","hadm_id",
                         "stay_id","intime","outtime","admittime","dischtime",
                         "deathtime","dod","next_admittime"]]
for col in str_cols:
    cohort[col] = cohort[col].astype("category").cat.codes

print("  Done ✅")


# ─────────────────────────────────────────────────────────────
# STEP 5: Charlson Comorbidity Index
# ─────────────────────────────────────────────────────────────
print("\nComputing Charlson Comorbidity Index...")

def compute_charlson(hadm_id_series, diagnoses_df):
    diag = diagnoses_df[diagnoses_df["hadm_id"].isin(hadm_id_series)].copy()
    diag["icd_code"] = diag["icd_code"].str.upper().str.strip()
    rows = []
    for hadm_id, grp in tqdm(diag.groupby("hadm_id"), desc="  Charlson", leave=False):
        codes = grp["icd_code"].tolist()
        score = sum(
            weight for _, (prefixes, weight) in CHARLSON_ICD10.items()
            if any(any(c.startswith(p.replace(".","")) or c.startswith(p)
                       for p in prefixes) for c in codes)
        )
        rows.append({"hadm_id": hadm_id, "charlson_score": score})
    return pd.DataFrame(rows)

charlson_df = compute_charlson(cohort["hadm_id"], diagnoses)
cohort = cohort.merge(charlson_df, on="hadm_id", how="left")
cohort["charlson_score"] = cohort["charlson_score"].fillna(0)
print(f"  Mean: {cohort['charlson_score'].mean():.2f} | Max: {cohort['charlson_score'].max():.0f}")


# ─────────────────────────────────────────────────────────────
# STEP 6: Lab features (chunked — memory safe)
# ─────────────────────────────────────────────────────────────
print("\nExtracting lab features (chunked)...")

cohort_hadm_ids = set(cohort["hadm_id"].tolist())
target_item_ids = set(LAB_ITEMS.keys())
lab_chunks      = []

for chunk in tqdm(
    pd.read_csv(
        os.path.join(MIMIC_DIR,"labevents.csv.gz"),
        usecols=["hadm_id","itemid","charttime","valuenum"],
        parse_dates=["charttime"], compression="gzip",
        chunksize=CHUNK_SIZE,
        dtype={"hadm_id":"Int32","itemid":"Int32","valuenum":"float32"},
    ), desc="  labevents chunks"
):
    chunk = chunk[
        chunk["hadm_id"].isin(cohort_hadm_ids) &
        chunk["itemid"].isin(target_item_ids)  &
        chunk["valuenum"].notna()
    ]
    if not chunk.empty:
        lab_chunks.append(chunk)

lab_cohort = pd.concat(lab_chunks, ignore_index=True)
del lab_chunks
print(f"  Loaded {len(lab_cohort):,} lab records")

lab_cohort = lab_cohort.merge(
    cohort[["hadm_id","intime","outtime"]], on="hadm_id", how="left"
)
lab_cohort = lab_cohort[
    (lab_cohort["charttime"] >= lab_cohort["intime"]) &
    (lab_cohort["charttime"] <= lab_cohort["outtime"])
].copy()

lab_cohort["lab_name"] = lab_cohort["itemid"].map(LAB_ITEMS)
lab_agg = (
    lab_cohort.sort_values("charttime")
    .groupby(["hadm_id","lab_name"])["valuenum"]
    .agg(["mean","last"]).reset_index()
)
lab_agg.columns = ["hadm_id","lab_name","lab_mean","lab_last"]

lab_mean_w = lab_agg.pivot(index="hadm_id", columns="lab_name", values="lab_mean")
lab_mean_w.columns = [f"{c}_mean" for c in lab_mean_w.columns]
lab_last_w = lab_agg.pivot(index="hadm_id", columns="lab_name", values="lab_last")
lab_last_w.columns = [f"{c}_last" for c in lab_last_w.columns]

lab_features = lab_mean_w.join(lab_last_w, how="outer").reset_index()
cohort = cohort.merge(lab_features, on="hadm_id", how="left")
del lab_cohort, lab_agg, lab_mean_w, lab_last_w
print(f"  Lab features added: {lab_features.shape[1]-1} columns")


# ─────────────────────────────────────────────────────────────
# STEP 7: Vital sign features (chunked — memory safe)
# ─────────────────────────────────────────────────────────────
print("\nExtracting vital sign features (chunked)...")

cohort_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())
vital_item_ids  = set(VITAL_ITEMS.keys())
chunk_list      = []

for chunk in tqdm(
    pd.read_csv(
        os.path.join(MIMIC_DIR,"chartevents.csv.gz"),
        usecols=["stay_id","itemid","charttime","valuenum"],
        parse_dates=["charttime"], compression="gzip",
        chunksize=CHUNK_SIZE,
        dtype={"stay_id":"Int32","itemid":"Int32","valuenum":"float32"},
    ), desc="  chartevents chunks"
):
    chunk = chunk[
        chunk["stay_id"].isin(cohort_stay_ids) &
        chunk["itemid"].isin(vital_item_ids)   &
        chunk["valuenum"].notna()
    ]
    if not chunk.empty:
        chunk_list.append(chunk)

chartevents = pd.concat(chunk_list, ignore_index=True)
del chunk_list
print(f"  Loaded {len(chartevents):,} vital records")

chartevents["vital_name"] = chartevents["itemid"].map(VITAL_ITEMS)
vital_agg = (
    chartevents.groupby(["stay_id","vital_name"])["valuenum"]
    .agg(["mean","min","max","std"]).reset_index()
)
vital_agg.columns = ["stay_id","vital_name","mean","min","max","std"]

vital_wide = vital_agg.pivot(
    index="stay_id", columns="vital_name",
    values=["mean","min","max","std"]
)
vital_wide.columns = [f"{stat}_{name}" for stat,name in vital_wide.columns]
vital_wide = vital_wide.reset_index()

cohort = cohort.merge(vital_wide, on="stay_id", how="left")
del chartevents, vital_agg
print(f"  Vital features added: {vital_wide.shape[1]-1} columns ")


# ─────────────────────────────────────────────────────────────
# STEP 8: Derived features
# ─────────────────────────────────────────────────────────────
print("\nCreating derived features...")

for col in ["mean_gcs_eye","mean_gcs_verbal","mean_gcs_motor"]:
    if col not in cohort.columns:
        cohort[col] = np.nan

cohort["gcs_total_mean"] = (
    cohort["mean_gcs_eye"].fillna(0) +
    cohort["mean_gcs_verbal"].fillna(0) +
    cohort["mean_gcs_motor"].fillna(0)
)
cohort.loc[
    cohort[["mean_gcs_eye","mean_gcs_verbal","mean_gcs_motor"]].isna().all(axis=1),
    "gcs_total_mean"
] = np.nan

if "mean_sbp" in cohort.columns and "mean_dbp" in cohort.columns:
    cohort["pulse_pressure_mean"] = cohort["mean_sbp"] - cohort["mean_dbp"]
if "mean_heart_rate" in cohort.columns and "mean_sbp" in cohort.columns:
    cohort["shock_index"] = (
        cohort["mean_heart_rate"] / cohort["mean_sbp"].replace(0, np.nan)
    )

cohort["age_group"] = pd.cut(
    cohort["age"], bins=[0,45,65,75,120],
    labels=["<45","45-65","65-75","75+"]
).cat.codes

print("  Done ")


# ─────────────────────────────────────────────────────────────
# STEP 9: Final cleanup + save
# subject_id kept — needed for StratifiedGroupKFold in Phase 2
# ─────────────────────────────────────────────────────────────
print("\nFinalizing dataset...")

DROP_COLS = [
    "hadm_id","stay_id",
    "intime","outtime","admittime","dischtime","deathtime",
    "dod","anchor_year","next_admittime",
    "hospital_expire_flag",
    "gender",
    "days_to_readmit",
]
DROP_COLS = [c for c in DROP_COLS if c in cohort.columns]
final_df  = cohort.drop(columns=DROP_COLS)

missing      = final_df.isna().mean().sort_values(ascending=False)
high_missing = missing[missing > 0.5]
if not high_missing.empty:
    print(f"\n  {len(high_missing)} features >50% missing:")
    print(high_missing.head(10).to_string())

final_df.to_csv(OUTPUT_PATH, index=False)

n_vitals = sum(1 for c in final_df.columns if any(v in c for v in VITAL_ITEMS.values()))
n_labs   = sum(1 for c in final_df.columns if any(l in c for l in LAB_ITEMS.values()))

print(f"\nSaved → {OUTPUT_PATH}")
print(f"   Shape  : {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
print(f"   Target : readmitted_30d  |  {final_df['readmitted_30d'].mean():.2%} positive")
print(f"   Vitals : {n_vitals} columns | Labs: {n_labs} columns")
print(f"   subject_id retained for Phase 2 patient-level split")
print("\n→ Next: phase2_baseline_models.py")