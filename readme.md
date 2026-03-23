# ─────────────────────────────────────────────────────────────
# README.md — ICU 30-Day Readmission Prediction
# Full reproducibility with Git + DVC + Docker
# ─────────────────────────────────────────────────────────────

# Predicting 30-Day ICU Readmission using Machine Learning
**GOWTHAM K (MA24C051) | IIT Madras | Supervisor: Prof. Neelesh Upadhye**

## Project Structure
```
icu-readmission/
├── src/
│   ├── phase1_data_prep.py        # MIMIC-IV feature engineering
│   ├── phase2_baseline_models.py  # LR, LASSO, Ridge
│   ├── phase3_ml_models.py        # RF, XGBoost, MLP
│   ├── phase4_shap.py             # SHAP interpretability
│   └── phase5_consolidate.py      # Final results table
├── data/
│   ├── raw/                       # MIMIC-IV files (DVC)
│   └── processed/                 # modeling_ready.csv (DVC)
├── models/                        # Trained .pkl files (DVC)
├── outputs/                       # Plots, tables (DVC)
├── Dockerfile                     # Frozen Python environment
├── docker-compose.yml             # Easy container management
├── dvc.yaml                       # Pipeline definition
├── params.yaml                    # All tunable parameters
└── requirements.txt               # Pinned dependencies
```

## Reproduce This Project (3 steps)

### Step 1 — Clone and pull data
```bash
git clone https://github.com/YOUR_USERNAME/icu-readmission.git
cd icu-readmission
dvc pull          # downloads data + models from Google Drive
```

### Step 2 — Build Docker image
```bash
docker-compose build
```

### Step 3 — Run full pipeline
```bash
docker-compose up pipeline
# OR run one stage:
docker-compose run dvc repro baseline_models
```

## First-Time Setup (for Gowtham only)

### 1. Install tools
```bash
# Install Git (already have)
# Install Docker Desktop for Windows: https://docs.docker.com/desktop/windows/
# Install DVC
pip install dvc dvc-gdrive
```

### 2. Initialize Git + DVC
```bash
git init
dvc init
git add .dvc .gitignore
git commit -m "Initialize Git + DVC"
```

### 3. Set up Google Drive remote
```bash
# Create folder "dvc-icu-readmission" in Drive
# Copy folder ID from URL: drive.google.com/drive/folders/FOLDER_ID_HERE
dvc remote add -d gdrive_remote gdrive://YOUR_FOLDER_ID_HERE
git add .dvc/config
git commit -m "Add Google Drive DVC remote"
```

### 4. Organize files
```
Move your files:
  phase1_complete.py          → src/phase1_data_prep.py
  phase2_baseline_models.py   → src/phase2_baseline_models.py
  admissions.csv.gz etc.      → data/raw/
  mimic_modeling_ready.csv    → data/processed/
  *.pkl                       → models/
  *.png, *.csv results        → outputs/
```

### 5. Track large files with DVC
```bash
# Track raw data
dvc add data/raw/admissions.csv.gz
dvc add data/raw/patients.csv.gz
dvc add data/raw/icustays.csv.gz
dvc add data/raw/diagnoses_icd.csv.gz
dvc add data/raw/labevents.csv.gz
dvc add data/raw/chartevents.csv.gz

# Track processed data
dvc add data/processed/mimic_modeling_ready.csv

# Track models
dvc add models/model_lr.pkl models/model_lasso.pkl
dvc add models/model_ridge.pkl models/model_xgb.pkl
dvc add models/model_rf.pkl models/model_mlp.pkl
dvc add models/mice_imputer.pkl models/standard_scaler.pkl

# Commit pointer files to Git
git add data/ models/ outputs/
git add .gitignore
git commit -m "Add DVC tracking for all large files"
```

### 6. Push to remotes
```bash
# Push large files to Google Drive
dvc push

# Push code to GitHub
git remote add origin https://github.com/YOUR_USERNAME/icu-readmission.git
git push -u origin main
```

### 7. Build Docker image
```bash
docker-compose build
# This creates image: icu-readmission:latest
```

## Daily Workflow

```bash
# Run full pipeline
docker-compose up pipeline

# Run one stage only
docker-compose run dvc repro baseline_models

# Run in Jupyter
docker-compose up jupyter
# Open: http://localhost:8888  Token: icu2025

# Check what changed
dvc status

# View pipeline diagram
dvc dag

# Compare experiment metrics
dvc metrics show
```

## Experiment Tracking

```bash
# Change a parameter in params.yaml then run:
dvc exp run --name "xgb_depth8"

# Compare all experiments
dvc exp show

# Promote best experiment
dvc exp apply xgb_depth8
```

## Pipeline DAG

```
data/raw/ ──→ [data_prep] ──→ data/processed/
                                     │
                     ┌───────────────┼───────────────┐
                     ↓               ↓               ↓
              [baseline_models] [ml_models]  [interpretability]
                     │               │
                     └───────┬───────┘
                             ↓
                  [consolidate_results]
                             │
                             ↓
                    outputs/final_*.png/.csv
```

## Key Results (from report)

| Model | AUC (95% CI) | Brier | F1 |
|---|---|---|---|
| XGBoost | 0.763 (0.752–0.775) | 0.097 | 0.422 |
| Random Forest | 0.760 (0.748–0.771) | 0.098 | 0.419 |
| Logistic Regression | 0.758 (0.747–0.769) | 0.191 | 0.238 |
| Neural Network | 0.731 (0.718–0.744) | 0.105 | 0.384 |

## Dataset
MIMIC-IV v2.2 (PhysioNet) — requires credentialed access.
Request access: https://physionet.org/content/mimiciv/

## References
1. Johnson AEW et al. MIMIC-IV. Sci Data. 2023.
2. James G et al. ISLR. Springer. 2021.
3. Chen T, Guestrin C. XGBoost. ACM SIGKDD. 2016.