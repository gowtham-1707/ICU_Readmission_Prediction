import os, warnings, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import mlflow
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold

# ── CONFIG ───────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

P           = params
SEED        = P["data"]["random_state"]
N_SPLITS    = P["data"]["n_splits"]
N_SAMPLES   = P["shap"]["n_samples"]
MAX_DISP    = P["shap"]["max_display"]
BEST_MODEL  = P["shap"]["best_model"]
INPUT_PATH  = "data/processed/mimic_modeling_ready.csv"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", P["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(P["mlflow"]["experiment_name"])

print("=" * 60)
print(f"Phase 4: SHAP — {BEST_MODEL.upper()}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load data + preprocessors + best model
# ─────────────────────────────────────────────────────────────
print("\nLoading data, preprocessors, and model...")

df     = pd.read_csv(INPUT_PATH)
mice   = joblib.load(os.path.join(MODELS_DIR, "mice_imputer.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "standard_scaler.pkl"))
model  = joblib.load(os.path.join(MODELS_DIR, f"model_{BEST_MODEL}.pkl"))

TARGET    = "readmitted_30d"
GROUP_COL = "subject_id"

# Same preprocessing as Phase 2/3
for stat in ["mean","min","max","std"]:
    for bp in ["sbp","dbp","map"]:
        inv, ni, merged = f"{stat}_{bp}", f"{stat}_{bp}_ni", f"{stat}_{bp}_merged"
        if inv in df.columns and ni in df.columns:
            df[merged] = df[inv].combine_first(df[ni])
            df.drop(columns=[inv,ni], inplace=True)

str_cols = [c for c in df.select_dtypes(include=["object","category"]).columns
            if c not in [TARGET, GROUP_COL]]
for col in str_cols:
    df[col] = df[col].astype("category").cat.codes

DROP_EXTRA = [c for c in ["anchor_age","los","next_admittime"] if c in df.columns]
df.drop(columns=DROP_EXTRA, inplace=True)

X      = df.drop(columns=[TARGET, GROUP_COL])
y      = df[TARGET]
groups = df[GROUP_COL]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# ─────────────────────────────────────────────────────────────
# STEP 2: Reproduce test set (same split as Phase 2/3)
# ─────────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for train_idx, test_idx in sgkf.split(X, y, groups=groups):
    pass

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()
X_test[num_cols] = mice.transform(X_test[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print(f"  Test set: {len(X_test):,} stays")


# ─────────────────────────────────────────────────────────────
# STEP 3: Compute SHAP values
# ─────────────────────────────────────────────────────────────
print(f"\nComputing SHAP values (n={N_SAMPLES})...")

rng        = np.random.default_rng(SEED)
sample_idx = rng.choice(len(X_test), size=min(N_SAMPLES,len(X_test)), replace=False)
X_shap     = X_test.iloc[sample_idx]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"  SHAP values shape: {shap_values.shape}")


with mlflow.start_run(run_name=f"SHAP_{BEST_MODEL}"):
    mlflow.log_params({
        "model"      : BEST_MODEL,
        "n_samples"  : N_SAMPLES,
        "max_display": MAX_DISP,
    })

    # ── SHAP summary (beeswarm) ───────────────────────────────
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X_shap, max_display=MAX_DISP, show=False)
    plt.title(
        f"SHAP Summary — 30-Day ICU Readmission Risk\n"
        f"({BEST_MODEL.upper()}, n={N_SAMPLES} samples)",
        fontsize=12, pad=15
    )
    plt.tight_layout()
    summary_path = os.path.join(OUTPUTS_DIR, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(summary_path)
    print(f"  Saved → {summary_path}")

    # ── SHAP bar plot ─────────────────────────────────────────
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df   = pd.DataFrame({
        "feature"   : X_shap.columns,
        "mean_shap" : mean_shap
    }).sort_values("mean_shap", ascending=False).head(20)

    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(i/len(shap_df)) for i in range(len(shap_df))]
    fig, ax = plt.subplots(figsize=(9,7))
    ax.barh(shap_df["feature"], shap_df["mean_shap"], color=colors)
    ax.set(xlabel="Mean |SHAP Value|",
           title="Top 20 Features — 30-Day ICU Readmission Risk")
    ax.invert_yaxis()
    plt.tight_layout()
    bar_path = os.path.join(OUTPUTS_DIR, "shap_bar_plot.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(bar_path)
    print(f"  Saved → {bar_path}")

    # ── Dependence plots (top 5) ──────────────────────────────
    top5 = shap_df["feature"].head(5).tolist()
    fig, axes = plt.subplots(1,5,figsize=(20,4))
    for ax, feat in zip(axes, top5):
        feat_idx = list(X_shap.columns).index(feat)
        ax.scatter(
            X_shap[feat], shap_values[:,feat_idx],
            c=shap_values[:,feat_idx],
            cmap="coolwarm", alpha=0.4, s=10
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set(xlabel=feat, ylabel="SHAP value", title=feat)
    plt.suptitle("SHAP Dependence — Top 5 Features\n30-Day ICU Readmission Risk",
                 fontsize=11)
    plt.tight_layout()
    dep_path = os.path.join(OUTPUTS_DIR, "shap_dependence_plots.png")
    plt.savefig(dep_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(dep_path)
    print(f"  Saved → {dep_path}")

    # ── Save SHAP values CSV ──────────────────────────────────
    shap_out = pd.DataFrame(shap_values, columns=X_shap.columns)
    shap_out.insert(0, "y_true", y_test.iloc[sample_idx].values)
    shap_csv = os.path.join(OUTPUTS_DIR, "shap_values.csv")
    shap_out.to_csv(shap_csv, index=False)
    mlflow.log_artifact(shap_csv)

    # ── Log top 10 feature importance as metrics ──────────────
    for _, row in shap_df.head(10).iterrows():
        mlflow.log_metric(f"shap_{row['feature']}", round(row["mean_shap"],4))

    print(f"\n  All SHAP artifacts logged to MLflow ")

print("\nPhase 4 Complete!")
print("→ Next: phase5_consolidate.py")