import os, sys, json, warnings, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

# ── CONFIG ───────────────────────────────────────────────────
with open("params.yml") as f:
    params = yaml.safe_load(f)

P           = params
SEED        = P["data"]["random_state"]
N_SPLITS    = P["data"]["n_splits"]
N_BOOT      = P["evaluation"]["n_bootstrap"]
INPUT_PATH  = "data/processed/mimic_modeling_ready.csv"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", P["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(P["mlflow"]["experiment_name"])

print("=" * 60)
print("Phase 3: ML Models")
print(f"MLflow  : {MLFLOW_URI}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load data + preprocessors from Phase 2
# ─────────────────────────────────────────────────────────────
print("\nLoading data and Phase 2 preprocessors...")

df     = pd.read_csv(INPUT_PATH)
mice   = joblib.load(os.path.join(MODELS_DIR, "mice_imputer.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "standard_scaler.pkl"))

# Same preprocessing as Phase 2
for stat in ["mean","min","max","std"]:
    for bp in ["sbp","dbp","map"]:
        inv, ni, merged = f"{stat}_{bp}", f"{stat}_{bp}_ni", f"{stat}_{bp}_merged"
        if inv in df.columns and ni in df.columns:
            df[merged] = df[inv].combine_first(df[ni])
            df.drop(columns=[inv,ni], inplace=True)

str_cols = [c for c in df.select_dtypes(include=["object","category"]).columns
            if c not in ["readmitted_30d","subject_id"]]
for col in str_cols:
    df[col] = df[col].astype("category").cat.codes

DROP_EXTRA = [c for c in ["anchor_age","los","next_admittime"] if c in df.columns]
df.drop(columns=DROP_EXTRA, inplace=True)

TARGET    = "readmitted_30d"
GROUP_COL = "subject_id"

X      = df.drop(columns=[TARGET, GROUP_COL])
y      = df[TARGET]
groups = df[GROUP_COL]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# ─────────────────────────────────────────────────────────────
# STEP 2: Reproduce same split as Phase 2 (same random_state)
# ─────────────────────────────────────────────────────────────
print("Reproducing Phase 2 split...")

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for train_idx, test_idx in sgkf.split(X, y, groups=groups):
    pass

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

X_train[num_cols] = mice.transform(X_train[num_cols])
X_test[num_cols]  = mice.transform(X_test[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

assert len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) == 0
print(f"  Same split reproduced | Train: {len(X_train):,} | Test: {len(X_test):,}")


# ─────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_prob, threshold=0.5):
    rng    = np.random.default_rng(SEED)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "AUC"        : roc_auc_score(y_true, y_prob),
        "PR_AUC"     : average_precision_score(y_true, y_prob),
        "F1"         : f1_score(y_true, y_pred),
        "Brier"      : brier_score_loss(y_true, y_prob),
        "Sensitivity": tp/(tp+fn) if (tp+fn)>0 else 0,
        "Specificity": tn/(tn+fp) if (tn+fp)>0 else 0,
    }

    boot = {k: [] for k in metrics}
    n, ya = len(y_true), np.array(y_true)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb, pb = ya[idx], y_prob[idx]
        yp     = (pb >= threshold).astype(int)
        if len(np.unique(yb)) < 2:
            continue
        tni,fpi,fni,tpi = confusion_matrix(yb,yp).ravel()
        boot["AUC"].append(roc_auc_score(yb, pb))
        boot["PR_AUC"].append(average_precision_score(yb, pb))
        boot["F1"].append(f1_score(yb, yp, zero_division=0))
        boot["Brier"].append(brier_score_loss(yb, pb))
        boot["Sensitivity"].append(tpi/(tpi+fni) if (tpi+fni)>0 else 0)
        boot["Specificity"].append(tni/(tni+fpi) if (tni+fpi)>0 else 0)

    print(f"\n{'─'*55}\n  {name}\n{'─'*55}")
    for k, v in metrics.items():
        lo = np.percentile(boot[k], 2.5)
        hi = np.percentile(boot[k], 97.5)
        print(f"  {k:<15} {v:.4f}   [{lo:.4f}, {hi:.4f}]")

    return metrics

all_probs   = {}
all_metrics = {}


# ─────────────────────────────────────────────────────────────
# MODEL 1: Random Forest
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="Random_Forest"):
    rfp = P["ml"]["random_forest"]
    mlflow.log_params({**rfp, "model": "random_forest"})

    rf = RandomForestClassifier(
        n_estimators   = rfp["n_estimators"],
        max_depth      = rfp["max_depth"],
        min_samples_leaf = rfp["min_samples_leaf"],
        max_features   = rfp["max_features"],
        class_weight   = rfp["class_weight"],
        n_jobs         = rfp["n_jobs"],
        random_state   = rfp["random_state"],
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    m = evaluate_model("Random Forest", y_test, y_prob_rf)
    mlflow.log_metrics({k: round(v,4) for k,v in m.items()})
    mlflow.sklearn.log_model(rf, "random_forest")

    all_probs["Random Forest"]   = y_prob_rf
    all_metrics["Random Forest"] = m

    joblib.dump(rf, os.path.join(MODELS_DIR, "model_rf.pkl"))
    print("  Logged to MLflow")


# ─────────────────────────────────────────────────────────────
# MODEL 2: XGBoost
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="XGBoost"):
    xgp = P["ml"]["xgboost"]
    mlflow.log_params({**xgp, "model": "xgboost"})

    xgb = XGBClassifier(
        n_estimators     = xgp["n_estimators"],
        max_depth        = xgp["max_depth"],
        learning_rate    = xgp["learning_rate"],
        subsample        = xgp["subsample"],
        colsample_bytree = xgp["colsample_bytree"],
        scale_pos_weight = xgp["scale_pos_weight"],
        eval_metric      = xgp["eval_metric"],
        random_state     = xgp["random_state"],
        n_jobs           = -1,
        use_label_encoder= False,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=50)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    m = evaluate_model("XGBoost", y_test, y_prob_xgb)
    mlflow.log_metrics({k: round(v,4) for k,v in m.items()})
    mlflow.xgboost.log_model(xgb, "xgboost")

    all_probs["XGBoost"]   = y_prob_xgb
    all_metrics["XGBoost"] = m

    joblib.dump(xgb, os.path.join(MODELS_DIR, "model_xgb.pkl"))
    print("  Logged to MLflow")


# ─────────────────────────────────────────────────────────────
# MODEL 3: MLP (Neural Network)
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="MLP_Neural_Network"):
    mlpp = P["ml"]["mlp"]
    mlflow.log_params({**{k:str(v) for k,v in mlpp.items()}, "model":"mlp"})

    mlp = MLPClassifier(
        hidden_layer_sizes = tuple(mlpp["hidden_layer_sizes"]),
        activation         = mlpp["activation"],
        max_iter           = mlpp["max_epochs"],
        batch_size         = mlpp["batch_size"],
        learning_rate_init = mlpp["learning_rate"],
        early_stopping     = True,
        n_iter_no_change   = mlpp["early_stopping_patience"],
        random_state       = mlpp["random_state"],
    )
    mlp.fit(X_train, y_train)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]

    m = evaluate_model("Neural Network (MLP)", y_test, y_prob_mlp)
    mlflow.log_metrics({k: round(v,4) for k,v in m.items()})
    mlflow.sklearn.log_model(mlp, "mlp")

    all_probs["Neural Network (MLP)"]   = y_prob_mlp
    all_metrics["Neural Network (MLP)"] = m

    joblib.dump(mlp, os.path.join(MODELS_DIR, "model_mlp.pkl"))
    print("  Logged to MLflow ")


# ─────────────────────────────────────────────────────────────
# STEP 3: Save predictions + metrics JSON
# ─────────────────────────────────────────────────────────────
preds_df = pd.DataFrame({
    "y_true"  : y_test.values,
    "prob_rf" : y_prob_rf,
    "prob_xgb": y_prob_xgb,
    "prob_mlp": y_prob_mlp,
})
preds_df.to_csv(os.path.join(OUTPUTS_DIR,"phase3_test_predictions.csv"), index=False)

with open(os.path.join(OUTPUTS_DIR,"metrics_ml.json"),"w") as f:
    json.dump({n:{k:round(v,4) for k,v in m.items()}
               for n,m in all_metrics.items()}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# STEP 4: ROC + PR curves
# ─────────────────────────────────────────────────────────────
colors = {"Random Forest":"#9C27B0","XGBoost":"#F44336","Neural Network (MLP)":"#FF9800"}
fig, axes = plt.subplots(1,2,figsize=(14,5))

ax = axes[0]
for name, y_prob in all_probs.items():
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[name], lw=2,
            label=f"{name} (AUC={roc_auc_score(y_test,y_prob):.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set(xlabel="FPR",ylabel="TPR",title="ROC — ML Models")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for name, y_prob in all_probs.items():
    prec,rec,_ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color=colors[name], lw=2,
            label=f"{name} (PR-AUC={average_precision_score(y_test,y_prob):.3f})")
ax.axhline(y_test.mean(), color="k", linestyle="--", lw=1)
ax.set(xlabel="Recall",ylabel="Precision",title="PR — ML Models")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(OUTPUTS_DIR,"phase3_roc_pr_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────
# STEP 5: Calibration plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
for name, y_prob in all_probs.items():
    pt,pp = calibration_curve(y_test, y_prob, n_bins=10)
    brier = brier_score_loss(y_test, y_prob)
    ax.plot(pp, pt, marker="o", color=colors[name], lw=2,
            label=f"{name} (Brier={brier:.4f})")
ax.set(xlabel="Mean Predicted Probability",
       ylabel="Fraction of Positives",
       title="Calibration — 30-Day ICU Readmission Risk")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(OUTPUTS_DIR,"phase3_calibration.png")
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close()

# Log plots to MLflow
with mlflow.start_run(run_name="Phase3_plots"):
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(cal_path)

print("\n Phase 3 Complete!")
print(f"   MLflow UI → {MLFLOW_URI}")
print("→ Next: phase4_shap.py")
