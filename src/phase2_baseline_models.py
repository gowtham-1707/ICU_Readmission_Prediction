
import os, sys, json, warnings, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

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

# ── MLflow setup ─────────────────────────────────────────────
MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI",
                             P["mlflow"]["tracking_uri"])
EXP_NAME    = P["mlflow"]["experiment_name"]

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXP_NAME)

print("=" * 60)
print("Phase 2: Baseline Models")
print(f"MLflow  : {MLFLOW_URI}")
print(f"Experiment: {EXP_NAME}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)

# Fix BP columns
for stat in ["mean","min","max","std"]:
    for bp in ["sbp","dbp","map"]:
        inv, ni, merged = f"{stat}_{bp}", f"{stat}_{bp}_ni", f"{stat}_{bp}_merged"
        if inv in df.columns and ni in df.columns:
            df[merged] = df[inv].combine_first(df[ni])
            df.drop(columns=[inv,ni], inplace=True)

# Encode any remaining string columns
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

print(f"  Features: {X.shape[1]} | Target: {y.mean():.2%} positive")


# ─────────────────────────────────────────────────────────────
# STEP 2: Patient-level stratified split
# ─────────────────────────────────────────────────────────────
print("\nPatient-level split (StratifiedGroupKFold)...")

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for train_idx, test_idx in sgkf.split(X, y, groups=groups):
    pass

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

assert len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) == 0
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | No patient overlap")


# ─────────────────────────────────────────────────────────────
# STEP 3: MICE imputation + scaling (fit on train only)
# ─────────────────────────────────────────────────────────────
print("\nMICE imputation + scaling (fit on train only)...")

mice = IterativeImputer(max_iter=P["preprocessing"]["mice_max_iter"],
                        random_state=P["preprocessing"]["mice_random_state"])
X_train[num_cols] = mice.fit_transform(X_train[num_cols])
X_test[num_cols]  = mice.transform(X_test[num_cols])

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

joblib.dump(mice,   os.path.join(MODELS_DIR, "mice_imputer.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "standard_scaler.pkl"))
print("  Preprocessors saved (fit on train only — no leakage)")


# ─────────────────────────────────────────────────────────────
# Evaluation helper with 95% bootstrap CIs
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

    return metrics, {k: np.percentile(boot[k], [2.5,97.5]).tolist()
                     for k in metrics}

all_probs   = {}
all_metrics = {}


# ─────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.log_params({
        "model"        : "logistic_regression",
        "penalty"      : "none",
        "solver"       : "lbfgs",
        "random_state" : SEED,
        "n_train"      : len(X_train),
        "n_test"       : len(X_test),
        "mice_iters"   : P["preprocessing"]["mice_max_iter"],
    })

    lr = LogisticRegression(penalty=None, solver="lbfgs",
                            max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    m, ci = evaluate_model("Logistic Regression", y_test, y_prob_lr)
    mlflow.log_metrics({k: round(v,4) for k,v in m.items()})
    mlflow.sklearn.log_model(lr, "logistic_regression")

    all_probs["Logistic Regression"]   = y_prob_lr
    all_metrics["Logistic Regression"] = m

    joblib.dump(lr, os.path.join(MODELS_DIR, "model_lr.pkl"))
    print("  Logged to MLflow")


# ─────────────────────────────────────────────────────────────
# MODEL 2: LASSO (L1)
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="LASSO_L1"):
    lp = P["baseline"]["lasso"]
    mlflow.log_params({
        "model"       : "lasso",
        "penalty"     : "l1",
        "Cs"          : lp["Cs"],
        "solver"      : lp["solver"],
        "max_iter"    : lp["max_iter"],
        "scoring"     : lp["scoring"],
        "random_state": SEED,
    })

    lasso = LogisticRegressionCV(
        Cs=lp["Cs"], cv=StratifiedGroupKFold(n_splits=N_SPLITS),
        penalty="l1", solver=lp["solver"], max_iter=lp["max_iter"],
        scoring=lp["scoring"], random_state=SEED, n_jobs=-1
    )
    lasso.fit(X_train, y_train)
    y_prob_lasso = lasso.predict_proba(X_test)[:, 1]

    n_nonzero = int(np.sum(lasso.coef_[0] != 0))
    m, ci = evaluate_model("LASSO (L1)", y_test, y_prob_lasso)
    mlflow.log_metrics({**{k: round(v,4) for k,v in m.items()},
                        "n_nonzero_features": n_nonzero,
                        "best_C": float(lasso.C_[0])})
    mlflow.sklearn.log_model(lasso, "lasso")

    all_probs["LASSO (L1)"]   = y_prob_lasso
    all_metrics["LASSO (L1)"] = m

    joblib.dump(lasso, os.path.join(MODELS_DIR, "model_lasso.pkl"))
    print(f"  Non-zero features: {n_nonzero} | Best C: {lasso.C_[0]:.4f}")
    print("  Logged to MLflow")


# ─────────────────────────────────────────────────────────────
# MODEL 3: Ridge (L2)
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="Ridge_L2"):
    rp = P["baseline"]["ridge"]
    mlflow.log_params({
        "model"       : "ridge",
        "penalty"     : "l2",
        "Cs"          : rp["Cs"],
        "solver"      : rp["solver"],
        "max_iter"    : rp["max_iter"],
        "random_state": SEED,
    })

    ridge = LogisticRegressionCV(
        Cs=rp["Cs"], cv=StratifiedGroupKFold(n_splits=N_SPLITS),
        penalty="l2", solver=rp["solver"], max_iter=rp["max_iter"],
        scoring=rp["scoring"], random_state=SEED, n_jobs=-1
    )
    ridge.fit(X_train, y_train)
    y_prob_ridge = ridge.predict_proba(X_test)[:, 1]

    m, ci = evaluate_model("Ridge (L2)", y_test, y_prob_ridge)
    mlflow.log_metrics({**{k: round(v,4) for k,v in m.items()},
                        "best_C": float(ridge.C_[0])})
    mlflow.sklearn.log_model(ridge, "ridge")

    all_probs["Ridge (L2)"]   = y_prob_ridge
    all_metrics["Ridge (L2)"] = m

    joblib.dump(ridge, os.path.join(MODELS_DIR, "model_ridge.pkl"))
    print("  Logged to MLflow ")


# ─────────────────────────────────────────────────────────────
# STEP 4: Save predictions + metrics
# ─────────────────────────────────────────────────────────────
preds_df = pd.DataFrame({
    "y_true"    : y_test.values,
    "prob_lr"   : y_prob_lr,
    "prob_lasso": y_prob_lasso,
    "prob_ridge": y_prob_ridge,
})
preds_df.to_csv(os.path.join(OUTPUTS_DIR,"phase2_test_predictions.csv"), index=False)

with open(os.path.join(OUTPUTS_DIR,"metrics_baseline.json"),"w") as f:
    json.dump({n: {k: round(v,4) for k,v in m.items()}
               for n,m in all_metrics.items()}, f, indent=2)

# ─────────────────────────────────────────────────────────────
# STEP 5: ROC + PR curves
# ─────────────────────────────────────────────────────────────
colors = {"Logistic Regression":"#2196F3","LASSO (L1)":"#03A9F4","Ridge (L2)":"#00BCD4"}
fig, axes = plt.subplots(1,2,figsize=(14,5))

ax = axes[0]
for name, y_prob in all_probs.items():
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[name], lw=2,
            label=f"{name} (AUC={roc_auc_score(y_test,y_prob):.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC — Baseline Models")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for name, y_prob in all_probs.items():
    prec,rec,_ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color=colors[name], lw=2,
            label=f"{name} (PR-AUC={average_precision_score(y_test,y_prob):.3f})")
ax.axhline(y_test.mean(), color="k", linestyle="--", lw=1)
ax.set(xlabel="Recall", ylabel="Precision", title="PR — Baseline Models")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(OUTPUTS_DIR,"phase2_roc_pr_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────
# STEP 6: Calibration plot (Brier = same source as metrics)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
for name, y_prob in all_probs.items():
    pt, pp = calibration_curve(y_test, y_prob, n_bins=10)
    brier  = brier_score_loss(y_test, y_prob)
    ax.plot(pp, pt, marker="o", color=colors[name], lw=2,
            label=f"{name} (Brier={brier:.4f})")
ax.set(xlabel="Mean Predicted Probability",
       ylabel="Fraction of Positives",
       title="Calibration — 30-Day ICU Readmission Risk")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(OUTPUTS_DIR,"phase2_calibration.png")
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────
# STEP 7: LASSO feature importance
# ─────────────────────────────────────────────────────────────
coef_df = (
    pd.DataFrame({"feature":X_train.columns,"coefficient":lasso.coef_[0]})
    .query("coefficient != 0")
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False).head(20)
)
colors_bar = ["#F44336" if c>0 else "#2196F3" for c in coef_df["coefficient"]]
fig, ax = plt.subplots(figsize=(9,7))
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors_bar)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(xlabel="LASSO Coefficient",
       title="Top 20 Features — LASSO (L1)\n30-Day ICU Readmission Risk")
ax.invert_yaxis()
plt.tight_layout()
feat_path = os.path.join(OUTPUTS_DIR,"phase2_lasso_features.png")
plt.savefig(feat_path, dpi=150, bbox_inches="tight")
plt.close()

# Log plots to MLflow
with mlflow.start_run(run_name="Phase2_plots"):
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(cal_path)
    mlflow.log_artifact(feat_path)

print("\n Phase 2 Complete!")
print(f"   MLflow UI → {MLFLOW_URI}")
print("→ Next: phase3_ml_models.py")
