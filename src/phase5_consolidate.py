import os, sys, json, warnings, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
N_BOOT      = P["evaluation"]["n_bootstrap"]
THR         = P["evaluation"]["threshold_default"]
OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", P["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(P["mlflow"]["experiment_name"])

print("=" * 60)
print("Phase 5: Consolidate All Results")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load all saved predictions
# Single source of truth — all metrics derived from here
# ─────────────────────────────────────────────────────────────
print("\nLoading saved predictions...")

p2 = pd.read_csv(os.path.join(OUTPUTS_DIR, "phase2_test_predictions.csv"))
p3 = pd.read_csv(os.path.join(OUTPUTS_DIR, "phase3_test_predictions.csv"))

assert np.array_equal(p2["y_true"].values, p3["y_true"].values), \
    "Test sets differ between Phase 2 and Phase 3 — recheck split!"
print("  Phase 2 + Phase 3 test sets match ")

y_true = p2["y_true"].values

all_probs = {
    "Logistic Regression" : p2["prob_lr"].values,
    "LASSO (L1)"          : p2["prob_lasso"].values,
    "Ridge (L2)"          : p2["prob_ridge"].values,
    "Random Forest"       : p3["prob_rf"].values,
    "XGBoost"             : p3["prob_xgb"].values,
    "Neural Network (MLP)": p3["prob_mlp"].values,
}

print(f"  Models: {list(all_probs.keys())}")
print(f"  Test size: {len(y_true):,} | Readmission: {y_true.mean():.2%}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Compute all metrics with 95% bootstrap CIs
# ─────────────────────────────────────────────────────────────
print("\nComputing metrics with bootstrap CIs...")

def full_evaluate(name, y_true, y_prob, threshold=THR):
    rng    = np.random.default_rng(SEED)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    point = {
        "AUC"        : roc_auc_score(y_true, y_prob),
        "PR_AUC"     : average_precision_score(y_true, y_prob),
        "F1"         : f1_score(y_true, y_pred),
        "Brier"      : brier_score_loss(y_true, y_prob),
        "Sensitivity": tp/(tp+fn) if (tp+fn)>0 else 0,
        "Specificity": tn/(tn+fp) if (tn+fp)>0 else 0,
    }

    boot = {k: [] for k in point}
    n    = len(y_true)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], y_prob[idx]
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

    row = {"Model": name}
    for k, v in point.items():
        lo = np.percentile(boot[k], 2.5)
        hi = np.percentile(boot[k], 97.5)
        row[k]         = round(v, 4)
        row[f"{k}_CI"] = f"[{lo:.4f}, {hi:.4f}]"
    return row

rows = []
for name, y_prob in all_probs.items():
    print(f"  Evaluating {name}...")
    rows.append(full_evaluate(name, y_true, y_prob))

results_df = pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# STEP 3: Print + save final results table
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL RESULTS TABLE — copy into report p.17")
print("="*80)
disp = ["Model","AUC","AUC_CI","PR_AUC","F1","Brier","Sensitivity","Specificity"]
print(results_df[disp].to_string(index=False))

table_path = os.path.join(OUTPUTS_DIR, "final_results_table.csv")
results_df.to_csv(table_path, index=False)

metrics_json = {
    row["Model"]: {k: row[k] for k in ["AUC","AUC_CI","PR_AUC","F1",
                                         "Brier","Sensitivity","Specificity"]}
    for _, row in results_df.iterrows()
}
json_path = os.path.join(OUTPUTS_DIR, "final_metrics.json")
with open(json_path, "w") as f:
    json.dump(metrics_json, f, indent=2)


# ─────────────────────────────────────────────────────────────
# STEP 4: Final ROC curve (all 6 models)
# ─────────────────────────────────────────────────────────────
colors  = {
    "Logistic Regression" :"#2196F3","LASSO (L1)"          :"#03A9F4",
    "Ridge (L2)"          :"#00BCD4","Random Forest"        :"#9C27B0",
    "XGBoost"             :"#F44336","Neural Network (MLP)" :"#FF9800",
}
styles  = {
    "Logistic Regression" :"--","LASSO (L1)":"-.","Ridge (L2)":":",
    "Random Forest"       :"-" ,"XGBoost"   :"-","Neural Network (MLP)":"-",
}

fig, ax = plt.subplots(figsize=(9,7))
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random (AUC=0.500)")
for name, y_prob in all_probs.items():
    fpr,tpr,_ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ci  = results_df[results_df.Model==name]["AUC_CI"].values[0]
    ax.plot(fpr, tpr, color=colors[name], linestyle=styles[name], lw=2,
            label=f"{name}  AUC={auc:.3f} {ci}")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curves — All Models\n30-Day ICU Readmission Risk")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUTS_DIR, "final_roc_plot.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────
# STEP 5: Final calibration plot
# Brier values here = Brier in results table (SAME y_prob)
# This makes p.15 and p.17 always consistent
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,7))
ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
for name, y_prob in all_probs.items():
    pt,pp = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    ax.plot(pp, pt, marker="o", color=colors[name], linestyle=styles[name], lw=2,
            label=f"{name} (Brier={brier:.4f})")
ax.set(xlabel="Mean Predicted Probability",
       ylabel="Fraction of Positives",
       title="Calibration Plot — 30-Day ICU Readmission Risk\n"
             "Brier scores = values in results table (same predictions)")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(OUTPUTS_DIR, "final_calibration_plot.png")
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────
# STEP 6: Log everything to MLflow
# ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="Final_Consolidated_Results"):
    # Log all model metrics
    for _, row in results_df.iterrows():
        name_clean = row["Model"].replace(" ","_").replace("(","").replace(")","")
        mlflow.log_metrics({
            f"{name_clean}_AUC"        : row["AUC"],
            f"{name_clean}_PR_AUC"     : row["PR_AUC"],
            f"{name_clean}_F1"         : row["F1"],
            f"{name_clean}_Brier"      : row["Brier"],
            f"{name_clean}_Sensitivity": row["Sensitivity"],
            f"{name_clean}_Specificity": row["Specificity"],
        })

    # Log all artifacts
    mlflow.log_artifact(table_path)
    mlflow.log_artifact(json_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(cal_path)

    # Tag best model
    best = results_df.loc[results_df["AUC"].idxmax()]
    mlflow.set_tag("best_model", best["Model"])
    mlflow.set_tag("best_AUC",   str(best["AUC"]))
    mlflow.set_tag("best_Brier", str(best["Brier"]))

    print(f"\n  All results logged to MLflow")
    print(f"  Best model: {best['Model']} | AUC={best['AUC']}")


# ─────────────────────────────────────────────────────────────
# STEP 7: Final summary
# ─────────────────────────────────────────────────────────────
best  = results_df.loc[results_df["AUC"].idxmax()]
worst = results_df.loc[results_df["AUC"].idxmin()]

print("\n" + "="*60)
print("REPORT SUMMARY")
print("="*60)
print(f"  Best  : {best['Model']}")
print(f"    AUC : {best['AUC']}  {best['AUC_CI']}")
print(f"    Brier: {best['Brier']} | F1: {best['F1']}")
print(f"    Sens : {best['Sensitivity']} | Spec: {best['Specificity']}")
print(f"\n  Worst : {worst['Model']}")
print(f"    AUC : {worst['AUC']}  {worst['AUC_CI']}")
print(f"\n  Test  : {len(y_true):,} stays | {y_true.mean():.2%} readmission")
print("="*60)

print("\n  For your report:")
print(f"   p.15 → {OUTPUTS_DIR}/final_calibration_plot.png")
print(f"   p.17 → {OUTPUTS_DIR}/final_results_table.csv")
print(f"   p.19 → {OUTPUTS_DIR}/shap_summary_plot.png  (from phase4)")
print(f"\n   MLflow UI → {MLFLOW_URI}")

print("\n Phase 5 Complete — Full pipeline done!")
