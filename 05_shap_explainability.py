"""
SHAP explainability analysis -- directly addresses the E-23 AI/ML-specific
requirement that institutions be able to explain model outputs and
detect "black box" opacity risk. We compute:
  1. Global feature importance (mean |SHAP value|) for the XGBoost model.
  2. A local explanation for a specific declined applicant, the kind of
     record-level explanation an adverse-action/reasons-for-denial
     process would need to produce.
  3. A side-by-side note on what the logistic model gives "for free"
     (coefficients = global AND local explanation, no extra tooling)
     versus what XGBoost requires (a whole additional explainability
     pipeline) -- this asymmetry is the actual governance cost of
     choosing the AI/ML model, and belongs in the risk-tiering doc.
"""
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/claude/osfi-e23-project/outputs/xgboost_model.pkl", "rb") as f:
    xgb_bundle = pickle.load(f)
model = xgb_bundle["model"]
FEATURES = xgb_bundle["features"]

test = pd.read_csv("/home/claude/osfi-e23-project/data/test.csv")
X_test = test[FEATURES]

# Use a manageable sample for SHAP computation speed
sample = X_test.sample(n=3000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

# 1. Global importance
mean_abs_shap = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
mean_abs_shap.to_csv("/home/claude/osfi-e23-project/outputs/shap_global_importance.csv", index=False)
print("Global SHAP importance:")
print(mean_abs_shap.to_string(index=False))

plt.figure()
shap.summary_plot(shap_values, sample, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("/home/claude/osfi-e23-project/outputs/shap_global_importance.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("/home/claude/osfi-e23-project/outputs/shap_summary_beeswarm.png", dpi=150)
plt.close()

# 2. Local explanation for one high-predicted-risk applicant
probs = model.predict_proba(X_test)[:, 1]
test = test.reset_index(drop=True)
test["pred_prob"] = probs
high_risk_idx = test["pred_prob"].idxmax()
high_risk_record = X_test.loc[[high_risk_idx]]
local_shap = explainer(high_risk_record)

local_explanation = pd.DataFrame({
    "feature": FEATURES,
    "value": high_risk_record.iloc[0].values,
    "shap_contribution": local_shap.values[0],
}).sort_values("shap_contribution", key=np.abs, ascending=False)

print(f"\nLocal explanation for record index {high_risk_idx} "
      f"(predicted PD = {test.loc[high_risk_idx, 'pred_prob']:.3f}):")
print(local_explanation.to_string(index=False))
local_explanation.to_csv("/home/claude/osfi-e23-project/outputs/shap_local_example.csv", index=False)

plt.figure()
shap.plots.waterfall(local_shap[0], show=False)
plt.tight_layout()
plt.savefig("/home/claude/osfi-e23-project/outputs/shap_local_waterfall.png", dpi=150)
plt.close()

print("\nSaved: shap_global_importance.png, shap_summary_beeswarm.png, shap_local_waterfall.png")
