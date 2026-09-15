"""
Bias / fairness assessment by age band.

Why age: this dataset carries no gender, ethnicity, or other demographic
field, so age is the only available proxy for a ground protected under
Canadian human-rights legislation (age discrimination in the provision of
credit is restricted, e.g., under provincial human rights codes and
federally). age is ALSO one of the strongest raw predictors in both
models here (see EDA / SHAP output) -- which is exactly the scenario
E-23's AI/ML-specific bias requirement is aimed at: a feature can be
statistically predictive and legally/ethically fraught at the same time,
and a governance framework has to make that tension visible rather than
resolve it silently inside the model.

This script does NOT attempt to "fix" the model. It reports where the two
models' error rates and selection rates diverge across age bands, at the
same 0.5 decision threshold used in the validation report, and flags
which divergences would need legal/compliance review before any real
deployment. It is deliberately conservative in its claims: with no
ground-truth demographic labels beyond age, and no legal opinion attached,
this is a screening-level check, not a compliance determination.
"""
import pandas as pd
import numpy as np
import pickle
import json

FEATURES_LOG = None
with open("outputs/logistic_model.pkl", "rb") as f:
    log_bundle = pickle.load(f)
with open("outputs/xgboost_model.pkl", "rb") as f:
    xgb_bundle = pickle.load(f)

test = pd.read_csv("data/test.csv")
FEATURES = xgb_bundle["features"]

X_test_xgb = test[FEATURES]
X_test_log = log_bundle["scaler"].transform(test[FEATURES])

test["prob_logistic"] = log_bundle["model"].predict_proba(X_test_log)[:, 1]
test["prob_xgboost"] = xgb_bundle["model"].predict_proba(X_test_xgb)[:, 1]

THRESH = 0.5
test["pred_logistic"] = (test["prob_logistic"] >= THRESH).astype(int)
test["pred_xgboost"] = (test["prob_xgboost"] >= THRESH).astype(int)

def group_metrics(df, pred_col, group_col="age_band"):
    rows = []
    for band, g in df.groupby(group_col):
        y = g["SeriousDlqin2yrs"]
        p = g[pred_col]
        n = len(g)
        selection_rate = p.mean()  # % flagged as high risk (declined)
        actual_event_rate = y.mean()
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        tn = ((p == 0) & (y == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        rows.append({
            "age_band": band, "n": n, "actual_event_rate": round(actual_event_rate, 4),
            "selection_rate_declined": round(selection_rate, 4),
            "false_positive_rate": round(fpr, 4), "false_negative_rate": round(fnr, 4),
            "true_positive_rate": round(tpr, 4),
        })
    out = pd.DataFrame(rows).set_index("age_band")
    ref_rate = out.loc["45_59", "selection_rate_declined"] if "45_59" in out.index else out["selection_rate_declined"].max()
    out["disparate_impact_ratio_vs_45_59"] = (out["selection_rate_declined"] / ref_rate).round(3)
    return out

print("=== LOGISTIC REGRESSION: outcomes by age band ===")
log_metrics = group_metrics(test, "pred_logistic")
print(log_metrics.to_string())

print("\n=== XGBOOST: outcomes by age band ===")
xgb_metrics = group_metrics(test, "pred_xgboost")
print(xgb_metrics.to_string())

log_metrics.to_csv("outputs/bias_logistic_by_age_band.csv")
xgb_metrics.to_csv("outputs/bias_xgboost_by_age_band.csv")

# Four-fifths rule screen (a common, if imperfect, adverse-impact yardstick;
# noted here as a screening heuristic only, not a legal standard under
# Canadian human rights law)
summary = {
    "logistic_min_disparate_impact_ratio": float(log_metrics["disparate_impact_ratio_vs_45_59"].min()),
    "xgboost_min_disparate_impact_ratio": float(xgb_metrics["disparate_impact_ratio_vs_45_59"].min()),
    "four_fifths_rule_flag_logistic": bool(log_metrics["disparate_impact_ratio_vs_45_59"].min() < 0.8),
    "four_fifths_rule_flag_xgboost": bool(xgb_metrics["disparate_impact_ratio_vs_45_59"].min() < 0.8),
}
print("\n=== SCREENING SUMMARY ===")
print(json.dumps(summary, indent=2))
with open("outputs/bias_screening_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
