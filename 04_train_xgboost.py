"""
Model 2: XGBoost -- the "AI/ML" model in this governance comparison.
Same feature set and same train/val/test split as the logistic baseline,
so any performance/behavioral differences are attributable to the model
class, not to different data handling (this comparability-by-design is
itself a validation principle worth stating explicitly in the docs).
"""
import pandas as pd
import numpy as np
import json
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "had_delinquency_sentinel",
    "revolving_util_capped",
    "debt_ratio_capped",
    "monthly_income_missing",
    "dependents_missing",
]
TARGET = "SeriousDlqin2yrs"

train = pd.read_csv("/home/claude/osfi-e23-project/data/train.csv")
val = pd.read_csv("/home/claude/osfi-e23-project/data/val.csv")
test = pd.read_csv("/home/claude/osfi-e23-project/data/test.csv")

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

def ks_statistic(y_true, y_prob):
    df_ = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    df_["cum_bad"] = (df_["y"] == 1).cumsum() / (df_["y"] == 1).sum()
    df_["cum_good"] = (df_["y"] == 0).cumsum() / (df_["y"] == 0).sum()
    return float(np.max(np.abs(df_["cum_bad"] - df_["cum_good"])))

results = {"best_iteration": int(model.best_iteration)}
for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    results[name] = {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y.values, prob),
        "brier": float(brier_score_loss(y, prob)),
    }

pred_test = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred_test)
results["test_confusion_matrix_at_0.5"] = cm.tolist()

importance = pd.DataFrame({
    "feature": FEATURES,
    "gain_importance": model.feature_importances_,
}).sort_values("gain_importance", ascending=False)

print(json.dumps(results, indent=2))
print("\nFeature importance (gain):")
print(importance.to_string(index=False))

with open("/home/claude/osfi-e23-project/outputs/xgboost_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
importance.to_csv("/home/claude/osfi-e23-project/outputs/xgboost_importance.csv", index=False)

with open("/home/claude/osfi-e23-project/outputs/xgboost_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)
