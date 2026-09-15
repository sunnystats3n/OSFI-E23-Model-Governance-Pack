"""
Model 1: Logistic Regression -- the "traditional" model in this governance
comparison. Chosen because it is the closest thing to an industry-standard
interpretable PD (probability of default) scorecard: coefficients map
directly to log-odds, no post-hoc explainability tooling is required to
justify a decision to a regulator or an adverse-action customer.
"""
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_val = scaler.transform(val[FEATURES])
X_test = scaler.transform(test[FEATURES])

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

def ks_statistic(y_true, y_prob):
    df_ = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    df_["cum_bad"] = (df_["y"] == 1).cumsum() / (df_["y"] == 1).sum()
    df_["cum_good"] = (df_["y"] == 0).cumsum() / (df_["y"] == 0).sum()
    return float(np.max(np.abs(df_["cum_bad"] - df_["cum_good"])))

results = {}
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

coef_table = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_[0],
    "abs_coefficient": np.abs(model.coef_[0]),
}).sort_values("abs_coefficient", ascending=False)

print(json.dumps(results, indent=2))
print("\nCoefficient ranking:")
print(coef_table.to_string(index=False))

with open("outputs/logistic_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
coef_table.to_csv("outputs/logistic_coefficients.csv", index=False)

with open("outputs/logistic_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": FEATURES}, f)
