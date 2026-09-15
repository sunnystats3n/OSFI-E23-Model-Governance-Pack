# Independent Validation Report (illustrative)

*Maps to OSFI Guideline E-23, "Validation & Review" — independent model reviews examining conceptual soundness, appropriateness, limitations, and performance, with frequency/intensity proportional to risk rating.*

Reviewed models: RETAIL-PD-001 (logistic regression variant) and RETAIL-PD-001 (XGBoost variant). Reproducibility: fixed `random_state=42` throughout; full pipeline in `notebooks/02_clean_and_split.py` → `03_train_logistic.py` / `04_train_xgboost.py`.

## 1. Conceptual soundness

- **Logistic regression.** Coefficients are directionally sensible and stable across train/val/test: higher revolving utilization, more 30–59/60–89/90+ day delinquencies, and the delinquency-sentinel flag all increase predicted risk; higher age, higher income, and higher debt ratio (after winsorization) decrease it. No sign flips between splits.
- **XGBoost.** Gain-based feature importance is directionally consistent with the logistic coefficients (utilization and delinquency counts dominate in both), which is a basic soundness check: two structurally different models agreeing on what drives risk is more reassuring than either model's standalone metrics.
- **Shared limitation.** Both models were trained on a single point-in-time snapshot with no explicit macroeconomic or vintage variable. Neither model has been through-the-cycle tested. This is flagged as an out-of-scope limitation for this exercise, not resolved.

## 2. Performance (test set, n = 29,878, event rate 6.70%)

| Metric | Logistic regression | XGBoost | Delta |
|---|---|---|---|
| AUC-ROC | 0.8527 | 0.8616 | +0.0089 |
| Gini coefficient | 0.7054 | 0.7232 | +0.0178 |
| KS statistic | 0.5554 | 0.5750 | +0.0196 |
| Brier score (lower is better) | 0.1468 | 0.1435 | −0.0033 |
| Precision @ 0.5 threshold | 21.5% | 21.0% | −0.5 pp |
| Recall @ 0.5 threshold | 74.7% | 78.2% | +3.5 pp |

Train/validation/test AUC were within 1.3 points of each other for both models (see `outputs/logistic_metrics.json`, `outputs/xgboost_metrics.json`), so neither shows material overfitting.

**Validator's read:** XGBoost's lift is real but modest — about 1 Gini point and 2 points of KS, and roughly 3.5 additional percentage points of recall (defaulters correctly flagged) at the cost of slightly lower precision. Whether that lift is "worth" a Tier 3 governance burden instead of Tier 2 (per `02_risk_tiering_rationale.md`) is a business decision this report surfaces but does not make — that call belongs to the model owner and risk committee, informed by this data, not to the validator.

## 3. Explainability (AI/ML-specific, per E-23)

SHAP analysis (`notebooks/05_shap_explainability.py`) confirms the XGBoost model is explainable at both the global level (`outputs/shap_global_importance.png`) and the individual-decision level (`outputs/shap_local_waterfall.png` — a single high-risk applicant's prediction is decomposed feature-by-feature). This satisfies the *technical* capability to produce a reasons-for-decision explanation.

It does **not** eliminate the governance cost difference from the logistic model: the logistic model's coefficients are the explanation, with zero additional tooling, computation, or approval of an explainability methodology required. The XGBoost model's explanation is a derived artifact of a separate library (SHAP) that itself has assumptions (background sample, Tree SHAP approximation) that would need their own validation sign-off in a real deployment. This asymmetry — not whether XGBoost *can* be explained, but what it costs to keep it explainable — is the point E-23 is making by singling out AI/ML models for extra scrutiny.

## 4. Bias / fairness screening

See `06_bias_fairness.py` output and `docs/04_monitoring_drift_plan.md`. Headline finding: both models' decline ("selection") rates fall off sharply with age (e.g., logistic: 48.0% of under-30 applicants flagged high-risk vs. 8.9% of 60+ applicants), which trips a four-fifths-rule screen for both models (min ratio 0.387 logistic, 0.358 XGBoost) against the 45–59 reference band. Actual event rates also fall with age in this data (11.9% under-30 vs. 3.3% for 60+), so part of the disparity reflects a genuine risk gradient rather than pure model artifact — but the screen doesn't distinguish those, and neither should this report pretend to on its own. **This requires a fair-lending/legal review before any real deployment decision; it is not resolved here.**

## 5. Overall recommendation (illustrative)

- Logistic regression: **passes validation for the stated use case**, with the age-disparity finding carried forward as a required legal/compliance follow-up rather than a validation blocker (since the model class itself is not the source of the disparity — the underlying risk gradient is).
- XGBoost: **conditional pass** — performance and explainability are adequate, but given the Tier 3 rating, deployment should not proceed without (a) a documented business case for why the incremental ~1-2 Gini points justifies the added complexity, and (b) the same fair-lending review as the logistic model, given the comparable-or-larger disparate-impact signal.
