# Model Inventory Entry

*Maps to OSFI Guideline E-23 (2027), "Model Inventory" — institutions must maintain a comprehensive, continuously updated record of all models carrying non-negligible risk, including identification data, risk ratings, ownership, and key metadata.*

> **Note on scope.** This inventory entry is written as if for a real institution, as a demonstration of the documentation E-23 expects. The underlying data is a public dataset (Kaggle "Give Me Some Credit"), not a real institution's book of business, and no real credit decisions are made by this model. See `docs/00_disclaimer.md`.

| Field | Value |
|---|---|
| Model ID | RETAIL-PD-001 (illustrative) |
| Model name | Retail Unsecured Credit — 2-Year Serious Delinquency PD Model |
| Model owner | Retail Credit Risk (illustrative business line) |
| Model developer | Data Science / Model Development (illustrative) |
| Independent validator | Model Validation function (illustrative — see `05_governance_raci.md`) |
| Business purpose | Estimate the probability that a retail credit applicant/borrower experiences a serious delinquency (90+ days past due, bankruptcy, or worse) within a 2-year horizon, to support underwriting and portfolio risk monitoring decisions |
| Model type | Two competing candidate models: (1) Logistic regression scorecard, (2) Gradient-boosted trees (XGBoost) |
| Model category under E-23 | Falls within E-23's definition of "model" (Section: Definition) — statistical/AI-ML technique processing input data to produce an output; the XGBoost variant is explicitly in scope of the AI/ML-specific expectations |
| Input data | 10 raw applicant/bureau-style attributes (utilization, age, delinquency counts, debt ratio, income, credit lines, real estate loans, dependents) plus 5 derived data-quality flags — see `02_risk_tiering_rationale.md` for full lineage |
| Output | Probability of serious delinquency in 24 months (PD score, 0–1) |
| Decision use | Illustrative use: underwriting decline/refer threshold and portfolio risk-tiering; **not connected to any live decision system** |
| Materiality / non-negligible risk? | Yes (illustrative) — a retail underwriting PD model that gates credit decisions would typically be inherently material under E-23's proportionality principle, regardless of institution size |
| Development date | This exercise, 2026 |
| Last validated | This exercise, 2026 (initial validation only — see `03_validation_report.md`) |
| Next scheduled review | Illustrative: 12 months, or earlier if a monitoring trigger fires (`04_monitoring_drift_plan.md`) |
| Third-party / vendor component? | No — open-source libraries only (scikit-learn, XGBoost); no vendor-supplied model or vendor data |
| Status | Illustrative / not deployed |

## Why two models are inventoried together

E-23 does not require a single model per use case — it requires that *every* model carrying non-negligible risk be inventoried, including ones under evaluation. Carrying both the logistic and XGBoost variants as linked inventory entries mirrors the real decision an institution faces before deployment: which model, if either, clears validation and risk-tiering well enough to justify production use. That comparison is the subject of `03_validation_report.md`.
