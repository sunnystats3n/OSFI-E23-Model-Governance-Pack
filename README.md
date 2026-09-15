# Applying OSFI Guideline E-23 to a Retail Credit PD Model: A Governance Case Study

**Ogbonnaya Nzie Ezichi** — [LinkedIn](https://www.linkedin.com/in/ogbonnaya-nzie-ezichi-msc-065687a6/) · [GitHub](https://github.com/sunnystats3n) · [Google Scholar](https://scholar.google.ca/citations?user=80PLXMkAAAAJ&hl=en)

> **Scope note (read first):** this is an illustrative application of [OSFI Guideline E-23](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/guideline-e-23-model-risk-management-2027) to a public dataset, built as a portfolio demonstration — not a claim of regulatory compliance for any real institution. Full disclaimer: [`docs/00_disclaimer.md`](docs/00_disclaimer.md).

## Why this project

OSFI's revised Guideline E-23 (final 2025, effective May 1, 2027) is the first Canadian federal supervisory guideline to explicitly fold AI/ML models into model risk management requirements — explainability, bias, and "black box" transparency are now named requirements, not implied ones. Federally regulated banks and insurers are actively building out compliance programs ahead of the 2027 deadline. Rather than write about E-23 in the abstract, this project builds the artifact a model risk management function would actually have to produce: a real model, on real (if public) data, governed end-to-end against the guideline's own structure.

## What's here

| Document | E-23 section it demonstrates |
|---|---|
| [`docs/01_model_inventory_entry.md`](docs/01_model_inventory_entry.md) | Model Inventory |
| [`docs/02_risk_tiering_rationale.md`](docs/02_risk_tiering_rationale.md) | Risk Rating & Assessment |
| [`docs/03_validation_report.md`](docs/03_validation_report.md) | Validation & Review + AI/ML explainability |
| [`docs/04_monitoring_drift_plan.md`](docs/04_monitoring_drift_plan.md) | Ongoing Monitoring |
| [`docs/05_governance_raci.md`](docs/05_governance_raci.md) | Governance & Organization |
| `notebooks/01_eda.py` → `06_bias_fairness.py` | The underlying analysis each document cites |
| `outputs/` | Every metric, chart, and CSV referenced in the docs above — nothing in the write-up is asserted without a file backing it |

## The setup

**Data:** Kaggle ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit/data), 150,000 retail credit records, target = serious delinquency (90+ days past due or worse) within 2 years.

**Two competing models, same features, same split** (60/20/20 train/val/test, stratified, `random_state=42`):
- **Logistic regression** — the "traditional" model: fully transparent, coefficients are the explanation.
- **XGBoost** — the "AI/ML" model E-23's 2025 update specifically targets: stronger performance, but opaque without additional tooling.

Building both on identical data and splits was a deliberate governance choice, not just a modeling one — it's the only way to attribute any behavioral difference between them to the model class itself rather than to inconsistent data handling.

## What the data quality pass found (and why it's the most governance-relevant part of the project)

Before any model was touched, this dataset needed six documented judgment calls: 609 duplicate rows, one impossible age value, 225 records carrying a known sentinel-value defect in the delinquency-count fields, two fields with implausible outlier maxima (utilization ratios up to 50,708; debt ratios up to 329,664), and — most materially — **19.5% of records missing income entirely**. Full rationale for each decision: [`docs/02_risk_tiering_rationale.md`](docs/02_risk_tiering_rationale.md).

The reason this matters more than any model metric: E-23's validation function exists largely to catch exactly this kind of undocumented judgment call. A model handed to a validator with these six decisions made silently — which is how most portfolio projects treat data cleaning — would be un-auditable by design, regardless of how well the resulting model performs.

## What the models found

| Metric (test set) | Logistic | XGBoost | Delta |
|---|---|---|---|
| AUC-ROC | 0.853 | 0.862 | +0.009 |
| Gini | 0.705 | 0.723 | +0.018 |
| KS statistic | 0.555 | 0.575 | +0.020 |
| Recall @ 0.5 | 74.7% | 78.2% | +3.5 pp |

XGBoost's lift is real but modest. Full validation writeup, including SHAP-based explainability and why "technically explainable" isn't the same as "as cheap to govern as logistic regression": [`docs/03_validation_report.md`](docs/03_validation_report.md).

## The finding that actually matters for E-23's AI/ML provisions

Both models' decline rates fall sharply with applicant age — logistic regression flags 48.0% of under-30 applicants as high-risk versus 8.9% of applicants 60+. That trips a four-fifths-rule screen for **both** models. Part of that gap tracks a genuine underlying risk gradient in the data (actual default rates: 11.9% under-30 vs. 3.3% for 60+) — but a screening tool doesn't know the difference between "the model reflects real risk" and "the model launders a legally protected characteristic into a decision," and neither does this write-up pretend to resolve that on its own. That's a fair-lending/legal call, not a data-science one — full detail in [`docs/03_validation_report.md`](docs/03_validation_report.md) and the raw numbers in `outputs/bias_*`.

## What I'd tell a hiring manager about this project

This isn't a claim that I can single-handedly run model risk management for a federally regulated institution — the [`docs/05_governance_raci.md`](docs/05_governance_raci.md) table exists specifically to show I understand that no real MRM framework lets one person fill every role in it. What this demonstrates is that I can read a supervisory guideline, translate its structure into the actual documents an institution would need to produce, and back every claim in those documents with a reproducible analysis rather than boilerplate. That's the skill the DAMA/CDMP study track and this project were both built to prove out.

## Reproducing this

```
pip install pandas numpy scikit-learn xgboost shap matplotlib
python3 notebooks/02_clean_and_split.py
python3 notebooks/03_train_logistic.py
python3 notebooks/04_train_xgboost.py
python3 notebooks/05_shap_explainability.py
python3 notebooks/06_bias_fairness.py
```

Source dataset (`data/cs-training.csv`) is the Kaggle "Give Me Some Credit" training file — not redistributed here; download from Kaggle directly.
