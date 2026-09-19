# Applying OSFI Guideline E-23 to a Retail Credit PD Model: An End-to-End Enterprise Governance Case Study

**Author:** Ogbonnaya Nzie Ezichi 

---

### 📌 Project Executive Summary
This project treats data governance and model risk governance as the same discipline applied to different objects. Using a retail credit Probability-of-Default (PD) model as the subject, it translates OSFI Guideline E-23's requirements — model inventory, risk tiering, data lineage, governance RACI, ongoing monitoring into the actual artifacts a governance function would own: the same categories DAMA-DMBOK2 already asks data governance teams to maintain for any other data asset (a registry, a stewardship map, lineage documentation, a quality assessment, a change-control process). E-23 just names them for models specifically.

OSFI's revised Guideline E-23 (finalized 2025, effective May 1, 2027) folds AI/ML models into federal supervisory requirements for the first time, pushing Canadian financial institutions to formalize explainability, bias screening, and data-quality lineage, the same governance disciplines this project demonstrates, applied here to a model instead of a database or report. This repository contains the structured artifacts themselves — model inventory, risk-tiering rationale, validation report, data-cleaning log, monitoring plan, and RACI matrix — not a description of what that documentation should contain in the abstract.

### 🛠️ Framework Mapping: OSFI E-23 Artifacts Deliverables

| Deliverable Asset | OSFI E-23 Framework Alignment | Core Objective & Risk Control Covered |
| :--- | :--- | :--- |
| [📄 Model Inventory Entry](docs/01_model_inventory_entry.md) | **Model Inventory & Registry** | Establishing centralized model metadata, ownership, and tracking. |
| [📄 Risk Tiering Rationale](docs/02_risk_tiering_rationale.md) | **Model Risk Rating & Assessment** | Categorizing model materiality and assigning Tier-level governance rigor. |
| [📄 Model Validation Report](docs/03_validation_report.md) | **Independent Validation & Review** | Stress-testing, challenge of assumptions, and AI/ML explainability. |
| [📄 Monitoring & Drift Plan](docs/04_monitoring_drift_plan.md) | **Ongoing Model Performance Monitoring** | Outlining thresholds for data drift, concept drift, and model recalibration. |
| [📄 Governance RACI Matrix](docs/05_governance_raci.md) | **Governance, Policies & Internal Controls** | Enforcing strict Segregation of Duties (SoD) between 1st and 2nd Line. |

---

### 📊 Model Architecture & Quantitative Approach
* **The Core Data Asset:** 150,000 retail credit profiles sourced from the public Kaggle *'Give Me Some Credit'* dataset. The objective function is a binary classification predicting serious delinquency (90+ days past due or worse) within a 2-year window.
* **The Controlled Modeling Environment:** To isolate and audit model behavior accurately, two distinct model classes were trained using identical data pipelines and stratified splits (60/20/20 train/validation/test; `random_state=42`):
  1. **Traditional Scorecard Model (Logistic Regression):** Inherently transparent, stable, and easily auditable via static coefficients.
  2. **Advanced AI/ML Model (XGBoost):** Highly performant but presents a traditional "black-box" risk profile—the exact target of OSFI's 2025 E-23 modernization updates.

---

### 🛡️ Core Governance Wins & Quantitative Findings

#### 1. Data Quality Assurance & Auditability Log (BCBS 239 Alignment)
Before training any models, a rigorous data quality screening identified six significant structural defects across the 150,000 records. Rather than applying silent pre-processing scripts, every judgment call was formally logged:
* **Identified Anomalies:** 609 duplicate profiles, 1 invalid age value, 225 records with critical sentinel-value corruption in delinquency-count arrays, and severe outlier maxima (utilization ratios up to 50,708 and debt ratios up to 329,664).
* **Materiality Impact:** 19.5% of the total dataset contained missing income values. 
* **The Risk Control Rationale:** In an institutional setting, passing undocumented data-cleaning choices to an independent validator renders a model un-auditable. This project demonstrates strict documentation of data-cleansing metadata to ensure end-to-end data lineage transparency.

#### 2. Quantitative Performance Trade-offs
The independent validation pass evaluated the trade-off between the model types:

| Evaluation Metric (Test Set) | Traditional (Logistic) | Advanced (XGBoost) | Operational Delta |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | 0.853 | 0.862 | +0.009 |
| **Gini Coefficient** | 0.705 | 0.723 | +0.018 |
| **KS Statistic** | 0.555 | 0.575 | +0.020 |
| **Recall @ 0.5 Threshold** | 74.7% | 78.2% | +3.5 pp |

* **Operational Assessment:** XGBoost yields a clear performance lift, but in an enterprise setting, this marginal optimization must be balanced against the increased compliance overhead required to govern a non-linear model class.

#### 3. AI/ML Explainability & Bias Screening (SHAP Integration)
Under the modern provisions of OSFI E-23, both models were screened for latent demographic bias:
* **The Operational Finding:** Both models tripped a standard **four-fifths-rule screening** based on applicant age. The traditional logistic model flagged 48.0% of under-30 applicants as high-risk, compared to just 8.9% of applicants over 60.
* **The Risk Interpretation:** While this disparity reflects a genuine statistical risk gradient in the raw historical credit data (actual baseline default rates are 11.9% for under-30 vs 3.3% for 60+), a machine learning model cannot distinguish between objective risk and the systemic laundering of a legally protected attribute. My validation report purposefully identifies this as a **Fair Lending and Credit Policy risk call** rather than a data-science optimization problem, providing a clear escalation pathway to senior legal and risk stakeholders.

---

### 🗂️ Core Pipeline Execution & Reproducibility
All metrics, visualizations, and documentation tables generated in this repository are 100% reproducible through structured Python modules and analytical scripts:

```bash
# Ingest dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib

# Execute the auditable data lineage pipeline
python3 notebooks/02_clean_and_split.py
python3 notebooks/03_train_logistic.py
python3 notebooks/04_train_xgboost.py
python3 notebooks/05_shap_explainability.py
python3 notebooks/06_bias_fairness.py
```
*Note: The source dataset (`data/cs-training.csv`) is managed under standard data privacy simulation protocols and must be sourced directly from the Kaggle repository.*

---

### ⚖️ Portfolio Disclaimer
*Scope Note: This repository serves as an illustrative portfolio demonstration of structural framework mapping applied to a public dataset. It does not represent financial, regulatory, or legal compliance advice for any real-world banking institution or commercial enterprise.*
