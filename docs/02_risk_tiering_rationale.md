# Risk Tiering Rationale

*Maps to OSFI Guideline E-23, "Risk Rating & Assessment" — a documented approach assessing inherent model risk using quantitative factors (portfolio size, financial impact) and qualitative factors (business purpose, complexity, data reliability, customer impact).*

## Quantitative factors

| Factor | Assessment | Contribution to rating |
|---|---|---|
| Portfolio scale | 149,390 records post-cleaning; illustrative retail book | High — retail underwriting models typically touch a large applicant volume |
| Financial impact per decision | Underwriting decline/approval and pricing-adjacent decisions | High — direct customer and revenue impact per decision |
| Event (bad) rate | 6.70% serious-delinquency rate | Moderate class imbalance; affects both model calibration and validation design (see `03_validation_report.md`) |

## Qualitative factors

| Factor | Assessment | Contribution to rating |
|---|---|---|
| Business purpose | Underwriting / portfolio risk decisioning | High — directly gates access to credit |
| Customer impact | Adverse decisions (decline, higher pricing) affect individual applicants | High — triggers adverse-action/explainability expectations |
| Model complexity | Two variants: logistic regression (low complexity) vs. gradient-boosted trees, 500 rounds, depth 4 (moderate-high complexity, non-linear, requires post-hoc explainability tooling) | **Complexity itself differs materially between the two candidate models** — this is the crux of the risk-tiering decision below |
| Data reliability | See data-quality findings below — meaningful missingness and several data-quality defects were found and required documented judgment calls | Moderate-high — data reliability concerns raise inherent risk regardless of which model consumes the data |
| Third-party dependency | None (open-source tooling only, in-house feature engineering) | Lowers inherent risk relative to a vendor "black box" |

## Data reliability findings feeding this rating

From `notebooks/01_eda.py` and `notebooks/02_clean_and_split.py` (full detail and rationale for each decision is in the cleaning script's docstring):

- **609 exact duplicate rows** removed — an extraction-quality issue, not a modeling choice.
- **1 record with age = 0** removed (impossible value); **13 records with age > 100** kept but flagged for monitoring rather than dropped, since deleting rare-but-plausible elderly applicants would itself introduce a selection bias.
- **225 records** carried sentinel values (96 or 98) in the three delinquency-count fields — a well-known defect in this dataset where these codes almost certainly represent a placeholder/error rather than a true count. These were recoded to the column median with a `had_delinquency_sentinel` flag retained, rather than silently imputed with no trace, so a validator can find and challenge the decision.
- **747 records** had an implausibly large `RevolvingUtilizationOfUnsecuredLines` (up to 50,708 against a ratio that should rarely exceed ~1.5) and **747 records** had extreme `DebtRatio` values (up to 329,664) — both winsorized at the 99.5th percentile with capped-value flags retained.
- **MonthlyIncome missing for 29,221 records (19.5% of the cleaned set)** — imputed by age-decile median with a `monthly_income_missing` flag retained, since non-disclosure of income is itself a potentially informative (and potentially bias-relevant) signal.
- **NumberOfDependents missing for 3,828 records (2.6%)** — imputed with the mode, flag retained.

None of these are edge-case rounding issues — nearly 20% of the applicant population had a missing income figure, and the sentinel-value issue affects records that, if taken at face value, would look like extreme outliers with disproportionate influence on any model that doesn't handle them explicitly. **A validator who received this dataset without this documentation would have no way to know these judgment calls were made**, which is exactly the failure mode E-23's documentation requirement exists to prevent.

## Risk tier recommendation

| Model | Recommended inherent risk tier (illustrative 1–3 scale, 3 = highest) | Rationale |
|---|---|---|
| Logistic regression | **Tier 2** | High business/customer impact, but low model complexity, fully transparent coefficients, no black-box risk |
| XGBoost | **Tier 3** | Same business/customer impact as above, *plus* elevated complexity, dependency on post-hoc explainability tooling (SHAP) rather than native interpretability, and (per `06_bias_fairness.py`) a marginally larger disparate-impact signal across age bands than the logistic model at the same threshold |

**A Tier 3 rating is not a rejection of the AI/ML model** — it is a statement that its validation, explainability documentation, and ongoing monitoring obligations under E-23 must be proportionally more rigorous than the logistic model's, given the modest performance lift it delivers (see `03_validation_report.md` for the actual size of that lift). That trade-off — a few points of AUC/Gini against materially higher governance overhead — is exactly the decision E-23 is designed to force into the open rather than leave to a developer's default choice of algorithm.
