# Ongoing Monitoring & Drift Plan (illustrative)

*Maps to OSFI Guideline E-23, "Monitoring" — standards for ongoing performance monitoring with defined thresholds, contingency plans, and escalation procedures to detect performance degradation or drift.*

## Metrics tracked and thresholds

| Metric | Cadence | Green | Amber (investigate) | Red (escalate / consider suspension) |
|---|---|---|---|---|
| AUC-ROC (rolling 3-month cohort) | Monthly | ≥ 0.84 | 0.80 – 0.84 | < 0.80 |
| Population Stability Index (PSI) on score distribution | Monthly | < 0.10 | 0.10 – 0.25 | ≥ 0.25 |
| Characteristic Stability Index (CSI), each input feature | Quarterly | < 0.10 | 0.10 – 0.25 | ≥ 0.25 |
| Observed vs. predicted event rate (calibration) | Quarterly | within ±10% relative | ±10–25% | > ±25% |
| Missingness rate, `MonthlyIncome` and `NumberOfDependents` | Monthly | within ±5 pp of training baseline (19.5% / 2.6%) | ±5–10 pp | > ±10 pp |
| Rate of `had_delinquency_sentinel` flag firing | Monthly | within ±2 pp of training baseline (0.25%) | 2–5 pp | > 5 pp — likely signals an upstream data feed change, not a real behavioral shift |
| Age-band disparate impact ratio (four-fifths screen) | Quarterly | ≥ 0.80 all bands | 0.65 – 0.80 | < 0.65 |

Baselines are taken from this exercise's training/test statistics (`outputs/cleaning_log.json`, `outputs/logistic_metrics.json`, `outputs/xgboost_metrics.json`, `outputs/bias_screening_summary.json`) — in a real deployment these would be locked at go-live, not retroactively adjusted.

## Why these specific triggers

- **PSI/CSI** are the standard credit-risk-industry drift metrics because AUC alone can look stable for months while the underlying population shifts in ways that only show up later (a classic model-risk blind spot).
- The **sentinel-flag rate** trigger is specific to this dataset's known data-quality defect (see `02_risk_tiering_rationale.md`) — a sudden change in how often that flag fires is far more likely to mean an upstream data pipeline or codebook change than a genuine shift in borrower behavior, so it is monitored as a **data-quality** trigger, not a **model-performance** trigger, and should route to the data owner first, not the model owner.
- The **disparate-impact ratio** is monitored on an ongoing basis, not just at initial validation, because a model that clears a fairness screen on today's population can drift into a flagged state as the applicant mix changes — this is explicitly called out in E-23's AI/ML section as an ongoing, not one-time, obligation.

## Escalation path (illustrative)

1. **Amber** on any metric → model owner investigates within 10 business days, documents root cause, decides whether a Red threshold is imminent.
2. **Red** on a performance or calibration metric → model owner notifies independent validation function within 2 business days; use of the model for new decisions is paused pending a re-validation decision by the risk committee (illustrative).
3. **Red** on the disparate-impact ratio → escalates directly to compliance/legal in parallel with model owner notification, regardless of performance metrics, since this is a legal-exposure trigger, not only a model-quality one.
4. Any Red condition triggers an out-of-cycle entry in the model inventory (`01_model_inventory_entry.md`) recording the trigger, root cause, and remediation decision — inventory entries are living records, not one-time filings, under E-23.

## Retraining/redevelopment triggers

- Two consecutive Amber readings on the same metric.
- Any single Red reading.
- A scheduled annual review, regardless of metric status (proportional to the Tier 2/3 ratings assigned in `02_risk_tiering_rationale.md`).
