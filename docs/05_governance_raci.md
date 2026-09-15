# Governance Roles (RACI) — illustrative

*Maps to OSFI Guideline E-23, "Governance & Organization" — enterprise-wide model risk management frameworks with clear roles (owners, developers, reviewers, approvers, users), adequate staffing, and board-level reporting.*

R = Responsible, A = Accountable, C = Consulted, I = Informed

| Activity | Model Owner (Retail Credit Risk) | Model Developer (Data Science) | Independent Validator | Compliance / Legal | Risk Committee | Board Risk Committee |
|---|---|---|---|---|---|---|
| Define business purpose & intended use | A | C | I | C | I | I |
| Model development & documentation | I | R/A | I | I | I | I |
| Data lineage & quality sign-off | C | R | A | I | I | I |
| Inherent risk tiering | C | C | R/A | C | I | I |
| Independent validation | I | C | R/A | I | I | I |
| Explainability methodology approval (AI/ML models) | I | C | R/A | C | I | I |
| Bias / fair-lending review | I | I | C | R/A | I | I |
| Go-live approval | A | I | C | C | R | I |
| Ongoing monitoring execution | R | C | I | I | I | I |
| Escalation on Red trigger | R | C | A | R (if bias-related) | I | I |
| Annual model risk report | C | I | R | C | A | I |
| Enterprise model risk appetite / policy | I | I | C | C | C | A |

## Why this matters for a solo portfolio project

Nobody builds this table alone — that's the point. A model risk management framework's value under E-23 comes precisely from *no single role* being able to approve their own work: the developer doesn't validate, the validator doesn't own the business case, and compliance signs off on fairness independent of whether the model "performs." Producing this RACI as part of a portfolio piece demonstrates understanding of *why* that separation exists, even though — as disclosed in `00_disclaimer.md` — one person did every role in this exercise. A real hiring conversation should surface that distinction directly rather than let the polished document imply otherwise.
