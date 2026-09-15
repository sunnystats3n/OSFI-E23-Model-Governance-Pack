"""
Data cleaning and train/val/test split.

Documented rationale for each cleaning decision (this doubles as the
"data lineage" input to the governance docs — E-23 requires model
documentation to record data provenance and transformation logic):

1. Exact duplicate rows (609) -> dropped. No legitimate reason for an
   identical record (all 11 fields, including target) to appear twice;
   most likely an extraction artifact.

2. age == 0 (1 record) -> dropped. Age cannot be zero for a credit
   applicant; treated as a data entry error, not a censored/missing code.
   Ages 100-109 (13 records) are implausible but not impossible -> kept,
   flagged for monitoring (see monitoring plan) rather than dropped,
   since arbitrarily deleting plausible-but-rare values would bias the
   elderly segment out of the training population.

3. NumberOfTime30-59DaysPastDueNotWorse / NumberOfTime60-89DaysPastDueNotWorse /
   NumberOfTimes90DaysLate contain sentinel values of 96 and 98 (264 and 5
   records respectively, identical counts across all three fields for the
   same records) that do not follow the otherwise-continuous distribution
   of these fields (which top out organically around 13-17). This is a
   well-documented data quality defect in this dataset: these look like a
   coded placeholder (e.g. "unknown"/"error") rather than a true count of
   98 late payments. Decision: recode 96/98 in these three fields to the
   column median AND add a binary flag `had_delinquency_sentinel` so the
   model can still use the fact that the record was flagged, without
   letting an implausible magnitude distort the fitted coefficients/splits.
   This is exactly the kind of undocumented data-quality assumption E-23's
   validation function is meant to catch if it is NOT documented -- so it
   is called out explicitly here rather than silently cleaned.

4. RevolvingUtilizationOfUnsecuredLines is defined as a utilization ratio
   (typically in [0, ~1.5]) but has a max of 50,708 (371 records > 2).
   Decision: winsorize (cap) at the 99.5th percentile, with a
   `revolving_util_capped` flag retained.

5. DebtRatio has a max of 329,664 and 28,877 records > 10. Many of these
   are legitimate artifacts of near-zero reported MonthlyIncome (ratio
   blows up when the denominator income is tiny), not data errors.
   Decision: winsorize at the 99.5th percentile (less aggressive judgment
   call than dropping), with a `debt_ratio_capped` flag retained, and
   document that this remains a monitored limitation.

6. MonthlyIncome missing (19.82%) -> median imputation BY age-decile
   (closer to true income than a single global median) plus a
   `monthly_income_missing` indicator flag, since "did not report income"
   may itself be predictive and is exactly the kind of signal an AI/ML
   model can pick up on but that needs to be surfaced explicitly for
   explainability/bias review (income non-disclosure can correlate with
   protected characteristics).

7. NumberOfDependents missing (2.62%) -> imputed with mode (0) plus a
   `dependents_missing` indicator flag.

Split: 60/20/20 train/validation/test, stratified on the target to
preserve the 6.68% event rate in each split, random_state fixed for
reproducibility (a validation requirement under E-23 -- results must be
reproducible by an independent reviewer).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

RANDOM_STATE = 42

df = pd.read_csv("data/cs-training.csv", index_col=0)

cleaning_log = {}

# 1. Drop exact duplicates
n0 = len(df)
df = df.drop_duplicates()
cleaning_log["duplicates_dropped"] = n0 - len(df)

# 2. Drop age == 0
n0 = len(df)
df = df[df["age"] > 0].copy()
cleaning_log["age_zero_dropped"] = n0 - len(df)
cleaning_log["age_over_100_kept"] = int((df["age"] > 100).sum())

# 3. Sentinel recoding for the three delinquency-count fields
sentinel_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
df["had_delinquency_sentinel"] = 0
for col in sentinel_cols:
    mask = df[col].isin([96, 98])
    df.loc[mask, "had_delinquency_sentinel"] = 1
cleaning_log["delinquency_sentinel_flagged"] = int(df["had_delinquency_sentinel"].sum())
for col in sentinel_cols:
    med = df.loc[~df[col].isin([96, 98]), col].median()
    df.loc[df[col].isin([96, 98]), col] = med

# 4. Winsorize RevolvingUtilizationOfUnsecuredLines at 99.5th pct
cap_revol = df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.995)
df["revolving_util_capped"] = (df["RevolvingUtilizationOfUnsecuredLines"] > cap_revol).astype(int)
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=cap_revol)
cleaning_log["revolving_util_cap_value"] = float(cap_revol)
cleaning_log["revolving_util_capped_n"] = int(df["revolving_util_capped"].sum())

# 5. Winsorize DebtRatio at 99.5th pct
cap_debt = df["DebtRatio"].quantile(0.995)
df["debt_ratio_capped"] = (df["DebtRatio"] > cap_debt).astype(int)
df["DebtRatio"] = df["DebtRatio"].clip(upper=cap_debt)
cleaning_log["debt_ratio_cap_value"] = float(cap_debt)
cleaning_log["debt_ratio_capped_n"] = int(df["debt_ratio_capped"].sum())

# 6. MonthlyIncome: missing flag + median-by-age-decile imputation
df["monthly_income_missing"] = df["MonthlyIncome"].isnull().astype(int)
df["age_decile"] = (df["age"] // 10) * 10
decile_medians = df.groupby("age_decile")["MonthlyIncome"].median()
df["MonthlyIncome"] = df.apply(
    lambda r: decile_medians[r["age_decile"]] if pd.isnull(r["MonthlyIncome"]) else r["MonthlyIncome"],
    axis=1,
)
df = df.drop(columns=["age_decile"])
cleaning_log["monthly_income_missing_n"] = int(df["monthly_income_missing"].sum())

# 7. NumberOfDependents: missing flag + mode imputation
df["dependents_missing"] = df["NumberOfDependents"].isnull().astype(int)
mode_dep = df["NumberOfDependents"].mode()[0]
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(mode_dep)
cleaning_log["dependents_missing_n"] = int(df["dependents_missing"].sum())

cleaning_log["final_n_rows"] = len(df)
cleaning_log["final_event_rate"] = float(df["SeriousDlqin2yrs"].mean())

# Age band for later bias/fairness analysis (not used as a model feature)
def age_band(a):
    if a < 30:
        return "under_30"
    elif a < 45:
        return "30_44"
    elif a < 60:
        return "45_59"
    else:
        return "60_plus"

df["age_band"] = df["age"].apply(age_band)

# Split: 60/20/20, stratified on target
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df["SeriousDlqin2yrs"], random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["SeriousDlqin2yrs"], random_state=RANDOM_STATE
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

cleaning_log["train_n"] = len(train_df)
cleaning_log["val_n"] = len(val_df)
cleaning_log["test_n"] = len(test_df)
cleaning_log["train_event_rate"] = float(train_df["SeriousDlqin2yrs"].mean())
cleaning_log["val_event_rate"] = float(val_df["SeriousDlqin2yrs"].mean())
cleaning_log["test_event_rate"] = float(test_df["SeriousDlqin2yrs"].mean())

with open("outputs/cleaning_log.json", "w") as f:
    json.dump(cleaning_log, f, indent=2)

print(json.dumps(cleaning_log, indent=2))
