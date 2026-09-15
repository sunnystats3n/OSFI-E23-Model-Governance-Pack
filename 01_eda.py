import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("/home/claude/osfi-e23-project/data/cs-training.csv", index_col=0)

print("=== SHAPE ===")
print(df.shape)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== TARGET BALANCE (SeriousDlqin2yrs) ===")
print(df["SeriousDlqin2yrs"].value_counts())
print(df["SeriousDlqin2yrs"].value_counts(normalize=True).round(4))

print("\n=== MISSINGNESS ===")
miss = df.isnull().sum()
miss_pct = (df.isnull().mean() * 100).round(2)
print(pd.DataFrame({"n_missing": miss, "pct_missing": miss_pct}))

print("\n=== DESCRIBE (numeric) ===")
print(df.describe().T)

print("\n=== SUSPICIOUS VALUE CHECKS ===")
print("age == 0 count:", (df["age"] == 0).sum())
print("age < 18 count:", (df["age"] < 18).sum())
print("age > 100 count:", (df["age"] > 100).sum())

for col in ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]:
    print(f"\n{col} value counts (top 10):")
    print(df[col].value_counts().sort_index(ascending=False).head(10))

print("\nRevolvingUtilizationOfUnsecuredLines > 2 count (should be a ratio, often <=1):", (df["RevolvingUtilizationOfUnsecuredLines"] > 2).sum())
print("RevolvingUtilizationOfUnsecuredLines max:", df["RevolvingUtilizationOfUnsecuredLines"].max())

print("\nDebtRatio > 10 count:", (df["DebtRatio"] > 10).sum())
print("DebtRatio max:", df["DebtRatio"].max())

print("\n=== DUPLICATES ===")
print("Exact duplicate rows (excluding index):", df.duplicated().sum())

print("\n=== CORRELATION WITH TARGET ===")
print(df.corr(numeric_only=True)["SeriousDlqin2yrs"].sort_values(ascending=False))
