##################################################
# Difference-in-Differences
##################################################

import os
import pandas as pd
import pyreadr

result = pyreadr.read_r("data/did/mpdta.rda")

df = result["mpdta"]
df["countyreal"] = df["countyreal"].astype(int)
df["first.treat"] = df["first.treat"].astype(int)
df["treat"] = df["treat"].astype(int)

import statsmodels.formula.api as smf

# Inspect available treatment cohorts
print(df["first.treat"].value_counts().sort_index())

# Choose the earliest treated cohort (excluding never-treated = 0)
cohort = int(df.loc[df["first.treat"] > 0, "first.treat"].min())
print(f"Using treatment cohort: {cohort}")

# Keep only the chosen cohort and never-treated counties
# Use one year before treatment and the treatment year
did = df[
    (
        (df["first.treat"] == cohort) |
        (df["first.treat"] == 0)
    ) &
    (df["year"].isin([cohort - 1, cohort]))
].copy()

# Create DiD indicators
did["treated"] = (did["first.treat"] == cohort).astype(int)
did["post"] = (did["year"] == cohort).astype(int)

# Check that all four cells exist
print("\n2x2 table:")
print(did.groupby(["treated", "post"]).size())

# Run the DiD regression
model = smf.ols(
    "lemp ~ treated + post + treated:post",
    data=did
).fit(cov_type="HC1")

print(model.summary())

##################################################
# Findings
# A

# Takeaways
# 1. A

# Assumptions:
# 1. A
##################################################