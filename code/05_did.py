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
# *** This is a synthetic dataset so there is little to be learned.

# Takeaways
# 1. Identify a treatment group, a comparable control group, and observations before and after treatment.
# 2. Compare how outcomes change over time in the treated group relative to the control group, rather than comparing levels directly.
# 3. Estimate a Difference-in-Differences model by interacting the treatment and post indicators. The interaction coefficient is the DiD estimate.
# 4. With multiple time periods or staggered treatment adoption, the DiD framework naturally extends to Two-Way Fixed Effects (TWFE) models.

# Assumptions:
# 1. Parallel trends: in the absence of treatment, the treated and control groups would have followed the same trend over time.
# 2. No anticipation: units do not change their behaviour before treatment because they expect to be treated.
# 3. Stable treatment composition: the treated and control groups remain comparable over time (no differential selection or attrition driven by treatment).
# 4. SUTVA: one unit's treatment does not directly affect another unit's outcome (no spillover effects).
##################################################