##################################################
# Randomized Controlled Trials (RCT)
##################################################

import pandas as pd

df = pd.read_csv("data/rct/project_star.csv") # found up to ~45% missing values for particular columns

balance_columns = ["stark", "gender", "ethnicity", "birth", "lunchk", "schoolk", "degreek", "ladderk", "experiencek", "tethnicityk"] # created a balance table

means = (
    df.groupby("stark")[["readk", "mathk"]]
      .mean()
      .round(2)
) # we compute the average math and reading scores for different class sizes

print(means)

from scipy.stats import f_oneway

# Split into treatment groups
small = df[df["stark"] == "small"]
regular = df[df["stark"] == "regular"]
aide = df[df["stark"] == "regular+aide"]

# Reading
F_read, p_read = f_oneway(
    small["readk"].dropna(),
    regular["readk"].dropna(),
    aide["readk"].dropna()
)

# Math
F_math, p_math = f_oneway(
    small["mathk"].dropna(),
    regular["mathk"].dropna(),
    aide["mathk"].dropna()
)

print(f"Reading: F = {F_read:.3f}, p = {p_read:.4f}")
print(f"Math:    F = {F_math:.3f}, p = {p_math:.4f}")
