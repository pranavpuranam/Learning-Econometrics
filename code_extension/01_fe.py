##################################################
# Fixed Effects
##################################################

import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("data_extension/fe/grunfeld.csv")

df = df.drop(columns=["rownames"]) # a regular pooled OLS ignores the panel structure of the data

twoway_fe = smf.ols( # firm fixed effects (FE) models control for time-invariant differences across firms
    formula="inv ~ value + capital + C(firm) + C(year)", # fixed effects for firm and for year *** Note to use C(year) rather than year to avoid estimating into the future with a fixed amount
    data=df
).fit(
    cov_type = "cluster",
    cov_kwds = {"groups": df["firm"]} # correct the inference for within-firm correlation over time, clustered standard errors (SEs)
)

# Print results
print(twoway_fe.summary())

##################################################
# Findings
# After controlling for firm and year fixed effects, just current capital and firm value explain ~0.952 of the variation in firm capital investment.
# Adding firm and year fixed effects substantially improves model fit relative to pooled OLS (R²: 0.812 → 0.952).

# Takeaways
# 1. It's almost boilerplate in empirical finance and applied micro to see "all regressions include firm and year fixed effects. standard errors are clustered at the firm level."
# 2. Fixed effects estimate relationships using within-entity variation, removing bias from time-invariant characteristics.
# 3. Year fixed effects absorb common macroeconomic shocks, preventing them from being attributed to the explanatory variables.

# Assumptions
# 1. Regressors are exogenous conditional on the fixed effects.
# 2. There are no important omitted variables that vary within firms over time.
# 3. There is sufficient within-firm variation in the explanatory variables to identify the coefficients.
##################################################