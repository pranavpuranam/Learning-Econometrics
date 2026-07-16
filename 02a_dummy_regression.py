##################################################
# Dummy Regression
##################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/regression/penguins.csv")

model = smf.ols("body_mass_g ~ C(sex) + C(species)", data = df).fit()

print(model.summary())

##################################################
# Findings
# Male penguins weigh, on average, 668 grams more than female penguins, holding species constant. 

# Takeaways
# 1. A regression with a single dummy variable is equivalent to a difference in group means.
# 2. The intercept is the mean of the reference group.
# 3. The dummy coefficient is the difference in means relative to the reference group.
# 4. Correlation does not imply causation.
# 5. Omitted Variable Bias (OMV) arises when an omitted variable affects both the regressor and the outcome.
# 6. Exogeneity requires E[u | X] = 0; equivalently, the regressors are uncorrelated with the error term.
# 7. Adding relavant control variables can reduce OMV, but cannot eliminate bias from unobserved confounders.
# 8. Every estimated coefficient has uncertainty, summarised by its standard error, t-statistic, p-value, and confidence interval.     

# Assumptions:
# 1. Linearity in Parameters
# 2. Random Sampling / Independent Observations
# 3. No perfect multicollinearity
# 4. Exogeneity: E[u | X] = 0
# 5. Homoskedasticity
# 6. Normality of Errors
##################################################