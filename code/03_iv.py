##################################################
# Instrumental Variables
##################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

df = pd.read_csv("data/iv/card.csv")

iv = IV2SLS.from_formula(
    """
    lwage ~ 1
          + exper + expersq + black
          + smsa + south + smsa66
          + reg662 + reg663 + reg664
          + reg665 + reg666 + reg667
          + reg668 + reg669
          + [educ ~ nearc4]
    """,
    data=df,
)

results = iv.fit(cov_type="robust")

print(results.summary)

##################################################
# Findings
# We first find nearc4 has a statistically significant positive coefficient of 0.3256 when regressed on educ.
# Using nearc4 as an IV, we find a statistically significant positive coefficient for educ when regressed on lwage.

# Takeaways
# 1. In this case, with education (educ) as the regressor of interest, we are concerned that it is correlated with the error term (u) in the wage equation.
# 2. Variables not of-interest, such as black, may be assumed exogenous for the model even if likely not to be in real life. 

# Assumptions:
# 1. Instrument Relevance: Cov(nearc4, educ) ≠ 0
# 2. Instrument Exogeneity: Cov(nearc4, u) = 0
# 3. Exclusion Restriction: nearc4 affects lwage only through educ.
# 4. No Perfect Multicollinearity: nearc4 is not a linear combination of other regressors.
# 5. Random Sampling / Independent Observations
##################################################