##################################################
# Regression Discontinuity Design
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdrobust import rdrobust

df = pd.read_stata("data/rdd/lee.dta")
df = df[df["use"] == 1].copy()

# Outcome
y = df["myoutcomenext"].values

# Running variable
x = df["difshare"].values

# Estimate sharp RDD
results = rdrobust(y=y, x=x, c=0)

print(results)

##################################################
# Findings
# Winning a very close election increases the probability of winning the next election by approximately 44 percentage points.

# Takeaways
# 1. To begin RDD, start by identifying a running variable and a numerical cutoff. Visualize the data, and look for a jump at cutoff.
# 2. Next, fit local linear regressions on LHS and RHS. Do NOT use higher-order polynomials as they can produce perturbations.
# 3. Pick a bandwidth, or use rdrobust to automatically decide these bandwidths and test for hyperparameter stability.
# 4. The treatment effect is the vertical difference between the fitted regressions at the cutoff.

# Assumptions:
# 1. No precise manipulation of the running variable around the cutoff (candidates cannot deliberately place themselves on one side).
# 2. Potential outcomes are continuous at the cutoff in the absence of treatment (without treatment we would see no jump).
# 3. Units just above and below the cutoff are otherwise comparable (directly above and below are essentially identical except for receiving treatment).
# 4. SUTVA, one unit's treatment does not affect another's outcome (one candidate becoming an incumbent doesn't directly change another result).
##################################################