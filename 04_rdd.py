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
# A

# Takeaways
# 1. To begin RDD, start by identifying a running variable and a numerical cutoff. Visualize the data, and look for a jump at cutoff.
# 2. Next, fit local linear regressions on LHS and RHS. Do NOT use higher-order polynomials as they can produce perturbations.
# 3. Pick a bandwidth 

# Assumptions:
# 1. A
##################################################