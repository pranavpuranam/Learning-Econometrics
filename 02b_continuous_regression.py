##################################################
# Continuous Regression
##################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/regression/cars.csv")

model = smf.ols("price ~ C(Make) + tax + mpg", data = df).fit()

# full model: price ~ C(model) + year + C(transmission) + mileage + C(fuelType) + tax + mpg + engineSize + C(Make)

print(model.summary())

##################################################
# Findings
# 

# Takeaways
# 1. 
# 2. 
# 3. 

# Assumptions:
# 1. Linearity in Parameters
# 2. Random Sampling / Independent Observations
# 3. No perfect multicollinearity
# 4. Exogeneity: E[u | X] = 0
# 5. Homoskedasticity
# 6. Normality of Errors
##################################################