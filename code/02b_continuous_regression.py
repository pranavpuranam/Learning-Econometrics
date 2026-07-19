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
# Using hedonic pricing, we can decompose the price of a product into constituent features and estimate the marginal contribution of each feature to the price.

# Takeaways
# 1. Regression coefficients measure ceteris paribus (holding all other variables constant) effects.
# 2. Statistical significance does not imply economic significance. Always consider the magnitude of the coefficient.
# 3. Highly correlated regressors (multicollinearity) make it difficult to estimate individual effects.
# 4. Residual diagnostics help assess model assumptions.
# 5. The condition number is a quick diagnostic for multicollinearity.
# 6. Linear-Linear means 1-unit increase in X -> β-unit increase in Y, Log-Linear means 1-unit increase in X -> 100*β% increase in Y, and Log-Log means 1% increase in X -> β% increase in Y. 

# Assumptions:
# 1. Linearity in Parameters
# 2. Random Sampling / Independent Observations
# 3. No perfect multicollinearity
# 4. Exogeneity: E[u | X] = 0
# 5. Homoskedasticity
# 6. Normality of Errors
##################################################