##################################################
# Dummy Regression
##################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# df = pd.read_csv("data/regression/nifty50_summary_statistics.csv")
# df_reg = df[["Company_Name", "Sector", "First_Date", "Last_Date", "Total_Trading_Days", "Avg_Daily_Volume", "Avg_Daily_Return_%", "Volatility_%"]]
# df_reg.to_csv("data/regression/nifty50_dummy_dataset.csv", index=False)

# Load data
df = pd.read_csv("data/regression/nifty50_dummy_dataset.csv")

# Log-transform average daily volume
df["Log_Avg_Daily_Volume"] = np.log(df["Avg_Daily_Volume"])

# Regression
model = smf.ols(
    formula='Q("Volatility_%") ~ Log_Avg_Daily_Volume + Total_Trading_Days + C(Sector)',
    data=df
).fit()

print(model.summary())


##################################################
# Findings
# 

# Lessons
# 1. Difference in means is the same as the coefficient of a dummy variable in a regression model.
# 2. High correlation does not imply causation.
# 3. Omitted Variable Bias (OMV) arises from variables which affect both treatment and outcome but are not included.
# 4. Exogeneity requires that explanatory variables are uncorrelated with the error term.
# 5. Controls can be used to reduce OMV, but do not guarantee exogeneity.
# 6. Every coefficient comes with standard error, t-stat, p-val, and confidence interval.
##################################################