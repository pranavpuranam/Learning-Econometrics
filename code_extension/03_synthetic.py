##################################################
# Synthetic Controls
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv(
    "data_extension/synthetic/meta_metaverse.csv",
    parse_dates=["Date"],
    index_col="Date"
)

df = df / df.iloc[0] * 100

event_date = pd.Timestamp("2021-10-28")

treated = "meta"
donors = [c for c in df.columns if c != treated]

pre = df[df.index < event_date]
post = df[df.index >= event_date]

y = pre[treated].values
X = pre[donors].values

def loss(w):
    return np.sum((y - X @ w) ** 2)

n = len(donors)

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1}
]

bounds = [(0, 1)] * n

w0 = np.ones(n) / n

result = minimize(
    loss,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

weights = result.x

synthetic = df[donors].values @ weights

comparison = pd.DataFrame(index=df.index)
comparison["Actual"] = df[treated]
comparison["Synthetic"] = synthetic
comparison["Gap"] = comparison["Actual"] - comparison["Synthetic"]

pre_fit = comparison.loc[comparison.index < event_date]

corr = pre_fit["Actual"].corr(pre_fit["Synthetic"])
mae = mean_absolute_error(pre_fit["Actual"], pre_fit["Synthetic"])
rmse = np.sqrt(mean_squared_error(pre_fit["Actual"], pre_fit["Synthetic"]))
r2 = r2_score(pre_fit["Actual"], pre_fit["Synthetic"])

print("\nSynthetic Control Weights")
print("-" * 40)

for donor, weight in zip(donors, weights):
    print(f"{donor:<12} {weight:.4f}")

print("\nTraining Fit")
print("-" * 40)
print(f"Correlation : {corr:.4f}")
print(f"R²          : {r2:.4f}")
print(f"MAE         : {mae:.2f}")
print(f"RMSE        : {rmse:.2f}")

plt.figure(figsize=(12,6))
plt.plot(comparison.index, comparison["Actual"], linewidth=2, label="Meta")
plt.plot(comparison.index, comparison["Synthetic"], "--", linewidth=2, label="Synthetic Meta")
plt.axvline(event_date, color="red", linestyle="--", label="Metaverse Announcement")
plt.ylabel("Adjusted Close")
plt.title("Meta vs Synthetic Meta")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(comparison.index, comparison["Gap"], linewidth=2)
plt.axhline(0, color="black")
plt.axvline(event_date, color="red", linestyle="--")
plt.ylabel("Treatment Effect")
plt.title("Gap: Actual − Synthetic")
plt.tight_layout()
plt.show()

##################################################
# Findings
# We can model META as an optimized synthetic form based on other comparable tech stocks.

# Takeaways
# 1. Synthetic controls provide a framework for estimating causal effects when there is one treated unit and no obvious control group.
# 2. The quality of the counterfactual depends heavily on the donor pool and the ability to replicate pre-treatment trends.
# 3. Unlike DiD, synthetic controls allow different weights to be assigned to control units rather than assuming an average treatment effect across groups.

# Extensions:
# Use better optimization algorithms, more data, longer timeperiod etc. to try and find a good synthetic form.
# Robust calibration of the synthetic model in the pre-event window.
##################################################