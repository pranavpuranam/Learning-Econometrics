##################################################
# Event Studies
##################################################

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(
    "data_extension/events/meta.csv",
    parse_dates=["Date"],
    index_col="Date"
)

df = df.dropna()

event_date = pd.Timestamp("2024-02-02")

event_idx = df.index.get_loc(event_date)

estimation = df.iloc[event_idx-250:event_idx-30]

event_window = df.iloc[event_idx-5:event_idx+6]

X = sm.add_constant(estimation["sp500_return"])
y = estimation["meta_return"]

market_model = sm.OLS(y, X).fit()

print(market_model.summary())

X_event = sm.add_constant(event_window["sp500_return"])

event_window = event_window.copy()

event_window["expected_return"] = market_model.predict(X_event)

event_window["abnormal_return"] = (
    event_window["meta_return"]
    - event_window["expected_return"]
)

event_window["CAR"] = event_window["abnormal_return"].cumsum()

print(event_window[[
    "meta_return",
    "expected_return",
    "abnormal_return",
    "CAR"
]])

total_car = event_window["abnormal_return"].sum()

print(f"\nCAR (-5,+5): {total_car:.4%}")

##################################################
# Findings
# Meta's 2023 Q4 Earnings led to a cumulative 12.76% abnormal return using a benchmark market model of the stock.

# Takeaways
# 1. We define a pre-event window (-250d to -30d) and regress META on MKT, we extend MKT model to predict return during event (event - 5d, +5d), then compare prediction to real to define abnormal return.
# 2. Cumulative Abnormal Return (CAR) is summed abnormal return over event window.

# Extensions:
# Multiple Events: e.g. every meta earnings call, or acquisition announcement, etc. We can collect a distribution of CARs for different earnings days.
# Cross-sectional Regression: we can regress CAR on market cap, size of earnings surprise, volatility, ESG, textual analysis of earnings call, etc.
# Expected Return Model: we would NOT usually use market model, alternatives are CAPM, Fama-French 3/5-factor model, Carhart 4-factor, etc.
# Different Event Windows: tinker with pre-event (-250, -30) and the event (-5, 5) windows.
# Statistical Tests: can test H0 of CAR = 0 or Average CAR = 0
# Compare Groups: fixed effect model for CARs across sector, or NASDAQ vs non-NASDAQ, etc.
##################################################