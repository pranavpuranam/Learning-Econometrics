##################################################
# Randomized Controlled Trials (RCT)
##################################################

import pandas as pd

df = pd.read_csv("data/rct/project_star.csv") # found up to ~45% missing values for particular columns

balance_columns = ["stark", "gender", "ethnicity", "birth", "lunchk", "schoolk", "degreek", "ladderk", "experiencek", "tethnicityk"] # created a balance table

means = (
    df.groupby("stark")[["readk", "mathk"]]
      .mean()
      .round(2)
) # we compute the average math and reading scores for different class sizes

print(means) # average reading and math scores are higher for students in smaller classes
