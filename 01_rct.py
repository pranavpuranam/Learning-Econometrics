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

##################################################
# Findings
# Employing smaller class sizes in kindergarten creates a positive effect on student performance in reading and math.

# Lessons
# 1. The experimental method is what creates an RCT, not shuffling or sampling of the data itself.
# 2. Before testing causal hypothesis, check for balance in the treatment and control groups.
# 3. A t-test or ANOVA test (with p-value) can be used to evaluate significance of causal findings.
##################################################