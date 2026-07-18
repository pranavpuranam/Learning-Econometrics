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

# Takeaways
# 1. The experimental method is what creates an RCT, not shuffling or sampling of the data itself.
# 2. Before testing causal hypothesis, check for balance in the treatment and control groups.
# 3. A t-test or ANOVA test (with p-value) can be used to evaluate significance of causal findings.

# Assumptions:
# 1. Random Assignment: Treatment is assigned independently of outcomes.
# 2. Independence (SUTVA): The treatment of one unit does not affect the outcome of another unit.
# 3. Compliance: Units assigned to treatment actually receive the treatment.
# 4. No differential attrition: Units do not drop out of the study in a way that is correlated with treatment assignment.
# 5. Random Sampling: The sample is representative of the population of interest (if you want to generalize).
##################################################