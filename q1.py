"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/3/2021
Homework #4 - Q1
Analyzing patient data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


# Reading in dataframe
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

# Separating into 2 dataframes
df_0 = df.loc[df["DEATH_EVENT"] == 0][
    ["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]
]
df_1 = df.loc[df["DEATH_EVENT"] == 1][
    ["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]
]

# Generating Matricies
corrMatrix = df_0.corr()
sn.heatmap(corrMatrix, annot=True)
plt.savefig("output/M_0.pdf")
plt.clf()
corrMatrix = df_1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.savefig("output/M_1.pdf")
