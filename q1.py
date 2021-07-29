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

# Reading in dataframe
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

# Separating into 2 dataframes
df_0 = df.loc[df["DEATH_EVENT"] == 0][
    ["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]
]
df_1 = df.loc[df["DEATH_EVENT"] == 1][
    ["creatinine_phosphokinase", "serum_creatinine", "serum_sodium", "platelets"]
]

# Generating matrices
fig = plt.figure(figsize=(17, 15))
ax1 = fig.add_subplot(111)
plt.imshow(df_0.corr(), cmap="hot")
plt.colorbar()
labels = df_0.columns.tolist()
ax1.set_xticks(np.arange(len(labels)))
ax1.set_yticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, rotation=90, fontsize=10)
ax1.set_yticklabels(labels, fontsize=10)
plt.savefig("output/M_0.pdf")

plt.imshow(df_1.corr(), cmap="hot")
plt.savefig("output/M_1.pdf")
