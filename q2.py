"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 8/3/2021
Homework #4 - Q2
Analyzing patient data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score


# Reading in dataframe
df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

# ==================
# Surviving Patients
# ==================

# Group 4 (x=platlets, y=serium creatinine)
df_0 = df.loc[df["DEATH_EVENT"] == 0]

# Separating into x and y
x = df_0[["platelets", "serum_creatinine"]]
y = df_0["serum_creatinine"]

# Splitting 50:50
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=1, shuffle=True
)

# Training and testing the model
reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

# Calculate sum of residuals squared
index = 0
sum = 0
residuals = []
for predicted_value in y_predict:
    test_value = y_test.values.tolist()[index]
    if predicted_value != test_value:
        sum += (test_value - predicted_value)**2
    index += 1
print("For surviving patients sum of residuals squared: " + str(sum))

# =================
# Deceased Patients
# =================

# Group 4 (x=platlets, y=serium creatinine)
df_1 = df.loc[df["DEATH_EVENT"] == 1]

# Separating into x and y
x = df_1[["platelets", "serum_creatinine"]]
y = df_1["serum_creatinine"]

# Splitting 50:50
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=1, shuffle=True
)

# Training and testing the model
reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

# Calculate sum of residuals squared
index = 0
sum = 0
residuals = []
for predicted_value in y_predict:
    test_value = y_test.values.tolist()[index]
    if predicted_value != test_value:
        sum += (test_value - predicted_value)**2
    index += 1
print("For deceased patients sum of residuals squared: " + str(sum))

# =================
# Quadradic
# =================


mymodel = np.poly1d(np.polyfit(x_train["platelets"], y_train, 3))
print(r2_score(y_test, mymodel(x_test["platelets"])))


