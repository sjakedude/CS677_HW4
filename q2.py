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


def main():

    # Reading in dataframe
    df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

    # Separating survivors from deceased
    df_0 = df.loc[df["DEATH_EVENT"] == 0]
    df_1 = df.loc[df["DEATH_EVENT"] == 1]

    # =================
    # Logistic
    # =================
    print("==================")
    print("Logistic")
    sum_0 = logistic_regression(df_0)
    sum_1 = logistic_regression(df_1)
    print("Survived: " + str(sum_0))
    print("Deceased: " + str(sum_1))

    # =================
    # Quadradic
    # =================
    print("==================")
    print("Quadradic")
    sum_0 = quadradic(df_0)
    sum_1 = quadradic(df_1)
    print("Survived: " + str(sum_0))
    print("Deceased: " + str(sum_1))

    # =================
    # Cubic
    # =================
    print("==================")
    print("Cubic")
    sum_0 = cubic(df_0)
    sum_1 = cubic(df_1)
    print("Survived: " + str(sum_0))
    print("Deceased: " + str(sum_1))
    
    # =================
    # Cubic
    # =================
    print("==================")
    print("GLM LOG X")
    sum_0 = glm_log_x(df_0)
    sum_1 = glm_log_x(df_1)
    print("Survived: " + str(sum_0))
    print("Deceased: " + str(sum_1))


def logistic_regression(df):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[["platelets", "serum_creatinine"]]
    y = df["serum_creatinine"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - y_predict) * (y_test - y_predict)).sum()

    return sum_of_errors_squared


def quadradic(df):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[["platelets", "serum_creatinine"]]
    y = df["serum_creatinine"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    x_train = x_train["platelets"]
    x_test = x_test["platelets"]

    degree = 2
    weights = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(weights)
    predicted = model(x_test)

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def cubic(df):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[["platelets", "serum_creatinine"]]
    y = df["serum_creatinine"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    x_train = x_train["platelets"]
    x_test = x_test["platelets"]

    degree = 3
    weights = np.polyfit(x_train, y_train, degree)
    model = np.poly1d(weights)
    predicted = model(x_test)

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def glm_log_x(df):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df[["platelets", "serum_creatinine"]]
    y = df["serum_creatinine"]

    x = np.log(x)

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - y_predict) * (y_test - y_predict)).sum()

    return sum_of_errors_squared


main()
