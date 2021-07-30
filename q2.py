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

    print("Q2")
    
    # Reading in dataframe
    df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

    # Separating survivors from deceased
    df_0 = df.loc[df["DEATH_EVENT"] == 0]
    df_1 = df.loc[df["DEATH_EVENT"] == 1]

    # =================
    # Linear
    # =================
    print("==================")
    print("Linear")
    linear_sum_0 = round(linear_regression(df_0, "Survived"), 2)
    print("Survived: " + str(linear_sum_0))
    print("---")
    linear_sum_1 = round(linear_regression(df_1, "Deceased"), 2)
    print("Deceased: " + str(linear_sum_1))

    # =================
    # Quadradic
    # =================
    print("==================")
    print("Quadradic")
    quadradic_sum_0 = round(quadradic(df_0, "Survived"), 2)
    print("Survived: " + str(quadradic_sum_0))
    print("---")
    quadradic_sum_1 = round(quadradic(df_1, "Deceased"), 2)
    print("Deceased: " + str(quadradic_sum_1))

    # =================
    # Cubic
    # =================
    print("==================")
    print("Cubic")
    cubic_sum_0 = round(cubic(df_0, "Survived"), 2)
    print("Survived: " + str(cubic_sum_0))
    print("---")
    cubic_sum_1 = round(cubic(df_1, "Deceased"), 2)
    print("Deceased: " + str(cubic_sum_1))

    # =================
    # GLM LOG(X)
    # =================
    print("==================")
    print("GLM LOG X")
    glm_x_sum_0 = round(glm_log_x(df_0, "Survived"), 2)
    print("Survived: " + str(glm_x_sum_0))
    print("---")
    glm_x_sum_1 = round(glm_log_x(df_1, "Deceased"), 2)
    print("Deceased: " + str(glm_x_sum_1))

    # =================
    # GLM LOG(Y)
    # =================
    print("==================")
    print("GLM LOG Y")
    glm_y_sum_0 = round(glm_log_y(df_0, "Survived"), 2)
    print("Survived: " + str(glm_y_sum_0))
    print("---")
    glm_y_sum_1 = round(glm_log_y(df_1, "Deceased"), 2)
    print("Deceased: " + str(glm_y_sum_1))
    print("==================")

    print("TABLE FOR Q3")
    print("==================")
    table = pd.DataFrame(
        [
            ["y = ax + b", str(linear_sum_0), str(linear_sum_1)],
            ["y = ax2 + bx + c", str(quadradic_sum_0), str(quadradic_sum_1)],
            ["y = ax3 + bx2 + cx + d", str(cubic_sum_0), str(cubic_sum_1)],
            ["y = a log x + b", str(glm_x_sum_0), str(glm_x_sum_1)],
            ["log y = a log x + b", str(glm_y_sum_0), str(glm_y_sum_1)],
        ],
        index=[1, 2, 3, 4, 5],
        columns=["Model", "SSE - DEATH_EVENT 0", "SSE - DEATH_EVENT 0"],
    )
    print(table)


def linear_regression(df, patient_type):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df["platelets"]
    y = df["serum_creatinine"]

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    degree = 1
    weights = np.polyfit(x_train, y_train, degree)
    print("Weights: " + str(weights))
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(y_test.values.tolist(), c='blue', label="Actual")
    plt.plot(predicted, c='red', label="Predicted")
    plt.title("Logistic Regression for " + patient_type + " Patients")
    plt.legend(loc="upper right")
    plt.show()

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def quadradic(df, patient_type):

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
    print("Weights: " + str(weights))
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(y_test.values.tolist(), c='blue', label="Actual")
    plt.plot(predicted, c='red', label="Predicted")
    plt.title("Quadradic for " + patient_type + " Patients")
    plt.legend(loc="upper right")
    plt.show()

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def cubic(df, patient_type):

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
    print("Weights: " + str(weights))
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(y_test.values.tolist(), c='blue', label="Actual")
    plt.plot(predicted, c='red', label="Predicted")
    plt.title("Cubic for " + patient_type + " Patients")
    plt.legend(loc="upper right")
    plt.show()

    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def glm_log_x(df, patient_type):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df["platelets"]
    y = df["serum_creatinine"]

    x = np.log(x)

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    degree = 1
    weights = np.polyfit(x_train, y_train, degree)
    print("Weights: " + str(weights))
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(y_test.values.tolist(), c='blue', label="Actual")
    plt.plot(predicted, c='red', label="Predicted")
    plt.title("GLM log(x) for " + patient_type + " Patients")
    plt.legend(loc="upper right")
    plt.show()

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


def glm_log_y(df, patient_type):

    # Group 4 (x=platlets, y=serium creatinine)
    # Separating into x and y
    x = df["platelets"]
    y = df["serum_creatinine"]

    y = np.log(y)

    # Splitting 50:50
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=1, shuffle=True
    )

    # Training and testing the model
    degree = 1
    weights = np.polyfit(x_train, y_train, degree)
    print("Weights: " + str(weights))
    model = np.poly1d(weights)
    predicted = model(x_test)

    plt.plot(y_test.values.tolist(), c='blue', label="Actual")
    plt.plot(predicted, c='red', label="Predicted")
    plt.title("GLM log(y) for " + patient_type + " Patients")
    plt.legend(loc="upper right")
    plt.show()

    # Calculate sum of residuals squared
    sum_of_errors_squared = ((y_test - predicted) * (y_test - predicted)).sum()

    return sum_of_errors_squared


main()
