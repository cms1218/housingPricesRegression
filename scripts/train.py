import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocessing import clean_data

# Read csv into pandas dataframe
df = pd.read_csv(os.path.join("data", "housing.csv"))

# Split into features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Clean data using function built from EDA of the dataset
X = clean_data(X)

# Convert features and target dataframes into numpy arrays. Ensure target is a column vector
Xmtrx = X.to_numpy()
ymtrx = y.to_numpy().reshape(-1, 1)

# concatenate a column of ones for the bias vector to the front of the feature matrix
Xmtrx = np.c_[np.ones((Xmtrx.shape[0], 1)), Xmtrx]

# calculate the parameter values using the normal equation
theta = np.linalg.inv(Xmtrx.T @ Xmtrx) @ (Xmtrx.T @ ymtrx)

# calculate the predicted value of y
y_pred = Xmtrx @ theta

# calculate the mean squared error
mse = np.mean((y_pred - ymtrx) ** 2)
print(mse)

# scale mean squared error down to same units as target
rmse = np.sqrt(mse)
print(rmse)



