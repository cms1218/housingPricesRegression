import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocessing import clean_data

df = pd.read_csv(os.path.join("data", "housing.csv"))

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X = clean_data(X)

Xmtrx = X.to_numpy()
ymtrx = y.to_numpy().reshape(-1, 1)

Xmtrx = np.c_[np.ones((Xmtrx.shape[0], 1)), Xmtrx]

theta = np.linalg.inv(Xmtrx.T @ Xmtrx) @ (Xmtrx.T @ ymtrx)

y_pred = Xmtrx @ theta

mse = np.mean((y_pred - ymtrx) ** 2)
print(mse)

rmse = np.sqrt(mse)
print(rmse)



