import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocessing import clean_data

learn_rate = 0.0001
epochs = 25

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
m = len(y)

# add bias column to feature matrix and set parameters to zero
Xmtrx = np.c_[np.ones((Xmtrx.shape[0], 1)), Xmtrx]
theta = np.zeros((Xmtrx.shape[1], 1))
error_per_epoch = []
for epoch in range(epochs):
    for i in range(m):
        xi = Xmtrx[i:i+1]
        yi = ymtrx[i:i+1]
        gradient = xi.T @ (xi @ theta - yi)
        theta = theta - learn_rate * gradient
    error_per_epoch.append(np.sqrt(np.mean((Xmtrx @ theta - ymtrx) ** 2)))
    
plt.plot(range(epochs), error_per_epoch)
plt.show()

print(error_per_epoch[-1])




