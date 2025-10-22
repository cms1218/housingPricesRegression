import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocessing import clean_data

df = pd.read_csv(os.path.join("data", "housing.csv"))

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X = clean_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))

