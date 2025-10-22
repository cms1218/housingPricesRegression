Housing Prices Regression Project

Goal:
Build a linear regression pipeline from scratch to predict housing prices using the California Housing Prices dataset. This project emphasizes understanding the full ML workflow: exploratory data analysis, model implementation, feature engineering, and performance evaluation.

Dataset:
Source: California Housing Prices Dataset (Kaggle)
Target Variable: median_house_value (median house value of a district)
Features (original):
Longitude
Latitude
Median Age of Residents
Total Number of Rooms
Total Number of Bedrooms
Population of District
Median Income
Number of Households
Ocean Proximity (categorical)

Notes:
Ocean Proximity was removed due to its categorical nature and the desire to simplify initial regression modeling.

Latitude and longitude were dropped after correlation analysis showed near-zero relationship with the target.

PART 1: Exploratory Data Analysis

Steps Taken:

Loaded the dataset and split into features (X) and target (y).

Generated summary statistics using df.describe() for all numeric features.

Visualized distributions with histograms to understand skewness and potential outliers.

Produced a correlation heatmap to assess relationships between features and the target.

Identified missing values: total_bedrooms contained some missing entries.

Filled missing total_bedrooms values using the median because histograms of related features (households) showed a right-skewed distribution.

Normalized numeric features to zero mean and unit variance for improved gradient descent performance.

Part 2: Model Development

Implemented a linear regression model from scratch using stochastic gradient descent (SGD).

Parameters: 
learning rate = 0.0001, epochs = 25.

Initial performance:
Mean Absolute Error (MAE) ≈ $76,240

Part 3: Model Optimization / Feature Engineering

New Features Added:

rooms_per_household = total rooms ÷ number of households (captures average room availability per household).

bedrooms_per_room = total bedrooms ÷ total rooms (captures room distribution).

Impact:

Model performance improved slightly:
MAE ≈ $75,528 (~1% improvement)