# housingPricesRegression
This is a project practicing ML building a linear regression model from scratch for a housing dataset.

The 'California Housing Prices' dataset (found here: https://www.kaggle.com/datasets/camnugent/california-housing-prices) is used for this project.

PART 1: Exploratory Data Analysis

This target value for this model is:
1. Median house value of a district

This dataset contains nine features:
1. Longitude
2. Latitude
3. Median Age of Residents
4. Total number of rooms
5. Total number of bedrooms
6. Population of district
7. Median income
8. Number of households
9. Ocean Proximity

Only ocean proximity is categorical, and because this is a regression model, ocean proximity was removed from the Data for training purposes. 

A correlation heatmap was generated using the remaining numerical data. Using the heatmap, it was noted that latitude and longitude had a near zero correlation with the target. Due to this, they were removed for the data in order to avoid wasting performance on training the model with them.

Missing values were detected in the column for total bedrooms. The number of missing values was comparatively insignificant to the total number of rows, so as opposed to removing the column the decision was made to fill the values with the mean or median of the column. In order to determine which measure of average to use, a histogram was created for the number of housholds column, which had the highest level of correlation with total bedrooms. This histogram contained a noticably right skew, so it was decided that the median should be used to replace the number of total bedrooms.
