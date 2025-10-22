import pandas as pd

# Function to clean and preprocess the housing data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values in total_bedrooms with median
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Scale median_income
    df['median_income'] = df['median_income'] * 10000
    
    # Drop categorical and irrelevant columns
    df = df.drop(columns=['ocean_proximity', 'latitude', 'longitude'])

    # Normalize numerical features
    for column in ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'housing_median_age']:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df