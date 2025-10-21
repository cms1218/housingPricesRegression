import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values in total_bedrooms with median
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    
    # Scale median_income
    df['median_income'] = df['median_income'] * 10000
    
    # Drop categorical and irrelevant columns
    df.drop(columns=['ocean_proximity', 'latitude', 'longitude'], inplace=True)

    # Normalize numerical features
    for column in ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df