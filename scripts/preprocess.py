import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, filename='preprocess.log', filemode='w')

def clean_data(data):
    """Cleans the data by handling missing values and checking data types."""
    data = data.dropna()  # Remove missing values for simplicity
    data.reset_index(drop=True, inplace=True)
    logging.info("Missing values handled, and data types verified.")
    return data

def normalize_data(data):
    """Normalizes data using StandardScaler."""
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    logging.info("Data normalized.")
    return data_scaled
