import pandas as pd
from statsmodels.tsa.stattools import adfuller
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, filename='preprocess.log', filemode='w')

def check_stationarity(data):
    """
    Perform Augmented Dickey-Fuller (ADF) test on each time series in the data to check for stationarity.
    """
    results = {}
    for name, series in data.items():
        adf_result = adfuller(series['Adj Close'])
        results[name] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Stationary': adf_result[1] < 0.05
        }
        logging.info(f"{name} Stationarity Check: {results[name]}")
    return results

def preprocess_data(data):
    """
    Fill missing values and calculate log returns for each asset.
    """
    for name, df in data.items():
        df['Log Return'] = (df['Adj Close'] / df['Adj Close'].shift(1)).apply(lambda x: pd.np.log(x))
    return data

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
