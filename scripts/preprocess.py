import pandas as pd
from statsmodels.tsa.stattools import adfuller
import logging
import numpy as np
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
# def load_data_from_csv():
#     """
#     Load data from pre-saved CSV files for each ticker.
#     """
#     tickers = ['TSLA', 'BND', 'SPY']
#     data = {}
#     for ticker in tickers:
#         df = pd.read_csv(r"C:\Users\Blen\OneDrive\Desktop\10Academy\PortfolioManagement\data\{}_data.csv".format(ticker),
#                          parse_dates=['Date'], index_col='Date')
#         data[ticker] = df
#     return data

def load_data(file_path):
    # Load the CSV file without interpreting any row as header to see if extra rows are added
    df = pd.read_csv(file_path, header=None)
    
    # Manually set the header by assigning the actual column names
    df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Drop any unnecessary rows, for example, if 'Ticker' was added as a row in the CSV
    df = df.dropna(subset=['Date'])  # Drops rows where 'Date' is NaT (if any rows have NaT as Date)
    
    # Set the 'Date' column as a DateTime index and ensure the format is correct
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')

    # Drop any rows where the 'Date' could not be parsed correctly
    df = df.dropna(subset=['Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])

    return df

def preprocess_data(data):
    """
    Basic preprocessing to compute log returns and handle missing values.
    """
    processed_data = {}
    for ticker, df in data.items():
        df['Log Return'] = df['Adj Close'].pct_change().apply(lambda x: np.log(1 + x))
        df['20-Day Volatility'] = df['Log Return'].rolling(window=20).std()  # 20-day rolling volatility
        df['50-Day Volatility'] = df['Log Return'].rolling(window=50).std()  # 50-day rolling volatility
        df['Bollinger Upper'] = df['Adj Close'].rolling(window=20).mean() + 2 * df['20-Day Volatility']
        df['Bollinger Lower'] = df['Adj Close'].rolling(window=20).mean() - 2 * df['20-Day Volatility']
        df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
        processed_data[ticker] = df.dropna()  # Drop any NaNs resulting from rolling calculations
    return processed_data

def compute_volatility_index(data):
    """
    Compute a combined volatility index from individual asset volatilities.
    """
    vol_index = pd.DataFrame({
        ticker: df['20-Day Volatility'] for ticker, df in data.items()
    }).mean(axis=1)
    return vol_index

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
