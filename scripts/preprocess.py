import pandas as pd
from statsmodels.tsa.stattools import adfuller
import logging
import numpy as np
import yfinance as yf
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

def load_data():
    """
    Load data for each ticker from separate CSV files and return individual DataFrames.
    Convert columns to appropriate data types after loading.
    """
    # Load each CSV file into separate DataFrames, while ensuring 'Date' is parsed as a datetime index
    tsla = pd.read_csv(r"C:\Users\Blen\OneDrive\Desktop\10Academy\PortfolioManagement\data\TSLA_data.csv", parse_dates=['Date'], index_col='Date')
    bnd = pd.read_csv(r"C:\Users\Blen\OneDrive\Desktop\10Academy\PortfolioManagement\data\BND_data.csv", parse_dates=['Date'], index_col='Date')
    spy = pd.read_csv(r"C:\Users\Blen\OneDrive\Desktop\10Academy\PortfolioManagement\data\SPY_data.csv", parse_dates=['Date'], index_col='Date')

    # Reset index if needed (to remove the 'Ticker' row)
    tsla.reset_index(drop=False, inplace=True)
    bnd.reset_index(drop=False, inplace=True)
    spy.reset_index(drop=False, inplace=True)

    # Ensure the columns are in the correct format (convert non-numeric values to NaN)
    for df in [tsla, bnd, spy]:
        # Convert relevant columns to numeric, forcing non-convertible values to NaN
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Ensure 'Date' is the correct datetime type
    tsla['Date'] = pd.to_datetime(tsla['Date'], errors='coerce')
    bnd['Date'] = pd.to_datetime(bnd['Date'], errors='coerce')
    spy['Date'] = pd.to_datetime(spy['Date'], errors='coerce')

    # Set 'Date' as the index for all DataFrames
    tsla.set_index('Date', inplace=True)
    bnd.set_index('Date', inplace=True)
    spy.set_index('Date', inplace=True)

    return tsla, bnd, spy

def preprocess_data(df):
    """
    Add volatility-related features to a single DataFrame.
    """
    df['Log Return'] = df['Adj Close'].pct_change().apply(lambda x: np.log(1 + x))
    df['20-Day Volatility'] = df['Log Return'].rolling(window=20).std()
    df['50-Day Volatility'] = df['Log Return'].rolling(window=50).std()
    df['Bollinger Upper'] = df['Adj Close'].rolling(window=20).mean() + 2 * df['20-Day Volatility']
    df['Bollinger Lower'] = df['Adj Close'].rolling(window=20).mean() - 2 * df['20-Day Volatility']
    df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
    return df.dropna()

def compute_volatility_index(tsla, bnd, spy):
    """
    Compute a combined volatility index from individual asset volatilities.
    """
    vol_index = pd.DataFrame({
        'TSLA': tsla['20-Day Volatility'],
        'BND': bnd['20-Day Volatility'],
        'SPY': spy['20-Day Volatility']
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
