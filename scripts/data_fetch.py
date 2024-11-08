import yfinance as yf
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, filename='data_fetch.log', filemode='w')

def fetch_data(ticker, start_date, end_date):
    """Fetches historical data for a given ticker using YFinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        logging.info(f"Data for {ticker} fetched successfully.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def save_data_to_csv(data, filename):
    """Saves data to a CSV file."""
    data.to_csv(filename)
    logging.info(f"Data saved to {filename}.")
