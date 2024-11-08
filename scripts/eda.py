import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

logging.basicConfig(level=logging.INFO, filename='eda.log', filemode='w')

def plot_closing_price(data, ticker):
    """Plots closing price over time with year formatting on the x-axis."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.title(f'{ticker} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Format the x-axis to show only the year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    plt.legend()
    plt.show()
    logging.info(f"Closing price plotted for {ticker}.")

def calculate_daily_returns(data):
    """Calculates daily returns."""
    data['Daily_Return'] = data['Close'].pct_change()
    logging.info("Daily returns calculated.")
    return data

def plot_volatility(data, window=30):
    """Plots rolling volatility with year formatting on the x-axis."""
    rolling_std = data['Daily_Return'].rolling(window=window).std()
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_std, label='Rolling Volatility')
    plt.title(f'Rolling {window}-Day Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    
    # Format the x-axis to show only the year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    plt.legend()
    plt.show()
    logging.info("Volatility plotted.")

def detect_outliers(data, threshold=0.05):
    """Identifies outliers based on returns above a certain threshold."""
    outliers = data[abs(data['Daily_Return']) > threshold]
    logging.info(f"Outliers detected: {len(outliers)}")
    return outliers
