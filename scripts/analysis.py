import logging
import statsmodels.api as sm
import numpy as np

logging.basicConfig(level=logging.INFO, filename='analysis.log', filemode='w')

def decompose_time_series(data):
    """Decomposes the time series into trend, seasonality, and residuals."""
    decomposition = sm.tsa.seasonal_decompose(data['Close'], model='multiplicative', period=252)
    logging.info("Time series decomposition completed.")
    return decomposition

def calculate_risk_metrics(data):
    """Calculates VaR and Sharpe Ratio for risk assessment."""
    var_99 = np.percentile(data['Daily_Return'], 1)
    sharpe_ratio = data['Daily_Return'].mean() / data['Daily_Return'].std() * np.sqrt(252)
    logging.info("Risk metrics (VaR, Sharpe Ratio) calculated.")
    return var_99, sharpe_ratio

def analyze_volatility(data, window=30):
    """Calculates rolling standard deviation to analyze short-term volatility."""
    data['Rolling_Volatility'] = data['Daily_Return'].rolling(window=window).std()
    logging.info("Volatility analysis completed.")
    return data
