import logging
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, filename='analysis.log', filemode='w')

def decompose_time_series(data):
    """Decomposes the time series into trend, seasonality, and residuals with year formatting on the x-axis."""
    decomposition = sm.tsa.seasonal_decompose(data['Close'], model='multiplicative', period=252)
    logging.info("Time series decomposition completed.")
    
    # Plot the decomposition with year formatting on the x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    # Set x-axis format for each subplot
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    plt.show()
    
    return decomposition

def calculate_risk_metrics(data):
    """Calculates VaR and Sharpe Ratio for risk assessment."""
    var_99 = np.percentile(data['Daily_Return'], 1)
    sharpe_ratio = data['Daily_Return'].mean() / data['Daily_Return'].std() * np.sqrt(252)
    logging.info("Risk metrics (VaR, Sharpe Ratio) calculated.")
    return var_99, sharpe_ratio

def analyze_volatility(data, window=30):
    """Calculates rolling standard deviation to analyze short-term volatility with year formatting on the x-axis."""
    data['Rolling_Volatility'] = data['Daily_Return'].rolling(window=window).std()
    
    # Plot the rolling volatility with year formatting
    plt.figure(figsize=(12, 6))
    plt.plot(data['Rolling_Volatility'], label='Rolling Volatility')
    plt.title(f'Rolling {window}-Day Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    
    # Format x-axis to display only years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    plt.legend()
    plt.show()
    logging.info("Volatility analysis completed.")
    
    return data
