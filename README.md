# Financial Time Series Analysis for Portfolio Optimization

## Overview

This project provides a comprehensive analysis of key financial assets to enhance portfolio management strategies. It focuses on three assets:

- **Tesla (TSLA)**
- **Vanguard Total Bond Market ETF (BND)**
- **S&P 500 ETF (SPY)**

We aim to assess their historical performance, volatility, and risk profiles to help GMF Investments make informed portfolio decisions. The analysis is divided into several tasks, including data extraction, preprocessing, exploratory data analysis (EDA), and advanced time series analysis with risk metrics.

---

## Project Structure

The project is organized into different modules, each handling specific aspects of the analysis:

- **data_fetch.py**: Handles data extraction from YFinance.
- **preprocess.py**: Contains functions for cleaning and normalizing data.
- **eda.py**: Provides exploratory data analysis, including plotting and outlier detection.
- **analysis.py**: Performs advanced time series analysis, including decomposition and risk metrics.
- **Notebook**: The Jupyter Notebook (`financial_analysis.ipynb`) integrates all the modules to carry out the tasks.

---

## Modules Overview

### 1. **data_fetch.py**

This module handles fetching historical stock data using the **YFinance** library. It defines the following functions:

- **`fetch_data(ticker, start_date, end_date)`**: Fetches historical stock data for the specified ticker and date range. Data includes Open, High, Low, Close, and Volume.
- **`save_data_to_csv(data, filename)`**: Saves the fetched data into a CSV file for future use.

### 2. **preprocess.py**

This module focuses on cleaning and preprocessing the data:

- **`clean_data(data)`**: Cleans the data by removing missing values and resetting the index.
- **`normalize_data(data)`**: Normalizes the data using **StandardScaler** to scale the features between 0 and 1.

### 3. **eda.py**

Exploratory Data Analysis (EDA) functions are defined in this module:

- **`plot_closing_price(data, ticker)`**: Plots the closing price of a given asset over time.
- **`calculate_daily_returns(data)`**: Calculates daily returns based on the closing price.
- **`plot_volatility(data, window=30)`**: Plots the rolling volatility of the asset over a given window (default 30 days).
- **`detect_outliers(data, threshold=0.05)`**: Identifies outliers in the daily returns based on a set threshold.

### 4. **analysis.py**

This module performs advanced statistical and financial analysis:

- **`decompose_time_series(data)`**: Decomposes the time series data into **trend**, **seasonality**, and **residuals** using **multiplicative** decomposition.
- **`calculate_risk_metrics(data)`**: Calculates key risk metrics such as **Value at Risk (VaR)** and the **Sharpe Ratio**.
- **`analyze_volatility(data, window=30)`**: Calculates rolling volatility to assess short-term risk.

---

## Installation & Requirements

To run this project, ensure you have the following libraries installed:

```bash
pip install yfinance pandas matplotlib scikit-learn statsmodels
```

Ensure that your Python environment has access to Jupyter for running the notebook.

---

## Usage

### 1. Data Extraction

The historical data for the three tickers **TSLA**, **BND**, and **SPY** is fetched using the `fetch_data()` function from **YFinance**. This data is saved locally into CSV files for further analysis.

Example:

```python
data = fetch_data('TSLA', '2015-01-01', '2024-10-31')
save_data_to_csv(data, 'TSLA_data.csv')
```

### 2. Data Preprocessing

Once the data is fetched, it is cleaned and normalized using the `clean_data()` and `normalize_data()` functions from **preprocess.py**.

Example:

```python
data = clean_data(data)
# data = normalize_data(data)  # Uncomment to normalize
```

### 3. Exploratory Data Analysis (EDA)

The **EDA** module provides various tools to analyze the data visually. The `plot_closing_price()` function plots the closing prices over time, while `calculate_daily_returns()` helps analyze price volatility, and `detect_outliers()` flags significant price movements.

Example:

```python
plot_closing_price(data, 'TSLA')
data = calculate_daily_returns(data)
plot_volatility(data)
outliers = detect_outliers(data)
```

### 4. Advanced Analysis

The **analysis.py** module includes advanced time series analysis and risk metrics calculation:

- **Time Series Decomposition**: Using the `decompose_time_series()` function, we decompose the time series into its components.
- **Risk Metrics**: The `calculate_risk_metrics()` function calculates **VaR (99%)** and **Sharpe Ratio** for each asset.

Example:

```python
decomposition = decompose_time_series(data)
fig = decomposition.plot()
fig.suptitle('TSLA Time Series Decomposition')

var_99, sharpe_ratio = calculate_risk_metrics(data)
print(f"VaR (99%): {var_99}, Sharpe Ratio: {sharpe_ratio}")
```

---

## Key Insights from the Analysis

- **Time Series Decomposition**: Identifies the underlying trend, seasonal patterns, and random noise in the asset price movement.
- **VaR (99%)**: Provides an estimate of the maximum expected loss with 99% confidence, helping investors understand downside risk.
- **Sharpe Ratio**: Measures the risk-adjusted return, offering insight into the asset's return relative to its risk.

---

## Conclusion

This project offers valuable insights into the historical performance and risk profile of Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and the S&P 500 ETF (SPY). By analyzing trends, volatility, and risk metrics, GMF Investments can optimize portfolio management strategies to minimize risks and maximize returns.

For further improvements, the project can be extended by including more assets or incorporating additional risk metrics and forecasting models.

---

## Logging

The project logs key actions and errors into separate log files for each module (e.g., `data_fetch.log`, `preprocess.log`, `eda.log`, `analysis.log`). This helps in tracking the progress and diagnosing any issues that arise during execution.

---

## License

This project is licensed under the Apache License. See the LICENSE file for more details.
