# Financial Time Series Analysis for Portfolio Optimization

## Overview

This project provides a comprehensive analysis of key financial assets to enhance portfolio management strategies. It focuses on three assets:

- **Tesla (TSLA)**
- **Vanguard Total Bond Market ETF (BND)**
- **S&P 500 ETF (SPY)**

We aim to assess their historical performance, volatility, and risk profiles to help GMF Investments make informed portfolio decisions. The analysis is divided into several tasks, including data extraction, preprocessing, exploratory data analysis (EDA), advanced time series analysis with risk metrics, and portfolio optimization. The project also implements forecasting models for return predictions and risk evaluation.

In addition to analyzing individual asset behavior, the project offers **portfolio optimization** by finding the optimal asset allocation that maximizes returns while minimizing risks.

---

## Key Features

- **Time Series Forecasting Models**:

  - **ARIMA (AutoRegressive Integrated Moving Average)** for univariate forecasting.
  - **SARIMAX (Seasonal ARIMA with eXogenous variables)** for incorporating external volatility and technical indicators.
  - **LSTM (Long Short-Term Memory)** for capturing complex, non-linear patterns in asset returns.

- **Portfolio Optimization**:

  - Optimizes portfolio weights to maximize the Sharpe ratio, balancing risk and return.
  - Calculates portfolio metrics like expected return, volatility, and Sharpe ratio.

- **Performance Evaluation**:
  - Evaluates forecasting models using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean Absolute Percentage Error).
  - Visualizes portfolio performance over time and provides detailed risk and return analysis.

---

## Project Structure

```
PortfolioOptimization/
│
├── data/                     # Folder containing input CSV files with historical data
│   ├── TSLA_data.csv              # Tesla stock price data
│   ├── BND_data.csv               # Vanguard Total Bond Market ETF data
│   └── SPY_data.csv               # S&P 500 ETF data
│
├── scripts/                  # Folder containing Python scripts for forecasting and optimization
│   ├── data_fetch.py         # Script for fetching historical data
│   ├── preprocess.py         # Script for cleaning and preprocessing data
│   ├── modeling.py         # Script for cleaning and preprocessing data
│   ├── eda.py                # Script for exploratory data analysis
│   ├── analysis.py           # Script for advanced analysis (decomposition and risk metrics)
│   ├── forecasting.py        # Script for time series forecasting models
│   ├── optimization.py       # Script for portfolio optimization
│   └── performance.py        # Script for calculating performance metrics and visualizations
│
├── notebooks/                # Folder containing Jupyter notebook for analysis
│   └── portfolio_analysis.ipynb
│   └── eda.ipynb
│
└── README.md                 # Project overview and setup instructions (this file)
```

---

## Modules Overview

### 1. **data_fetch.py**

This module handles fetching historical stock data. It defines the following functions:

- **`fetch_data(ticker, start_date, end_date)`**: Fetches historical stock data for the specified ticker and date range.
- **`save_data_to_csv(data, filename)`**: Saves the fetched data into a CSV file for future use.

### 2. **preprocess.py**

Handles cleaning and preprocessing of the data:

- **`clean_data(data)`**: Removes missing values and resets the index.
- **`normalize_data(data)`**: Normalizes the data using **StandardScaler** to scale the features between 0 and 1.

### 3. **eda.py**

Exploratory Data Analysis (EDA) tools to analyze the data:

- **`plot_closing_price(data, ticker)`**: Plots the closing price of a given asset over time.
- **`calculate_daily_returns(data)`**: Calculates daily returns based on the closing price.
- **`plot_volatility(data, window=30)`**: Plots rolling volatility over a given window (default 30 days).
- **`detect_outliers(data, threshold=0.05)`**: Identifies outliers in the daily returns based on a set threshold.

### 4. **analysis.py**

Advanced statistical and financial analysis:

- **`decompose_time_series(data)`**: Decomposes the time series into trend, seasonality, and residuals using multiplicative decomposition.
- **`calculate_risk_metrics(data)`**: Calculates key risk metrics such as **Value at Risk (VaR)** and the **Sharpe Ratio**.
- **`analyze_volatility(data, window=30)`**: Calculates rolling volatility to assess short-term risk.

### 5. **forecasting.py**

Forecasting models for return predictions:

- **ARIMA, SARIMAX, and LSTM models** are implemented to predict future returns.
- Predictions are used to assess potential future performance and adjust portfolio weights.
- **Evaluation**: Models are evaluated using MAE, RMSE, and MAPE to assess the prediction accuracy.

### 6. **optimization.py**

Portfolio optimization script:

- **`optimize_portfolio()`**: Optimizes portfolio weights based on the forecasted returns and historical risk, maximizing the Sharpe ratio.
- **Outputs**: The optimal portfolio weights for TSLA, BND, and SPY, along with metrics like expected return, portfolio volatility, and Sharpe ratio.

### 7. **performance.py**

Performance evaluation and visualization:

- **`calculate_performance_metrics()`**: Calculates portfolio metrics like expected return, volatility, and Sharpe ratio.
- **`plot_performance()`**: Visualizes portfolio performance over time using Matplotlib.

---

## Installation & Requirements

To run this project, ensure you have the following libraries installed:

```bash
pip install yfinance pandas matplotlib scikit-learn statsmodels tensorflow
```

Ensure that your Python environment has access to Jupyter for running the notebook.

---

## Usage

### 1. **Data Extraction**

The historical data for the three tickers **TSLA**, **BND**, and **SPY** is fetched using the `fetch_data()` function from **YFinance**. This data is saved locally into CSV files for further analysis.

Example:

```python
data = fetch_data('TSLA', '2015-01-01', '2024-10-31')
save_data_to_csv(data, 'TSLA_data.csv')
```

### 2. **Data Preprocessing**

Once the data is fetched, it is cleaned and normalized using the `clean_data()` and `normalize_data()` functions from **preprocess.py**.

Example:

```python
data = clean_data(data)
# data = normalize_data(data)  # Uncomment to normalize
```

### 3. **Exploratory Data Analysis (EDA)**

The **EDA** module provides various tools to analyze the data visually. The `plot_closing_price()` function plots the closing prices over time, while `calculate_daily_returns()` helps analyze price volatility, and `detect_outliers()` flags significant price movements.

Example:

```python
plot_closing_price(data, 'TSLA')
data = calculate_daily_returns(data)
plot_volatility(data)
outliers = detect_outliers(data)
```

### 4. **Advanced Analysis**

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

### 5. **Forecasting**

The **forecasting.py** script trains and applies ARIMA, SARIMAX, and LSTM models to predict future returns. The models are evaluated using error metrics such as MAE, RMSE, and MAPE.

Example for ARIMA:

```python
from forecasting import forecast_arima
forecast = forecast_arima(data)
```

### 6. **Portfolio Optimization**

Run the **optimization.py** script to calculate the optimal portfolio weights, expected return, volatility, and Sharpe ratio:

```bash
python scripts/optimization.py
```

Output Example:

```
Optimized Portfolio Weights:
TSLA: 0.14, BND: 0.47, SPY: 0.39

Portfolio Expected Return: 0.0005
Portfolio Volatility (Risk): 0.0084
Portfolio Sharpe Ratio: 0.0608
```

### 7. **Performance Evaluation**

Use the **performance.py** script to evaluate portfolio performance and visualize the results over time.

Example:

```python
from performance import plot_performance
plot_performance(portfolio_data)
```

---

## Key Insights from the Analysis

- **Time Series Decomposition**: Identifies the underlying trend, seasonal patterns, and random noise in asset price movements.
- **VaR (99%)**: Provides an estimate of the maximum expected loss with 99% confidence, helping investors understand downside risk.
- **Sharpe Ratio**: Measures the risk-adjusted return, offering insight into the asset's return relative to its risk.
- **Portfolio Optimization**: The optimal portfolio weights are designed to maximize risk-adjusted returns while minimizing volatility.

---

## Limitation

ations & Future Work

- **Data Coverage**: The analysis is limited to the three assets (TSLA, BND, and SPY). Future versions can include more assets or incorporate additional risk metrics.
- **Forecasting Accuracy**: The accuracy of predictions may vary based on the model and data quality. Exploring other machine learning models for better predictions is an option.
- **Scenario Analysis**: Further improvements can include running scenario analyses for different economic conditions and their impact on the portfolio.

---

## Logging

The project logs key actions and errors into separate log files for each module (e.g., `data_fetch.log`, `preprocess.log`, `eda.log`, `analysis.log`). This helps in tracking the progress and diagnosing any issues that arise during execution.
