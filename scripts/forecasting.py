# forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='forecasting.log', level=logging.INFO)

def generate_forecast(model, steps):
    """
    Generate forecasts for the given number of steps.
    """
    forecast = model.forecast(steps=steps)
    logging.info(f"Generated Forecast: {forecast}")
    return forecast

def plot_forecast(train_data, forecast, model_name, confidence_interval=None):
    """
    Plot the forecast alongside historical data.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='Historical')
    plt.plot(forecast, label=f'{model_name} Forecast')
    if confidence_interval:
        plt.fill_between(forecast.index, confidence_interval[:, 0], confidence_interval[:, 1], color='pink', alpha=0.3)
    plt.title(f'{model_name} Forecast vs Historical Data')
    plt.legend()
    plt.show()

def plot_all_forecasts(train_data, arima_forecast, sarimax_forecast, lstm_forecast):
    """
    Plot forecasts from ARIMA, SARIMAX, and LSTM models for comparison.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Historical')
    plt.plot(arima_forecast, label='ARIMA Forecast', linestyle='--')
    plt.plot(sarimax_forecast, label='SARIMAX Forecast', linestyle=':')
    plt.plot(lstm_forecast, label='LSTM Forecast', linestyle='-.')
    plt.title('Model Forecast Comparisons')
    plt.legend()
    plt.show()
