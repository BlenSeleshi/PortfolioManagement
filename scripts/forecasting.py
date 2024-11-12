# forecasting.py

import numpy as np
import matplotlib.pyplot as plt

def generate_forecast(model, steps, exog=None):
    forecast = model.forecast(steps=steps, exog=exog)
    return forecast

def plot_forecast(actual, forecast, model_name):
    plt.figure(figsize=(10,6))
    plt.plot(actual.index, actual.values, label="Actual", color='blue')
    plt.plot(forecast.index, forecast.values, label=f"Forecast ({model_name})", color='red')
    plt.title(f'{model_name} Forecast vs Actual')
    plt.legend()
    plt.show()

def plot_all_forecasts(actual, arima_forecast, sarimax_forecast, lstm_forecast):
    plt.figure(figsize=(14,8))
    
    plt.subplot(311)
    plt.plot(actual.index, actual.values, label="Actual", color='blue')
    plt.plot(arima_forecast.index, arima_forecast.values, label="ARIMA Forecast", color='red')
    plt.title("ARIMA Forecast vs Actual")
    plt.legend()

    plt.subplot(312)
    plt.plot(actual.index, actual.values, label="Actual", color='blue')
    plt.plot(sarimax_forecast.index, sarimax_forecast.values, label="SARIMAX Forecast", color='green')
    plt.title("SARIMAX Forecast vs Actual")
    plt.legend()

    plt.subplot(313)
    plt.plot(actual.index, actual.values, label="Actual", color='blue')
    plt.plot(lstm_forecast.index, lstm_forecast, label="LSTM Forecast", color='purple')
    plt.title("LSTM Forecast vs Actual")
    plt.legend()

    plt.tight_layout()
    plt.show()
