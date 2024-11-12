# modeling.py

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import logging

logging.basicConfig(filename='modeling.log', level=logging.INFO)

def train_arima(data, order=(1, 1, 1)):
    """
    Train an ARIMA model on the given data.
    """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    logging.info(f"ARIMA Model Summary:\n{model_fit.summary()}")
    return model_fit

def train_sarimax(data, seasonal_order=(1, 1, 1, 12), exog=None):
    """
    Train a SARIMAX model on the given data with optional exogenous variables.
    """
    model = SARIMAX(data, seasonal_order=seasonal_order, exog=exog)
    model_fit = model.fit(disp=False)
    logging.info(f"SARIMAX Model Summary:\n{model_fit.summary()}")
    return model_fit

def train_lstm(data, epochs=10, batch_size=1):
    """
    Train an LSTM model on the given data.
    """
    data = np.array(data).reshape(-1, 1)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(data.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(data[:-1], data[1:], epochs=epochs, batch_size=batch_size, verbose=0)
    logging.info("LSTM Model Trained")
    return model
