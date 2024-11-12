# portfolio_optimization.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logging.basicConfig(filename='optimization.log', level=logging.INFO)

def calculate_portfolio_metrics(data, weights):
    """
    Calculate portfolio return, volatility, and Sharpe ratio.
    """
    returns = np.dot(weights, data.mean())
    risk = np.sqrt(np.dot(weights.T, np.dot(data.cov(), weights)))
    sharpe_ratio = returns / risk
    return returns, risk, sharpe_ratio

def optimize_portfolio(data):
    """
    Optimize portfolio weights to maximize Sharpe Ratio.
    """
    num_assets = data.shape[1]
    initial_weights = np.ones(num_assets) / num_assets

    def neg_sharpe(weights):
        return -calculate_portfolio_metrics(data, weights)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, initial_weights, bounds=bounds, constraints=constraints)
    logging.info(f"Optimized Portfolio Weights: {result.x}")
    return result.x
