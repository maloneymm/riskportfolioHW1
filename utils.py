import numpy as np
import pandas as pd

ANN = 12


def summary_stats(rets, ann=ANN):
    """Annualized mean, volatility and Sharpe ratio of each column of excess returns."""
    out = pd.DataFrame({
        'Mean': rets.mean() * ann,
        'Vol': rets.std() * np.sqrt(ann),
    })
    out['Sharpe'] = out['Mean'] / out['Vol']
    return out


def tangency_weights(mu, sigma):
    """w_tan = Sigma^-1 mu / (1' Sigma^-1 mu), from monthly mu and sigma."""
    w = np.linalg.solve(sigma, mu)
    return pd.Series(w / w.sum(), index=mu.index, name='w_tan')


def portfolio_stats(w, mu, sigma, ann=ANN):
    """Annualized mean, vol, Sharpe of portfolio w, given monthly mu and sigma."""
    mean = w @ mu * ann
    vol = np.sqrt(w @ sigma @ w * ann)
    return pd.Series({'Mean': mean, 'Vol': vol, 'Sharpe': mean / vol})
