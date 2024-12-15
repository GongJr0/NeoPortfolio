import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import plotly

from CustomTypes import StockSymbol, Portfolio


class Markowitz:

    def __init__(self, portfolio: Portfolio, market: StockSymbol, horizon: int = 21, lookback: int = 252, rf_rate: float = 0.05717):
        self.portfolio: Portfolio = portfolio
        self.market: yf.Ticker = yf.Ticker(market)
        self.rf = rf_rate

        self.raw_close, self.periodic_return, self.expected_returns = self._construct_data(horizon=horizon, lookback=lookback)
        self.market_returns = self._construct_market_data(horizon=horizon, lookback=lookback)

        self.cov_matrix = self._covariance_matrix()
        self.beta = self._beta()

    def _construct_data(self, horizon: int, lookback: int):
        data = self.portfolio.tickers.history(period=f"{lookback}d", interval='1d')
        data = data['Close']

        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        expected_returns = periodic_return.ewm(span=horizon).mean().iloc[-1]

        for i in expected_returns.index:
            self.portfolio.results['expected_returns'][i] = expected_returns[i]

        return data, periodic_return, expected_returns

    def _construct_market_data(self, horizon: int, lookback: int):
        data = self.market.history(period=f"{lookback}d", interval='1d')
        data = data['Close']

        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        return periodic_return

    def _covariance_matrix(self):
        return self.periodic_return.cov()

    def _beta(self):
        betas = []
        for stock in self.periodic_return.columns:
            b = np.cov(self.periodic_return[stock], self.market_returns)[0][1] / np.var(self.market_returns, ddof=1)
            self.portfolio.results['beta'][stock] = b
            betas.append(b)
        return betas

    def optimize(self, target_return: float, *, bounds: tuple[float, float] = (0.0, 1.0), with_beta: bool = True):
        mu = np.array(self.expected_returns.values)
        beta = np.array(self.beta)

        def _objective_no_beta(weights):
            return weights @ self.cov_matrix @ weights

        def _objective_with_beta(weights):
            portfolio_variance = weights @ self.cov_matrix @ weights
            beta_penalty = sum((weights * beta))

            return portfolio_variance + 0.1 * beta_penalty

        if with_beta:
            objective = _objective_with_beta
        else:
            objective = _objective_no_beta

        n = len(self.portfolio)
        initial_guess = np.array([1/n for _ in range(n)])
        constraints = np.array([{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.sum(x * mu) - target_return}])

        opt = optimize.minimize(objective, initial_guess, constraints=constraints, bounds=[bounds for _ in range(n)])

        for i in range(len(opt.x)):
            self.portfolio.results['weights'][self.portfolio[i]] = opt.x[i]

        return opt.x
