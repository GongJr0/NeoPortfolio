import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import optimize
import matplotlib.pyplot as plt
import plotly

import warnings

from CustomTypes import StockSymbol, Portfolio, Days
from typing import cast, Optional

class Markowitz:

    def __init__(self, portfolio: Portfolio,
                 market: StockSymbol,
                 horizon: Days = 21,
                 lookback: Days = 252,
                 rf_rate: float = 0.05717):

        # Portfolio, market, and environment definition
        self.portfolio: Portfolio = portfolio
        self.market: yf.Ticker = yf.Ticker(market)
        self.rf = rf_rate

        # Portfolio data construction
        self.raw_close, self.periodic_return, self.expected_returns, self.volatility = self._construct_data(horizon=horizon, lookback=lookback)

        # Market data construction
        self.market_returns, self.market_volatility = self._construct_market_data(horizon=horizon, lookback=lookback)

        # Portfolio statistics
        self.cov_matrix = self._covariance_matrix()
        self.beta = self._beta()

    def _construct_data(self, horizon: Days, lookback: Days) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get and process historical data for the portfolio stocks.
        :param horizon: investment horizon in days
        :param lookback: number of days to look back
        :return: historical price data, periodic returns (per horizon), expected returns, and volatility
        """
        now = datetime.now()
        start = now - timedelta(days=lookback)

        data = self.portfolio.tickers.history(start=start, end=now, interval='1d')
        data = data['Close']

        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        expected_returns = periodic_return.ewm(span=horizon).mean().iloc[-1]
        volatility = periodic_return.std()

        for i in expected_returns.index:
            self.portfolio.results['expected_returns'][i] = expected_returns[i]
            self.portfolio.results['volatility'][i] = volatility[i]
            self.portfolio.results['sharpe_ratio'][i] = (expected_returns[i] - self.rf) / volatility[i]

        return data, periodic_return, expected_returns, volatility

    def _construct_market_data(self, horizon: Days, lookback: Days) -> tuple[pd.Series, pd.Series]:
        """
        Get and process historical data for the market.
        :param horizon: investment horizon in days
        :param lookback: number of days to look back
        :return: periodic returns (per horizon) and volatility
        """
        now = datetime.now()
        start = now - timedelta(days=lookback)

        data = self.market.history(start=start, end=now, interval='1d')
        data = data['Close']

        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        volatility = periodic_return.std()

        return periodic_return, volatility

    def _covariance_matrix(self) -> pd.DataFrame:
        """
        Calculate the covariance matrix for the portfolio.
        :return: covariance matrix
        """
        return self.periodic_return.cov()

    def _beta(self) -> list[np.float64]:
        """
        Calculate the beta for each stock in the portfolio.
        :return: list of betas
        """
        betas = []
        for stock in self.periodic_return.columns:
            b: np.float64 = np.cov(self.periodic_return[stock], self.market_returns)[0][1] / np.var(self.market_returns, ddof=1) # type: ignore
            self.portfolio.results['beta'][stock] = b
            betas.append(b)
        return betas

    def optimize(self, target_return: float, *,
                 bounds: tuple[float, float] = (0.0, 1.0),
                 additional_constraints: tuple[dict] = (),
                 with_beta: bool = True) -> tuple[dict[StockSymbol, float], optimize.OptimizeResult]:
        """
        Optimize the portfolio weights to achieve a target return.
        :param target_return: target return
        :param bounds: upper and lower bounds for the weights
        :param additional_constraints: additional constraints
        :param with_beta: include beta in the optimization
        :return: optimized weights and optimization results
        """
        mu = np.array(self.expected_returns.values)
        beta = np.array(self.beta)

        def _objective_no_beta(weights):
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            return_deviation = (mu @ weights - target_return) ** 2
            return portfolio_variance + return_deviation

        def _objective_with_beta(weights):
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            portfolio_beta = weights @ beta
            market_beta = 1  # cov(market, market) / var(market) = 1
            beta_penalty = (portfolio_beta - market_beta) ** 2
            return_penalty = (mu @ weights - target_return) ** 2

            return portfolio_variance + beta_penalty + return_penalty

        if with_beta:
            objective = _objective_with_beta
        else:
            objective = _objective_no_beta

        n = len(self.portfolio)
        initial_guess = np.array([1/n for _ in range(n)])
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.sum(x * mu) - target_return},
                       *additional_constraints]

        opt = optimize.minimize(objective, initial_guess, constraints=constraints, bounds=[bounds for _ in range(n)])  # type: ignore

        for i in range(len(opt.x)):
            self.portfolio.results['weights'][self.portfolio[i]] = opt.x[i].round(4)

        if opt.success:
            return {self.portfolio[i]: opt.x[i].round(4) for i in range(len(opt.x))}, opt
        else:
            warnings.warn(f"Optimization for the portfolio {self.portfolio} did not converge.", category=UserWarning)

    def efficient_frontier(self, n: int = 1000, *, save: bool = False) -> None:
        """
        Plot the efficient frontier.
        :param n: number of points to plot
        :param save: save the plot
        """
        mu = self.expected_returns.values
        mus = np.linspace(mu.min(), mu.max(), n)
        sigmas_with_beta = np.zeros(n)
        sigmas_no_beta = np.zeros(n)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Efficient Frontier (With Beta)
        weights_record_beta = []
        for i in range(n):
            target_return = mus[i]
            weights, _ = self.optimize(target_return)
            weights = np.array(list(weights.values()))
            weights_record_beta.append(weights)
            returns, volatility = weights @ mu, np.sqrt(weights @ self.cov_matrix @ weights)
            sigmas_with_beta[i] = volatility

        scatter1 = ax[0].scatter(
            sigmas_with_beta, mus, c=(mus - self.rf) / sigmas_with_beta, cmap='viridis'
        )
        ax[0].set_title("Efficient Frontier (With Beta)")
        ax[0].set_xlabel("Volatility")
        ax[0].set_ylabel("Return")
        ax[0].grid(True)

        # Efficient Frontier (No Beta)
        weights_record_no_beta = []
        for i in range(n):
            target_return = mus[i]
            weights, _ = self.optimize(target_return, with_beta=False)
            weights = np.array(list(weights.values()))
            weights_record_no_beta.append(weights)
            returns, volatility = weights @ mu, np.sqrt(weights @ self.cov_matrix @ weights)
            sigmas_no_beta[i] = volatility

        scatter2 = ax[1].scatter(
            sigmas_no_beta, mus, c=(mus - self.rf) / sigmas_no_beta, cmap='viridis'
        )
        ax[1].set_title("Efficient Frontier (No Beta)")
        ax[1].set_xlabel("Volatility")
        ax[1].set_ylabel("Return")
        ax[1].grid(True)

        # Add shared colorbar
        cbar = fig.colorbar(scatter1, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label("Sharpe Ratio")

        if save:
            plt.savefig("efficient_frontier.png")
        else:
            plt.show()