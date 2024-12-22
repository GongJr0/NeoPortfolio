import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import optimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import plotly

import warnings

from CustomTypes import StockSymbol, Portfolio, Days
from typing import cast, Optional, Literal

from _class.Sentiment import Sentiment

class Markowitz:

    def __init__(self, portfolio: Portfolio,
                 market: StockSymbol,
                 horizon: Days = 21,
                 lookback: Days = 252,
                 rf_rate: float = 0.05717):

        # Sentiment Analysis Module
        self.sentiment = Sentiment()

        # Portfolio, market, and environment definition
        self.portfolio: Portfolio = portfolio
        self.names = {symbol: portfolio.tickers.tickers[symbol].info['shortName'] for symbol in portfolio}

        self.market: yf.Ticker = yf.Ticker(market)
        self.market_name: str = self.market.info['shortName']

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

        # Get historical data
        data = self.portfolio.tickers.history(start=start, end=now, interval='1d')
        data = data['Close']

        # Calculate periodic returns
        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        # Calculate expected returns and volatility
        expected_returns = periodic_return.ewm(span=horizon).mean().iloc[-1]
        volatility = periodic_return.std()

        # Adjust return with sentiment analysis
        for stock in self.portfolio:
            sentiment_score = self.sentiment.get_sentiment(f"{self.names[stock]} Stock", n=10, lookback=horizon)
            self.portfolio.results['sentiment'][stock] = round(sentiment_score, 4)
            expected_returns[stock] = expected_returns[stock] * (1 + 0.5 * (sentiment_score - 0.5))  # 50% weight on sentiment

        for i in expected_returns.index:
            self.portfolio.results['expected_returns'][i] = expected_returns[i].round(4)
            self.portfolio.results['volatility'][i] = volatility[i].round(4)
            self.portfolio.results['sharpe_ratio'][i] = ((expected_returns[i] - self.rf) / volatility[i]).round(4)

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
            self.portfolio.results['beta'][stock] = b.round(4)
            betas.append(b)
        return betas

    def optimize(self,
                 target: float, *,
                 target_input: Literal['return', 'volatility'] = 'return',
                 bounds: tuple[float, float] = (0.0, 1.0),
                 additional_constraints: tuple[dict] = (),
                 with_beta: bool = True,
                 record: bool = True) -> tuple[dict[str, float], OptimizeResult]:
        """
        Optimize the portfolio weights to achieve a target return.
        :param target_input: optimization input (return or volatility)
        :param target: target return
        :param bounds: upper and lower bounds for the weights
        :param additional_constraints: additional constraints
        :param with_beta: include beta in the optimization
        :param record: record the optimized weights and results (disabled when optimizing for efficient frontier)
        :return: optimized weights and optimization results
        """
        mu = np.array(self.expected_returns.values)
        beta = np.array(self.beta)

        def _objective_no_beta(weights) -> float:
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            return_deviation = (mu @ weights - target) ** 2

            return portfolio_variance + return_deviation

        def _objective_with_beta(weights) -> float:
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            portfolio_beta = weights @ beta
            market_beta = 1  # cov(market, market) / var(market) = 1
            beta_penalty = (portfolio_beta - market_beta) ** 2
            return_penalty = (mu @ weights - target) ** 2

            return portfolio_variance + beta_penalty + return_penalty

        def _objective_no_beta_volatility(weights) -> float:
            return -(mu @ weights)

        def _objective_with_beta_volatility(weights) -> float:
            portfolio_return = mu @ weights
            portfolio_beta = weights @ beta
            market_beta = 1  # cov(market, market) / var(market) = 1
            beta_penalty = (portfolio_beta - market_beta) ** 2

            return -(portfolio_return) + beta_penalty

        if target_input == 'return':
            objective = _objective_with_beta if with_beta else _objective_no_beta
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                           {'type': 'eq', 'fun': lambda x: mu @ x - target},
                           *additional_constraints]
        elif target_input == 'volatility':
            objective = _objective_with_beta_volatility if with_beta else _objective_no_beta_volatility
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                           {'type': 'eq', 'fun': lambda x: np.sqrt(x.T @ self.cov_matrix @ x) - target},
                           *additional_constraints]

        n = len(self.portfolio)
        initial_guess = np.array([1/n for _ in range(n)])

        opt = optimize.minimize(objective, initial_guess, constraints=constraints, bounds=[bounds for _ in range(n)])  # type: ignore

        if opt.success:

            if record:
                for i in range(len(opt.x)):
                    self.portfolio.results['weights'][self.portfolio[i]] = opt.x[i].round(4)

                self.portfolio.optimum_portfolio_info['target_return'] = mu @ opt.x
                self.portfolio.optimum_portfolio_info['weights'] = self.portfolio.results['weights']

                coef_of_variance = [(np.std(self.periodic_return[stock]) / np.mean(self.periodic_return[stock])).round(4)
                                    for stock in self.portfolio]

                self.portfolio.optimum_portfolio_info['risk_per_return'] = {self.portfolio[i]: coef_of_variance[i] for i in range(n)}

            return {self.portfolio[i]: opt.x[i].round(4) for i in range(len(opt.x))}, opt

        else:
            warnings.warn(f"Optimization for the portfolio {self.portfolio} did not converge.", category=UserWarning)
            return {self.portfolio[i]: 1/n for i in range(n)}, opt

    def efficient_frontier(self, target_input: Literal['return', 'volatility'], n: int = 1000, *,
                           save: bool = False) -> None:
        """
        Plot the efficient frontier.
        :param target_input: optimization input (return or volatility)
        :param n: number of points to plot
        :param save: save the plot
        """
        mu = self.expected_returns.values
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        if target_input == 'return':
            mus = np.linspace(mu.min(), mu.max(), n)
            sigmas_with_beta = np.zeros(n)
            sigmas_no_beta = np.zeros(n)

            # Efficient Frontier (With Beta)
            for i, target_return in enumerate(mus):
                weights, _ = self.optimize(target_return, target_input=target_input, record=False)
                weights = np.array(list(weights.values()))
                volatility = np.sqrt(weights @ self.cov_matrix @ weights)
                sigmas_with_beta[i] = volatility

            scatter1 = ax[0].scatter(
                sigmas_with_beta, mus, c=(mus - self.rf) / sigmas_with_beta, cmap='viridis'
            )
            ax[0].set_title("Efficient Frontier (With Beta)")
            ax[0].set_xlabel("Volatility")
            ax[0].set_ylabel("Return")
            ax[0].grid(True)

            # Efficient Frontier (No Beta)
            for i, target_return in enumerate(mus):
                weights, _ = self.optimize(target_return, with_beta=False, record=False)
                weights = np.array(list(weights.values()))
                volatility = np.sqrt(weights @ self.cov_matrix @ weights)
                sigmas_no_beta[i] = volatility

            scatter2 = ax[1].scatter(
                sigmas_no_beta, mus, c=(mus - self.rf) / sigmas_no_beta, cmap='viridis'
            )
            ax[1].set_title("Efficient Frontier (No Beta)")
            ax[1].set_xlabel("Volatility")
            ax[1].set_ylabel("Return")
            ax[1].grid(True)

        elif target_input == 'volatility':
            volatilities = np.linspace(0.01, np.sqrt(self.cov_matrix.max()), n)
            returns_with_beta = np.zeros(n)
            returns_no_beta = np.zeros(n)

            # Efficient Frontier (With Beta)
            for i, target_volatility in enumerate(volatilities):
                weights, _ = self.optimize(target_volatility, target_input=target_input, record=False)
                weights = np.array(list(weights.values()))
                returns = weights @ mu
                returns_with_beta[i] = returns

            scatter1 = ax[0].scatter(
                volatilities, returns_with_beta, c=(returns_with_beta - self.rf) / volatilities, cmap='viridis'
            )
            ax[0].set_title("Efficient Frontier (With Beta)")
            ax[0].set_xlabel("Volatility")
            ax[0].set_ylabel("Return")
            ax[0].grid(True)

            # Efficient Frontier (No Beta)
            for i, target_volatility in enumerate(volatilities):
                weights, _ = self.optimize(target_volatility, with_beta=False, record=False)
                weights = np.array(list(weights.values()))
                returns = weights @ mu
                returns_no_beta[i] = returns

            scatter2 = ax[1].scatter(
                volatilities, returns_no_beta, c=(returns_no_beta - self.rf) / volatilities, cmap='viridis'
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

