from .Portfolio import Portfolio
from .ReturnPred import ReturnPred

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scipy import optimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

import warnings

from .CustomTypes import StockSymbol, Days, IndexSymbol
from typing import Optional, Literal
from os import PathLike

from .Sentiment import Sentiment


class Markowitz:

    def __init__(self, portfolio: Portfolio,
                 market: IndexSymbol,
                 horizon: Days = 21,
                 lookback: Days = 252,
                 rf_rate_pa: Optional[float] = None,
                 api_key_path: Optional[PathLike | str] = ...,
                 api_key_var: Optional[str] = ...,
                 ) -> None:

        assert horizon > 0, "Horizon must be a positive integer."
        assert lookback > 0, "Lookback must be a positive integer."
        assert lookback > horizon, "Lookback must be greater than the horizon."
        assert rf_rate_pa is None or rf_rate_pa >= 0, "Risk-free rate must be a non-negative float."

        # Sentiment Analysis Module
        if api_key_var and api_key_path:
            self.sentiment = Sentiment(api_key_path=api_key_path, api_key_var=api_key_var)
        else:
            self.sentiment = None

        # Portfolio, market, and environment definition
        self.portfolio: Portfolio = portfolio
        self.names = {symbol: portfolio.tickers.tickers[symbol].info['shortName'] for symbol in portfolio}

        self.market: yf.Ticker = yf.Ticker(market)
        self.market_name: str = self.market.info['shortName']

        if rf_rate_pa is None:
            us_bond_10y = yf.Ticker('^TNX')
            us_bond_10y = us_bond_10y.history(period='1d')['Close']
            rf_rate_pa = us_bond_10y.iloc[-1] / 100

        self.rf = (1 + (rf_rate_pa / 2))**(lookback / 365) - 1  # semi-annual compounding with ACT/365 (approximate for US Treasury Bonds)

        # Portfolio data construction
        self.raw_close, self.periodic_return, self.expected_returns, self.volatility = self._construct_data(horizon=horizon, lookback=lookback)

        # Market data construction
        self.market_returns, self.market_volatility, self.rm = self._construct_market_data(horizon=horizon, lookback=lookback)

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

        return_pred = ReturnPred(data, horizon)
        expected_returns = return_pred.all_stocks_pred()

        return_dict = {}
        for stock in expected_returns.keys():
            if expected_returns[stock]['success']:
                return_dict[stock] = expected_returns[stock]['expected_return']
            else:
                print(f"Model performance for {stock} is low. Falling back to historical EWMA.")
                return_dict[stock] = expected_returns[stock]['expected_return'] # EWMA is already calculated if success == False

        price_dict = {stock: expected_returns[stock]['expected_price'] for stock in expected_returns.keys()}
        print(f"Expected Prices at {horizon} days:\n{price_dict}")

        return_series = pd.Series(return_dict.values(), index=return_dict.keys())

        # Calculate periodic returns
        periodic_return = (data - data.shift(horizon)) / data.shift(horizon)
        periodic_return = periodic_return.dropna()

        # Calculate expected returns and volatility
        expected_returns = return_series
        volatility = periodic_return.std()

        # Adjust return with sentiment analysis
        sentiment_period = min(30, horizon)

        if self.sentiment:
            for stock in self.portfolio:
                sentiment_score = self.sentiment.get_sentiment(f"{self.names[stock]} Stock", n=10, lookback=sentiment_period)
                self.portfolio.results['sentiment'][stock] = round(sentiment_score, 4)
                expected_returns[stock] = expected_returns[stock] * (1 + 0.33 * sentiment_score)  # 33% weight on sentiment

        if horizon >= 30:
            warnings.warn(f"Sentiment analysis is done for a maximum period of 30 days due to API limitations."
                          f"You will see this warning if the investment horizon is greater than 30 days.",
                          category=UserWarning)

        for i in expected_returns.index:
            self.portfolio.results['expected_returns'][i] = expected_returns[i].round(4)
            self.portfolio.results['volatility'][i] = volatility[i].round(4)
            self.portfolio.results['sharpe_ratio'][i] = ((expected_returns[i] - self.rf) / volatility[i]).round(4)

        return data, periodic_return, expected_returns, volatility

    def _construct_market_data(self, horizon: Days, lookback: Days) -> tuple[pd.Series, pd.Series, np.float64]:
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

        span = np.round(np.sqrt(horizon), 0) * 2
        rm = periodic_return.ewm(span=span).mean().iloc[-1]

        periodic_return.index = pd.to_datetime(periodic_return.index.date)
        common_index = periodic_return.index.intersection(self.periodic_return.index)

        periodic_return = periodic_return.loc[common_index]



        return periodic_return, volatility, rm

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
            try:
                b: np.float64 = np.cov(self.periodic_return[stock], self.market_returns)[0][1] / np.var(self.market_returns, ddof=1) # type: ignore
            except ZeroDivisionError:
                b = 0

            self.portfolio.results['beta'][stock] = b.round(4)
            betas.append(b)
        return betas

    def min_volatility(self) -> float:
        n = len(self.portfolio)
        initial_guess = np.array([1/n for _ in range(n)])

        def check_sum(weights):
            return np.sum(weights) - 1

        def objective(weights):
            return weights.T @ self.cov_matrix @ weights

        opt = optimize.minimize(objective,
                                initial_guess,
                                constraints={'type': 'eq', 'fun': check_sum},
                                bounds=[(0.0, 1.0) for _ in range(n)],
                                method='SLSQP')
        return np.sqrt(opt.fun)

    def optimize_return(self,
                        target_return: float,
                        *,
                        additional_constraints: Optional[list[dict]] = None,
                        include_beta: bool = True,
                        bounds: tuple[float, float] = (0.0, 1.0),
                        record: bool = True,
                        ) -> tuple[dict[str, float], OptimizeResult]:

        mu = np.array(self.expected_returns.values)
        betas = np.array(self.beta)
        n = len(mu)

        initial_guess = np.array([1/n for _ in range(n)])

        rm = self.rm
        rf = self.rf

        def check_sum(weights):
            return np.sum(weights) - 1

        def beta_return_constraint(weights):
            portfolio_return = mu @ weights
            beta_adjustment = (weights @ betas) * (rm - rf)

            return portfolio_return + beta_adjustment - target_return

        def no_beta_return_constraint(weights):
            return mu @ weights - target_return

        if include_beta:
            constraints = [{'type': 'eq', 'fun': check_sum},
                           {'type': 'eq', 'fun': beta_return_constraint}
                           ]

        elif not include_beta:
            constraints = [{'type': 'eq', 'fun': check_sum},
                           {'type': 'eq', 'fun': no_beta_return_constraint},
                           ]

        if additional_constraints:
            constraints += additional_constraints


        def objective(weights):
            return weights.T @ self.cov_matrix @ weights

        opt = optimize.minimize(objective,
                                initial_guess,
                                constraints=constraints,
                                bounds=[bounds for _ in range(n)]
                                )  # type: ignore

        if opt.success:

            if record:
                for i in range(len(opt.x)):
                    self.portfolio.results['weights'][self.portfolio[i]] = opt.x[i].round(4)

                self.portfolio.optimum_portfolio_info['target_return'] = mu @ opt.x
                self.portfolio.optimum_portfolio_info['target_volatility'] = np.sqrt(opt.x @ self.cov_matrix @ opt.x)
                self.portfolio.optimum_portfolio_info['weights'] = self.portfolio.results['weights']


            return {self.portfolio[i]: opt.x[i].round(4) for i in range(len(opt.x))}, opt

        else:
            warnings.warn(f"Optimization for the portfolio {self.portfolio} did not converge.", category=UserWarning)
            return {self.portfolio[i]: 1/n for i in range(n)}, opt

    def optimize_volatility(self,
                            target_volatility: float,
                            *,
                            additional_constraints: Optional[list[dict]] = None,
                            include_beta: bool = True,
                            bounds: tuple[float, float] = (0.0, 1.0),
                            record: bool = True,
                            ) -> tuple[dict[str, float], OptimizeResult]:

        mu = np.array(self.expected_returns.values)
        betas = np.array(self.beta)
        n = len(mu)

        initial_guess = np.array([1/n for _ in range(n)])

        rm = self.rm
        rf = self.rf

        def check_sum(weights):
            return np.sum(weights) - 1

        def volatility_constraint(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights) - target_volatility

        constraints = [{'type': 'eq', 'fun': check_sum},
                       {'type': 'eq', 'fun': volatility_constraint}
                       ]

        if additional_constraints:
            constraints += additional_constraints

        def objective_beta(weights):
            portfolio_return = mu @ weights
            beta_adjustment = (weights @ betas) * (rm - rf)

            return -(portfolio_return + beta_adjustment)

        def objective_no_beta(weights):
            return -mu @ weights

        if include_beta:
            objective = objective_beta

        elif not include_beta:
            objective = objective_no_beta

        opt = optimize.minimize(objective, initial_guess,
                                constraints=constraints,
                                bounds=[bounds for _ in range(n)]
                                )

        if opt.success:

                if record:
                    for i in range(len(opt.x)):
                        self.portfolio.results['weights'][self.portfolio[i]] = opt.x[i].round(4)

                    self.portfolio.optimum_portfolio_info['target_return'] = mu @ opt.x
                    self.portfolio.optimum_portfolio_info['target_volatility'] = np.sqrt(opt.x @ self.cov_matrix @ opt.x)
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
        beta = np.array(self.beta)
        mu = self.expected_returns.values
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Efficient Frontier (With Beta)
        if target_input == 'return':
            min_mu = mu.min() if mu.min() > 0 else 0
            max_mu = mu.max()
            mus = np.linspace(min_mu, max_mu, n)

            adjusted_mus = self.expected_returns.values - beta * (self.rm - self.rf)
            adjusted_min = adjusted_mus.min() if adjusted_mus.min() > 0 else 0
            adjusted_max = adjusted_mus.max()
            mus_beta = np.linspace(adjusted_min, adjusted_max, n)

            sigmas = np.zeros(n)
            sigmas_no_beta = np.zeros(n)

            for i, (target_w_beta, target_no_beta) in enumerate(zip(mus_beta, mus)):
                weights, _ = self.optimize_return(target_w_beta, record=False)
                weights = np.array(list(weights.values()))
                volatility = np.sqrt(weights @ self.cov_matrix @ weights)
                sigmas[i] = volatility

                weights_no_beta, _ = self.optimize_return(target_no_beta, include_beta=False, record=False)
                weights_no_beta = np.array(list(weights_no_beta.values()))
                volatility_no_beta = np.sqrt(weights_no_beta @ self.cov_matrix @ weights_no_beta)
                sigmas_no_beta[i] = volatility_no_beta

            scatter1 = ax[0].scatter(
                    sigmas, mus, c=(mus - self.rf) / sigmas, cmap='viridis'
            )

            scatter2 = ax[1].scatter(
                    sigmas_no_beta, mus, c=(mus - self.rf) / sigmas_no_beta, cmap='viridis'
            )

        if target_input == 'volatility':
            min_sigma = self.min_volatility() if self.min_volatility() > 0 else 0
            max_sigma = self.volatility.max()

            sigmas = np.linspace(min_sigma, max_sigma, n)
            mus_with_beta = np.zeros(n)
            mus_no_beta = np.zeros(n)

            for i, target_volatility in enumerate(sigmas):
                weights, _ = self.optimize_volatility(target_volatility, record=False)
                weights = np.array(list(weights.values()))
                mus_with_beta[i] = self.expected_returns.values @ weights

                weights_no_beta, _ = self.optimize_volatility(target_volatility, include_beta=False, record=False)
                weights_no_beta = np.array(list(weights_no_beta.values()))
                mus_no_beta[i] = self.expected_returns.values @ weights_no_beta

            scatter1 = ax[0].scatter(
                    sigmas, mus_with_beta, c=(mus_with_beta - self.rf) / sigmas, cmap='viridis'
            )

            scatter2 = ax[1].scatter(
                    sigmas, mus_no_beta, c=(mus_no_beta - self.rf) / sigmas, cmap='viridis'
            )

        ax[0].set_title("Efficient Frontier (With Beta)")
        ax[0].set_xlabel("Volatility")
        ax[0].set_ylabel("Return")
        ax[0].grid(True)

        ax[1].set_title("Efficient Frontier (No Beta)")
        ax[1].set_xlabel("Volatility")
        ax[1].set_ylabel("Return")
        ax[1].grid(True)

        fig.colorbar(scatter2, ax=ax[1], label='Sharpe Ratio')

        if save:
            plt.savefig(f"{' '.join(self.portfolio)}.png")

        plt.show()
