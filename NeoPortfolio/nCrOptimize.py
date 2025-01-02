from NeoPortfolio import Portfolio
from NeoPortfolio.ReturnPred import ReturnPred
from NeoPortfolio.nCrEngine import nCrEngine
import yfinance as yf

import numpy as np
from scipy.optimize import minimize

from PortfolioCache import PortfolioCache
import datetime as dt

from typing import Optional, Any


class nCrOptimize(nCrEngine):
    """Find the optimal portfolio for a target return in a combination pool created from index components."""
    def __init__(self,
                 market: str,
                 n: int,
                 target_return: float = 0.1,
                 horizon: int = 21,
                 lookback: int = 252,
                 max_pool_size: Optional[int] = None) -> None:

        super().__init__(market, n, horizon, lookback, max_pool_size, target_return)

        self.portfolio_cache = PortfolioCache()
        self.portfolios = self._get_portfolios()
        self.market_returns = self._get_market()

    def _get_portfolios(self) -> list:
        """
        Get Portfolio objects from string combinations.
        """
        portfolios = []
        for comb in self.ncr_gen:
            portfolio = Portfolio(*comb)
            portfolios.append(portfolio)
        return portfolios

    def _get_market(self) -> None:
        """Get the market returns for the given horizon."""
        start = dt.datetime.today() - dt.timedelta(days=self.lookback)
        start = start.date()
        end = dt.datetime.today().date()

        market_close = yf.Ticker(self.market).history(start=start, end=end)["Close"]
        market_returns = (market_close - market_close.shift(self.horizon)) / market_close.shift(self.horizon)
        market_returns = market_returns.dropna()

        return market_returns

    def _iteration_optimize(self, portfolio) -> dict[str, Any]:
        """Optimization function ran in parallel iteration of portfolios.

        :param portfolio: Portfolio object
        :return: tuple of Portfolio object and optimized weights
        """
        # Cache query
        cached = self.portfolio_cache.get(portfolio)
        if cached:
            return cached

        # Get the historical data for the portfolio
        periodic_returns = self.periodic_returns.loc[:, list(portfolio)]
        historical_close = self.historical_close.loc[:, list(portfolio)]

        return_preds = ReturnPred(historical_close).all_stocks_pred(comb=True)
        expected_returns = np.array([return_dict['expected_return'] for return_dict in return_preds.values()])

        cov_matrix = periodic_returns.cov()

        betas = [np.cov(periodic_returns[stock], self.market_returns)[0, 1] for stock in portfolio]

        # Constraints
        def check_sum(weights):
            return np.sum(weights) - 1

        def target_return(weights):
            return np.dot(weights, expected_returns) - self.target

        initial_guess = np.array([1/len(portfolio) for _ in range(len(portfolio))])
        bounds = [(0.0, 1.0) for _ in range(len(portfolio))]
        constraints = [{'type': 'eq', 'fun': check_sum},
                       {'type': 'eq', 'fun': target_return}]

        # Objective function
        def objective(weights):
            return weights.T @ cov_matrix @ weights

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

        out = {
            "portfolio": " - ".join(portfolio),
            "weights": result.x.round(4).tolist(),  # Convert to list
            "return": np.dot(result.x, expected_returns).round(4),
            "portfolio_variance": result.fun.round(4),
            "expected_returns": expected_returns.tolist(),  # Convert to list
            "cov_matrix": cov_matrix.values.tolist(),  # Convert to a nested list
            "betas": betas
        }

        self.portfolio_cache.cache(portfolio, self.target, out)
        return out

    def optimize_space(self) -> list[dict[str, Any]]:
        """
        Optimize the combination space.

        :return: List of optimized portfolios (best to worst)
        """
        results = [self._iteration_optimize(portfolio) for portfolio in self.portfolios]
        results = sorted(results, key=lambda x: x["portfolio_variance"])

        return results
