from NeoPortfolio import Portfolio
from NeoPortfolio.ReturnPred import ReturnPred
from ncrEngine.nCrEngine import CombinationEngine
from PortfolioCache import PortfolioCache

from joblib import Parallel, delayed
from scipy.optimize import minimize

import datetime as dt

import pandas as pd
import numpy as np

import yfinance as yf


class nCrOptimize(CombinationEngine):
    def __init__(self,
                 market: str,
                 n: int,
                 horizon: int = 21,
                 lookback: int = 252,
                 target_return: float = 0.1) -> None:

        super().__init__(market, n, horizon, lookback, target_return)

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

    def _get_market(self):
        start = dt.today() - dt.timedelta(days=self.lookback)
        start = start.date()
        end = dt.today().date()

        market_close = yf.Ticker(self.market).history(start=start, end=end)["Close"]
        market_returns = (market_close - market_close.shift(self.horizon)) / market_close.shift(self.horizon)
        market_returns = market_returns.dropna()

        return market_returns

    def _iterative_optimize(self, portfolio):

        # Get the historical data for the portfolio
        periodic_returns = self.periodic_returns.loc[:, portfolio.components]
        historical_close = self.historical_close.loc[:, portfolio.components]

        expected_returns = ReturnPred(historical_close).all_stocks_pred()

        cov_matrix = periodic_returns.cov()
