from BtStrategy import BtStrategy

import pandas as pd
import numpy as np

from datetime import datetime as dt

from typing import Optional, Literal, Generator

class BtEngine:
    def __init__(self, portfolio_close: pd.DataFrame, strategy: BtStrategy):
        self.cash: int = 100_000

        self.min_trade_price = 1000  # At least 1000$ per transaction
        self.trade_proportion = 0.05 # baseline trade proportion: 5% of liquid assets

        self.price_data = portfolio_close

        self.sma_period = 5
        self.lma_period = 20

        self._iter = self._iterate()
        self.strat = strategy
        self.arg_signature = strategy.arg_signature

        self.arg_map = {
            'sma': [self._ma, [self.sma_period]],
            'lma': [self._ma, [self.lma_period]]
        }

        self.holdings = {stock: 0 for stock in self.price_data.columns}  # init with 0 holdings

    def _get_args(self):
        out = []
        for arg in self.arg_signature:
            out.append(self.arg_map[arg][0](*self.arg_map[arg][1]))
        return out

    def _ma(self, window):
        return self.price_data.rolling(window=window).mean()

    def _iterate(self) -> Generator[pd.Series, None, None]:
        """Generator that yields the next row of the price data"""
        for i in range(len(self.price_data)):
            yield self.price_data.iloc[i]

    def _trade(self,
               stock_name: str,
               stock_price: float,
               signal: Literal[-1, 0, 1]) -> None:
        if signal == 1:
            traded_cash = min(self.trade_proportion * self.cash, self.min_trade_price)  # Check for sufficient cash before buy
            self.cash -= traded_cash
            self.holdings[stock_name] += traded_cash / stock_price

        elif signal == -1:
            signal_magnitude = self.trade_proportion * self.holdings[stock_name]  # In shares (not USD)
            signal_magnitude = signal_magnitude if signal_magnitude > (self.min_trade_price/stock_price) else self.min_trade_price/stock_price

            self.cash += signal_magnitude * stock_price
            self.holdings[stock_name] -= signal_magnitude

        elif signal == 0:
            pass

        else:
            raise ValueError("Invalid signal value")

    def run(self, strat: Literal['crossover'] = 'crossover'):
        holdings = self.holdings
        cash = self.cash
        price_data = self.price_data.reset_index(drop=True)
        strat = self.strat
        objective = strat.objective

        args = self._get_args()

        signal_format = {
            stock: 0 for stock in price_data.columns
        }
        signals = {}
        for i in price_data.index:
            signal = signal_format.copy()
            for stock in signal_format.keys():
                signal[stock] = objective(*[arg[stock] for arg in args], index=i)
            signals[i] = signal

        for i in signals.keys():
            for stock in signals[i].keys():
                self._trade(stock, price_data.loc[i, stock], signals[i][stock])

        return self.cash, self.holdings