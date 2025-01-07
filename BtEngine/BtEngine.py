from BtStrategy import BtStrategy

import pandas as pd
import numpy as np

from datetime import datetime as dt

from collections import Generator
from typing import Optional

class BtEngine:
    def __init__(self, portfolio_close: pd.DataFrame, strategy: BtStrategy):
        self.cash: int = 100_000

        self.min_trade_price = 1000  # At least 1000$ per transaction
        self.trade_proportion = 0.05 # baseline trade proportion: 5% of liquid assets

        self.price_data = portfolio_close

        self.sma_period = 10
        self.lma_period = 50

        self._iter = self._iterate()
        self.strat = strategy
        self.arg_signature = strategy.arg_signature

        self.arg_map = {
            'sma': [self._ma, [self.sma_period]],
            'lma': [self._ma, [self.lma_period]]
        }


    def _get_args(self):
        out = {}
        for arg in self.arg_signature:
            out[arg] = self.arg_map[arg][0](*self.arg_map[arg][1])
        return out

    def _ma(self, window):
        return self.price_data.rolling(window=window).mean()

    def _iterate(self) -> Generator[pd.Series, None, None]:
        """Generator that yields the next row of the price data"""
        for i in range(len(self.price_data)):
            yield self.price_data.iloc[i]

    def _trade(self):
        ...

    def run(self):
        ...