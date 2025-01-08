from BtStrategy import BtStrategy

import pandas as pd
import numpy as np

from datetime import datetime as dt

from typing import Optional, Literal, Generator

class BtEngine:
    def __init__(self, portfolio_close: pd.DataFrame, strategy: BtStrategy):
        self.cash: int = 100_000

        self.min_trade_price = 1000  # At least 1000$ per transaction
        self.trade_proportion = 0.1  # baseline trade proportion: 10% of liquid assets

        self.price_data = portfolio_close
        self.dt_index = portfolio_close.index

        self.sma_period = 16
        self.lma_period = 128

        self._iter = self._iterate()
        self.strat = strategy
        self.arg_signature = strategy.arg_signature

        self.arg_map = {
            'sma': [self._ma, [self.sma_period]],
            'lma': [self._ma, [self.lma_period]],
            'diff': [self._diff, []],
            'window': [self._get_window, []]
        }

        self.holdings = {stock: 0 for stock in self.price_data.columns}  # init with 0 holdings
        self._all_signals = None

        # Processed signals dict
        self.buy = None
        self.sell = None



    def _get_args(self):
        out = []
        for arg in self.arg_signature:
            out.append(self.arg_map[arg][0](*self.arg_map[arg][1]))
        return out
    @staticmethod
    def _arg_indexer(arg, locator):
        if isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series):
            return arg[locator]
        else:
            return arg

    def _ma(self, window):
        return self.price_data.rolling(window=window).mean()

    def _diff(self):
        return self.price_data.diff()

    def _get_window(self):
        return self.sma_period

    def _iterate(self) -> Generator[pd.Series, None, None]:
        """Generator that yields the next row of the price data"""
        for i in range(len(self.price_data)):
            yield self.price_data.iloc[i]

    def _process_signals(self):
        assert self._all_signals is not None, "No signals to process"

        buy = {index: [stock[0] for stock in signal.items() if stock[1] == 1]
               for index, signal in self._all_signals.items()}

        sell = {index: [stock[0] for stock in signal.items() if stock[1] == -1] for index,
                signal in self._all_signals.items()}

        buy = {self.dt_index[key].strftime(format='%d %b, %y - %H:%M'): value for key, value in buy.items() if value}
        sell = {self.dt_index[key].strftime(format='%d %b, %y - %H:%M'): value for key, value in sell.items() if value}

        return buy, sell

    def _trade(self,
               stock_name: str,
               stock_price: float,
               signal: Literal[-1, 0, 1]) -> None:

        if signal == 1:
            expected_trade = self.trade_proportion * self.cash
            min_trade = self.min_trade_price
            traded_cash = min(expected_trade, min_trade, self.cash)  # Check for cash
                                                                     # balance before buy

            self.cash -= traded_cash
            self.holdings[stock_name] += traded_cash / stock_price

        elif signal == -1:
            expected_trade = self.trade_proportion * self.holdings[stock_name] * stock_price
            min_trade = self.min_trade_price
            traded_cash = min(expected_trade, min_trade, self.holdings[stock_name] * stock_price)
            traded_stock = traded_cash / stock_price

            if self.holdings[stock_name] < traded_stock:
                traded_cash = self.holdings[stock_name] * stock_price

            self.cash += traded_cash
            self.holdings[stock_name] -= traded_stock

        elif signal == 0:
            pass

        else:
            raise ValueError("Invalid signal value")

    def run(self, strat: Literal['crossover'] = 'crossover'):

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
                signal[stock] = objective(*[self._arg_indexer(arg, stock) for arg in args], index=i)
            signals[i] = signal

        for i in signals.keys():
            for stock in signals[i].keys():
                self._trade(stock, price_data.loc[i, stock], signals[i][stock])

        self._all_signals = signals
        self.buy, self.sell = self._process_signals()

        asset_distribution = {
            'Liquid': self.cash,
            'Holdings': self.holdings
        }

        last_prices = price_data.iloc[-1]
        assets = list(self.holdings.values())

        cash_equivalent = self.cash + np.sum(assets * last_prices)

        out = {
            'Total Value': cash_equivalent,
            'Asset Distribution': asset_distribution
        }

        return out
