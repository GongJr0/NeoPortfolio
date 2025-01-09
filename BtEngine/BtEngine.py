from BtStrategy import BtStrategy

import pandas as pd
import numpy as np

from datetime import datetime as dt

from typing import Optional, Literal, Generator

class BtEngine:
    def __init__(self, portfolio_close: pd.DataFrame, strategy: BtStrategy):
        self.cash: int = 100_000

        self.min_trade_price = 1000  # At least 1000$ per transaction
        self.max_trade_proportion = 0.33  # 33% of the cash can be used for a single trade

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
        self.total_buys = None
        self.total_sells = None

    @staticmethod
    def _arg_indexer(arg, loc):
        if isinstance(arg, pd.DataFrame):
            return arg[loc]
        else:
            return arg

    def _get_args(self):
        out = []
        for arg in self.arg_signature:
            out.append(self.arg_map[arg][0](*self.arg_map[arg][1]))
        return out


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

        buy = {index: [(stock[0], stock[1][1]) for stock in signal.items() if stock[1][0] == 1]
               for index, signal in self._all_signals.items()}

        sell = {index: [(stock[0], stock[1][1]) for stock in signal.items() if stock[1][0] == -1] for index,
                signal in self._all_signals.items()}

        buy = {self.dt_index[key].strftime(format='%d %b, %y - %H:%M'): value for key, value in buy.items() if value}
        sell = {self.dt_index[key].strftime(format='%d %b, %y - %H:%M'): value for key, value in sell.items() if value}

        total_buys = sum([len(b) for b in buy.values()])
        total_sells = sum([len(s) for s in sell.values()])

        return buy, sell, total_buys, total_sells

    def _trade(self,
               stock_name: str,
               stock_price: float,
               signal: Literal[-1, 0, 1],
               signal_strength: float) -> None:
        if any([pd.isna(signal), pd.isna(stock_price), pd.isna(signal_strength)]):
            return

        if signal == 1:
            if self.cash <= 0:
                return

            trade_amount = self.max_trade_proportion * signal_strength * self.cash

            if trade_amount >= self.cash:
                self.holdings[stock_name] += self.cash / stock_price
                self.cash = 0
            else:
                self.holdings[stock_name] += trade_amount / stock_price
                self.cash -= trade_amount

        elif signal == -1:
            if self.holdings[stock_name] <= 0:
                return

            trade_amount = self.max_trade_proportion * signal_strength * self.holdings[stock_name]

            if trade_amount >= self.holdings[stock_name]:
                self.cash += self.holdings[stock_name] * stock_price
                self.holdings[stock_name] = 0
            else:
                self.cash += trade_amount * stock_price
                self.holdings[stock_name] -= trade_amount

        elif signal == 0:
            pass

        else:
            raise ValueError("Invalid signal value")

    def run(self, strat: Literal['crossover'] = 'crossover'):

        price_data = self.price_data.reset_index(drop=True)
        strat = self.strat
        objective = strat.objective
        scaler = strat.signal_scaler

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
                sig = signals[i][stock][0]
                mag = signals[i][stock][1]
                self._trade(stock, price_data.loc[i, stock], sig, scaler(sig, mag))

        self._all_signals = signals
        self.buy, self.sell, self.total_buys, self.total_sells = self._process_signals()

        asset_distribution = {
            'Liquid ($)': self.cash.__round__(2),
            'Holdings': self.holdings
        }

        last_prices = price_data.iloc[-1]
        assets = list(self.holdings.values())

        cash_equivalent = self.cash + np.sum(assets * last_prices)

        out = {
            'Total Value ($)': cash_equivalent.__round__(2),
            'Asset Distribution': asset_distribution
        }

        return out
