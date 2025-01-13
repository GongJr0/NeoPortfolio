from .BtStrategy import BtStrategy
from .BtStateRecord import record_state

import pandas as pd
import numpy as np

from contextlib import contextmanager


from typing import Optional, Literal, Generator
import matplotlib.pyplot as plt


class BtEngine:
    def __init__(self,
                 portfolio_close: pd.DataFrame,
                 strategy: BtStrategy,
                 *,
                 # Supplementary data for strategies
                 hi_lo: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
                 vol: Optional[pd.DataFrame] = None,
                 horizon: Optional[int] = None
                 ) -> None:

        self.history = None

        self.cash: float = 100_000.0

        self.max_trade_proportion = 0.33  # 33% of the cash can be used for a single trade

        self.price_data = portfolio_close
        self.dt_index = portfolio_close.index

        if hi_lo:
            self.hi, self.lo = hi_lo

        if vol:
            self.vol = vol

        self.sma_period = 16
        self.lma_period = 128
        self.horizon = horizon or 7

        self._iter = self._iterate()
        self.strat = strategy
        self._arg_signature = strategy.arg_signature

        self._arg_map = {
            'sma': [self._ma, [self.sma_period]],
            'lma': [self._ma, [self.lma_period]],
            'diff': [self._diff, []],
            'window': [self._get_window, []],
            'fib_percentile': [self._fib_retracement, [self.horizon]]
        }

        self.holdings = {stock: 0 for stock in self.price_data.columns}  # init with 0 holdings
        self._all_signals = None

        # Processed signals dict
        self.buy = None
        self.sell = None
        self.total_buys = None
        self.total_sells = None

        self.raw_signals = {}
        self.cash_history = {}

    @staticmethod
    def _arg_indexer(arg, loc):
        if isinstance(arg, pd.DataFrame):
            return arg[loc]
        else:
            return arg

    def _get_args(self):
        out = []
        for arg in self._arg_signature:
            out.append(self._arg_map[arg][0](*self._arg_map[arg][1]))
        return out

    def _ma(self, window):
        return self.price_data.rolling(window=window, min_periods=0).mean()

    def _diff(self):
        return self.price_data.diff()

    def _fib_retracement(self, horizon):
        assert isinstance(self.lo, (pd.DataFrame, pd.Series)) and isinstance(self.hi, (pd.DataFrame, pd.Series)), "Provide Hi/Lo data to use the fib retracement strategy"
        fib_df = pd.DataFrame()
        for stock in self.price_data.columns:
            p = self.price_data[stock]
            lo = self.lo[stock]
            hi = self.hi[stock]
            fib_df[stock] = (p - lo.rolling(horizon).min()) / (hi.rolling(horizon).max() - lo.rolling(horizon).min())

        return fib_df

    def _get_window(self):
        return self.sma_period

    def _get_horizon(self):
        return self.horizon

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

    def run(self) -> dict[str, float | dict]:
        price_data = self.price_data.reset_index(drop=True)
        strat = self.strat
        objective = strat.objective
        scaler = strat.signal_scaler

        args = self._get_args()
        signal_format = {stock: 0 for stock in price_data.columns}
        signals = {}

        with record_state() as recorder:
            for i in price_data.index:
                # Calculate signals
                signal = signal_format.copy()
                for stock in signal_format.keys():
                    signal[stock] = objective(*[self._arg_indexer(arg, stock) for arg in args], index=i)
                signals[i] = signal

                # Execute trades
                current_signals = {}
                for stock in signals[i].keys():
                    sig = signals[i][stock][0]
                    mag = signals[i][stock][1]
                    current_signals[stock] = (sig, mag)
                    self._trade(stock, price_data.loc[i, stock], sig, scaler(sig, mag))

                # Record state
                recorder.record(
                    iteration=i,
                    cash=self.cash,
                    holdings=self.holdings,
                    signals=current_signals,
                    current_prices=price_data.loc[i]
                )

        # Store recorded history in instance
        self.history = recorder.get_history()

        # Rest of the original method
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

    def plot_history(self):
        if self.history is None:
            raise ValueError("No history to plot, call run() before plotting")

        total_value = self.history['portfolio_value'].values
        cash_equivalent = self.history['cash'].values
        dt_index = self.dt_index

        def get_signal(_iter, signal: int):
            sigs = [1 if _iter[0] == signal else 0 for _iter in _iter.values()]
            return sum(sigs)

        buys = [get_signal(_iter, 1) for _iter in list(self._all_signals.values())]
        sells = [get_signal(_iter, -1) for _iter in list(self._all_signals.values())]

        cumulative_pl = ((total_value - 100_000) / 100_000) * 100

        window = self.horizon if self.strat.strat == 'fib_retracement' else self.sma_period

        fig, ax = plt.subplots(3, 2, figsize=(15, 12))

        # Total Value, liquid value line plot
        ax[0, 0].plot(dt_index, total_value, label='Total Value')
        ax[0, 0].plot(dt_index, cash_equivalent, label='Liquid Assets')
        ax[0, 0].set_title('Portfolio Value')
        ax[0, 0].set_ylabel('Cash Equivalent ($)')
        ax[0, 0].set_xlabel('Date')
        ax[0, 0].grid(True)
        ax[0, 0].tick_params(axis='x', rotation=45)
        ax[0, 0].legend()

        sell_ax = ax[0, 1].inset_axes([0.1, 0.1, 0.8, 0.35])  # [x, y, width, height]
        buy_ax = ax[0, 1].inset_axes([0.1, 0.55, 0.8, 0.35])  # [x, y, width, height]

        # Plot buys and sells in the respective inset axes
        sell_ax.plot(dt_index, sells, color='red', label=f'Sells')
        sell_ax.tick_params(axis='x', rotation=45)
        sell_ax.legend()

        buy_ax.plot(dt_index, buys, color='green', label=f'Buys')
        buy_ax.set_title('Buy/Sell Signals')
        buy_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        buy_ax.legend()

        pos = ax[0, 1].get_position()
        # Adjust the position by increasing the bottom margin
        ax[0, 1].set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height - 0.1])

        # Optionally, remove the original grid cell's axes for cleaner presentation
        ax[0, 1].axis('off')  # Optional: to hide the base axes (i.e., ax[0, 1])

        hist_keys = self.history.keys()
        holding_keys = [key.split("_")[-1] for key in hist_keys if 'holdings' in key]

        for stock in holding_keys:
            ax[1, 0].plot(dt_index, self.history[f'holdings_{stock}'], label=stock)

        ax[1, 0].set_title('Stock Holdings')
        ax[1, 0].set_ylabel('Shares')
        ax[1, 0].set_xlabel('Date')
        ax[1, 0].grid(True)
        ax[1, 0].tick_params(axis='x', rotation=45)
        ax[1, 0].legend()

        ax[1, 1].plot(dt_index, self.history['profit'])
        ax[1, 1].set_title('Profit')
        ax[1, 1].set_ylabel('Profit ($)')
        ax[1, 1].set_xlabel('Date')
        ax[1, 1].grid(True)
        ax[1, 1].tick_params(axis='x', rotation=45)

        ax[2, 0].plot(dt_index, cumulative_pl, label=f'Cumulative P/L ($)')
        ax[2, 0].set_title('Cumulative P/L')
        ax[2, 0].set_ylabel('Profit (%)')
        ax[2, 0].set_xlabel('Date')
        ax[2, 0].grid(True)
        ax[2, 0].tick_params(axis='x', rotation=45)

        ax[2, 1].axis('off')

        plt.tight_layout()
        plt.show()
