import numpy as np
import pandas as pd

from typing import Literal


class BtStrategy:
    def __init__(self,
                 strat: Literal['crossover', 'rsi_ma', 'rsi_ewma', 'fib_retracement']  # More literals will be added as
                 ) -> None:                                                             # the strategies are implemented

        self.strat = strat
        self._arg_signature = {
            'crossover': ['sma', 'lma'],  # prev averages can be accessed with pd.Series.iloc
            'rsi_ma': ['diff', 'window'],
            'rsi_ewma': ['diff', 'window'],
            'fib_retracement': ['fib_percentile']
        }
        self._func_map = {
            'crossover': self._crossover,
            'rsi_ma': self._rsi_ma,
            'rsi_ewma': self._rsi_ewma,
            'fib_retracement': self._fib_retracement
        }

        self._signal_scalers = {
            self._crossover: self._no_scale,
            self._rsi_ma: self._rsi_strength_log,
            self._rsi_ewma: self._rsi_strength_log,
            self._fib_retracement: self._fib_magnitude_log,
        }

        self.objective = self._func_map[self.strat]
        self.signal_scaler = self._signal_scalers[self.objective]

        self.rsi_buy_threshold = 30
        self.rsi_sell_threshold = 70

    def set_thresholds(self, threshold: int) -> None:
        self.rsi_buy_threshold = threshold
        self.rsi_sell_threshold = 100 - threshold

    @staticmethod
    def _crossover(sma: pd.Series,
                   lma: pd.Series,
                   *,
                   index: int  # Enumerate date indices to support index locator (iloc).
                   ) -> tuple[int, int]:

        curr_sma = sma.iloc[index]
        curr_lma = lma.iloc[index]

        prev_sma = sma.iloc[index-1]
        prev_lma = lma.iloc[index-1]

        if curr_sma > curr_lma and prev_sma <= prev_lma:
            return 1, 1
        elif curr_sma < curr_lma and prev_sma >= prev_lma:
            return -1, 1
        else:
            return 0, 1

    def _rsi_ma(
            self,
            diff: pd.Series,
            window: int,
            *,
            index: int
            ) -> tuple[int, float]:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[index] > sell:
            return -1, rsi.iloc[index]

        elif rsi.iloc[index] < buy:
            return 1, rsi.iloc[index]
        else:
            return 0, 0

    def _rsi_ewma(
            self,
            diff: pd.Series,
            window: int,
            *,
            index: int
    ) -> tuple[int, float]:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.ewm(span=window, adjust=False, min_periods=0).mean()
        avg_loss = loss.ewm(span=window, adjust=False, min_periods=0).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[index] > sell:
            return -1, rsi.iloc[index]
        elif rsi.iloc[index] < buy:
            return 1, rsi.iloc[index]
        else:
            return 0, rsi.iloc[index]

    @staticmethod
    def _fib_retracement(fib_percentile: pd.Series,
                         *,
                         index: int):
        fib_buy = 0.236
        fib_sell = 0.618
        percent = fib_percentile.iloc[index].round(3)

        # Use un-rounded percentile to avoid rounding errors in conditionals
        if fib_percentile.iloc[index] >= fib_sell:
            return -1, percent
        elif fib_percentile.iloc[index] <= fib_buy:
            return 1, percent
        else:
            return 0, percent

    @staticmethod
    def _fib_magnitude_lin(signal: int, level: float) -> float:
        fib_buy = 0.382
        fib_sell = 0.618

        if signal == 1:
            return 1 - (level / fib_buy)

        elif signal == -1:
            return (level - fib_sell) / fib_buy

        return 0

    @staticmethod
    def _fib_magnitude_exp(signal: int, level: float, k: float = 3) -> float:
        fib_buy = 0.382
        fib_sell = 0.618

        if signal == 1:
            return (1 - level / fib_buy) ** k
        elif signal == -1:
            return ((level - fib_sell) / (1 - fib_sell)) ** k
        return 0

    @staticmethod
    def _fib_magnitude_log(signal: int, level: float, k=20) -> float:
        fib_buy = 0.382
        fib_sell = 0.618

        buy_mid = fib_buy / 2
        sell_mid = (1 + fib_sell) / 2

        if signal == 1:
            return 1 - 1 / (1 + np.exp(-k * (level - buy_mid)))
        elif signal == -1:
            return 1 / (1 + np.exp(-k * (level - sell_mid)))
        return 0

    @staticmethod
    def _no_scale(signal: int, score: float) -> float:
        return 1

    def _rsi_strength_lin(self, signal: int, score: float) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            return (buy-score)/buy

        elif signal == -1:
            return (score-sell)/buy

        elif signal == 0:
            return 0

    def _rsi_strength_exp(self, signal: int, score: float, k: int = 3) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            return (1 - score / buy) ** k

        elif signal == -1:
            return ((score - sell) / (100 - sell)) ** k

        elif signal == 0:
            return 0

    def _rsi_strength_log(self, signal: int, score: float, k=0.1) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold
        buy_reference = buy / 2
        sell_reference = sell + (100 - sell) / 2

        if signal == 1:
            return 1 / (1 + np.exp(k * (score - buy_reference)))

        elif signal == -1:
            return 1 / (1 + np.exp(k * (sell_reference - score)))
            
        elif signal == 0:
            return 0
        
    @property
    def arg_signature(self) -> list:
        return self._arg_signature[self.strat]
