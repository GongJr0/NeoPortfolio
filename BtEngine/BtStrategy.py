import numpy as np
import pandas as pd

from math import log

from typing import Literal
class BtStrategy:
    def __init__(self,
                 strat: Literal['crossover']  # More literals will be added as the strategies are implemented
                 ) -> None:
        self.strat = strat
        self._arg_signature = {
            'crossover': ['sma', 'lma'],  # prev averages can be accessed with pd.Series.iloc
            'rsi_ma': ['diff', 'window'],
            'rsi_ewma': ['diff', 'window']
        }
        self.func_map = {
            'crossover': self._crossover,
            'rsi_ma': self._rsi_ma,
            'rsi_ewma': self._rsi_ewma
        }

        self.signal_scalers = {
            self._rsi_ma: self.rsi_strength_exp,
            self._rsi_ewma: self.rsi_strength_exp
        }

        self.objective = self.func_map[self.strat]
        self.signal_scaler = self.signal_scalers[self.objective]

        self.rsi_buy_threshold = 30
        self.rsi_sell_threshold = 70

    def set_thresholds(self, threshold: int) -> None:
        self.rsi_buy_threshold = threshold
        self.rsi_sell_threshold = 100 - threshold

    @staticmethod
    def _crossover(sma: pd.DataFrame,
                  lma: pd.DataFrame,
                  *,
                  index: int  # Enumerate date indices to support iloc. BtEngine._iterate won't traverse DateIndex
                  ) -> int:

        curr_sma = sma.iloc[index]
        curr_lma = lma.iloc[index]

        prev_sma = sma.iloc[index-1]
        prev_lma = lma.iloc[index-1]

        if curr_sma > curr_lma and prev_sma <= prev_lma:
            return 1
        elif curr_sma < curr_lma and prev_sma >= prev_lma:
            return -1
        else:
            return 0

    @staticmethod
    def _rsi_ma(
            diff: pd.DataFrame,
            window: int,
            *,
            index: int
            ) -> tuple[float, float]:
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

    @staticmethod
    def _rsi_ewma(
            diff: pd.DataFrame,
            window: int,
            *,
            index: int
    ) -> tuple[float, float]:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[index] > sell:
            return -1, rsi.iloc[index]
        elif rsi.iloc[index] < buy:
            return 1, rsi.iloc[index]
        else:
            return 0, rsi.iloc[index]

    def rsi_strength_linear(self, signal: int, score: float) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            return (buy-score)/buy

        elif signal == -1:
            return (score-sell)/buy

        elif signal == 0:
            return 0

    def rsi_strength_exp(self, signal: int, score: float) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            upper = np.exp(-0.05*(score-buy))
            lower = np.exp(-0.05*(0-buy))
            return upper/lower

        elif signal == -1:
            upper = np.exp(-0.05*(sell-score))
            lower = np.exp(-0.05*(sell-100))
            return upper/lower

        elif signal == 0:
            return 0

    def rsi_strength_log(self, signal: int, score: float) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            upper = log(1 + (buy - score)/buy)
            lower = log(2)

            return upper/lower

        elif signal == -1:
            upper = log(1 + (score-sell)/(100-sell))
            lower = log(2)

            return upper/lower

        elif signal == 0:
            return 0



    @property
    def arg_signature(self) -> list:
        return self._arg_signature[self.strat]
