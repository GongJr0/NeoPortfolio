import numpy as np
import pandas as pd

from math import log

from typing import Literal


class BtStrategy:
    def __init__(self,
                 strat: Literal['crossover', 'rsi_ma', 'rsi_ewma']  # More literals will be added as the strategies are implemented
                 ) -> None:
        self.strat = strat
        self._arg_signature = {
            'crossover': ['sma', 'lma'],  # prev averages can be accessed with pd.Series.iloc
            'rsi_ma': ['diff', 'window'],
            'rsi_ewma': ['diff', 'window']
        }
        self._func_map = {
            'crossover': self._crossover,
            'rsi_ma': self._rsi_ma,
            'rsi_ewma': self._rsi_ewma
        }

        self._signal_scalers = {
            self._crossover: self._no_scale,
            self._rsi_ma: self._rsi_strength_exp,
            self._rsi_ewma: self._rsi_strength_exp
        }

        self.objective = self._func_map[self.strat]
        self.signal_scaler = self._signal_scalers[self.objective]

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
            diff: pd.DataFrame,
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
            diff: pd.DataFrame,
            window: int,
            *,
            index: int
    ) -> tuple[int, float]:

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

    def _rsi_strength_exp(self, signal: int, score: float) -> float:

        buy = self.rsi_buy_threshold
        sell = self.rsi_sell_threshold

        if signal == 1:
            return np.exp(((buy - score)/buy) - 1) / (np.exp(1) - 1)

        elif signal == -1:
            return - np.exp(((score - sell)/(100 - sell)) - 1) / (np.exp(1) - 1)
        
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
