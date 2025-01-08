import numpy as np
import pandas as pd

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
        self.objective = self.func_map[self.strat]

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
            ) -> int:

        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[index] > 70:
            return -1
        elif rsi.iloc[index] < 30:
            return 1
        else:
            return 0

    @staticmethod
    def _rsi_ewma(
            diff: pd.DataFrame,
            window: int,
            *,
            index: int
    ) -> int:

        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)

        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[index] > 70:
            return -1
        elif rsi.iloc[index] < 30:
            return 1
        else:
            return 0

    @property
    def arg_signature(self) -> list:
        return self._arg_signature[self.strat]
