import numpy as np
import pandas as pd

from typing import Literal
class BtStrategy:
    def __init__(self,
                 strat: Literal['crossover']  # More literals will be added as the strategies are implemented
                 ) -> None:
        self.strat = strat
        self._arg_signature = {
            'crossover': ['sma', 'lma']  # prev averages can be accessed with pd.Series.iloc
        }
        self.func_map = {
            'crossover': self.crossover
        }
        self.objective = self.func_map[self.strat]

    @staticmethod
    def crossover(sma: pd.Series,
                  lma: pd.Series,
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

    @property
    def arg_signature(self) -> list:
        return self._arg_signature[self.strat]
