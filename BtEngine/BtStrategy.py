import numpy as np
import pandas as pd

from typing import Literal

class BtStrategy:
    def __init__(self, strat: Literal['crossover', ...]):
        self._arg_signature = {
            'crossover': ['sma', 'lma']  # prev averages can be accessed with .iloc
        }


    @staticmethod
    def crossover(sma, lma, prev_sma, prev_lma):
        if prev_sma <= prev_lma and sma > lma:
            return 1
        elif prev_sma >= prev_lma and sma < lma:
            return -1
        else:
            return 0


    @property
    def arg_signature(self) -> list:
        return self._arg_signature[self.strat]
