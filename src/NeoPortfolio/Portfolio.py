from typing import Union, Iterable
import yfinance as yf
import math

from .CustomTypes import StockSymbol

class Portfolio(tuple):
    def __new__(cls, *args):
        obj = super().__new__(cls, sorted(args))

        obj._weights = [1/len(obj) for _ in range(len(obj))]

        obj._tickers = yf.Tickers(' '.join(obj))

        obj._results = {
            'weights': {obj[i]: -1 for i in range(len(obj))},
            'expected_returns': {obj[i]: -1 for i in range(len(obj))},
            'volatility': {obj[i]: -1 for i in range(len(obj))},
            'beta': {obj[i]: -1 for i in range(len(obj))},
            'sharpe_ratio': {obj[i]: -1 for i in range(len(obj))},
            'sentiment': {obj[i]: -1 for i in range(len(obj))}
        }

        obj.optimum_portfolio_info = {
            'target_return': None,
            'target_volatility': None,
            'weights': None,
        }

        return obj

    def _stock_results(self, stock: StockSymbol):
        return {
            'weight': self._results['weights'][stock],
            'expected_return': self._results['expected_returns'][stock],
            'volatility': self._results['volatility'][stock],
            'beta': self._results['beta'][stock],
            'sharpe_ratio': self._results['sharpe_ratio'][stock]
        }

    def __getitem__(self, key: Union[int, StockSymbol]):
        if isinstance(key, int):
            return super().__getitem__(key)

        elif isinstance(key, StockSymbol):
            return self._stock_results(key)

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, sub_category: str, value: dict):
        assert sub_category in self._results.keys(), f"Invalid sub_category: {sub_category}, must be one of {self._results.keys()}"
        self._results[sub_category] = value

    @property
    def tickers(self):
        return self._tickers

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: Iterable[float]):
        assert len(value) == len(self), f"Invalid number of weights: {len(value)}, must be {len(self)}"
        self._weights = value
