from typing import NewType, Literal, Tuple, Union
import yfinance as yf
import math

StockSymbol = NewType('StockSymbol', str)
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

Days = NewType('Days', int)


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
            'sharpe_ratio': {obj[i]: -1 for i in range(len(obj))}
        }

        obj.optimum_portfolio_info = {
            'target_return': None,
            'weights': None,
            'risk_per_return': None,
        }

        return obj

    def stock_results(self, stock: StockSymbol):
        res = {
            'weights': self._results['weights'][stock],
            'expected_returns': self._results['expected_returns'][stock],
            'volatility': self._results['volatility'][stock],
            'beta': self._results['beta'][stock],
            'sharpe_ratio': self._results['sharpe_ratio'][stock]
        }
        return res

    def __getitem__(self, key: Union[int, StockSymbol]):
        if isinstance(key, int):
            return super().__getitem__(key)

        elif isinstance(key, StockSymbol.__supertype__):
            return self.stock_results(key)

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
    def weights(self, value):
        self._weights = value

