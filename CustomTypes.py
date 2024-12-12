from typing import NewType, Literal, Tuple

StockSymbol = NewType('StockSymbol', str)
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]