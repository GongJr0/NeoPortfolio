from typing import NewType, Literal, Tuple, Union

# Type aliases
StockSymbol = str
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

Days = int
