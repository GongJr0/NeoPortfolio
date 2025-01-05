from typing import NewType, Literal, Tuple, Union, NewType

# Type aliases
Days = int

StockSymbol = str
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
IndexSymbol = str