from typing import NewType, Literal, Tuple, Union
import yfinance as yf
import math

StockSymbol = NewType('StockSymbol', str)
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

Days = NewType('Days', int)