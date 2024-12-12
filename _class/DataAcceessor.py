# Imports
import pandas as pd
import numpy as np
import yfinance as yf

from zoneinfo import ZoneInfo
from datetime import datetime

from CustomTypes import StockSymbol, StockDataSubset
from typing import Optional

from _class.Session import Session

from bs4 import BeautifulSoup
import requests


class DataAccessor:
    """Accesses and Formats Stock Data using Yahoo Finance API.
    Limited to SP500 Stocks as per the goal of the project."""

    _default_subset = ('Open', 'High', 'Low', 'Close', 'Volume')

    def __init__(self,
                 symbols: Optional[list[StockSymbol]] = None,
                 session: Optional[Session] = None) -> None:

        # Get Session
        self.session = session if session else Session()

        # Get Tickers
        if symbols:
            self.market = yf.Tickers(symbols, session=self.session.get_session)
        else:
            self.market = self._get_SP500_market_tickers()

        # Get Initialization Date
        self.__init_date = datetime.now(tz=ZoneInfo('US/Eastern'))  # Init Date in Eastern Time

    def _get_SP500_market_tickers(self) -> yf.Tickers:
        """Get the Tickers for the S&P 500 Market."""
        url = 'https://stockanalysis.com/list/sp-500-stocks/'

        try:
            response = self.session.get(url)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise f"Error fetching SP500 Symbols, provide a list of symbols on initialization if error persists: {e}"

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'main-table'})
        headers = table.find_all('th')
        symbol_index = None
        for i, header in enumerate(headers):
            if 'symbol' in header.text.strip().lower():  # This is case-insensitive matching
                symbol_index = i
                break

        if symbol_index is None:
            raise Exception("Symbol column not found in the table.")

        # Extract stock symbols from the corresponding column in each row
        symbols = []
        for row in table.find_all('tr')[1:]:  # Skip the header row
            cols = row.find_all('td')
            if len(cols) > symbol_index:  # Ensure the row has the expected number of columns
                symbol = cols[symbol_index].text.strip()
                symbols.append(symbol)

        symbols = " ".join(symbols)

        return yf.Tickers(symbols, session=self.session.get_session)


    def get_stock(self,
                  symbol: StockSymbol,
                  start: datetime, end: datetime,
                  subset: StockDataSubset = _default_subset,
                  interval: str = '1d') -> pd.DataFrame:
        data = self.market

        """Returns the Stock Data for a given Stock Symbol and Date Range."""
        stock = yf.Ticker(symbol, session=self.session.get_session())
        data = stock.history(start=start, end=end, interval=interval)[subset]
        return data

    def get_market(self,
                   start: Optional[datetime] = None, end: Optional[datetime] = None,
                   period: Optional[str] = None,
                   subset: StockDataSubset = _default_subset,
                   interval: str = '1d'):

        # Ensure that if start and end are specified, both are provided
        if start and not end:
            raise ValueError("Please specify both start and end parameters if you wish to fetch a specific period.")

        if end and not start:
            raise ValueError("Please specify both start and end parameters if you wish to fetch a specific period.")

        # Ensure that if start and end are not specified, period must be provided
        if not start and not end and not period:
            raise ValueError("Please specify either a period or both start and end parameters.")


        if start and end:
            return self.market.history(start=start, end=end, interval=interval)[subset]
        elif period:
            return self.market.history(period=period, interval=interval)[subset]

    @property
    def init_date(self):
        """Returns the Initialization Date of the DataAccessor Object."""
        return self.__init_date.astimezone().strftime('%d-%m-%Y - %H:%M')


    @property
    def default_subset(self):
        """Returns the Default Subset of Data to Retrieve."""
        return self._default_subset

    @default_subset.setter
    def default_subset(self, subset: StockDataSubset):
        """Sets the Default Subset of Data to Retrieve."""
        self._default_subset = subset
