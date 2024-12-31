from nCrCache import nCrCache

import pandas as pd
import numpy as np

from itertools import combinations

import yfinance as yf
import requests
from bs4 import BeautifulSoup
from io import StringIO
import datetime as dt

class CombinationEngine:
    """
    Class to perform portfolio selection from all possible nCr combinations of a list of index components.
    """
    INDEX_MAP = {
        "^GSPC": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "^DJI": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "^IXIC": "https://en.wikipedia.org/wiki/NASDAQ-100",
        "^RUT": "https://en.wikipedia.org/wiki/Russell_2000_Index",
        "^FTSE": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "^GDAXI": "https://en.wikipedia.org/wiki/DAX",
        "^N225": "https://en.wikipedia.org/wiki/Nikkei_225",
        "^HSI": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "^FCHI": "https://en.wikipedia.org/wiki/CAC_40",
        "URTH": "https://en.wikipedia.org/wiki/MSCI_World_Index",
        "^AXJO": "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        "^BSESN": "https://en.wikipedia.org/wiki/SENSEX",
        "^N100": "https://en.wikipedia.org/wiki/Euronext_100",
        "^TSE60": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "GLOBAL": "https://en.wikipedia.org/wiki/List_of_stock_market_indices",
        "GLOBAL_INDICES": "https://en.wikipedia.org/wiki/List_of_global_indices"
    }

    def __init__(self, market: str, n: int, horizon: int = 21, lookback: int = 252):
        assert n > 0, "n must be greater than 0"
        if market not in self.INDEX_MAP.keys():
            raise ValueError(f"Invalid market: {market}, must be one of {self.INDEX_MAP.keys()}")

        self.cache = nCrCache(expire_days=1)

        self.horizon = horizon
        self.lookback = lookback

        self.market = market
        self.n = n

        self.components = self._get_components(market)
        self.tickers = self._get_tickers(self.components)

        self.historical_close = self._get_historical_close(lookback=lookback)
        self.periodic_returns = self._get_periodic_returns(horizon=horizon)
        self.expected_returns = self._get_ewma_expected(horizon=horizon)
        self.volatility = self._get_volatility(horizon=horizon)

        self.comb_space_components = min(max(n * 24, 48), len(self.components))
        self.high_ret = self._high_return_stock_proportion(n)
        self.low_vol = 1 - self.high_ret

        self.combination_space = self._compose_combination_space()
        self.ncr_gen = self._get_nCr_generator(self.combination_space, n)

    @classmethod
    def _get_components(cls, market: str) -> list:
        """
        Get the components of the index.
        """
        url = cls.INDEX_MAP[market]
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find_all('table')[0]
        table = StringIO(str(table))
        df = pd.read_html(table)[0]

        components = pd.Series(df['Symbol'])

        # Fix known exceptions
        if components.isin(["BRK.B"]).any():
            components = components.replace("BRK.B", "BRK-B")

        if components.isin(["BF.B"]).any():
            components = components.replace("BF.B", "BF-B")

        return components.tolist()


    @staticmethod
    def _get_nCr_generator(components: list, n: int) -> list:
        """
        Get all possible nCr combinations of a list of components.
        """
        for comb in combinations(components, n):
            yield comb

    def _get_tickers(self, components: list) -> yf.Tickers:
        """
        Get yf.Tickers object from a list of components.
        """
        assert self.components, "Components could not be found."

        return yf.Tickers(' '.join(components))

    def _get_historical_close(self, lookback: int) -> pd.DataFrame:
        """
        Get historical close prices for all components.
        """
        assert self.components, "Components could not be found."

        # Cache Check
        query_id = f"{self.market}_{lookback}"
        response = self.cache.get(query_id)

        if response is not None:
            return response

        start = dt.datetime.today() - dt.timedelta(days=lookback)
        start = start.date()

        end = dt.datetime.today().date()

        data = yf.download(' '.join(self.components), start=start, end=end)['Close']
        self.cache.cache(query_id, data)

        return data

    def _get_periodic_returns(self, horizon: int) -> pd.DataFrame:
        """
        Get periodic returns for all components.
        """
        periodic_returns = self.historical_close.copy()

        periodic_returns = (periodic_returns - periodic_returns.shift(horizon)) / periodic_returns.shift(horizon)
        periodic_returns = periodic_returns.dropna()

        return periodic_returns

    def _get_ewma_expected(self, horizon: int) -> dict[str, float]:
        """
        Get the exponentially weighted moving average expected returns.
        """
        ewma_expected = self.periodic_returns.ewm(halflife=horizon).mean().iloc[-1].to_dict()
        return ewma_expected

    def _get_volatility(self, horizon: int) -> dict[str, float]:
        """
        Get the volatility of the components.
        """
        ewma_var = self.periodic_returns.pow(2).ewm(span=horizon).mean()
        volatility = ewma_var.apply(np.sqrt).iloc[-1].to_dict()
        return volatility

    @staticmethod
    def _high_return_stock_proportion(n: int = 5) -> pd.Series:
        """
        Get the proportion of high return stocks in the portfolio.
        """
        return 0.7/(1+(n-5)/5)  # Assume 5 to be the average portfolio size and 0.7
                                # to be the proportion of high return stocks in the average portfolio


    def _compose_combination_space(self) -> list:
        """
        Compose the combination space of all possible nCr combinations of components.
        """
        hi_ret = round(self.high_ret * self.comb_space_components)
        lo_vol = round(self.low_vol * self.comb_space_components)

        ret_based = list(sorted(self.expected_returns, key=self.expected_returns.get, reverse=True))[:hi_ret]
        vol_based = list(sorted(self.volatility, key=self.volatility.get, reverse=False))[:lo_vol]

        return [*ret_based, *vol_based]
