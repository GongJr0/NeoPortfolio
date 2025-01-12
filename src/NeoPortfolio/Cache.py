from .CacheConstructor import CacheConstructor

from pandas import DataFrame
from datetime import datetime as dt, timedelta
import pickle

from typing import Union, Optional, Any


class SentimentCache(CacheConstructor):

    def __init__(self, name: Optional[str] = None, exp_seconds: int = 3600) -> None:
        super().__init__(name or "sentiment")
        self.exp_seconds: int = exp_seconds

        self.create({
            "symbol": "TEXT PRIMARY KEY",
            "sentiment": "REAL",
            "createdAt": "TEXT",
            "expireAfter": "INTEGER"
        })

    def cache(self, symbol: str, sentiment: float) -> None:
        now = dt.now().isoformat()
        expire = self.exp_seconds
        q = f"INSERT OR REPLACE INTO {self.name} (symbol, sentiment, createdAt, expireAfter) VALUES (?, ?, ?, ?)"
        self.exec(q, (symbol, sentiment, now, expire))

    def get(self, symbol: str) -> Optional[Union[dict, float]]:
        # Check if the symbol is in the cache
        q = f"SELECT * FROM {self.name} WHERE symbol = ?"
        self.exec(q, (symbol,))
        data = self.curr.fetchone()

        if data is None:
            return None

        # Check if the data is expired
        now = dt.now()
        expire = dt.fromisoformat(data[2]) + timedelta(seconds=data[3])
        if now - expire > timedelta(seconds=0):
            return None

        return data[1]


class PortfolioCache(CacheConstructor):
    def __init__(self, name: Optional[str] = None, expire_days: int = 1) -> None:
        super().__init__(name or "portfolio")
        self.expire_days = expire_days

        self.create({
                "portfolio": "TEXT PRIMARY KEY",
                "data": "BLOB",
                "createdAt": "DATE",
                "expiresAt": "DATE"
            })

    def cache(self, portfolio: tuple, target_return: float, bounds: tuple[float, float], data: Any) -> None:
        portfolio_id = f'({", ".join(portfolio)})_{target_return}_{bounds}'
        created_at = dt.today().date()

        expires_at = created_at + timedelta(days=self.expire_days)

        data = pickle.dumps(data)
        self.exec(
            f"INSERT OR REPLACE INTO {self.name} (portfolio, data, createdAt, expiresAt) VALUES (?, ?, ?, ?)",
            (portfolio_id, data, created_at, expires_at))


    def get(self, portfolio: tuple, target_return: float, bounds: tuple[float, float]) -> Any:
        portfolio_id = f'({", ".join(portfolio)})_{target_return}_{bounds}'

        self.exec(f"""SELECT data, expiresAt FROM {self.name}
                             WHERE portfolio=?""", (portfolio_id,))
        response = self.curr.fetchone()

        if response:
            data, expires_at = response
            if dt.now() < dt.strptime(expires_at, "%Y-%m-%d"):
                return pickle.loads(data)
            else:
                # Cache expired, remove the entry
                self.exec(f"DELETE FROM {self.name} WHERE portfolio=?",
                          (portfolio_id,))

        return None


class nCrCache(CacheConstructor):
    def __init__(self, name: Optional[str] = None, expire_days: int = 1) -> None:
        super().__init__(name or "nCr")

        self.expire_days = expire_days

        self.create({
            "id": "TEXT PRIMARY KEY",
            "data": "BLOB",
            "createdAt": "DATE",
            "expiresAt": "DATE"
        })

    def cache(self, _id: str, data: Any) -> None:
        """
        Cache the data.

        :param _id: identifier for with format: f"{market_name}_{lookback}"
        :param data: BLOB of price data
        """

        created_at = dt.now().date()
        expires_at = created_at + timedelta(days=self.expire_days)

        data = pickle.dumps(data)

        query = f"""
        INSERT OR REPLACE INTO {self.name} (id, data, createdAt, expiresAt)
        VALUES (?, ?, ?, ?)
        """
        self.exec(
            query,
            (_id, data, created_at, expires_at))

    def get(self, _id: str) -> Optional[DataFrame]:
        """
        Get the data from the cache.

        :param _id: identifier for with format: f"{market_name}_{lookback}"
        """
        query = f"""
        SELECT * FROM {self.name} WHERE id = ?
        """
        self.exec(query, (_id,))
        data = self.curr.fetchone()

        if data is None:
            return None

        if dt.fromisoformat(data[3]).date() < dt.now().date():
            return None
        else:
            return pickle.loads(data[1])
