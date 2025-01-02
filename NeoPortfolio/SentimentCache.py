import pandas as pd
import numpy as np

from datetime import datetime as dt, timedelta

import sqlite3 as sql

from typing import Union, Optional

class SentimentCache:

    def __init__(self, name: str, exp_after: int) -> None:
        """

        :param name: name of the database
        :param exp_after: expiry time for cached data in seconds
        """
        self.name: str = name
        self.exp_after: int = exp_after
        self.conn, self.curr = self._connect(name)

    @staticmethod
    def _connect(name: str) -> tuple[sql.Connection, sql.Cursor]:
        extension = name.split('.')[-1]
        if extension != 'db' and extension != 'sqlite':
            name = name + '.db'

        conn = sql.connect(name)
        curr = conn.cursor()

        # CREATE TABLE Query
        q = "CREATE TABLE IF NOT EXISTS cache (symbol TEXT PRIMARY KEY, sentiment FLOAT, createdAt DATE, expireAfter INTEGER)"
        curr.execute(q)
        conn.commit()

        return conn, curr

    def close(self) -> None:
        self.conn.close()

    def cache(self, symbol: str, sentiment: float) -> None:
        now = dt.now().isoformat()
        expire = self.exp_after
        q = "INSERT OR REPLACE INTO cache (symbol, sentiment, createdAt, expireAfter) VALUES (?, ?, ?, ?)"
        self.curr.execute(q, (symbol, sentiment, now, expire))
        self.conn.commit()

    def get(self, symbol: str) -> Optional[Union[dict, float]]:
        # Check if the symbol is in the cache
        q = "SELECT * FROM cache WHERE symbol = ?"
        self.curr.execute(q, (symbol,))
        data = self.curr.fetchone()

        if data is None:
            return None

        # Check if the data is expired
        now = dt.now()
        expire = dt.fromisoformat(data[2]) + timedelta(seconds=data[3])
        if now - expire > timedelta(seconds=0):
            return None

        return data[1]
