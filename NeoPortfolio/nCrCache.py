import sqlite3 as sql
import os
import datetime as dt
from os import PathLike
import pickle

from typing import Any, Optional
from pandas import DataFrame


class nCrCache:
    def __init__(self, name: Optional[PathLike] = None, expire_days: int = 1) -> None:

        self.conn, self.curr = self._connect(name)
        self.expire_days = expire_days

    @staticmethod
    def _connect(name: Optional[PathLike] = None) -> tuple[sql.Connection, sql.Cursor]:
        """
        Initialize the cache database.
        """
        if not name:
            name = "nCr.db"
        elif (not str(name).endswith(".db")) or (not str(name).endswith(".sqlite")):
            name += ".db"

        conn = sql.connect(name)
        curr = conn.cursor()

        curr.execute("CREATE TABLE IF NOT EXISTS nCr (id TEXT PRIMARY KEY, data BLOB, createdAt DATE, expiresAt DATE)")
        conn.commit()

        return conn, curr

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.conn.close()

    def cache(self, id: str, data: Any) -> None:
        """
        Cache the data.

        :param id: identifier for with format: f"{market_name}_{lookback}"
        :param data: BLOB of price data
        """

        created_at = dt.datetime.now().date()
        expires_at = created_at + dt.timedelta(days=self.expire_days)

        data = pickle.dumps(data)

        query = """
        INSERT OR REPLACE INTO nCr (id, data, createdAt, expiresAt)
        VALUES (?, ?, ?, ?)
        """
        self.curr.execute(query, (id, data, created_at, expires_at))
        self.conn.commit()

    def clear(self) -> None:
        """
        Clear the cache.
        """
        self.curr.execute("DELETE FROM nCr")
        self.conn.commit()

    def get(self, query_id: str) -> Optional[DataFrame]:
        """
        Get the data from the cache.

        :param query_id: identifier for with format: f"{market_name}_{lookback}"
        """
        query = """
        SELECT * FROM nCr WHERE id = ?
        """
        self.curr.execute(query, (query_id,))
        data = self.curr.fetchone()

        if data is None:
            return None

        if dt.datetime.fromisoformat(data[3]).date() < dt.datetime.now().date():
            return None
        else:
            return pickle.loads(data[1])
