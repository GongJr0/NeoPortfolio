import sqlite3 as sql
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional
from os import PathLike


class PortfolioCache:
    def __init__(self, name: Optional[str] = None, expire_days: int = 1) -> None:
        self.conn, self.curr = self._connect(name)
        self.expire_days = expire_days

    @staticmethod
    def _connect(name: Optional[PathLike] = None) -> tuple[sql.Connection, sql.Cursor]:
        if name is None:
            name = "portfolioCache.db"
        elif not str(name).endswith(".db") and not str(name).endswith(".sqlite"):
            name = f"{str(name).split('.')[0]}.db"

        conn = sql.connect(name)
        curr = conn.cursor()

        curr.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            portfolio TEXT PRIMARY KEY, 
            data BLOB, 
            createdAt DATETIME, 
            expiresAt DATETIME
        )
        """)

        return conn, curr

    def close(self) -> None:
        self.conn.close()

    def cache(self, portfolio: tuple, target_return: float, bounds: tuple[float, float] , data: Any) -> None:
        portfolio_id = f'({", ".join(portfolio)})_{target_return}_{bounds}'
        created_at = datetime.today().date()

        expires_at = created_at + timedelta(days=self.expire_days)

        data = pickle.dumps(data)
        self.curr.execute(
            "INSERT OR REPLACE INTO portfolio (portfolio, data, createdAt, expiresAt) VALUES (?, ?, ?, ?)",
            (portfolio_id, data, created_at, expires_at))

        self.conn.commit()

    def get(self, portfolio: tuple, target_return: float, bounds: tuple[float, float]) -> Any:
        portfolio_id = f'({", ".join(portfolio)})_{target_return}_{bounds}'
        self.curr.execute("""SELECT data, expiresAt FROM portfolio
                             WHERE portfolio=?""", (portfolio_id,))
        response = self.curr.fetchone()

        if response:
            data, expires_at = response
            if datetime.now() < datetime.strptime(expires_at, "%Y-%m-%d"):
                return pickle.loads(data)
            else:
                # Cache expired, remove the entry
                self.curr.execute("DELETE FROM portfolio WHERE portfolio=?", (portfolio_id,))
                self.conn.commit()

        return None

    def clear(self) -> None:
        self.curr.execute("DELETE FROM portfolio")
        self.conn.commit()

    def __del__(self):
        self.close()
