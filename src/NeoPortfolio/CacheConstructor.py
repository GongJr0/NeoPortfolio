import sqlite3 as sql
from typing import Optional, Any


class CacheConstructor:

    def __init__(self, name: str) -> None:
        self.name = name
        self.conn, self.curr = self._connect()

    @staticmethod
    def _connect() -> tuple[sql.Connection, sql.Cursor]:
        conn = sql.connect('cache.db')
        curr = conn.cursor()
        return conn, curr

    def pass_connection(self) -> tuple[sql.Connection, sql.Cursor]:
        return self.conn, self.curr

    def exec(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        try:
            # Execute the query with parameters
            if params:
                self.curr.execute(query, params)
            else:
                self.curr.execute(query)

            # Commit changes for write operations
            self.conn.commit()

            # Return results for read operations
            if query.strip().lower().startswith("select"):
                return self.curr.fetchall()

            return None

        except Exception as e:
            print(f"Error executing query: {query}\nParameters: {params}\nError: {e}")
            return None

    def create(self, cols: dict[str, str]) -> None:
        query = f"CREATE TABLE IF NOT EXISTS {self.name} ("
        for col, dtype in cols.items():
            query += f"{col} {dtype}, "
        query = query[:-2] + ")"

        self.exec(query)

    def clear(self) -> None:
        self.exec(f"DELETE FROM {self.name}")


    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        self.close()
