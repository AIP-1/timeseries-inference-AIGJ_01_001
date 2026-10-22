"""InfluxDB 에 쓰기 전 디스크에 먼저 쌓아 두는 큐.

InfluxDB 가 잠시 죽거나 네트워크가 끊겨도, 수집기가 재시작돼도 데이터가 사라지지 않게 한다.
"""
from __future__ import annotations

import sqlite3
import threading


class Spool:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS q (id INTEGER PRIMARY KEY, line TEXT NOT NULL)")

    def put(self, lines: list[str]) -> None:
        if not lines:
            return
        with self._lock:
            self._db.executemany("INSERT INTO q(line) VALUES (?)", [(l,) for l in lines])

    def peek(self, n: int) -> list[tuple[int, str]]:
        with self._lock:
            return self._db.execute("SELECT id, line FROM q ORDER BY id LIMIT ?", (n,)).fetchall()

    def ack(self, last_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM q WHERE id <= ?", (last_id,))

    def size(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM q").fetchone()[0]
