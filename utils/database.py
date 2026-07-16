"""SQLite history utilities."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/analyses.db")


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT,
                spectrum_type TEXT,
                extraction_method TEXT,
                n_points INTEGER,
                n_peaks INTEGER,
                quality TEXT
            )
            """
        )
        conn.commit()


def insert_analysis(
    filename: str,
    spectrum_type: str,
    extraction_method: str,
    n_points: int,
    n_peaks: int,
    quality: str,
    db_path: Path = DB_PATH,
) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analyses (created_at, filename, spectrum_type, extraction_method, n_points, n_peaks, quality)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                filename,
                spectrum_type,
                extraction_method,
                int(n_points),
                int(n_peaks),
                quality,
            ),
        )
        conn.commit()


def load_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM analyses ORDER BY id DESC", conn)
