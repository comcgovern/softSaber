"""Parquet-based local store for processed tables.

Tables live under ``data/processed/<table>/<partition>.parquet``. The shape of
the data here is stable across pipeline stages, so downstream stats code can
read tables by name without re-running ingest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, ensure_dirs


def _table_dir(name: str) -> Path:
    return PROCESSED_DIR / name


def write_partition(table: str, partition: str, df: pd.DataFrame) -> Path:
    ensure_dirs()
    tdir = _table_dir(table)
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"{partition}.parquet"
    df.to_parquet(out, index=False)
    return out


def read_table(table: str, *, partitions: list[str] | None = None) -> pd.DataFrame:
    tdir = _table_dir(table)
    if not tdir.exists():
        return pd.DataFrame()
    files = sorted(tdir.glob("*.parquet"))
    if partitions is not None:
        wanted = {f"{p}.parquet" for p in partitions}
        files = [f for f in files if f.name in wanted]
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
