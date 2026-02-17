"""Transparent caching layer that wraps any DataProvider.

Data is persisted as Parquet files so that repeated runs do not
re-download from the remote API.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.base_provider import DataProvider


def _resolve_cache_dir() -> Path:
    """Return a writable cache directory.

    Prefers ``cache/`` next to the project root.  Falls back to the
    system temp directory (needed on Streamlit Cloud where the app
    directory is read-only).
    """
    project_cache = Path(__file__).resolve().parent.parent / "cache"
    try:
        project_cache.mkdir(parents=True, exist_ok=True)
        return project_cache
    except OSError:
        return Path(tempfile.gettempdir()) / "treasury_risk_cache"


_DEFAULT_CACHE_DIR = _resolve_cache_dir()


class CachedProvider:
    """Wraps a :class:`DataProvider` and caches results as Parquet files.

    Parameters
    ----------
    provider : The underlying data provider to wrap.
    cache_dir : Directory where cached Parquet files are stored.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, instruments: List[str]) -> Path:
        key = "_".join(sorted(instruments))
        return self._cache_dir / f"{self._provider.source_name()}_{key}.parquet"

    def get(
        self,
        instruments: List[str],
        start_date: date,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return data for *instruments*, using the cache when possible.

        Parameters
        ----------
        instruments : List of series identifiers.
        start_date, end_date : Requested date range.
        force_refresh : If ``True``, always re-fetch from the remote source.

        Returns
        -------
        DataFrame of raw yield levels.
        """
        path = self._cache_path(instruments)

        if path.exists() and not force_refresh:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            # Trim to requested range
            mask = df.index >= pd.Timestamp(start_date)
            if end_date:
                mask &= df.index <= pd.Timestamp(end_date)
            return df.loc[mask]

        # Fetch fresh data from the provider
        df = self._provider.fetch(instruments, start_date, end_date)
        df.to_parquet(path)
        return df
