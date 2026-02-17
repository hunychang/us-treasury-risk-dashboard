"""FRED data provider — pulls daily Treasury yields via the FRED API."""

from __future__ import annotations

import os
from datetime import date
from typing import List, Optional

import pandas as pd
from fredapi import Fred

from data.base_provider import DataProvider


class FREDProvider(DataProvider):
    """Pulls daily U.S. Treasury constant-maturity yields from FRED.

    Requires a FRED API key.  Set the ``FRED_API_KEY`` environment
    variable or pass the key directly.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")

        # Fallback: read from Streamlit secrets (used on Streamlit Cloud)
        if not self._api_key:
            try:
                import streamlit as st
                self._api_key = st.secrets.get("FRED_API_KEY")
            except Exception:
                pass

        if not self._api_key:
            raise ValueError(
                "FRED API key required. Set the FRED_API_KEY environment "
                "variable, add it to .streamlit/secrets.toml, or pass "
                "api_key= to FREDProvider()."
            )
        self._fred = Fred(api_key=self._api_key)

    def fetch(
        self,
        instruments: List[str],
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch yield series for each instrument from FRED.

        Parameters
        ----------
        instruments : FRED series IDs (e.g. ``["DGS1", "DGS2", "DGS5", "DGS10"]``).
        start_date, end_date : Date range for the query.

        Returns
        -------
        DataFrame indexed by date with one column per instrument.
        """
        frames: dict[str, pd.Series] = {}
        for series_id in instruments:
            s = self._fred.get_series(
                series_id,
                observation_start=start_date.isoformat(),
                observation_end=end_date.isoformat() if end_date else None,
            )
            frames[series_id] = s

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        # Convert string "." entries (FRED missing-data markers) to NaN
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def source_name(self) -> str:
        return "FRED"
