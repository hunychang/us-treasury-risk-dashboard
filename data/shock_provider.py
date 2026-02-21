"""Romer & Romer monetary policy shock data provider.

Loads the shock series from a CSV file.  The shocks are irregularly
spaced (FOMC meeting dates, ~8 per year) and represent the intended
federal funds rate change residual from the Romer & Romer (2004)
narrative approach.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.base_provider import DataProvider


class ShockProvider(DataProvider):
    """Load Romer & Romer monetary policy shocks from CSV.

    Expected CSV format::

        date,rr_shock
        1969-03-04,-0.17
        1969-04-01,0.08
        ...

    Parameters
    ----------
    csv_path : Path to the shock CSV file.
    shock_column : Column name for the shock magnitude (default ``rr_shock``).
    cumulate : If True, cumulate shocks within a rolling window.
    shock_window_months : Rolling window (months) for cumulated shocks.
    """

    def __init__(
        self,
        csv_path: str,
        shock_column: str = "rr_shock",
        cumulate: bool = False,
        shock_window_months: int = 3,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._shock_column = shock_column
        self._cumulate = cumulate
        self._window_months = shock_window_months

    def fetch(
        self,
        instruments: List[str],
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load shock series, filtered to [start_date, end_date].

        Parameters
        ----------
        instruments : Ignored (the CSV has a single shock column).
        start_date, end_date : Date range filter.

        Returns
        -------
        DataFrame with DatetimeIndex and a single column for the shock.
        Non-meeting dates are NaN (sparse representation).
        """
        df = pd.read_csv(
            self._csv_path,
            parse_dates=["date"],
            index_col="date",
        )

        if self._shock_column not in df.columns:
            raise ValueError(
                f"Column {self._shock_column!r} not found in "
                f"{self._csv_path}.  Available: {list(df.columns)}"
            )

        # Keep only the shock column
        shocks = df[[self._shock_column]].copy()

        # Date filtering
        mask = shocks.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= shocks.index <= pd.Timestamp(end_date)
        shocks = shocks.loc[mask]

        if self._cumulate:
            # Rolling sum over the specified window
            shocks = shocks.rolling(
                f"{self._window_months * 30}D", min_periods=1
            ).sum()

        return shocks

    def source_name(self) -> str:
        return "romer_romer"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def load_as_series(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """Load shock data as a single pd.Series (convenience)."""
        df = self.fetch([], start_date, end_date)
        return df[self._shock_column]

    def reindex_to_daily(
        self,
        shocks: pd.Series,
        daily_index: pd.DatetimeIndex,
        fill_value: float = 0.0,
    ) -> pd.Series:
        """Reindex sparse shock series to a daily calendar.

        Non-meeting dates are filled with ``fill_value`` (default 0).
        """
        return shocks.reindex(daily_index, fill_value=fill_value)
