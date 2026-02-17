"""Abstract data provider interface.

Concrete implementations (FRED, futures, etc.) inherit from DataProvider
so the rest of the system is data-source agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def fetch(
        self,
        instruments: List[str],
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch raw data for the given instruments.

        Returns
        -------
        DataFrame with DatetimeIndex and one column per instrument.
        Values are raw yield levels (e.g., 4.25 means 4.25 %).
        """
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Human-readable data source identifier."""
        ...
