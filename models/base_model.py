"""Abstract base class for all covariance-estimation risk models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RiskModel(ABC):
    """Interface that every risk / covariance model must implement.

    The ``estimate`` method receives the *full* return history but also an
    ``as_of_date`` cutoff -- implementations must not use any data after
    that date, which prevents look-ahead bias during backtesting.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (used as dict key, chart label)."""
        ...

    @abstractmethod
    def estimate(
        self,
        returns: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> np.ndarray:
        """Estimate the covariance matrix using data up to *as_of_date*.

        Parameters
        ----------
        returns : Full return-history DataFrame.
        as_of_date : Only rows with ``index <= as_of_date`` may be used.

        Returns
        -------
        (n_assets, n_assets) annualized covariance matrix.
        """
        ...

    @abstractmethod
    def min_history(self) -> int:
        """Minimum number of return observations needed before the model
        can produce a valid covariance estimate."""
        ...
