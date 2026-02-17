"""Date utilities for rebalancing schedules."""

from __future__ import annotations

import pandas as pd
from typing import List


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> List[pd.Timestamp]:
    """Return the last available date of each period within *index*.

    Parameters
    ----------
    index : A DatetimeIndex (typically the return series index).
    frequency : ``'daily'`` | ``'weekly'`` | ``'monthly'``

    Returns
    -------
    Sorted list of rebalance timestamps.
    """
    if frequency == "daily":
        return sorted(index.tolist())

    if frequency == "weekly":
        period_key = "W"
    elif frequency == "monthly":
        period_key = "M"
    else:
        raise ValueError(f"Unknown rebalance frequency: {frequency!r}")

    series = index.to_series()
    grouped = series.groupby(index.to_period(period_key))
    dates = [group.index[-1] for _, group in grouped]
    return sorted(dates)
