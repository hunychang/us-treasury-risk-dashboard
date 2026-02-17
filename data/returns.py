"""Return computation from raw yield (or price) levels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    yields_df: pd.DataFrame,
    method: str = "log",
    interpolation: str = "linear",
    missing_handling: str = "drop",
) -> pd.DataFrame:
    """Compute returns from yield / price levels.

    Parameters
    ----------
    yields_df : DataFrame of raw levels (e.g., 4.25 means 4.25 %).
    method :
        * ``"log"``    -- ln(y_t / y_{t-1})
        * ``"diff"``   -- y_t - y_{t-1}
        * ``"simple"`` -- (y_t - y_{t-1}) / y_{t-1}
    interpolation : How to fill interior NaN gaps before computing returns.
        ``"linear"`` uses pandas linear interpolation.
    missing_handling : What to do with remaining NaN values after return
        computation.  ``"drop"`` removes any row with a NaN.

    Returns
    -------
    DataFrame of daily returns with the same column names.
    """
    df = yields_df.copy()

    # Interpolate interior gaps
    if interpolation == "linear":
        df = df.interpolate(method="linear", limit_direction="forward")
    elif interpolation == "ffill":
        df = df.ffill()

    # Compute returns
    if method == "log":
        returns = np.log(df / df.shift(1))
    elif method == "diff":
        returns = df.diff()
    elif method == "simple":
        returns = df.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method!r}")

    # Handle remaining missing values
    if missing_handling == "drop":
        returns = returns.dropna()
    elif missing_handling == "ffill":
        returns = returns.ffill().dropna()

    return returns
