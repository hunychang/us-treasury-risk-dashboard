"""Return computation from raw yield (or price) levels."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# Default instrument metadata for the duration-adjusted return method.
# Treasury yields: approximate modified duration for zero-coupon equivalent.
# Spreads: scale factor to convert percentage-point changes to decimals.
# Indices: return computation method on the level itself.
_DEFAULT_METADATA: Dict[str, dict] = {
    "DGS1":         {"type": "treasury_yield", "duration": 1.0},
    "DGS2":         {"type": "treasury_yield", "duration": 1.9},
    "DGS5":         {"type": "treasury_yield", "duration": 4.5},
    "DGS10":        {"type": "treasury_yield", "duration": 8.5},
    "DGS30":        {"type": "treasury_yield", "duration": 19.5},
    "T10Y2Y":       {"type": "spread", "spread_duration": 7.0, "scale": 0.01},
    "T10Y3M":       {"type": "spread", "spread_duration": 8.0, "scale": 0.01},
    "BAMLC0A0CM":   {"type": "spread", "spread_duration": 7.0, "scale": 0.01},
    "BAMLH0A0HYM2": {"type": "spread", "spread_duration": 4.0, "scale": 0.01},
    "BAA10Y":       {"type": "spread", "spread_duration": 7.0, "scale": 0.01},
    "T5YIE":        {"type": "spread", "spread_duration": 5.0, "scale": 0.01},
    "T10YIE":       {"type": "spread", "spread_duration": 8.5, "scale": 0.01},
    "VIXCLS":       {"type": "index", "return_method": "simple"},
}


def compute_returns(
    yields_df: pd.DataFrame,
    method: str = "duration_adj",
    interpolation: str = "linear",
    missing_handling: str = "drop",
    instrument_metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute returns from yield / price levels.

    Parameters
    ----------
    yields_df : DataFrame of raw levels (e.g., 4.25 means 4.25 %).
    method :
        * ``"duration_adj"`` -- economically correct method (default):
            - Treasury yields: ``-D_mod * dy / 100``
            - Spreads: ``diff * scale``
            - Indices: simple percent change
        * ``"log"``    -- ln(y_t / y_{t-1})
        * ``"diff"``   -- y_t - y_{t-1}
        * ``"simple"`` -- (y_t - y_{t-1}) / y_{t-1}
    interpolation : How to fill interior NaN gaps before computing returns.
        ``"linear"`` uses pandas linear interpolation.
    missing_handling : What to do with remaining NaN values after return
        computation.  ``"drop"`` removes any row with a NaN.
    instrument_metadata : Per-instrument type and parameters for
        ``duration_adj``.  Falls back to built-in defaults for known
        FRED series when a column is not found in the mapping.

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
    elif method == "duration_adj":
        returns = _duration_adjusted_returns(df, instrument_metadata)
    else:
        raise ValueError(f"Unknown return method: {method!r}")

    # Handle remaining missing values
    if missing_handling == "drop":
        returns = returns.dropna()
    elif missing_handling == "ffill":
        returns = returns.ffill().dropna()

    return returns


def _duration_adjusted_returns(
    df: pd.DataFrame,
    instrument_metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute economically meaningful returns by instrument type.

    * **Treasury yields** (``type='treasury_yield'``): First-order
      bond-price approximation ``dP/P ~ -D_mod * dy``, where *dy* is the
      daily yield change expressed as a decimal (FRED percentage-point
      value / 100).
    * **Spreads** (``type='spread'``): Duration-scaled first difference:
      ``-spread_duration * diff * scale``.  The negative sign mirrors the
      Treasury convention (spread widening ≈ price loss).  ``spread_duration``
      is the effective spread duration of the instrument (e.g., ~7 yr for IG
      credit); ``scale`` (default 0.01) converts percentage-point changes
      to decimals.
    * **Indices** (``type='index'``): Simple percent change on the raw
      level, appropriate for volatility indices like VIX.
    """
    if instrument_metadata is None:
        instrument_metadata = {}

    parts: Dict[str, pd.Series] = {}
    for col in df.columns:
        meta = instrument_metadata.get(col, _DEFAULT_METADATA.get(col, {}))
        inst_type = meta.get("type", "spread")  # safe fallback

        if inst_type == "treasury_yield":
            duration = meta.get("duration", 1.0)
            # dy in percentage points -> decimal: divide by 100
            dy = df[col].diff() / 100.0
            parts[col] = -duration * dy

        elif inst_type == "spread":
            spread_dur = meta.get("spread_duration", 1.0)
            scale = meta.get("scale", 0.01)
            parts[col] = -spread_dur * df[col].diff() * scale

        elif inst_type == "index":
            ret_method = meta.get("return_method", "simple")
            if ret_method == "log":
                parts[col] = np.log(df[col] / df[col].shift(1))
            else:
                parts[col] = df[col].pct_change()

        else:
            # Unknown type: fallback to scaled diff
            parts[col] = df[col].diff() * 0.01

    return pd.DataFrame(parts, index=df.index)
