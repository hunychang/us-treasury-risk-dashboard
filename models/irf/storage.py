"""Persistence layer for fitted IRF results.

Currently uses pickle for local storage.  A future version will
support S3-backed storage with version hashing.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.irf.local_projection import IRFResult


class IRFStore:
    """Persist and load fitted IRF results."""

    def __init__(self, output_dir: str = "output/irf") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: Dict[str, IRFResult],
        filename: str = "irf_results.pkl",
    ) -> Path:
        """Save IRF results to disk.

        Returns the path to the saved file.
        """
        path = self._output_dir / filename
        with open(path, "wb") as f:
            pickle.dump(results, f)
        return path

    def load(
        self,
        filename: str = "irf_results.pkl",
    ) -> Dict[str, IRFResult]:
        """Load IRF results from disk."""
        path = self._output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"IRF results not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(self, filename: str = "irf_results.pkl") -> bool:
        """Check whether a saved result file exists."""
        return (self._output_dir / filename).exists()

    def version_hash(
        self,
        returns: pd.DataFrame,
        shocks: pd.Series,
    ) -> str:
        """Compute an MD5 hash of the input data for cache invalidation.

        If the data changes, the hash changes and stale IRF results
        should be re-estimated.
        """
        h = hashlib.md5()

        # Hash returns shape + sample of values
        h.update(str(returns.shape).encode())
        h.update(returns.values.tobytes()[:8192])

        # Hash shocks
        shock_vals = shocks.dropna().values
        h.update(str(len(shock_vals)).encode())
        h.update(shock_vals.tobytes()[:4096])

        return h.hexdigest()[:12]
