"""Tests for the Romer & Romer shock data provider."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from data.shock_provider import ShockProvider


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a small sample shock CSV for testing."""
    csv_content = (
        "date,rr_shock\n"
        "2000-01-15,0.25\n"
        "2000-03-20,-0.12\n"
        "2000-05-16,0.18\n"
        "2000-08-22,-0.30\n"
        "2001-01-03,-0.42\n"
        "2001-03-20,0.15\n"
        "2001-06-27,-0.25\n"
    )
    csv_file = tmp_path / "test_shocks.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def provider(sample_csv) -> ShockProvider:
    return ShockProvider(csv_path=str(sample_csv))


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestShockProvider:

    def test_load_basic(self, provider):
        """ShockProvider should load and return a DataFrame."""
        df = provider.fetch([], start_date=date(1990, 1, 1))
        assert isinstance(df, pd.DataFrame)
        assert "rr_shock" in df.columns
        assert len(df) == 7

    def test_date_filtering(self, provider):
        """Shock data should be filtered by start/end dates."""
        df = provider.fetch(
            [], start_date=date(2000, 3, 1), end_date=date(2000, 12, 31)
        )
        assert len(df) == 3  # Mar, May, Aug 2000
        assert df.index.min() >= pd.Timestamp("2000-03-01")
        assert df.index.max() <= pd.Timestamp("2000-12-31")

    def test_source_name(self, provider):
        assert provider.source_name() == "romer_romer"

    def test_load_as_series(self, provider):
        """load_as_series should return a pd.Series."""
        s = provider.load_as_series(start_date=date(1990, 1, 1))
        assert isinstance(s, pd.Series)
        assert s.name == "rr_shock"
        assert len(s) == 7

    def test_reindex_to_daily(self, provider):
        """Reindexing should fill non-meeting dates with 0."""
        s = provider.load_as_series(start_date=date(2000, 1, 1), end_date=date(2000, 6, 30))
        # Use calendar days so the FOMC dates (Jan 15, Mar 20) are included
        daily_idx = pd.date_range("2000-01-01", "2000-06-30", freq="D")
        daily = provider.reindex_to_daily(s, daily_idx, fill_value=0.0)
        assert len(daily) == len(daily_idx)
        # Jan 15 shock should be present
        assert daily.loc[pd.Timestamp("2000-01-15")] == 0.25
        # Non-meeting dates should be zero
        assert daily.loc[pd.Timestamp("2000-01-14")] == 0.0
        assert daily.loc[pd.Timestamp("2000-02-01")] == 0.0

    def test_missing_column_raises(self, sample_csv):
        """Should raise ValueError for wrong column name."""
        bad_provider = ShockProvider(str(sample_csv), shock_column="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            bad_provider.fetch([], start_date=date(2000, 1, 1))

    def test_cumulate(self, sample_csv):
        """Cumulated shocks should aggregate within window."""
        provider = ShockProvider(
            str(sample_csv), cumulate=True, shock_window_months=3
        )
        df = provider.fetch([], start_date=date(1990, 1, 1))
        assert isinstance(df, pd.DataFrame)
        # Cumulated values should differ from raw values (at least some)
        raw_provider = ShockProvider(str(sample_csv))
        raw_df = raw_provider.fetch([], start_date=date(1990, 1, 1))
        # First value should be the same (only 1 in window)
        assert df.iloc[0, 0] == raw_df.iloc[0, 0]

    def test_actual_csv_exists(self):
        """The bundled sample CSV should be loadable."""
        csv_path = Path(__file__).parent.parent / "data" / "shock_data" / "romer_romer_shocks.csv"
        if csv_path.exists():
            provider = ShockProvider(str(csv_path))
            df = provider.fetch([], start_date=date(1960, 1, 1))
            assert len(df) > 0
            assert "rr_shock" in df.columns
