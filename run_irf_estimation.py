"""Offline IRF estimation script.

Loads market returns and Romer & Romer shocks, runs Local Projection
estimation, and saves the IRF results for use by the dashboard and
shock-conditioned risk model.

Usage::

    python run_irf_estimation.py [--config config/default_config.yaml]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger

from config.config_loader import load_config
from data.cache_manager import CachedProvider
from data.fred_provider import FREDProvider
from data.returns import compute_returns
from data.shock_provider import ShockProvider
from models.irf import LPEstimator, IRFStore


def main(config_path: str = None) -> None:
    cfg = load_config(config_path)

    if not cfg.shocks.enabled:
        logger.warning("Shocks are disabled in config.  Enable with shocks.enabled: true")
        print("Shocks disabled.  Set shocks.enabled: true in config to run IRF estimation.")
        return

    # ------------------------------------------------------------------
    # 1. Load market data
    # ------------------------------------------------------------------
    logger.info("Loading market data from FRED...")
    provider = CachedProvider(FREDProvider())
    yields_df = provider.fetch(
        cfg.data.instruments,
        cfg.data.start_date,
        cfg.data.end_date,
    )
    returns = compute_returns(
        yields_df,
        method=cfg.data.return_type,
        interpolation=cfg.data.interpolation,
        missing_handling=cfg.data.missing_handling,
        instrument_metadata=cfg.data.instrument_metadata,
    )
    logger.info(f"Returns: {returns.shape[0]} days × {returns.shape[1]} instruments")

    # ------------------------------------------------------------------
    # 2. Load shock data
    # ------------------------------------------------------------------
    logger.info(f"Loading shock data from {cfg.shocks.csv_path}...")
    shock_provider = ShockProvider(
        csv_path=cfg.shocks.csv_path,
        shock_column=cfg.shocks.shock_column,
        cumulate=cfg.shocks.cumulate,
        shock_window_months=cfg.shocks.shock_window_months,
    )
    shocks = shock_provider.load_as_series(
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    # Reindex to daily returns calendar
    shocks_daily = shock_provider.reindex_to_daily(
        shocks, returns.index, fill_value=0.0
    )
    logger.info(f"Shocks: {(shocks_daily != 0).sum()} non-zero events in sample")

    # ------------------------------------------------------------------
    # 3. Run LP estimation
    # ------------------------------------------------------------------
    logger.info(
        f"Running LP estimation (H={cfg.irf.max_horizon}, "
        f"lags={cfg.irf.n_lags}, CI={cfg.irf.confidence_level:.0%})..."
    )
    estimator = LPEstimator(
        max_horizon=cfg.irf.max_horizon,
        n_lags=cfg.irf.n_lags,
        confidence_level=cfg.irf.confidence_level,
    )
    irf_results = estimator.estimate(returns, shocks_daily)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    store = IRFStore(output_dir=cfg.irf.output_dir)
    data_hash = store.version_hash(returns, shocks_daily)
    filename = f"irf_{data_hash}.pkl"
    path = store.save(irf_results, filename)

    # Also save as the default filename for easy loading
    store.save(irf_results, "irf_results.pkl")
    logger.info(f"IRF results saved to {path}")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("IRF Estimation Summary")
    print("=" * 72)
    print(f"{'Instrument':<16} {'h=1':>8} {'h=6':>8} {'h=12':>8} {'h=24':>8}  {'Sig?':>5}")
    print("-" * 72)

    for name, irf in irf_results.items():
        sig_count = (irf.p_values < (1 - cfg.irf.confidence_level)).sum()
        sig_str = f"{sig_count}/{len(irf.horizons)}"

        row_vals = []
        for h_target in [1, 6, 12, 24]:
            if h_target <= cfg.irf.max_horizon:
                idx = h_target - 1
                beta = irf.coefficients[idx]
                stars = ""
                if irf.p_values[idx] < 0.01:
                    stars = "***"
                elif irf.p_values[idx] < 0.05:
                    stars = "**"
                elif irf.p_values[idx] < 0.10:
                    stars = "*"
                row_vals.append(f"{beta:+.4f}{stars}")
            else:
                row_vals.append("   --  ")

        print(f"{name:<16} {'  '.join(row_vals)}  {sig_str:>5}")

    print("=" * 72)
    print(f"\nResults saved to: {cfg.irf.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IRF estimation")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()
    main(args.config)
