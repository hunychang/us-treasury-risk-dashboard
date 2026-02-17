"""CLI entry point: run the full walk-forward backtest and print metrics.

Usage:
    python run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on sys.path
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from config.config_loader import load_config
from data.fred_provider import FREDProvider
from data.cache_manager import CachedProvider
from data.returns import compute_returns
from models import build_models
from optimizer.min_variance import MinVarianceOptimizer
from backtester.engine import BacktestEngine
from backtester.benchmarks import equal_weight_benchmark, sixty_forty_benchmark
from metrics.performance import metrics_comparison_table
from utils.logging_setup import setup_logging


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    # ---- Data ---------------------------------------------------------------
    print("Fetching data from FRED...")
    provider = CachedProvider(FREDProvider())
    yields = provider.get(
        cfg.data.instruments, cfg.data.start_date, cfg.data.end_date
    )
    returns = compute_returns(
        yields,
        method=cfg.data.return_type,
        interpolation=cfg.data.interpolation,
        missing_handling=cfg.data.missing_handling,
    )
    print(
        f"Data loaded: {returns.shape[0]} observations, "
        f"{returns.shape[1]} instruments "
        f"({returns.index[0].date()} to {returns.index[-1].date()})"
    )

    # ---- Models -------------------------------------------------------------
    models = build_models(cfg.models)
    print(f"Models: {[m.name() for m in models]}")

    # ---- Optimizer ----------------------------------------------------------
    optimizer = MinVarianceOptimizer(
        long_only=cfg.portfolio.long_only,
        weight_sum=cfg.portfolio.weight_sum,
        transaction_cost_bps=cfg.portfolio.transaction_cost_bps,
    )

    # ---- Backtest -----------------------------------------------------------
    oos_start = pd.Timestamp(cfg.evaluation.oos_start)
    engine = BacktestEngine(
        returns, models, optimizer,
        cfg.portfolio.rebalance_frequency, oos_start,
    )
    print(f"Running walk-forward backtest (OOS from {oos_start.date()})...")
    results = engine.run()

    # ---- Benchmarks ---------------------------------------------------------
    if "equal_weight" in cfg.evaluation.benchmark:
        results["equal_weight"] = equal_weight_benchmark(returns, oos_start)
    if "60_40_proxy" in cfg.evaluation.benchmark:
        results["60_40_proxy"] = sixty_forty_benchmark(returns, oos_start)

    # ---- Metrics ------------------------------------------------------------
    table = metrics_comparison_table(results)

    print("\n" + "=" * 70)
    print("RISK METRICS COMPARISON")
    print("=" * 70)
    print(table.to_string(float_format="%.4f"))
    print("=" * 70)

    # ---- Save outputs -------------------------------------------------------
    if cfg.logging.save_weights:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        for name, res in results.items():
            res.weights_history.to_csv(out_dir / f"weights_{name}.csv")
        table.to_csv(out_dir / "metrics.csv")

        # Also save portfolio returns
        rets_df = pd.DataFrame(
            {name: res.portfolio_returns for name, res in results.items()}
        )
        rets_df.to_csv(out_dir / "portfolio_returns.csv")

        print(f"\nResults saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
