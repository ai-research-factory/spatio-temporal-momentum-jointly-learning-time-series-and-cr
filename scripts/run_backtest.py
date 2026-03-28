"""
Walk-forward backtest runner for Spatio-Temporal Momentum strategy.

Executes the full walk-forward validation with integrated transaction costs:
  - 5 folds, 60-month training window, 12-month test window
  - Trains model on each fold's training data
  - Generates out-of-sample positions and computes portfolio returns
  - Applies per-asset transaction cost model (fee + slippage)
  - Performs cost sensitivity analysis across multiple cost levels
  - Saves per-fold and aggregate metrics to reports/cycle_4/
"""
import json
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import load_monthly_returns, build_features, TICKERS
from src.model import train_model, predict_positions
from src.evaluation import compute_fold_metrics, sharpe_ratio, portfolio_turnover
from src.backtest import (
    BacktestConfig, BacktestResult,
    calculate_portfolio_costs, cost_sensitivity_analysis,
    generate_metrics_json,
)


def run_walkforward(
    n_splits: int = 5,
    train_months: int = 60,
    test_months: int = 12,
) -> tuple:
    """Run walk-forward validation with per-asset transaction cost model.

    Args:
        n_splits: Number of walk-forward folds.
        train_months: Training window in months.
        test_months: Test window in months.

    Returns:
        Tuple of (fold_results, backtest_results, config, all_fold_positions, all_fold_returns).
    """
    print("Loading monthly returns...")
    returns = load_monthly_returns()
    print(f"  Returns shape: {returns.shape} ({returns.index[0].date()} to {returns.index[-1].date()})")

    tickers = list(returns.columns)
    n_assets = len(tickers)

    print("Building features...")
    features = build_features(returns)
    print(f"  Features shape: {features.shape}")

    # Align features and forward returns (target = next month's return)
    aligned_features = features.iloc[:-1]
    forward_returns = returns.loc[features.index[1:]].copy()
    forward_returns.index = aligned_features.index

    common_idx = aligned_features.index.intersection(forward_returns.index)
    aligned_features = aligned_features.loc[common_idx]
    forward_returns = forward_returns.loc[common_idx]

    total_months = len(aligned_features)
    required = train_months + test_months
    print(f"  Total aligned months: {total_months}, required per fold: {required}")

    if total_months < required:
        print(f"WARNING: Not enough data ({total_months} months) for {train_months}+{test_months} month folds.")
        print("Reducing train_months to fit available data.")
        train_months = total_months - test_months * n_splits
        if train_months < 24:
            train_months = 24
        print(f"  Adjusted train_months={train_months}")

    fold_results = []
    backtest_results = []
    all_fold_positions = []
    all_fold_returns = []
    config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0, n_splits=n_splits)

    feat_array = aligned_features.values
    ret_array = forward_returns.values

    last_test_end = total_months
    first_test_start = last_test_end - n_splits * test_months

    for fold in range(n_splits):
        test_start = first_test_start + fold * test_months
        test_end = test_start + test_months
        train_start = max(0, test_start - train_months)
        train_end = test_start

        if train_end - train_start < 12:
            print(f"  Fold {fold+1}: skipping (insufficient training data, {train_end - train_start} months)")
            continue
        if test_end > total_months:
            test_end = total_months

        train_dates = aligned_features.index[train_start:train_end]
        test_dates = aligned_features.index[test_start:test_end]

        print(f"\n  Fold {fold+1}/{n_splits}:")
        print(f"    Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} months)")
        print(f"    Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} months)")

        X_train = feat_array[train_start:train_end].copy()
        Y_train = ret_array[train_start:train_end].copy()
        X_test = feat_array[test_start:test_end].copy()
        Y_test = ret_array[test_start:test_end].copy()

        # Normalize features using train-only statistics (no look-ahead bias)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model = train_model(
            X_train, Y_train, n_assets=n_assets,
            epochs=200, lr=1e-3, hidden_size=64, batch_size=32,
        )

        positions = predict_positions(model, X_test)  # (test_months, n_assets)

        # Store for aggregate cost sensitivity analysis
        all_fold_positions.append(positions)
        all_fold_returns.append(Y_test)

        # Per-asset transaction cost calculation
        net_portfolio_ret, cost_info = calculate_portfolio_costs(
            Y_test, positions, n_assets, config,
        )

        # Gross portfolio returns (no costs)
        gross_portfolio_ret = (positions * Y_test).sum(axis=1) / n_assets
        port_ret_gross_series = pd.Series(gross_portfolio_ret, index=test_dates)
        port_ret_net_series = pd.Series(net_portfolio_ret, index=test_dates)

        gross_metrics = compute_fold_metrics(port_ret_gross_series, periods_per_year=12)
        net_metrics = compute_fold_metrics(port_ret_net_series, periods_per_year=12)

        turnover = portfolio_turnover(positions)
        n_trades = int(np.abs(np.diff(positions, axis=0)).sum())

        fold_result = {
            "fold": fold + 1,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "n_train_months": len(train_dates),
            "n_test_months": len(test_dates),
            "gross_sharpe": gross_metrics["sharpe_ratio"],
            "gross_annual_return": gross_metrics["annual_return"],
            "gross_max_drawdown": gross_metrics["max_drawdown"],
            "net_sharpe": net_metrics["sharpe_ratio"],
            "net_annual_return": net_metrics["annual_return"],
            "net_max_drawdown": net_metrics["max_drawdown"],
            "hit_rate": gross_metrics["hit_rate"],
            "avg_monthly_turnover": cost_info["avg_monthly_turnover"],
            "avg_monthly_cost_bps": cost_info["avg_monthly_cost_bps"],
            "total_cost_drag": cost_info["total_cost_drag"],
            "total_trades": n_trades,
        }
        fold_results.append(fold_result)
        print(f"    Gross Sharpe: {fold_result['gross_sharpe']:.4f}, "
              f"Net Sharpe: {fold_result['net_sharpe']:.4f}, "
              f"Hit Rate: {fold_result['hit_rate']:.4f}")
        print(f"    Avg Monthly Turnover: {fold_result['avg_monthly_turnover']:.4f}, "
              f"Cost Drag: {fold_result['total_cost_drag']:.6f}")

        backtest_results.append(BacktestResult(
            window=fold + 1,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            gross_sharpe=gross_metrics["sharpe_ratio"],
            net_sharpe=net_metrics["sharpe_ratio"],
            annual_return=net_metrics["annual_return"],
            max_drawdown=net_metrics["max_drawdown"],
            total_trades=n_trades,
            hit_rate=gross_metrics["hit_rate"],
            pnl_series=port_ret_net_series,
        ))

    return fold_results, backtest_results, config, all_fold_positions, all_fold_returns


def main():
    print("=" * 60)
    print("Spatio-Temporal Momentum Walk-Forward Backtest (Phase 4: Transaction Costs)")
    print("=" * 60)

    fold_results, backtest_results, config, all_positions, all_returns = run_walkforward(
        n_splits=5, train_months=60, test_months=12,
    )

    n_assets = len(TICKERS)

    # Cost sensitivity analysis across all OOS periods
    print("\n" + "=" * 60)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 60)
    combined_positions = np.vstack(all_positions)
    combined_returns = np.vstack(all_returns)
    sensitivity = cost_sensitivity_analysis(
        combined_returns, combined_positions, n_assets,
        fee_levels_bps=[0, 5, 10, 15, 20, 30, 50],
    )
    for s in sensitivity:
        print(f"  {s['total_cost_bps']:3d} bps -> Net Sharpe: {s['net_sharpe']:.4f}, "
              f"Net Annual Return: {s['net_annual_return']:.4f}")

    # Save reports to cycle_4
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "cycle_4")
    os.makedirs(report_dir, exist_ok=True)

    # Save walkforward metrics
    wf_metrics_path = os.path.join(report_dir, "walkforward_gross_metrics.json")
    with open(wf_metrics_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"\nSaved fold metrics to {wf_metrics_path}")

    # Compute aggregate turnover stats
    avg_turnover = np.mean([r["avg_monthly_turnover"] for r in fold_results]) if fold_results else 0.0
    avg_cost_bps = np.mean([r["avg_monthly_cost_bps"] for r in fold_results]) if fold_results else 0.0

    # Generate ARF-standard metrics.json
    custom_metrics = {
        "n_splits": 5,
        "train_months": 60,
        "test_months": 12,
        "model": "SpatioTemporalMomentumNet",
        "features": "TSM+CSM (lookbacks: 1,3,6,12 months)",
        "per_fold_gross_sharpe": [r["gross_sharpe"] for r in fold_results],
        "per_fold_net_sharpe": [r["net_sharpe"] for r in fold_results],
        "per_fold_turnover": [r["avg_monthly_turnover"] for r in fold_results],
        "avg_monthly_turnover": round(avg_turnover, 4),
        "avg_monthly_cost_bps": round(avg_cost_bps, 2),
        "cost_sensitivity": sensitivity,
    }
    metrics = generate_metrics_json(backtest_results, config, custom_metrics)
    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved ARF metrics to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Folds completed: {len(fold_results)}")
    if fold_results:
        avg_gross = np.mean([r["gross_sharpe"] for r in fold_results])
        avg_net = np.mean([r["net_sharpe"] for r in fold_results])
        pos_windows_gross = sum(1 for r in fold_results if r["gross_sharpe"] > 0)
        pos_windows_net = sum(1 for r in fold_results if r["net_sharpe"] > 0)
        print(f"  Avg Gross Sharpe: {avg_gross:.4f}")
        print(f"  Avg Net Sharpe:   {avg_net:.4f}")
        print(f"  Positive windows (gross): {pos_windows_gross}/{len(fold_results)}")
        print(f"  Positive windows (net):   {pos_windows_net}/{len(fold_results)}")
        print(f"  Avg Monthly Turnover:     {avg_turnover:.4f}")
        print(f"  Avg Monthly Cost (bps):   {avg_cost_bps:.2f}")
        print(f"  Transaction costs: {config.fee_bps} bps fee + {config.slippage_bps} bps slippage")


if __name__ == "__main__":
    main()
