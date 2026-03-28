"""
ARF Standard Backtest Framework
Walk-forward validation with transaction cost accounting.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_bps: float = 10.0       # Transaction fee in basis points
    slippage_bps: float = 5.0   # Slippage in basis points
    train_ratio: float = 0.7    # Train window ratio for walk-forward
    n_splits: int = 10          # Number of walk-forward windows
    gap: int = 1                # Gap between train and test (prevent leakage)
    min_train_size: int = 252   # Minimum training samples (~1 year daily)


@dataclass
class BacktestResult:
    """Results from a single walk-forward window."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    pnl_series: Optional[pd.Series] = field(default=None, repr=False)


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Usage:
        validator = WalkForwardValidator(config)
        for train_idx, test_idx in validator.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            # Train model on train_df, evaluate on test_df
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def split(self, data: pd.DataFrame):
        """Generate train/test index pairs for walk-forward validation."""
        n = len(data)
        cfg = self.config
        test_size = max(1, (n - cfg.min_train_size) // cfg.n_splits)

        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap
            train_start = max(0, int(train_end * (1 - cfg.train_ratio))) if cfg.train_ratio < 1.0 else 0

            if train_end - train_start < cfg.min_train_size:
                continue
            if test_start >= test_end:
                continue

            yield (
                list(range(train_start, train_end)),
                list(range(test_start, test_end)),
            )


def calculate_costs(returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate transaction costs from position changes (1D portfolio-level).

    Args:
        returns: Gross returns series
        positions: Position series (-1, 0, 1 or continuous)
        config: Backtest configuration with fee/slippage settings

    Returns:
        Net returns after costs
    """
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = (config.fee_bps + config.slippage_bps) / 10000
    costs = trades * cost_per_trade
    return returns - costs


def calculate_portfolio_costs(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    n_assets: int,
    config: BacktestConfig,
) -> tuple[np.ndarray, dict]:
    """
    Calculate per-asset transaction costs and net portfolio returns.

    Computes turnover as the sum of absolute position changes across all assets,
    then deducts proportional costs from portfolio returns each period.

    Args:
        gross_returns: Array of shape (T, n_assets) — asset returns per period.
        positions: Array of shape (T, n_assets) — position weights in [-1, 1].
        n_assets: Number of assets.
        config: Backtest configuration with fee/slippage settings.

    Returns:
        Tuple of (net_portfolio_returns array of shape (T,), cost_info dict).
    """
    cost_bps = config.fee_bps + config.slippage_bps
    cost_rate = cost_bps / 10000.0

    # Per-asset position changes (first period assumes entry from zero)
    prev_positions = np.vstack([np.zeros((1, n_assets)), positions[:-1]])
    turnover_per_asset = np.abs(positions - prev_positions)  # (T, n_assets)

    # Total turnover per period: sum across assets, normalized by n_assets
    turnover_per_period = turnover_per_asset.sum(axis=1) / n_assets  # (T,)

    # Cost per period
    costs_per_period = turnover_per_period * cost_rate  # (T,)

    # Gross portfolio returns
    gross_portfolio = (positions * gross_returns).sum(axis=1) / n_assets  # (T,)

    # Net portfolio returns
    net_portfolio = gross_portfolio - costs_per_period

    # Aggregate cost statistics
    total_turnover = float(turnover_per_asset.sum())
    avg_monthly_turnover = float(turnover_per_period.mean())
    total_cost_drag = float(costs_per_period.sum())

    cost_info = {
        "total_turnover": round(total_turnover, 4),
        "avg_monthly_turnover": round(avg_monthly_turnover, 4),
        "total_cost_drag": round(total_cost_drag, 6),
        "avg_monthly_cost_bps": round(avg_monthly_turnover * cost_bps, 2),
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
    }

    return net_portfolio, cost_info


def cost_sensitivity_analysis(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    n_assets: int,
    fee_levels_bps: list[float] = None,
) -> list[dict]:
    """
    Evaluate net performance across different transaction cost levels.

    Args:
        gross_returns: Array of shape (T, n_assets).
        positions: Array of shape (T, n_assets).
        n_assets: Number of assets.
        fee_levels_bps: List of total cost levels in bps to test.

    Returns:
        List of dicts with cost level and resulting net Sharpe.
    """
    if fee_levels_bps is None:
        fee_levels_bps = [0, 5, 10, 15, 20, 30, 50]

    results = []
    for total_bps in fee_levels_bps:
        cfg = BacktestConfig(fee_bps=total_bps, slippage_bps=0.0)
        net_ret, _ = calculate_portfolio_costs(gross_returns, positions, n_assets, cfg)
        net_series = pd.Series(net_ret)
        if len(net_series) > 1 and net_series.std() > 0:
            sharpe = float(np.sqrt(12) * net_series.mean() / net_series.std())
        else:
            sharpe = 0.0
        results.append({
            "total_cost_bps": total_bps,
            "net_sharpe": round(sharpe, 4),
            "net_annual_return": round(float((1 + net_series.mean()) ** 12 - 1), 4) if len(net_series) > 0 else 0.0,
        })
    return results


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics from a returns series.

    Args:
        returns: Daily (or periodic) returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

    Returns:
        Dict with sharpeRatio, annualReturn, maxDrawdown, hitRate, totalTrades
    """
    if len(returns) == 0:
        return {"sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0, "hitRate": 0.0}

    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    annual_return = float(cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1)

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "sharpeRatio": round(sharpe, 4),
        "annualReturn": round(annual_return, 4),
        "maxDrawdown": round(max_drawdown, 4),
        "hitRate": round(hit_rate, 4),
    }


def generate_metrics_json(
    results: list[BacktestResult],
    config: BacktestConfig,
    custom_metrics: Optional[dict] = None,
) -> dict:
    """
    Generate ARF-standard metrics.json from walk-forward results.

    Args:
        results: List of BacktestResult from each window
        config: Backtest configuration
        custom_metrics: Optional paper-specific metrics

    Returns:
        Dict matching ARF metrics.json schema
    """
    if not results:
        return {
            "sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0,
            "hitRate": 0.0, "totalTrades": 0,
            "transactionCosts": {"feeBps": config.fee_bps, "slippageBps": config.slippage_bps, "netSharpe": 0.0},
            "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
            "customMetrics": custom_metrics or {},
        }

    net_sharpes = [r.net_sharpe for r in results]
    positive_windows = sum(1 for s in net_sharpes if s > 0)

    return {
        "sharpeRatio": round(float(np.mean([r.gross_sharpe for r in results])), 4),
        "annualReturn": round(float(np.mean([r.annual_return for r in results])), 4),
        "maxDrawdown": round(float(min(r.max_drawdown for r in results)), 4),
        "hitRate": round(float(np.mean([r.hit_rate for r in results])), 4),
        "totalTrades": sum(r.total_trades for r in results),
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "walkForward": {
            "windows": len(results),
            "positiveWindows": positive_windows,
            "avgOosSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "customMetrics": custom_metrics or {},
    }
