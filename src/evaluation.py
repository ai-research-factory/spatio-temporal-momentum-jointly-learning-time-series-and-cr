"""
Evaluation metrics for Spatio-Temporal Momentum strategy.
Computes Sharpe ratio, annualized return, and maximum drawdown.
"""
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from a returns series.

    Args:
        returns: Periodic returns (e.g., monthly).
        periods_per_year: Number of periods per year (12 for monthly, 252 for daily).
        risk_free_rate: Annual risk-free rate.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compute annualized return from a returns series.

    Args:
        returns: Periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized return as a decimal.
    """
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).prod()
    n_periods = len(returns)
    return float(cumulative ** (periods_per_year / n_periods) - 1)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a returns series.

    Args:
        returns: Periodic returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.25 for 25% drawdown).
    """
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def hit_rate(returns: pd.Series) -> float:
    """Compute fraction of positive return periods."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def portfolio_turnover(positions: np.ndarray) -> float:
    """Compute average monthly portfolio turnover from position array.

    Turnover = mean of sum of absolute position changes across assets per period.

    Args:
        positions: Array of shape (T, n_assets) with values in [-1, 1].

    Returns:
        Average turnover per period.
    """
    if len(positions) < 2:
        return 0.0
    prev = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
    changes = np.abs(positions - prev)
    return float(changes.sum(axis=1).mean())


def compute_fold_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Compute all evaluation metrics for a single fold.

    Args:
        returns: Out-of-sample returns for this fold.
        periods_per_year: Number of periods per year.

    Returns:
        Dict with sharpe, annual_return, max_drawdown, hit_rate.
    """
    return {
        "sharpe_ratio": round(sharpe_ratio(returns, periods_per_year), 4),
        "annual_return": round(annualized_return(returns, periods_per_year), 4),
        "max_drawdown": round(max_drawdown(returns), 4),
        "hit_rate": round(hit_rate(returns), 4),
    }
