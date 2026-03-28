"""
Data loading and feature engineering for Spatio-Temporal Momentum.
Loads ETF data from ARF Data API cached CSV files and computes
TSM (time-series momentum) and CSM (cross-sectional momentum) features.
"""
import os
import pandas as pd
import numpy as np

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"
TICKERS = ["XLE", "XLF", "XLK", "XLI", "XLU", "XLY", "XLP", "XLV", "XLC", "XLRE", "XLB"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fetch_and_cache(ticker: str, interval: str = "1d", period: str = "10y") -> pd.DataFrame:
    """Load ticker data from local cache, fetching from API if needed."""
    cache_path = os.path.join(DATA_DIR, f"{ticker}_{interval}.csv")
    if not os.path.exists(cache_path):
        import urllib.request
        url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    df = pd.read_csv(cache_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def load_monthly_returns(tickers: list[str] = None) -> pd.DataFrame:
    """Load daily data for all tickers and resample to monthly returns.

    Returns:
        DataFrame with monthly returns, columns = tickers, index = month-end dates.
    """
    tickers = tickers or TICKERS
    close_prices = {}
    for ticker in tickers:
        df = fetch_and_cache(ticker)
        close_prices[ticker] = df["close"]

    # Align all tickers on common dates
    prices = pd.DataFrame(close_prices).dropna()

    # Resample to monthly (last business day) closing prices, then compute returns
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def compute_tsm_features(returns: pd.DataFrame, lookbacks: list[int] = None) -> pd.DataFrame:
    """Compute time-series momentum features.

    For each asset, compute cumulative return over lookback windows.
    Default lookbacks: [1, 3, 6, 12] months (paper standard).

    Args:
        returns: Monthly returns DataFrame (assets as columns).
        lookbacks: List of lookback periods in months.

    Returns:
        DataFrame with TSM features. Columns named like 'XLE_tsm_1', 'XLE_tsm_3', etc.
    """
    lookbacks = lookbacks or [1, 3, 6, 12]
    features = {}
    for col in returns.columns:
        for lb in lookbacks:
            # Cumulative return over lookback window (momentum signal)
            features[f"{col}_tsm_{lb}"] = returns[col].rolling(lb).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
    return pd.DataFrame(features, index=returns.index)


def compute_csm_features(returns: pd.DataFrame, lookbacks: list[int] = None) -> pd.DataFrame:
    """Compute cross-sectional momentum features.

    For each asset, compute its rank relative to peers over lookback windows.
    Ranks are normalized to [-1, 1].

    Args:
        returns: Monthly returns DataFrame.
        lookbacks: List of lookback periods in months.

    Returns:
        DataFrame with CSM features. Columns named like 'XLE_csm_1', etc.
    """
    lookbacks = lookbacks or [1, 3, 6, 12]
    n_assets = returns.shape[1]
    features = {}

    for lb in lookbacks:
        # Cumulative returns over lookback
        cum_ret = returns.rolling(lb).apply(lambda x: (1 + x).prod() - 1, raw=False)
        # Cross-sectional rank, normalized to [-1, 1]
        ranked = cum_ret.rank(axis=1, pct=True) * 2 - 1
        for col in returns.columns:
            features[f"{col}_csm_{lb}"] = ranked[col]

    return pd.DataFrame(features, index=returns.index)


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Build combined TSM + CSM feature matrix.

    Args:
        returns: Monthly returns DataFrame.

    Returns:
        Combined feature DataFrame (TSM and CSM features concatenated).
    """
    tsm = compute_tsm_features(returns)
    csm = compute_csm_features(returns)
    features = pd.concat([tsm, csm], axis=1).dropna()
    return features
