# Open Questions

## Data Constraints

- **XLC inception date**: XLC (Communication Services Select Sector SPDR) began trading in June 2018, limiting our full 11-ETF universe to ~7.5 years of history. This means fold 1 of the walk-forward validation only has 21 months of training data instead of the target 60 months. The paper likely uses a longer history or different ETFs.

- **Universe size**: The paper may use a broader universe beyond 11 sector ETFs. Our universe is constrained to the 11 SPDRs specified in the design brief.

- **Monthly resampling**: Daily data is resampled to monthly using month-end close prices. The paper's exact resampling convention (last trading day vs calendar month-end) is not specified.

## Model Architecture

- **Hidden size and depth**: We use hidden_size=64 and 2 hidden layers as a reasonable default. The paper's exact architecture hyperparameters may differ.

- **Loss function**: We implement a Sharpe-like loss (mean/std of portfolio returns). The paper may use a different differentiable objective or MSE on future returns.

- **Training epochs**: 200 epochs per fold is a heuristic. No early stopping is currently implemented.

## Evaluation Protocol

- **Expanding vs. rolling window**: Our implementation uses an expanding training window (all data from start up to the fold boundary) for folds 1-4, switching to a 60-month rolling window only for fold 5 when enough history exists. The paper specifies a rolling 60-month window, which would require more historical data than we have.

- **Fold 1 training data**: Only 21 months of training data for fold 1. Results from this fold should be interpreted with caution.

## ARF Data API

- No issues encountered with the ARF Data API. All 11 tickers returned valid OHLCV data.
