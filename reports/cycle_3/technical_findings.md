# Cycle 3: Walk-Forward Validation Framework — Technical Findings

## Objective

Implement and execute a walk-forward backtest engine following the paper's evaluation protocol for the Spatio-Temporal Momentum strategy.

## Implementation

### Walk-Forward Configuration
- **Folds**: 5
- **Training window**: 60 months (target; fold 1 used 21 months due to data availability)
- **Test window**: 12 months per fold
- **Step**: Non-overlapping 12-month OOS windows, stepping forward chronologically

### Model
- `SpatioTemporalMomentumNet`: 2-layer MLP (hidden_size=64, ReLU, dropout=0.1, tanh output)
- Loss: Negative Sharpe-like objective (mean/std of portfolio returns)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4), 200 epochs per fold

### Features
- **TSM (Time-Series Momentum)**: Cumulative returns over [1, 3, 6, 12]-month lookbacks per asset
- **CSM (Cross-Sectional Momentum)**: Percentile rank (normalized to [-1, 1]) over [1, 3, 6, 12]-month lookbacks per asset
- Total: 88 features (11 assets x 4 lookbacks x 2 types)
- Normalization: Train-only mean/std per fold (no look-ahead bias)

### Data
- Universe: 11 US Sector SPDRs (XLE, XLF, XLK, XLI, XLU, XLY, XLP, XLV, XLC, XLRE, XLB)
- Source: ARF Data API, daily OHLCV, resampled to monthly returns
- Period: 2018-07 to 2026-03 (93 monthly observations after alignment)
- XLC (Communication Services) limited history start to mid-2018

## Results

### Per-Fold Metrics

| Fold | Train Period | Test Period | Gross Sharpe | Net Sharpe | Hit Rate |
|------|-------------|-------------|-------------|-----------|----------|
| 1 | 2019-06 to 2021-02 (21mo) | 2021-03 to 2022-02 | 0.2970 | 0.2688 | 0.5833 |
| 2 | 2019-06 to 2022-02 (33mo) | 2022-03 to 2023-02 | 0.0935 | 0.0791 | 0.4167 |
| 3 | 2019-06 to 2023-02 (45mo) | 2023-03 to 2024-02 | 1.6662 | 1.6414 | 0.6667 |
| 4 | 2019-06 to 2024-02 (57mo) | 2024-03 to 2025-02 | -0.4661 | -0.5304 | 0.5000 |
| 5 | 2020-03 to 2025-02 (60mo) | 2025-03 to 2026-02 | 2.5067 | 2.4629 | 0.8333 |

### Aggregate Metrics
- **Average Gross Sharpe**: 0.8195
- **Average Net Sharpe**: 0.7844
- **Positive windows**: 4/5 (80%)
- **Average Hit Rate**: 0.60
- **Average Annual Return**: 1.75%
- **Worst Max Drawdown**: -5.69%
- **Transaction Costs**: 10 bps fee + 5 bps slippage

## Observations

1. **High variance across folds**: Sharpe ratios range from -0.47 to 2.51, indicating significant regime dependence. The model performs well in trending markets (folds 3, 5) but struggles in volatile/choppy periods (fold 4 covering 2024 market rotation).

2. **Transaction costs have minimal impact**: The difference between gross and net Sharpe is small (~0.04) because monthly rebalancing generates few position changes relative to daily strategies.

3. **Data limitation on fold 1**: Only 21 months of training data for fold 1 (vs target 60 months) due to XLC's late inception (June 2018). This is documented in open_questions.md.

4. **Conservative positions**: The tanh output and Sharpe-maximizing loss tend to produce small positions, resulting in modest absolute returns but controlled drawdowns.

5. **No look-ahead bias**: Feature normalization uses train-only statistics; walk-forward structure prevents any future information leakage.

## Files Produced
- `src/evaluation.py` — Sharpe ratio, annualized return, max drawdown, hit rate functions
- `src/data.py` — ETF data loading, monthly resampling, TSM/CSM feature engineering
- `src/model.py` — SpatioTemporalMomentumNet MLP and training/prediction functions
- `scripts/run_backtest.py` — Walk-forward backtest execution script
- `reports/cycle_3/walkforward_gross_metrics.json` — Per-fold Sharpe ratios (5 folds)
- `reports/cycle_3/metrics.json` — ARF-standard metrics
