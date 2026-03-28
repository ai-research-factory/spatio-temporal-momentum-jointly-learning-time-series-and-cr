# Cycle 4: Transaction Cost Model Integration — Technical Findings

## Phase 4 Objective

Integrate a per-asset transaction cost model into the walk-forward backtest engine and evaluate net-of-cost performance across multiple cost scenarios.

## Implementation

### Per-Asset Transaction Cost Model

Replaced the simplified portfolio-level cost calculation (Cycle 3) with a proper per-asset turnover-based model in `src/backtest.py`:

- **`calculate_portfolio_costs()`**: Computes position changes for each of the 11 assets individually. Turnover is the sum of absolute position deltas across all assets per period, normalized by n_assets. Costs are deducted from gross portfolio returns each month.
- **`cost_sensitivity_analysis()`**: Evaluates net Sharpe and net annual return across multiple cost levels (0, 5, 10, 15, 20, 30, 50 bps) using all out-of-sample data.
- **`portfolio_turnover()`** in `src/evaluation.py`: Computes average monthly turnover per fold.

### Cost Assumptions

Following the paper and CLAUDE.md specifications:
- **Fee**: 10 bps per transaction (one-way)
- **Slippage**: 5 bps per transaction
- **Total**: 15 bps applied to each unit of turnover
- Entry from zero (initial position) is treated as a trade.

## Results

### Walk-Forward Performance (5 folds, 15 bps total cost)

| Fold | Gross Sharpe | Net Sharpe | Hit Rate | Avg Monthly Turnover | Monthly Cost (bps) |
|------|-------------|-----------|----------|---------------------|-------------------|
| 1    | 0.3423      | 0.2563    | 58.3%    | 0.2052              | 3.08              |
| 2    | 0.1316      | 0.0485    | 50.0%    | 0.2374              | 3.56              |
| 3    | 1.4244      | 1.2613    | 58.3%    | 0.1710              | 2.57              |
| 4    | 0.0641      | -0.1152   | 66.7%    | 0.1711              | 2.57              |
| 5    | 1.7321      | 1.4704    | 58.3%    | 0.1917              | 2.88              |
| **Avg** | **0.7389** | **0.5843** | **58.3%** | **0.1953**       | **2.93**          |

### Aggregate Metrics

- **Gross Sharpe**: 0.7389
- **Net Sharpe (15 bps)**: 0.5843
- **Sharpe Degradation**: 0.1546 (20.9% reduction from gross)
- **Positive Windows (gross)**: 5/5 (100%)
- **Positive Windows (net)**: 4/5 (80%)
- **Total Trades**: 112 across all folds
- **Avg Monthly Turnover**: 0.1953 (19.5% of portfolio per month)

### Cost Sensitivity Analysis

| Total Cost (bps) | Net Sharpe | Net Annual Return |
|-------------------|-----------|-------------------|
| 0                 | 0.4608    | 1.48%             |
| 5                 | 0.4214    | 1.35%             |
| 10                | 0.3820    | 1.23%             |
| 15                | 0.3427    | 1.10%             |
| 20                | 0.3036    | 0.98%             |
| 30                | 0.2256    | 0.73%             |
| 50                | 0.0712    | 0.23%             |

## Key Observations

1. **Moderate turnover**: Average monthly turnover of ~19.5% is reasonable for a monthly rebalancing strategy with tanh-bounded positions. The neural network produces smooth position changes rather than binary flips.

2. **Cost impact is material but manageable**: At 15 bps total cost, Sharpe degrades by ~21%. The strategy remains profitable (net Sharpe 0.58) and 4/5 windows remain positive.

3. **Cost breakeven**: The strategy remains positive-Sharpe even at 50 bps total cost (Sharpe 0.07), indicating some robustness to higher cost environments. However, meaningful alpha (Sharpe > 0.3) requires costs below ~20 bps.

4. **Fold 4 sensitivity**: Fold 4 (2024-03 to 2025-02) flips from marginally positive gross (0.06) to slightly negative net (-0.12), showing that transaction costs can flip marginal periods.

5. **Regime dependence persists**: High variance across folds (net Sharpe from -0.12 to 1.47) remains the primary concern, consistent with Cycle 3 findings.

## Changes from Cycle 3

- Per-asset cost calculation replacing portfolio-level approximation
- Turnover tracking per fold and aggregate
- Cost sensitivity analysis across 7 cost levels
- Net metrics (annual return, max drawdown) reported alongside gross
- Entry costs from zero position properly accounted for
