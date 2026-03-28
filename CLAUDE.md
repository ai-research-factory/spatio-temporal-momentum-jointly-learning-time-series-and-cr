# Spatio-Temporal Momentum: Jointly Learning Time-Series and Cross-Sectional Strategies

## Project ID
proj_cc30609a

## Taxonomy
StatArb, Other

## Current Cycle
3

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Traditional momentum strategies are typically divided into two distinct types: time-series momentum (TSM), which bets on an asset's own past performance, and cross-sectional momentum (CSM), which bets on an asset's performance relative to its peers. These are often modeled and implemented as separate, independent strategies. This paper argues that this separation is suboptimal as it fails to capture the potentially complex, non-linear interactions between TSM and CSM signals.
The paper introduces 'Spatio-Temporal Momentum', a unified model class that uses a single neural network to learn trading signals for an entire portfolio of assets simultaneously. By feeding both TSM and CSM-derived features into the network, the model is designed to learn a combined strategy that dynamically weights the two momentum types, potentially leading to more robust and profitable signals than either strategy in isolation.

### Datasets
US Sector ETFs (11 SPDRs: XLE, XLF, XLK, XLI, XLU, XLY, XLP, XLV, XLC, XLRE, XLB) via yfinance API

### Targets
The primary target is the sign of the next month's return for each asset in the portfolio. The model will output a continuous value between -1 and 1 for each asset, representing the desired portfolio weight (position).

### Model
The model is a neural network, likely a Multi-Layer Perceptron (MLP), that takes concatenated time-series momentum (TSM) and cross-sectional momentum (CSM) features for all assets in the portfolio as input. For a universe of N assets and F features per asset, the input is a flattened vector or a 2D tensor of shape (N, F). The output layer has N neurons, one for each asset, with a `tanh` activation function to produce position weights between -1 (full short) and +1 (full long). The key architectural feature is that the single network processes all assets jointly, allowing it to learn interactions between them.

### Training
The model is trained and evaluated using a walk-forward validation scheme. For each fold, the model is trained on a rolling window of historical data (e.g., 60 months) to predict the next month's returns. The loss function is likely a variation of Mean Squared Error on future returns or a direct optimization of a risk-adjusted return metric like the Sharpe Ratio (e.g., using a custom differentiable Sharpe loss). The trained model is then used to generate signals for the subsequent out-of-sample period (e.g., 12 months). The training window then slides forward, and the process is repeated.

### Evaluation
The primary evaluation metric is the out-of-sample Sharpe Ratio of the portfolio strategy, calculated after applying a transaction cost model (e.g., 5-10 bps per transaction). This net Sharpe Ratio will be compared against several benchmarks: a simple 1/N combination of standalone TSM and CSM strategies, each strategy individually, and a buy-and-hold portfolio of the asset universe. Other reported metrics will include annualized return, maximum drawdown, portfolio turnover, and the percentage of profitable walk-forward windows.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 3)


### Phase 3: ウォークフォワード検証フレームワークの実装 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: 論文の評価プロトコルに沿った、ウォークフォワード方式のバックテストエンジンを構築する。

**具体的な作業指示**:
`src/backtest.py`に`WalkForwardValidator`クラスを実装します。このクラスは、`n_splits=5`、`train_period_months=60`、`test_period_months=12`をデフォルトパラメータとして初期化します。その`run`メソッドは、データセット全体をウォークフォワードでループ処理します。各ループで、1) 訓練/テストデータを分割、2) 訓練データでモデルを再学習、3) テストデータでシグナルを生成、4) ポートフォリオのリターンを計算します。`src/evaluation.py`にSharpe比、年率リターン、最大ドローダウンを計算する関数を実装します。`scripts/run_backtest.py`を作成し、この検証を実行して、各フォールドの指標を`reports/cycle_3/walkforward_gross_metrics.json`に保存します。

**期待される出力ファイル**:
- src/backtest.py
- src/evaluation.py
- reports/cycle_3/walkforward_gross_metrics.json

**受入基準 (これを全て満たすまで完了としない)**:
- `walkforward_gross_metrics.json`が生成され、5つのfoldに対応するSharpe比が記録されている
- バックテストがエラーなく完了する




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

✓ Phase 1: コアモデルのスケルトン実装と健全性チェック — 合成データ上で動作する、基本的なSpatio-Temporal Momentum NNモデルを実装する。
✓ Phase 2: データパイプラインと特徴量エンジニアリング — yfinanceからETFデータを取得し、TSMおよびCSM特徴量を計算して保存する。
→ Phase 3: ウォークフォワード検証フレームワークの実装 — 論文の評価プロトコルに沿った、ウォークフォワード方式のバックテストエンジンを構築する。
  Phase 4: 取引コストモデルの統合 — バックテストエンジンに取引コストを組み込み、ネットパフォーマンスを評価する。
  Phase 5: ハイパーパラメータ最適化 — Optunaを用いて、最初の訓練フォールド上でニューラルネットワークの主要なハイパーパラメータを最適化する。
  Phase 6: ロバスト性検証（全長バックテスト） — 最適化されたハイパーパラメータを使用して、利用可能な全期間にわたるウォークフォワードバックテストを実行する。
  Phase 7: アブレーションスタディ：モデル比較 — 論文の核心的な主張を検証するため、統合モデルと個別戦略のパフォーマンスを比較する。
  Phase 8: 代替特徴量の探求 — ボラティリティやモメンタムの加速度など、論文で明示されていない追加特徴量の有効性を検証する。
  Phase 9: レポート生成と可視化 — すべての実験結果を統合し、発見事項をまとめた技術レポートを生成する。
  Phase 10: 最終化とエグゼクティブサマリー — 非技術者向けの要約を作成し、コードベースの品質を確保する。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_3/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_3/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
