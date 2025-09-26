# Smart Portfolio Optimizer

## Overview

This is an advanced portfolio optimization system that automatically
discovers the strongest alpha factors and beta risk models, then
optimizes asset allocation to maximize risk-adjusted returns.

## Core Features

### 1. Automated Alpha Factor Mining

-   Technical factors: Momentum, mean reversion, technical indicators
-   Fundamental factors: P/E ratio, revenue growth, dividend yield
-   Microstructure factors: Liquidity, price impact, volatility
    clustering
-   Cross-sectional factors: Relative strength, sector rotation
-   Machine learning factors: Predictive signals using RF, XGBoost, etc.

### 2. Multiple Beta Estimation Methods

-   CAPM Beta: Traditional market beta
-   Multi-factor Beta: Fama-French and similar models
-   Copula Beta: Dependency modeling via copulas
-   CVaR Beta: Conditional Value-at-Risk models

### 3. Optimization Objectives

-   Maximum Sharpe Ratio: Maximize risk-adjusted return
-   Minimum Variance: Minimize risk
-   Maximum Utility: Utility maximization with risk aversion
-   Risk Parity: Equal risk contribution allocation

### 4. Comprehensive Backtesting & Reporting

-   Dynamic rebalancing backtests
-   Risk decomposition analysis
-   Automated report generation with charts

## Quick Start

### Run the Optimizer

``` bash
cd scripts
python smart_portfolio_optimizer.py
```

### Customize the Ticker Universe

Edit the `tickers` list in `smart_portfolio_optimizer.py`:

``` python
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'JPM', 'BAC', 'WFC',
    'PG', 'KO', 'WMT',
]
```

### Adjust Optimization Parameters

``` python
optimizer = SmartPortfolioOptimizer(
    start_date="2020-01-01",
    end_date="2024-01-01",
    risk_free_rate=0.03
)

alpha_factors = optimizer.mine_alpha_factors(
    min_ic_threshold=0.015,
    top_n_factors=25
)

result = optimizer.optimize_portfolio(
    objective='max_sharpe',
    constraints={
        'max_weight': 0.25,
        'min_weight': 0.01
    },
    alpha_weight=0.7
)
```

## Output Files

All outputs are saved under `results/smart_optimizer/`:

1.  `optimization_report.txt` -- Full optimization report
2.  `optimal_weights.csv` -- Portfolio weights and expected returns
3.  `backtest_results.csv` -- Detailed backtest results
4.  `portfolio_weights.png` -- Allocation visualization
5.  `backtest_performance.png` -- Performance chart with cumulative
    return and rolling Sharpe

## Advanced Usage

### Programmatic API

``` python
from smart_portfolio_optimizer import SmartPortfolioOptimizer

optimizer = SmartPortfolioOptimizer()
optimizer.fetch_market_data(['AAPL', 'MSFT', 'GOOGL'])
optimizer.mine_alpha_factors()
optimizer.estimate_risk_models()
result = optimizer.optimize_portfolio(objective='max_sharpe')
backtest = optimizer.backtest_strategy(rebalance_frequency='monthly')
report = optimizer.generate_report()
```

### Batch Testing

``` python
objectives = ['max_sharpe', 'min_variance', 'max_utility']
results = {}

for obj in objectives:
    result = optimizer.optimize_portfolio(objective=obj)
    results[obj] = result
    print(f"{obj}: Sharpe = {result.get('sharpe_ratio', 0):.3f}")
```

### Custom Constraints

``` python
custom_constraints = {
    'max_weight': 0.20,
    'min_weight': 0.02,
    'max_concentration': 0.6,
    'min_diversification': 8
}
result = optimizer.optimize_portfolio(constraints=custom_constraints)
```

## Performance Metrics

-   Total Return
-   Annual Return
-   Annual Volatility
-   Sharpe Ratio = (Return − Risk-free rate) / Volatility
-   Max Drawdown
-   Win Rate

### Risk Metrics

-   Effective Assets = 1 / Σ(weight²)
-   Concentration Index (Herfindahl index)
-   Risk Contributions by asset

## Notes

1.  Data quality matters: ensure stable connection
2.  Long histories may take time to compute
3.  Past performance does not guarantee future results

## Troubleshooting

-   Import errors:

``` bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn yfinance
```

-   Data fetch errors: Check tickers and network
-   Optimization errors: Relax overly strict constraints

Enable debug logs:

``` python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## System Architecture

    SmartPortfolioOptimizer
    ├── Data Layer
    │   ├── EnhancedDataFetcher
    │   └── yfinance fallback
    ├── Factor Layer
    │   ├── RealAlphaMiner
    │   ├── RealBetaEstimator
    │   └── FactorValidator
    ├── Optimization Layer
    │   ├── Expected return calc
    │   ├── Covariance matrix
    │   └── Constrained solver
    └── Output Layer
        ├── Backtest engine
        ├── Report generation
        └── Visualization

## Changelog

### v1.0 (Current)

-   Initial release
-   Multiple alpha factor categories
-   Integrated beta estimation methods
-   Full backtesting and reporting pipeline

------------------------------------------------------------------------
