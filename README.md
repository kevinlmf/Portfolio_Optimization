# Advanced Portfolio Optimization System

> **End-to-end, production-ready system for portfolio construction ---
> integrating ML alpha factors, advanced risk models, and intelligent
> optimization.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

## Features

-   Alpha Mining: 100+ technical, fundamental, macro, and ML-driven
    factors\
-   Macro Integration: Yield curves, VIX, GDP, inflation\
-   Risk Models: CAPM, multi-factor, Copula, CVaR\
-   Optimization: Sharpe, min-variance, risk parity, utility\
-   Backtesting: Rolling-window analysis & performance metrics\
-   Enhanced Alpha Combination: IC-weighted & regime-aware methods\
-   Production Ready: Modular, clean, and extensible design

## Architecture

    Portfolio_Optimization_System/
    ├── data/           # Market & macro data fetchers
    ├── strategy/       # Alpha & beta modeling
    ├── risk_control/   # Risk validation
    ├── execution_engine/ # Trading environment
    ├── scripts/        # Main entry points
    └── results/        # Reports & analytics

## Quick Start

``` bash
git clone https://github.com/kevinlmf/Portfolio_Optimization
cd Portfolio_Optimization_System
pip install -r requirements.txt

# Run main optimizer
python scripts/smart_portfolio_optimizer.py
```

## Sample Results

  Metric          Smart Optimizer   SPY
  --------------- ----------------- --------
  Annual Return   14.2%             12.1%
  Sharpe Ratio    0.93              0.71
  Max Drawdown    -12.3%            -18.7%

## Advanced Usage

-   Custom Factors\
    Easily extend `strategy/factor/alpha/` with your own signals.\
-   Risk Models\
    Switch between CAPM, Copula, CVaR with a single config.\
-   Optimization Objectives\
    Choose Sharpe, variance, risk parity, or utility.

## Output

Results saved to `results/`: - `optimization_report.txt`\
- `optimal_weights.csv`\
- `backtest_results.csv`\
- Performance plots

## Changelog

-   v2.1 (Sep 2024): Macro factors, enhanced alpha combiner,
    regime-aware weighting\
-   v2.0: System refactor, advanced risk modeling\
-   v1.0: Initial release

------------------------------------------------------------------------



