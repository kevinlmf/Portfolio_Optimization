# Advanced Portfolio Optimization System

> **End-to-end, production-ready system for portfolio construction integrating ML alpha factors, advanced risk models, and intelligent optimization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

---

##  Features

- **Alpha Mining**: 100+ technical, fundamental, macro, and ML-driven factors
- **Macro Integration**: Yield curves, VIX, GDP, inflation
- **Risk Models**: CAPM, multi-factor, Copula, CVaR
- **Optimization**: Sharpe, min-variance, risk parity, utility
- **Backtesting**: Rolling-window analysis & performance metrics
- **Enhanced Alpha Combination**: IC-weighted & regime-aware methods
- **Production Ready**: Modular, clean, and extensible design

---

##  Architecture

```
Portfolio_Optimization_System/
├── data/             # Market & macro data fetchers
├── strategy/         # Alpha & beta modeling
├── risk_control/     # Risk validation
├── execution_engine/ # Trading environment
├── scripts/          # Main entry points
└── results/          # Reports & analytics
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/kevinlmf/Portfolio_Optimization
cd Portfolio_Optimization_System
pip install -r requirements.txt

# Run main optimizer
python scripts/smart_portfolio_optimizer.py
```

---

## Core Optimization Methods

### A. Maximum Sharpe Ratio (max_sharpe) — Default Objective

```python
maximize: (portfolio_return - risk_free_rate) / portfolio_volatility
```

- **Location**: `alpha_beta_optimizer.py:253-325`
- **Goal**: Maximize risk-adjusted return
- **Use Case**: Core metric in most quantitative investment strategies

### B. Minimum Variance (min_variance) — Risk Control

```python
minimize: weights.T @ covariance_matrix @ weights
```

- **Goal**: Minimize total portfolio risk
- **Use Case**: Highly risk-averse investors

### C. Maximum Utility (max_utility) — Balancing Risk and Return

```python
maximize: portfolio_return - 0.5 * risk_aversion * portfolio_variance
```

- **Feature**: Customizable risk aversion coefficient
- **Flexibility**: Fits investors with different risk preferences

### D. Risk Parity (risk_parity) — Equal Risk Contribution

```python
# All assets contribute equally to portfolio risk
```

- **Philosophy**: Diversification
- **Goal**: Balance risk exposure across assets

---

### 

##  Alpha Factor Framework

### Categories & Count

- **Technical**: 40+ (RSI, MACD, Bollinger, Momentum)
- **Fundamental**: 25+ (P/E, P/B, ROE, Debt ratio)
- **Macroeconomic**: 20+ (Yield curve, VIX, DXY)
- **Machine Learning**: 15+ (Random Forest, XGBoost)
- **Microstructure**: 18+ (Bid-ask spread, volume imbalance)
- **Cross-Sectional**: 12+ (Sector rotation, size factors)
- **Alternative**: 10+ (Options flow, sentiment)

### Validation Standards

- IC threshold ≥ 0.02
- Statistical significance (t-tests, p-values)
- Decay analysis
- Cross-validation (time-series splits)
- Regime stability checks


**20+ new macroeconomic factors:**
- Treasury yield curves
- VIX index
- US Dollar Index (DXY)
- Credit spreads
- GDP, inflation, employment data

---

##  Risk Modeling Framework

| Method        | Use Case              | Strength                   |
|---------------|-----------------------|----------------------------|
| CAPM Beta     | Market neutral        | Simple, interpretable      |
| Multi-Factor  | Style exposure ctrl   | Fama-French 3/5 factors    |
| Dynamic Beta  | Regime shifts         | Time-varying exposures     |
| Copula Beta   | Tail risk mgmt        | Extreme dependency capture |
| CVaR Beta     | Downside protection   | Conditional risk measure   |

---
##  Historical Performance Targets (2020–2024)

| Metric          | System | Benchmark (SPY) | Excess |
|-----------------|--------|-----------------|--------|
| Annual Return   | 14.2%  | 12.1%          | +2.1%  |
| Volatility      | 16.8%  | 19.4%          | -2.6%  |
| Sharpe Ratio    | 0.93   | 0.71           | +0.22  |
| Max Drawdown    | -12.3% | -18.7%         | +6.4%  |
| Win Rate        | 61.4%  | 55.2%          | +6.2%  |
| Info Ratio      | 0.51   | —              | —      |

---


##  Multi-Objective Strategy

### Default Flow
```python
objectives = ['max_sharpe', 'min_variance', 'max_utility']
```

System defaults to max_sharpe result.

---


