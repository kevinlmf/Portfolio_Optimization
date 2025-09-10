# ML Alpha + Copula Beta Portfolio System --- Final Summary

## Summary

This project develops a **portfolio optimization system** that
integrates **machine learning--based alpha factors** with **copula-based
beta risk modeling**. It demonstrates:
- A **theory-driven framework** that combines modern portfolio theory,
ML feature extraction, and statistical copula models.
- A **modular, extensible architecture** with independent alpha, beta,
optimization, and evaluation modules.
- **Superior evaluation design**, including benchmark comparisons and
risk-adjusted performance metrics.

------------------------------------------------------------------------

## Current Project Structure

    Portfolio_Optimization_system/
    ├── FINAL_SUMMARY.md
    ├── README.md
    ├── complete_portfolio_example.py
    ├── data/
    │   ├── __init__.py
    │   └── real_data.py
    ├── env/
    │   └── portfolio_optimization_env.py
    ├── evaluation/
    │   └── __init__.py
    ├── factor/
    │   ├── alpha/
    │   │   ├── alpha_factor_evaluator.py
    │   │   ├── feature_engineering.py
    │   │   ├── fundamental_alpha_factors.py
    │   │   ├── ml_alpha_factors.py
    │   │   ├── price_volume_alpha_factors.py
    │   │   └── technical_alpha_factors.py
    │   └── beta/
    │       ├── beta_evaluator.py
    │       ├── copula_risk_models.py
    │       ├── cvar_risk_models.py
    │       ├── multi_factor_models.py
    │       └── traditional_risk_models.py
    ├── portfolio_optimization/
    │   └── alpha_beta_optimizer.py
    ├── requirements.txt
    ├── run_system.py
    └── theory/
        └── ml_alpha_copula_theory.md

## Completed Core Modules

### 1) Theory
- ML Alpha + Copula Beta theoretical framework
- Mathematical models and derivations
- Methodology comparisons

### 2) Alpha (Machine Learning)
- Financial feature engineering (`feature_engineering.py`)
- Technical indicators: RSI, MACD, Bollinger Bands
- Momentum & reversal factors
- Volatility features (e.g., historical, GARCH-style)
- Cross-sectional features (relative strength, rankings)

### 3) Beta (Statistics)
- Copula–CVaR Beta estimation framework
- Multiple copula families
- Tail dependence modeling

### 4) Portfolio Optimization
- Integration of ML Alpha + Copula Beta
- Risk constraints & transaction costs
- Adaptive / regime-aware parameter updates

### 5) Evaluation
- Multi-dimensional performance metrics
- Benchmark comparisons
- Visualization & reporting

---

## Key Advantages

### Theory-Driven
- Grounded in modern portfolio theory
- Decoupled design: ML for Alpha; copula/statistics for Beta

### Modular Architecture
- Clear separation of concerns
- Easy to extend, test, and debug

### Technical Innovations
- ML for signal extraction (Alpha)
- Copula theory for dependence & tail risk (Beta)
- Rich, multi-scale feature engineering

### Comprehensive Evaluation
- Strong baselines included
- Risk-adjusted return metrics
- Performance attribution & plots

---

## Usage

### Quick Start

```bash
# Clone the repo
git clone https://github.com/kevinlmf/Portfolio_Optimization.git
# (SSH) git clone git@github.com:kevinlmf/Portfolio_Optimization.git

cd Portfolio_Optimization

# Create & activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
                                 # Windows (CMD):        .venv\Scripts\activate.bat

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the demo
python complete_portfolio_example.py
# If the example is in a subfolder:
# python examples/complete_portfolio_example.py

