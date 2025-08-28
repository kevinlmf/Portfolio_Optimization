#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complete_portfolio_example.py

End-to-end demo:
1) Load real adjusted prices (via data/real_data or yfinance fallback)
2) Build a simple Alpha (momentum or ML if your factor modules are available)
3) Build a Copula-style covariance (Gaussianized ranks -> correlation)
4) Solve mean-variance portfolio with box + budget constraints (SLSQP)
5) Print weights & basic backtest on the last 252 trading days

This script is robust:
- If your factor modules exist (factor/alpha, factor/beta), it will try them first.
- If not, it falls back to a lightweight baseline that still runs.
"""

import os
import sys
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ========= Paths =========
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========= Config =========
TICKERS: List[str] = [
    "SPY","QQQ","DIA","AAPL","MSFT","GOOGL","AMZN","TSLA",
    "JPM","BAC","XOM","CVX","GE","TLT","LQD","GLD","SLV","USO","XLF","XLK",
]
START = "2020-01-01"
END   = "2023-12-31"

GAMMA  = 5.0      # risk aversion for mean-variance
W_MAX  = 0.15     # box constraint |w_i| <= W_MAX
TC_LMB = 0.0      # transaction cost coefficient (set >0 to penalize turnover)
LOOKBACK_ALPHA = 60  # days for momentum alpha
BACKTEST_DAYS  = 252 # evaluate last N days

# ========= Utils =========
def _rank_gaussianize(df: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-observations -> N(0,1) via inverse normal CDF on ranks."""
    from scipy.stats import norm, rankdata
    X = df.copy()
    for c in X.columns:
        r = rankdata(X[c].values, method="average")
        u = r / (len(r) + 1.0)
        X[c] = norm.ppf(u)
    return X.replace([np.inf, -np.inf], np.nan).dropna()

def _cov_from_copula(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Copula-style covariance:
    - Gaussianize each series using rank -> N(0,1)
    - Correlation on gaussianized series
    - Scale back with original volatilities
    """
    Z = _rank_gaussianize(returns.dropna(how="any"))
    if Z.empty:
        raise ValueError("Not enough data to estimate copula-style correlation.")
    corr = np.corrcoef(Z.values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, -0.999, 0.999)

    vol = returns.loc[Z.index].std().values  # sample stdev
    D   = np.diag(vol)
    Sigma = D @ corr @ D
    return pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns)

def _solve_mv(alpha: np.ndarray,
              Sigma: np.ndarray,
              w_prev: Optional[np.ndarray] = None,
              gamma: float = 5.0,
              w_max: float = 0.15,
              tc_lambda: float = 0.0) -> np.ndarray:
    """
    Maximize: alpha^T w - 0.5*gamma * w^T Sigma w - tc_lambda * ||w - w_prev||_1
    s.t. sum w = 1, -w_max <= w_i <= w_max
    We approximate L1 cost with small-slope smooth abs (Huber-like), or set tc_lambda=0 for clean QP.
    """
    from scipy.optimize import minimize

    n = alpha.shape[0]
    if w_prev is None:
        w_prev = np.zeros(n)

    # smooth abs
    eps = 1e-4
    def smooth_l1(x):
        return np.sum(np.sqrt((x - w_prev)**2 + eps))

    def objective(w):
        quad = 0.5 * gamma * w @ Sigma @ w
        lin  = - alpha @ w
        tc   = tc_lambda * smooth_l1(w)
        return quad + lin + tc

    cons = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    },)

    bounds = [(-w_max, w_max)] * n
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9, "disp": False})
    if not res.success:
        raise RuntimeError(f"Optimizer failed: {res.message}")
    return res.x

# ========= Data loading =========
def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Try to use data/real_data.fetch_and_save_real_data (if exists) for consistency.
    Fallback to yfinance (auto_adjust=True).
    """
    # attempt package import
    try:
        sys.path.append(ROOT)
        from data.real_data import fetch_and_save_real_data  # type: ignore
        meta = fetch_and_save_real_data(tickers, start, end)
        prices_path = os.path.join(ROOT, "data", "real", "prices.csv")
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        # Keep requested tickers intersection
        keep = [t for t in tickers if t in prices.columns]
        prices = prices[keep]
        return prices
    except Exception as e:
        print(f"[INFO] Fallback to yfinance due to: {e}")
        import yfinance as yf
        df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            closes = {t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns}
            prices = pd.DataFrame(closes).sort_index().ffill().bfill()
        else:
            prices = pd.DataFrame({tickers[0]: df["Close"]}).sort_index().ffill().bfill()
        return prices

# ========= Alpha builders =========
def alpha_from_factors(prices: pd.DataFrame) -> pd.Series:
    """
    Try to use your factor.alpha.* modules for ML alpha; otherwise use momentum alpha.
    Returns alpha vector for the LAST date (one-step-ahead signal).
    """
    # Try user modules
    try:
        from factor.alpha.ml_alpha_factors import MLAlphaModel  # your module (if implemented)
        from factor.alpha.feature_engineering import FinancialFeatureEngineer
        returns = prices.pct_change().dropna()
        fe = FinancialFeatureEngineer()
        X = fe.create_features(prices.loc[returns.index])
        # Example: simple per-asset next-day forecasting (placeholder if your MLAlphaModel has .fit/.predict)
        model = MLAlphaModel()
        # Fit on all but last day; predict last day
        X_train, y_train = X.iloc[:-1], returns.iloc[1:]
        model.fit(X_train, y_train)
        alpha_vec = pd.Series(model.predict(X.iloc[[-1]]).ravel(), index=returns.columns, name=prices.index[-1])
        return alpha_vec
    except Exception as e:
        # Fallback: momentum alpha — last LOOKBACK_ALPHA days mean return
        returns = prices.pct_change()
        alpha_vec = returns.iloc[-LOOKBACK_ALPHA:].mean().rename(prices.index[-1])
        return alpha_vec

# ========= Main pipeline =========
def main() -> None:
    print("=== ML Alpha + Copula Beta: Complete Portfolio Example ===")
    prices = load_prices(TICKERS, START, END)
    prices = prices.dropna(how="any")
    if prices.shape[1] < 3:
        raise ValueError("Not enough valid tickers after download/cleaning.")

    returns = prices.pct_change().dropna()

    # Split train/backtest
    if returns.shape[0] <= BACKTEST_DAYS + 60:
        print("[WARN] Few rows; using all data for both estimation and eval.")
        est_rets = returns
        test_rets = returns.iloc[-BACKTEST_DAYS:]
    else:
        est_rets = returns.iloc[:-BACKTEST_DAYS]
        test_rets = returns.iloc[-BACKTEST_DAYS:]

    # Alpha (vector for the *start* of backtest)
    alpha_vec = alpha_from_factors(prices.loc[est_rets.index.union(test_rets.index)])
    alpha_vec = alpha_vec.reindex(est_rets.columns).fillna(0.0)

    # Beta/Copula risk (covariance on estimation window)
    Sigma = _cov_from_copula(est_rets)
    Sigma = Sigma.reindex(index=est_rets.columns, columns=est_rets.columns).fillna(0.0)

    # Optimize weights
    w = _solve_mv(alpha_vec.values, Sigma.values, w_prev=None, gamma=GAMMA, w_max=W_MAX, tc_lambda=TC_LMB)
    weights = pd.Series(w, index=est_rets.columns, name="weight").sort_values(ascending=False)

    # Backtest (constant weights over test window)
    port_ret = (test_rets @ weights).rename("portfolio")
    cum = (1 + port_ret).cumprod()
    summary = {
        "start": str(test_rets.index.min().date()),
        "end": str(test_rets.index.max().date()),
        "days": int(test_rets.shape[0]),
        "ann_ret": float(np.power((1 + port_ret.mean())**252, 1) - 1),
        "ann_vol": float(port_ret.std() * np.sqrt(252)),
        "sharpe": float((port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252) + 1e-12)),
        "max_dd": float((cum / cum.cummax() - 1).min()),
    }

    # Save outputs
    weights.to_csv(os.path.join(RESULTS_DIR, "weights.csv"))
    port_ret.to_csv(os.path.join(RESULTS_DIR, "backtest_returns.csv"))
    pd.Series(summary).to_json(os.path.join(RESULTS_DIR, "summary.json"), indent=2)

    print("\n--- Optimal Weights (top 10) ---")
    print(weights.head(10).to_string(float_format=lambda x: f"{x: .4f}"))
    print("\n--- Backtest Summary (last 252 days) ---")
    for k, v in summary.items():
        print(f"{k:>8}: {v}")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        cum.plot(title="Cumulative Return (Constant Weights)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "cumulative_return.png"), dpi=140)
        # plt.show()  # enable if you want a popup
        print(f"\nSaved results to: {RESULTS_DIR}")
    except Exception as e:
        print(f"[INFO] Skipped plotting: {e}")

if __name__ == "__main__":
    main()

