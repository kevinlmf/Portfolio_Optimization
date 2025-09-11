#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Portfolio Optimizer
Advanced portfolio optimization system that finds the strongest alpha factors and beta models,
then optimizes asset allocation for maximum risk-adjusted returns.

Key Features:
1. Automatic factor mining and selection
2. Multi-method beta estimation
3. Advanced portfolio optimization with multiple objectives
4. Comprehensive backtesting and reporting
5. Real-time factor validation and performance monitoring

Author: Claude AI Assistant
Version: 1.0
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from data.enhanced_data_fetcher import EnhancedDataFetcher
    from strategy.factor.alpha.real_alpha_miner import RealAlphaMiner
    from strategy.factor.beta.real_beta_estimator import RealBetaEstimator
    from risk_control.factor_validation import FactorValidator
except ImportError as e:
    logger.warning(f"Some modules not found: {e}")
    logger.info("Will use fallback implementations")

class SmartPortfolioOptimizer:
    """
    Intelligent portfolio optimization system that automatically:
    1. Mines and validates alpha factors
    2. Estimates beta risk models
    3. Optimizes portfolio weights
    4. Provides comprehensive analysis and backtesting
    """
    
    def __init__(self, 
                 start_date: str = "2020-01-01",
                 end_date: str = None,
                 risk_free_rate: float = 0.02,
                 results_dir: str = None):
        """
        Initialize the Smart Portfolio Optimizer.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection (default: today)
            risk_free_rate: Annual risk-free rate
            results_dir: Directory to save results
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.risk_free_rate = risk_free_rate
        
        # Setup results directory
        self.results_dir = results_dir or os.path.join(ROOT_DIR, "results", "smart_optimizer")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.data_fetcher = None
        self.alpha_miner = None
        self.beta_estimator = None
        self.factor_validator = None
        
        # Storage for results
        self.market_data = pd.DataFrame()
        self.alpha_factors = pd.DataFrame()
        self.beta_estimates = pd.DataFrame()
        self.selected_factors = []
        self.portfolio_weights = pd.DataFrame()
        self.backtest_results = pd.DataFrame()
        self.performance_metrics = {}
        
        logger.info(f"Smart Portfolio Optimizer initialized")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def fetch_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch comprehensive market data for analysis.
        
        Args:
            tickers: List of stock tickers to analyze
            
        Returns:
            DataFrame with comprehensive market data
        """
        logger.info(f"Fetching market data for {len(tickers)} tickers...")
        
        try:
            # Use enhanced data fetcher if available
            self.data_fetcher = EnhancedDataFetcher(self.start_date, self.end_date)
            self.market_data = self.data_fetcher.create_alpha_research_dataset(
                tickers, 
                include_fundamentals=True, 
                include_market_data=True
            )
        except Exception as e:
            logger.warning(f"Enhanced data fetcher failed: {e}")
            # Fallback to basic yfinance data
            self.market_data = self._fetch_basic_data(tickers)
        
        if self.market_data.empty:
            raise ValueError("Failed to fetch market data")
        
        logger.info(f"Market data fetched: {self.market_data.shape}")
        logger.info(f"Date range: {self.market_data['date'].min()} to {self.market_data['date'].max()}")
        logger.info(f"Tickers: {self.market_data['ticker'].nunique()}")
        
        return self.market_data
    
    def _fetch_basic_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fallback method to fetch basic market data using yfinance."""
        import yfinance as yf
        
        data_list = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                if hist.empty:
                    continue
                
                for date, row in hist.iterrows():
                    data_list.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'close': row['Close'],
                        'volume': row['Volume'],
                        'high': row['High'],
                        'low': row['Low'],
                        'open': row['Open']
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df['returns'] = df.groupby('ticker')['close'].pct_change()
        
        return df
    
    def mine_alpha_factors(self, 
                          min_ic_threshold: float = 0.02,
                          top_n_factors: int = 20) -> pd.DataFrame:
        """
        Mine and validate alpha factors from market data.
        
        Args:
            min_ic_threshold: Minimum information coefficient threshold
            top_n_factors: Number of top factors to select
            
        Returns:
            DataFrame with validated alpha factors
        """
        logger.info("Mining alpha factors...")
        
        if self.market_data.empty:
            raise ValueError("Market data not available. Call fetch_market_data() first.")
        
        try:
            # Use real alpha miner if available
            self.alpha_miner = RealAlphaMiner(
                data=self.market_data,
                min_ic_threshold=min_ic_threshold
            )
            self.alpha_factors = self.alpha_miner.mine_all_alpha_factors()
            
            # Get factor performance metrics
            factor_performance = self.alpha_miner.factor_performance
            if isinstance(factor_performance, pd.DataFrame) and not factor_performance.empty:
                # Select top factors based on IC
                top_factors = factor_performance.sort_values(
                    'ic_1d', key=abs, ascending=False
                ).head(top_n_factors)
                self.selected_factors = top_factors['factor'].tolist()
                
                logger.info(f"Top {len(self.selected_factors)} alpha factors selected:")
                for i, (_, row) in enumerate(top_factors.head(10).iterrows(), 1):
                    logger.info(f"  {i}. {row['factor']}: IC={row.get('ic_1d', 0):.4f}")
            
        except Exception as e:
            logger.warning(f"Alpha miner failed: {e}")
            # Fallback to basic momentum factors
            self.alpha_factors = self._create_basic_alpha_factors()
            self.selected_factors = [col for col in self.alpha_factors.columns 
                                   if col not in ['ticker', 'date']]
        
        logger.info(f"Alpha factors mining completed: {len(self.selected_factors)} factors")
        return self.alpha_factors
    
    def _create_basic_alpha_factors(self) -> pd.DataFrame:
        """Create basic alpha factors as fallback."""
        logger.info("Creating basic alpha factors...")
        
        factors_list = []
        
        for ticker in self.market_data['ticker'].unique():
            ticker_data = self.market_data[self.market_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            if len(ticker_data) < 60:
                continue
            
            # Calculate basic momentum factors
            prices = ticker_data['close']
            returns = prices.pct_change()
            
            factors_df = pd.DataFrame()
            factors_df['ticker'] = ticker
            factors_df['date'] = ticker_data['date']
            
            # Momentum factors
            for window in [5, 10, 20, 60]:
                factors_df[f'momentum_{window}d'] = returns.rolling(window).mean()
                factors_df[f'volatility_{window}d'] = returns.rolling(window).std()
                factors_df[f'sharpe_{window}d'] = (
                    factors_df[f'momentum_{window}d'] / 
                    (factors_df[f'volatility_{window}d'] + 1e-8)
                )
            
            # Price-based factors
            for window in [10, 20]:
                sma = prices.rolling(window).mean()
                factors_df[f'price_to_sma_{window}'] = prices / sma - 1
                
                # Bollinger bands position
                bb_std = prices.rolling(window).std()
                bb_upper = sma + 2 * bb_std
                bb_lower = sma - 2 * bb_std
                factors_df[f'bb_position_{window}'] = (
                    (prices - bb_lower) / (bb_upper - bb_lower + 1e-8)
                )
            
            # Volume factors if available
            if 'volume' in ticker_data.columns:
                volume = ticker_data['volume']
                for window in [5, 20]:
                    vol_ma = volume.rolling(window).mean()
                    factors_df[f'volume_ratio_{window}'] = volume / (vol_ma + 1e-8)
            
            factors_list.append(factors_df)
        
        if not factors_list:
            return pd.DataFrame()
        
        all_factors = pd.concat(factors_list, ignore_index=True)
        return all_factors
    
    def estimate_risk_models(self, 
                           methods: List[str] = None) -> pd.DataFrame:
        """
        Estimate beta risk models using multiple methods.
        
        Args:
            methods: List of beta estimation methods to use
            
        Returns:
            DataFrame with beta estimates
        """
        logger.info("Estimating beta risk models...")
        
        if methods is None:
            methods = ['capm_beta', 'multi_factor_beta', 'copula_beta', 'cvar_beta']
        
        try:
            # Use real beta estimator if available
            self.beta_estimator = RealBetaEstimator(data=self.market_data)
            tickers = self.market_data['ticker'].unique().tolist()
            self.beta_estimates = self.beta_estimator.estimate_all_betas(
                tickers=tickers, 
                methods=methods
            )
            
        except Exception as e:
            logger.warning(f"Beta estimator failed: {e}")
            # Fallback to basic CAPM beta
            self.beta_estimates = self._estimate_basic_beta()
        
        logger.info(f"Beta estimation completed for {len(self.beta_estimates)} assets")
        return self.beta_estimates
    
    def _estimate_basic_beta(self) -> pd.DataFrame:
        """Estimate basic CAPM beta as fallback."""
        logger.info("Estimating basic CAPM beta...")
        
        # Calculate market returns (equal-weighted)
        market_returns = (self.market_data.groupby('date')['returns']
                         .mean().dropna())
        
        beta_results = []
        
        for ticker in self.market_data['ticker'].unique():
            ticker_data = self.market_data[self.market_data['ticker'] == ticker].copy()
            ticker_returns = ticker_data.set_index('date')['returns'].dropna()
            
            # Align dates
            common_dates = market_returns.index.intersection(ticker_returns.index)
            if len(common_dates) < 60:
                continue
            
            market_ret = market_returns.loc[common_dates]
            stock_ret = ticker_returns.loc[common_dates]
            
            # Calculate beta using linear regression
            if len(market_ret) > 0 and market_ret.std() > 0:
                beta = np.cov(stock_ret, market_ret)[0, 1] / np.var(market_ret)
                alpha = stock_ret.mean() - beta * market_ret.mean()
                
                # R-squared
                y_pred = alpha + beta * market_ret
                ss_res = np.sum((stock_ret - y_pred) ** 2)
                ss_tot = np.sum((stock_ret - stock_ret.mean()) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                
                beta_results.append({
                    'ticker': ticker,
                    'capm_beta': beta,
                    'alpha': alpha,
                    'r_squared': r_squared,
                    'volatility': stock_ret.std() * np.sqrt(252)
                })
        
        return pd.DataFrame(beta_results)
    
    def optimize_portfolio(self, 
                         objective: str = 'max_sharpe',
                         constraints: Dict = None,
                         alpha_weight: float = 0.6,
                         beta_weight: float = 0.4) -> Dict[str, Any]:
        """
        Optimize portfolio using alpha signals and beta risk models.
        
        Args:
            objective: Optimization objective ('max_sharpe', 'min_variance', 'max_utility')
            constraints: Portfolio constraints
            alpha_weight: Weight given to alpha signals in expected returns
            beta_weight: Weight given to beta models in risk estimation
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing portfolio with objective: {objective}")
        
        if self.alpha_factors.empty or self.beta_estimates.empty:
            raise ValueError("Alpha factors and beta estimates required. Run mining and estimation first.")
        
        # Prepare data for optimization
        assets = self._get_common_assets()
        if len(assets) < 2:
            raise ValueError("Need at least 2 assets for optimization")
        
        # Calculate expected returns using alpha signals
        expected_returns = self._calculate_expected_returns(assets, alpha_weight)
        
        # Build covariance matrix using beta estimates
        covariance_matrix = self._build_covariance_matrix(assets, beta_weight)
        
        # Set default constraints
        default_constraints = {
            'max_weight': 0.3,
            'min_weight': 0.0,
            'max_concentration': 0.8,
            'min_diversification': 5  # Minimum effective number of assets
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        # Optimize portfolio
        result = self._solve_optimization(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            assets=assets,
            objective=objective,
            constraints=default_constraints
        )
        
        # Calculate portfolio metrics
        if result['success']:
            metrics = self._calculate_portfolio_metrics(
                weights=result['weights'],
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                assets=assets
            )
            result.update(metrics)
        
        # Store results
        if result['success']:
            weights_df = pd.DataFrame({
                'ticker': assets,
                'weight': result['weights'],
                'expected_return': expected_returns,
                'contribution': result['weights'] * expected_returns
            })
            weights_df = weights_df.sort_values('weight', ascending=False)
            self.portfolio_weights = weights_df
        
        logger.info(f"Portfolio optimization {'succeeded' if result['success'] else 'failed'}")
        return result
    
    def _get_common_assets(self) -> List[str]:
        """Get assets that have both alpha factors and beta estimates."""
        alpha_assets = set(self.alpha_factors['ticker'].unique())
        beta_assets = set(self.beta_estimates['ticker'].unique())
        common_assets = list(alpha_assets.intersection(beta_assets))
        
        # Filter out assets with insufficient data
        valid_assets = []
        for asset in common_assets:
            alpha_data = self.alpha_factors[self.alpha_factors['ticker'] == asset]
            beta_data = self.beta_estimates[self.beta_estimates['ticker'] == asset]
            
            if len(alpha_data) > 60 and not beta_data.empty:
                valid_assets.append(asset)
        
        return valid_assets
    
    def _calculate_expected_returns(self, assets: List[str], alpha_weight: float) -> np.ndarray:
        """Calculate expected returns using alpha factors."""
        expected_returns = np.zeros(len(assets))
        
        for i, asset in enumerate(assets):
            # Get latest alpha signals
            asset_factors = self.alpha_factors[self.alpha_factors['ticker'] == asset]
            if asset_factors.empty:
                continue
            
            # Use most recent factor values
            latest_factors = asset_factors.iloc[-1]
            
            # Calculate alpha score using selected factors
            alpha_score = 0.0
            valid_factors = 0
            
            for factor_name in self.selected_factors:
                if factor_name in latest_factors and pd.notna(latest_factors[factor_name]):
                    alpha_score += latest_factors[factor_name]
                    valid_factors += 1
            
            if valid_factors > 0:
                alpha_score /= valid_factors  # Average alpha score
            
            # Combine with risk-free rate and market risk premium
            market_risk_premium = 0.08  # 8% annual market risk premium
            beta = self._get_beta_for_asset(asset)
            
            expected_return = (
                self.risk_free_rate / 252 +  # Daily risk-free rate
                alpha_weight * alpha_score * 0.01 +  # Alpha contribution (scaled)
                (1 - alpha_weight) * beta * market_risk_premium / 252  # Beta contribution
            )
            
            expected_returns[i] = expected_return
        
        return expected_returns
    
    def _get_beta_for_asset(self, asset: str) -> float:
        """Get beta estimate for an asset."""
        asset_beta = self.beta_estimates[self.beta_estimates['ticker'] == asset]
        if asset_beta.empty:
            return 1.0  # Default beta
        
        # Use CAPM beta if available, otherwise use the first available beta
        if 'capm_beta' in asset_beta.columns and pd.notna(asset_beta['capm_beta'].iloc[0]):
            return asset_beta['capm_beta'].iloc[0]
        else:
            # Find first non-null beta column
            for col in asset_beta.columns:
                if 'beta' in col.lower() and pd.notna(asset_beta[col].iloc[0]):
                    return asset_beta[col].iloc[0]
        
        return 1.0  # Default beta
    
    def _build_covariance_matrix(self, assets: List[str], beta_weight: float) -> np.ndarray:
        """Build covariance matrix using factor model approach."""
        n_assets = len(assets)
        
        # Get beta values
        betas = np.array([self._get_beta_for_asset(asset) for asset in assets])
        
        # Market factor variance (annual)
        market_variance = 0.04
        
        # Factor model: Cov = β * σ_m² * β' + Ω (idiosyncratic risk)
        systematic_cov = market_variance * np.outer(betas, betas)
        
        # Idiosyncratic risk (diagonal matrix)
        idiosyncratic_vars = []
        for asset in assets:
            # Use volatility from beta estimates if available
            asset_data = self.beta_estimates[self.beta_estimates['ticker'] == asset]
            if not asset_data.empty and 'volatility' in asset_data.columns:
                vol = asset_data['volatility'].iloc[0]
                if pd.notna(vol):
                    idiosyncratic_vars.append((vol ** 2 - market_variance * betas[len(idiosyncratic_vars)] ** 2) / 252)
                else:
                    idiosyncratic_vars.append(0.1 / 252)  # Default daily variance
            else:
                idiosyncratic_vars.append(0.1 / 252)  # Default daily variance
        
        idiosyncratic_cov = np.diag(np.maximum(idiosyncratic_vars, 1e-6))  # Ensure positive
        
        # Convert to daily
        systematic_cov /= 252
        
        # Combine
        covariance_matrix = beta_weight * systematic_cov + (1 - beta_weight) * idiosyncratic_cov
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return covariance_matrix
    
    def _solve_optimization(self, 
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          assets: List[str],
                          objective: str,
                          constraints: Dict) -> Dict[str, Any]:
        """Solve portfolio optimization problem."""
        n_assets = len(assets)
        
        # Define objective function
        if objective == 'max_sharpe':
            def objective_func(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = weights.T @ covariance_matrix @ weights
                portfolio_std = np.sqrt(portfolio_variance)
                if portfolio_std == 0:
                    return 1e6  # Large penalty
                sharpe_ratio = (portfolio_return - self.risk_free_rate/252) / portfolio_std
                return -sharpe_ratio  # Negative because we minimize
                
        elif objective == 'min_variance':
            def objective_func(weights):
                return weights.T @ covariance_matrix @ weights
                
        elif objective == 'max_utility':
            risk_aversion = 2.0
            def objective_func(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = weights.T @ covariance_matrix @ weights
                utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                return -utility
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Diversification constraint (minimum effective number of assets)
        if constraints.get('min_diversification', 0) > 1:
            min_eff_assets = constraints['min_diversification']
            def diversification_constraint(weights):
                effective_assets = 1 / np.sum(weights**2)
                return effective_assets - min_eff_assets
            cons.append({'type': 'ineq', 'fun': diversification_constraint})
        
        # Concentration constraint
        if constraints.get('max_concentration', 1) < 1:
            max_conc = constraints['max_concentration']
            def concentration_constraint(weights):
                herfindahl = np.sum(weights**2)
                return max_conc - herfindahl
            cons.append({'type': 'ineq', 'fun': concentration_constraint})
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) 
                  for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Solve optimization
        try:
            result = optimize.minimize(
                objective_func, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            return {
                'success': result.success,
                'weights': result.x if result.success else x0,
                'objective_value': result.fun if result.success else objective_func(x0),
                'message': result.message if hasattr(result, 'message') else 'Completed',
                'assets': assets
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'success': False,
                'weights': x0,
                'objective_value': objective_func(x0),
                'message': f'Error: {str(e)}',
                'assets': assets
            }
    
    def _calculate_portfolio_metrics(self, 
                                   weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   assets: List[str]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        # Basic metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Annualized metrics
        ann_return = portfolio_return * 252
        ann_volatility = portfolio_std * np.sqrt(252)
        sharpe_ratio = (ann_return - self.risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
        
        # Diversification metrics
        effective_assets = 1 / np.sum(weights**2)
        max_weight = np.max(weights)
        concentration = np.sum(weights**2)  # Herfindahl index
        
        # Risk decomposition
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_variance if portfolio_variance > 0 else weights
        
        return {
            'expected_return': ann_return,
            'volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'effective_assets': effective_assets,
            'max_weight': max_weight,
            'concentration': concentration,
            'risk_contributions': dict(zip(assets, risk_contrib))
        }
    
    def backtest_strategy(self, 
                        rebalance_frequency: str = 'monthly',
                        lookback_window: int = 252,
                        out_of_sample_days: int = 60) -> pd.DataFrame:
        """
        Backtest the portfolio optimization strategy.
        
        Args:
            rebalance_frequency: How often to rebalance ('weekly', 'monthly', 'quarterly')
            lookback_window: Days of historical data for optimization
            out_of_sample_days: Days to test each portfolio
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Starting backtest with {rebalance_frequency} rebalancing...")
        
        if self.market_data.empty:
            raise ValueError("Market data required for backtesting")
        
        # Determine rebalancing frequency
        freq_map = {'weekly': 7, 'monthly': 21, 'quarterly': 63}
        rebalance_days = freq_map.get(rebalance_frequency, 21)
        
        # Get date range for backtesting
        dates = sorted(self.market_data['date'].unique())
        start_idx = lookback_window
        
        if len(dates) < start_idx + out_of_sample_days:
            raise ValueError("Insufficient data for backtesting")
        
        backtest_results = []
        current_weights = None
        
        # Iterate through time periods
        for i in range(start_idx, len(dates) - out_of_sample_days, rebalance_days):
            rebalance_date = dates[i]
            
            try:
                # Get historical data for optimization
                hist_end_date = dates[i]
                hist_start_date = dates[max(0, i - lookback_window)]
                
                hist_data = self.market_data[
                    (self.market_data['date'] >= hist_start_date) & 
                    (self.market_data['date'] <= hist_end_date)
                ].copy()
                
                if len(hist_data) < lookback_window // 2:
                    continue
                
                # Create temporary optimizer with historical data
                temp_optimizer = SmartPortfolioOptimizer(
                    start_date=hist_start_date.strftime('%Y-%m-%d'),
                    end_date=hist_end_date.strftime('%Y-%m-%d')
                )
                temp_optimizer.market_data = hist_data
                
                # Mine factors and estimate risk models
                temp_optimizer.mine_alpha_factors(top_n_factors=15)
                temp_optimizer.estimate_risk_models()
                
                # Optimize portfolio
                opt_result = temp_optimizer.optimize_portfolio()
                
                if opt_result['success']:
                    assets = opt_result['assets']
                    weights = opt_result['weights']
                    current_weights = dict(zip(assets, weights))
                
            except Exception as e:
                logger.warning(f"Optimization failed at {rebalance_date}: {e}")
                # Keep previous weights if optimization fails
            
            # Calculate returns for the next period
            if current_weights:
                end_period = min(i + rebalance_days, len(dates) - 1)
                
                for j in range(i + 1, end_period + 1):
                    if j >= len(dates):
                        break
                    
                    current_date = dates[j]
                    
                    # Get returns for this date
                    day_data = self.market_data[self.market_data['date'] == current_date]
                    
                    portfolio_return = 0.0
                    total_weight = 0.0
                    
                    for ticker, weight in current_weights.items():
                        ticker_data = day_data[day_data['ticker'] == ticker]
                        if not ticker_data.empty and 'returns' in ticker_data.columns:
                            ticker_return = ticker_data['returns'].iloc[0]
                            if pd.notna(ticker_return):
                                portfolio_return += weight * ticker_return
                                total_weight += weight
                    
                    # Normalize if some assets missing
                    if total_weight > 0:
                        portfolio_return /= total_weight
                    
                    backtest_results.append({
                        'date': current_date,
                        'portfolio_return': portfolio_return,
                        'rebalance_date': rebalance_date,
                        'num_assets': len(current_weights)
                    })
        
        if not backtest_results:
            logger.warning("No backtest results generated")
            return pd.DataFrame()
        
        # Convert to DataFrame and calculate cumulative returns
        self.backtest_results = pd.DataFrame(backtest_results)
        self.backtest_results['cumulative_return'] = (
            1 + self.backtest_results['portfolio_return']
        ).cumprod()
        
        # Calculate performance metrics
        returns = self.backtest_results['portfolio_return'].dropna()
        
        self.performance_metrics = {
            'total_return': self.backtest_results['cumulative_return'].iloc[-1] - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(self.backtest_results['cumulative_return']),
            'win_rate': (returns > 0).mean(),
            'num_periods': len(returns)
        }
        
        logger.info("Backtest completed successfully")
        logger.info(f"Total Return: {self.performance_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {self.performance_metrics['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {self.performance_metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.4f}")
        logger.info(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
        
        return self.backtest_results
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def generate_report(self, save_plots: bool = True) -> str:
        """
        Generate comprehensive optimization and backtesting report.
        
        Args:
            save_plots: Whether to save visualization plots
            
        Returns:
            Report text
        """
        logger.info("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SMART PORTFOLIO OPTIMIZER - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Analysis Period: {self.start_date} to {self.end_date}")
        report_lines.append("")
        
        # Data Summary
        report_lines.append("DATA SUMMARY")
        report_lines.append("-" * 40)
        if not self.market_data.empty:
            report_lines.append(f"Total Observations: {len(self.market_data):,}")
            report_lines.append(f"Number of Assets: {self.market_data['ticker'].nunique()}")
            report_lines.append(f"Date Range: {self.market_data['date'].min()} to {self.market_data['date'].max()}")
        
        # Alpha Factors Summary
        if not self.alpha_factors.empty:
            report_lines.append("\nALPHA FACTORS SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Factors Generated: {len(self.alpha_factors.columns) - 2}")
            report_lines.append(f"Selected Top Factors: {len(self.selected_factors)}")
            
            if self.selected_factors:
                report_lines.append("\nTop 10 Alpha Factors:")
                for i, factor in enumerate(self.selected_factors[:10], 1):
                    report_lines.append(f"  {i:2d}. {factor}")
        
        # Portfolio Optimization Results
        if not self.portfolio_weights.empty:
            report_lines.append("\nPORTFOLIO OPTIMIZATION RESULTS")
            report_lines.append("-" * 40)
            
            report_lines.append("Optimal Portfolio Weights:")
            for _, row in self.portfolio_weights.head(10).iterrows():
                report_lines.append(f"  {row['ticker']:>6}: {row['weight']:7.2%} "
                                  f"(Expected Return: {row['expected_return']*252:6.2%})")
            
            # Portfolio metrics
            if hasattr(self, '_last_optimization_result'):
                metrics = self._last_optimization_result
                report_lines.append(f"\nPortfolio Metrics:")
                report_lines.append(f"  Expected Return: {metrics.get('expected_return', 0):7.2%}")
                report_lines.append(f"  Volatility:      {metrics.get('volatility', 0):7.2%}")
                report_lines.append(f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):7.4f}")
                report_lines.append(f"  Effective Assets: {metrics.get('effective_assets', 0):6.2f}")
                report_lines.append(f"  Max Weight:      {metrics.get('max_weight', 0):7.2%}")
        
        # Backtesting Results
        if not self.backtest_results.empty and self.performance_metrics:
            report_lines.append("\nBACKTESTING RESULTS")
            report_lines.append("-" * 40)
            
            metrics = self.performance_metrics
            report_lines.append(f"Total Return:     {metrics['total_return']:7.2%}")
            report_lines.append(f"Annual Return:    {metrics['annual_return']:7.2%}")
            report_lines.append(f"Annual Volatility: {metrics['annual_volatility']:6.2%}")
            report_lines.append(f"Sharpe Ratio:     {metrics['sharpe_ratio']:7.4f}")
            report_lines.append(f"Max Drawdown:     {metrics['max_drawdown']:7.2%}")
            report_lines.append(f"Win Rate:         {metrics['win_rate']:7.2%}")
            report_lines.append(f"Number of Periods: {metrics['num_periods']:,}")
        
        # Beta Estimates Summary
        if not self.beta_estimates.empty:
            report_lines.append("\nRISK MODEL SUMMARY")
            report_lines.append("-" * 40)
            
            beta_cols = [col for col in self.beta_estimates.columns if 'beta' in col.lower()]
            if beta_cols:
                report_lines.append("Beta Estimates by Asset:")
                for _, row in self.beta_estimates.head(10).iterrows():
                    ticker = row['ticker']
                    beta_val = row.get('capm_beta', row.get(beta_cols[0], 1.0))
                    report_lines.append(f"  {ticker:>6}: {beta_val:6.3f}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.results_dir, "optimization_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save data
        if not self.portfolio_weights.empty:
            self.portfolio_weights.to_csv(
                os.path.join(self.results_dir, "optimal_weights.csv"), 
                index=False
            )
        
        if not self.backtest_results.empty:
            self.backtest_results.to_csv(
                os.path.join(self.results_dir, "backtest_results.csv"), 
                index=False
            )
        
        # Generate plots
        if save_plots:
            self._save_plots()
        
        logger.info(f"Report saved to: {report_path}")
        return report_text
    
    def _save_plots(self):
        """Save visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Portfolio weights plot
        if not self.portfolio_weights.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            weights_data = self.portfolio_weights.head(15)
            bars = ax.barh(range(len(weights_data)), weights_data['weight'], 
                          color='steelblue', alpha=0.7)
            
            ax.set_yticks(range(len(weights_data)))
            ax.set_yticklabels(weights_data['ticker'])
            ax.set_xlabel('Portfolio Weight')
            ax.set_title('Optimal Portfolio Weights', fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add weight labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1%}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "portfolio_weights.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Backtest performance plot
        if not self.backtest_results.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Cumulative returns
            dates = pd.to_datetime(self.backtest_results['date'])
            cumulative_returns = self.backtest_results['cumulative_return']
            
            ax1.plot(dates, cumulative_returns, linewidth=2, color='navy', label='Portfolio')
            ax1.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Rolling Sharpe ratio
            returns = self.backtest_results['portfolio_return']
            rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252))
            
            ax2.plot(dates, rolling_sharpe, linewidth=2, color='darkgreen', 
                    label='60-Day Rolling Sharpe')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            ax2.set_title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "backtest_performance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Plots saved to results directory")


def main():
    """
    Main execution function demonstrating the Smart Portfolio Optimizer.
    """
    print("=" * 80)
    print("SMART PORTFOLIO OPTIMIZER")
    print("=" * 80)
    print("Advanced portfolio optimization using alpha factor mining and beta risk modeling")
    print()
    
    # Configuration
    tickers = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        # Financial Services
        'JPM', 'BAC', 'WFC', 'GS',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT',
        # Industrial & Energy
        'CAT', 'BA', 'XOM', 'CVX',
        # ETFs
        'SPY', 'QQQ', 'TLT', 'GLD'
    ]
    
    # Initialize optimizer
    optimizer = SmartPortfolioOptimizer(
        start_date="2020-01-01",
        end_date="2024-01-01",
        risk_free_rate=0.03
    )
    
    try:
        # Step 1: Fetch market data
        print("Step 1: Fetching market data...")
        market_data = optimizer.fetch_market_data(tickers)
        print(f"✓ Market data fetched: {market_data.shape}")
        
        # Step 2: Mine alpha factors
        print("\nStep 2: Mining alpha factors...")
        alpha_factors = optimizer.mine_alpha_factors(min_ic_threshold=0.015, top_n_factors=25)
        print(f"✓ Alpha factors mined: {len(optimizer.selected_factors)} factors selected")
        
        # Step 3: Estimate risk models
        print("\nStep 3: Estimating beta risk models...")
        beta_estimates = optimizer.estimate_risk_models()
        print(f"✓ Beta models estimated for {len(beta_estimates)} assets")
        
        # Step 4: Portfolio optimization
        print("\nStep 4: Optimizing portfolio...")
        
        # Try different optimization objectives
        objectives = ['max_sharpe', 'min_variance', 'max_utility']
        results = {}
        
        for objective in objectives:
            print(f"  Optimizing with {objective} objective...")
            result = optimizer.optimize_portfolio(
                objective=objective,
                constraints={'max_weight': 0.25, 'min_weight': 0.01},
                alpha_weight=0.7
            )
            results[objective] = result
            
            if result['success']:
                print(f"    ✓ Success - Sharpe: {result.get('sharpe_ratio', 0):.3f}")
            else:
                print(f"    ✗ Failed: {result.get('message', 'Unknown error')}")
        
        # Use the best result (max_sharpe by default)
        best_result = results.get('max_sharpe', results[objectives[0]])
        optimizer._last_optimization_result = best_result
        
        # Step 5: Backtesting
        print("\nStep 5: Running backtest...")
        backtest_results = optimizer.backtest_strategy(
            rebalance_frequency='monthly',
            lookback_window=252,
            out_of_sample_days=30
        )
        
        if not backtest_results.empty:
            print("✓ Backtest completed")
        else:
            print("✗ Backtest failed or returned no results")
        
        # Step 6: Generate report
        print("\nStep 6: Generating comprehensive report...")
        report = optimizer.generate_report(save_plots=True)
        
        # Display summary results
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if not optimizer.portfolio_weights.empty:
            print("Top 10 Holdings:")
            for _, row in optimizer.portfolio_weights.head(10).iterrows():
                print(f"  {row['ticker']:>6}: {row['weight']:>7.2%}")
        
        if optimizer.performance_metrics:
            print(f"\nBacktest Performance:")
            metrics = optimizer.performance_metrics
            print(f"  Total Return:     {metrics['total_return']:>7.2%}")
            print(f"  Annual Return:    {metrics['annual_return']:>7.2%}")
            print(f"  Annual Volatility: {metrics['annual_volatility']:>6.2%}")
            print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>7.3f}")
            print(f"  Max Drawdown:     {metrics['max_drawdown']:>7.2%}")
        
        print(f"\n✓ All results saved to: {optimizer.results_dir}")
        print("\nFiles generated:")
        print("  - optimization_report.txt")
        print("  - optimal_weights.csv")
        print("  - backtest_results.csv")
        print("  - portfolio_weights.png")
        print("  - backtest_performance.png")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"\n✗ Error: {e}")
        return 1
    
    print("\n" + "="*60)
    print("SMART PORTFOLIO OPTIMIZER COMPLETED")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)