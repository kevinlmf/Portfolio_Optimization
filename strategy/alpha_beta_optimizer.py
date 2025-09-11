"""
Alpha-Beta Portfolio Optimizer
Advanced portfolio optimization combining alpha factor signals and beta risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import our alpha and beta modules
from .factor.alpha.alpha_factor_evaluator import AlphaFactorEvaluator
from .factor.alpha.technical_alpha_factors import TechnicalAlphaFactors
from .factor.alpha.fundamental_alpha_factors import FundamentalAlphaFactors



class AlphaBetaOptimizer:
    """
    Advanced portfolio optimizer that combines alpha signals and beta risk models.
    
    This class integrates alpha factor analysis with comprehensive beta risk modeling
    to create optimal portfolios that maximize risk-adjusted returns.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 market_index: str = 'SPY',
                 risk_free_rate: float = 0.02,
                 benchmark_weights: Optional[Dict] = None):
        """
        Initialize the Alpha-Beta optimizer.
        
        Args:
            data: Market data DataFrame
            market_index: Market index symbol
            risk_free_rate: Annual risk-free rate
            benchmark_weights: Benchmark portfolio weights
        """
        self.data = data.copy()
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        self.benchmark_weights = benchmark_weights or {}
        
        # Initialize alpha and beta evaluators
        self.alpha_evaluator = AlphaFactorEvaluator(data)
        self.beta_evaluator = BetaEvaluator(data, market_index, risk_free_rate)
        
        # Storage for results
        self.alpha_scores = {}
        self.beta_estimates = {}
        self.optimization_results = {}
        
    def generate_alpha_signals(self, 
                             assets: List[str],
                             include_fundamental: bool = False,
                             top_factors: int = 20) -> pd.DataFrame:
        """
        Generate comprehensive alpha signals for assets.
        
        Args:
            assets: List of asset symbols
            include_fundamental: Whether to include fundamental factors
            top_factors: Number of top factors to use
            
        Returns:
            DataFrame with alpha scores
        """
        print("Generating alpha signals...")
        
        # Generate all factors
        all_factors = self.alpha_evaluator.generate_all_factors(
            tickers=assets,
            include_fundamental=include_fundamental
        )
        
        # Evaluate and rank factors
        evaluation_results = self.alpha_evaluator.evaluate_factors()
        ranked_factors = self.alpha_evaluator.rank_factors(evaluation_results)
        top_factor_names = ranked_factors.head(top_factors)['factor'].tolist()
        
        # Calculate alpha scores for each asset
        alpha_scores = {}
        
        for asset in assets:
            if asset not in all_factors.columns:
                alpha_scores[asset] = {
                    'alpha_score': 0.0,
                    'factor_count': 0,
                    'signal_strength': 0.0
                }
                continue
            
            asset_data = all_factors[all_factors['tic'] == asset].copy()
            
            if len(asset_data) == 0:
                alpha_scores[asset] = {
                    'alpha_score': 0.0,
                    'factor_count': 0,
                    'signal_strength': 0.0
                }
                continue
            
            # Calculate weighted alpha score
            factor_scores = []
            factor_weights = []
            
            for i, factor_name in enumerate(top_factor_names):
                if factor_name in asset_data.columns:
                    factor_values = asset_data[factor_name].dropna()
                    
                    if len(factor_values) > 0:
                        # Use recent factor value
                        factor_value = factor_values.iloc[-1]
                        
                        # Get factor IC as weight
                        factor_info = ranked_factors[ranked_factors['factor'] == factor_name]
                        if not factor_info.empty:
                            ic = abs(factor_info['ic_1d'].iloc[0])
                            factor_weight = ic if not np.isnan(ic) else 0.01
                        else:
                            factor_weight = 0.01
                        
                        # Standardize factor value
                        factor_std = factor_values.std()
                        if factor_std > 0:
                            standardized_value = (factor_value - factor_values.mean()) / factor_std
                        else:
                            standardized_value = 0
                        
                        factor_scores.append(standardized_value)
                        factor_weights.append(factor_weight)
            
            if factor_scores:
                # Weighted average alpha score
                factor_weights = np.array(factor_weights)
                factor_weights = factor_weights / factor_weights.sum()
                
                alpha_score = np.sum(np.array(factor_scores) * factor_weights)
                signal_strength = np.sqrt(np.sum(factor_weights))  # Diversification measure
                
                alpha_scores[asset] = {
                    'alpha_score': alpha_score,
                    'factor_count': len(factor_scores),
                    'signal_strength': signal_strength
                }
            else:
                alpha_scores[asset] = {
                    'alpha_score': 0.0,
                    'factor_count': 0,
                    'signal_strength': 0.0
                }
        
        self.alpha_scores = alpha_scores
        
        # Convert to DataFrame
        alpha_df = pd.DataFrame.from_dict(alpha_scores, orient='index')
        alpha_df.index.name = 'asset'
        alpha_df = alpha_df.reset_index()
        
        print(f"Alpha signals generated for {len(assets)} assets")
        return alpha_df
    
    def estimate_risk_models(self, 
                           assets: List[str],
                           beta_methods: List[str] = None) -> pd.DataFrame:
        """
        Estimate comprehensive beta risk models.
        
        Args:
            assets: List of asset symbols
            beta_methods: List of beta estimation methods to use
            
        Returns:
            DataFrame with beta estimates
        """
        print("Estimating beta risk models...")
        
        if beta_methods is None:
            beta_methods = [
                'capm_beta', 'ff_market_beta', 'gaussian_copula_beta', 'cvar_beta_conditional'
            ]
        
        # Estimate all betas
        all_betas = self.beta_evaluator.estimate_all_betas(assets)
        
        # Extract selected methods and calculate consensus
        beta_results = {}
        
        for asset in assets:
            if asset not in all_betas:
                beta_results[asset] = {
                    'beta_consensus': 1.0,
                    'beta_uncertainty': 0.5,
                    'primary_beta': 1.0
                }
                continue
            
            asset_betas = all_betas[asset]
            
            # Extract betas from selected methods
            selected_betas = []
            for method in beta_methods:
                if method in asset_betas and not np.isnan(asset_betas[method]):
                    selected_betas.append(asset_betas[method])
            
            if selected_betas:
                beta_array = np.array(selected_betas)
                
                # Consensus beta (median)
                beta_consensus = np.median(beta_array)
                
                # Uncertainty (standard deviation)
                beta_uncertainty = np.std(beta_array) if len(beta_array) > 1 else 0.1
                
                # Primary beta (prefer multi-factor or advanced methods)
                if 'ff_market_beta' in asset_betas and not np.isnan(asset_betas['ff_market_beta']):
                    primary_beta = asset_betas['ff_market_beta']
                elif 'gaussian_copula_beta' in asset_betas and not np.isnan(asset_betas['gaussian_copula_beta']):
                    primary_beta = asset_betas['gaussian_copula_beta']
                else:
                    primary_beta = beta_consensus
                
                beta_results[asset] = {
                    'beta_consensus': beta_consensus,
                    'beta_uncertainty': beta_uncertainty,
                    'primary_beta': primary_beta
                }
            else:
                # Default values
                beta_results[asset] = {
                    'beta_consensus': 1.0,
                    'beta_uncertainty': 0.5,
                    'primary_beta': 1.0
                }
        
        self.beta_estimates = beta_results
        
        # Convert to DataFrame
        beta_df = pd.DataFrame.from_dict(beta_results, orient='index')
        beta_df.index.name = 'asset'
        beta_df = beta_df.reset_index()
        
        print(f"Beta estimates completed for {len(assets)} assets")
        return beta_df
    
    def optimize_portfolio(self, 
                         assets: List[str],
                         method: str = 'max_sharpe',
                         constraints: Dict = None,
                         risk_aversion: float = 1.0,
                         alpha_weight: float = 0.7,
                         beta_weight: float = 0.3) -> Dict:
        """
        Optimize portfolio using alpha signals and beta risk models.
        
        Args:
            assets: List of asset symbols
            method: Optimization method ('max_sharpe', 'min_variance', 'max_utility', 'risk_parity')
            constraints: Additional constraints
            risk_aversion: Risk aversion parameter
            alpha_weight: Weight given to alpha signals
            beta_weight: Weight given to beta risk models
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Optimizing portfolio using {method} method...")
        
        # Ensure we have alpha and beta estimates
        if not self.alpha_scores:
            self.generate_alpha_signals(assets)
        if not self.beta_estimates:
            self.estimate_risk_models(assets)
        
        # Filter available assets
        available_assets = [asset for asset in assets 
                          if asset in self.alpha_scores and asset in self.beta_estimates]
        
        if len(available_assets) < 2:
            raise ValueError("Need at least 2 assets with valid alpha and beta estimates")
        
        n_assets = len(available_assets)
        
        # Prepare data for optimization
        alpha_signals = np.array([self.alpha_scores[asset]['alpha_score'] for asset in available_assets])
        beta_estimates = np.array([self.beta_estimates[asset]['primary_beta'] for asset in available_assets])
        beta_uncertainties = np.array([self.beta_estimates[asset]['beta_uncertainty'] for asset in available_assets])
        
        # Create covariance matrix
        covariance_matrix = self._build_covariance_matrix(available_assets, beta_estimates, beta_uncertainties)
        
        # Expected returns (combination of alpha and beta)
        market_risk_premium = 0.08  # Assumed market risk premium
        expected_returns = (alpha_weight * alpha_signals + 
                          beta_weight * beta_estimates * market_risk_premium)
        
        # Default constraints
        default_constraints = {
            'max_weight': 0.4,
            'min_weight': 0.0,
            'max_sector_weight': 0.6,
            'turnover_limit': None
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        # Optimization
        if method == 'max_sharpe':
            result = self._maximize_sharpe_ratio(
                expected_returns, covariance_matrix, available_assets, default_constraints
            )
        elif method == 'min_variance':
            result = self._minimize_variance(
                covariance_matrix, available_assets, default_constraints
            )
        elif method == 'max_utility':
            result = self._maximize_utility(
                expected_returns, covariance_matrix, available_assets, default_constraints, risk_aversion
            )
        elif method == 'risk_parity':
            result = self._risk_parity(
                covariance_matrix, available_assets, default_constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Add performance metrics
        if result['success']:
            weights = result['weights']
            result.update(self._calculate_portfolio_metrics(
                weights, expected_returns, covariance_matrix, available_assets
            ))
        
        self.optimization_results = result
        return result
    
    def _build_covariance_matrix(self, 
                               assets: List[str], 
                               betas: np.ndarray, 
                               beta_uncertainties: np.ndarray) -> np.ndarray:
        """Build covariance matrix using factor model approach."""
        n_assets = len(assets)
        
        # Market factor variance
        market_variance = 0.04  # Assumed annual market variance
        
        # Factor model covariance matrix: Cov = β * σ_m² * β' + Ω
        # where Ω is the idiosyncratic risk matrix
        
        # Systematic risk component
        beta_matrix = betas.reshape(-1, 1)
        systematic_cov = market_variance * (beta_matrix @ beta_matrix.T)
        
        # Idiosyncratic risk component
        # Use beta uncertainty as a proxy for idiosyncratic risk
        idiosyncratic_var = beta_uncertainties * 0.1  # Scale factor
        idiosyncratic_cov = np.diag(idiosyncratic_var)
        
        # Total covariance matrix
        covariance_matrix = systematic_cov + idiosyncratic_cov
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return covariance_matrix
    
    def _maximize_sharpe_ratio(self, 
                             expected_returns: np.ndarray,
                             covariance_matrix: np.ndarray,
                             assets: List[str],
                             constraints: Dict) -> Dict:
        """Maximize Sharpe ratio optimization."""
        n_assets = len(assets)
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = weights.T @ covariance_matrix @ weights
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate/252) / portfolio_std
            return -sharpe_ratio  # Negative because we minimize
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimization
        try:
            result = optimize.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            return {
                'success': result.success,
                'weights': result.x if result.success else x0,
                'assets': assets,
                'method': 'max_sharpe',
                'message': result.message if hasattr(result, 'message') else 'Completed'
            }
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {
                'success': False,
                'weights': x0,
                'assets': assets,
                'method': 'max_sharpe',
                'message': f'Error: {str(e)}'
            }
    
    def _minimize_variance(self, 
                         covariance_matrix: np.ndarray,
                         assets: List[str],
                         constraints: Dict) -> Dict:
        """Minimum variance optimization."""
        n_assets = len(assets)
        
        # Objective function (portfolio variance)
        def objective(weights):
            return weights.T @ covariance_matrix @ weights
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            return {
                'success': result.success,
                'weights': result.x if result.success else x0,
                'assets': assets,
                'method': 'min_variance',
                'message': result.message if hasattr(result, 'message') else 'Completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'weights': x0,
                'assets': assets,
                'method': 'min_variance',
                'message': f'Error: {str(e)}'
            }
    
    def _maximize_utility(self, 
                        expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray,
                        assets: List[str],
                        constraints: Dict,
                        risk_aversion: float) -> Dict:
        """Maximize utility optimization."""
        n_assets = len(assets)
        
        # Objective function (negative utility)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = weights.T @ covariance_matrix @ weights
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
            return -utility
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            return {
                'success': result.success,
                'weights': result.x if result.success else x0,
                'assets': assets,
                'method': 'max_utility',
                'message': result.message if hasattr(result, 'message') else 'Completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'weights': x0,
                'assets': assets,
                'method': 'max_utility',
                'message': f'Error: {str(e)}'
            }
    
    def _risk_parity(self, 
                   covariance_matrix: np.ndarray,
                   assets: List[str],
                   constraints: Dict) -> Dict:
        """Risk parity optimization."""
        n_assets = len(assets)
        
        # Objective function (sum of squared risk contributions)
        def objective(weights):
            portfolio_variance = weights.T @ covariance_matrix @ weights
            marginal_contributions = covariance_matrix @ weights
            risk_contributions = weights * marginal_contributions / portfolio_variance
            
            # Target: equal risk contributions (1/n each)
            target_contributions = np.ones(n_assets) / n_assets
            return np.sum((risk_contributions - target_contributions)**2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            return {
                'success': result.success,
                'weights': result.x if result.success else x0,
                'assets': assets,
                'method': 'risk_parity',
                'message': result.message if hasattr(result, 'message') else 'Completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'weights': x0,
                'assets': assets,
                'method': 'risk_parity',
                'message': f'Error: {str(e)}'
            }
    
    def _calculate_portfolio_metrics(self, 
                                   weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   assets: List[str]) -> Dict:
        """Calculate portfolio performance metrics."""
        # Portfolio return and risk
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate/252) / portfolio_std if portfolio_std > 0 else 0
        
        # Risk contributions
        marginal_contributions = covariance_matrix @ weights
        risk_contributions = weights * marginal_contributions / portfolio_variance if portfolio_variance > 0 else weights
        
        # Diversification metrics
        effective_assets = 1 / np.sum(weights**2)  # Effective number of assets
        max_weight = np.max(weights)
        concentration = np.sum(weights**2)  # Herfindahl index
        
        return {
            'expected_return': portfolio_return * 252,  # Annualized
            'volatility': portfolio_std * np.sqrt(252),  # Annualized
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
            'risk_contributions': dict(zip(assets, risk_contributions)),
            'effective_assets': effective_assets,
            'max_weight': max_weight,
            'concentration': concentration
        }
    
    def backtest_strategy(self, 
                        assets: List[str],
                        rebalance_frequency: str = 'monthly',
                        lookback_period: int = 252,
                        method: str = 'max_sharpe') -> pd.DataFrame:
        """
        Backtest the alpha-beta optimization strategy.
        
        Args:
            assets: List of asset symbols
            rebalance_frequency: Rebalancing frequency
            lookback_period: Lookback period for estimation
            method: Optimization method
            
        Returns:
            DataFrame with backtest results
        """
        print(f"Backtesting strategy with {rebalance_frequency} rebalancing...")
        
        # Get return data
        returns_data = self.beta_evaluator.traditional_models.returns_data
        
        # Filter available assets
        available_assets = [asset for asset in assets if asset in returns_data.columns]
        
        if len(available_assets) < 2:
            raise ValueError("Need at least 2 assets with return data")
        
        # Prepare backtest
        backtest_results = []
        rebalance_dates = []
        
        # Determine rebalancing dates
        if rebalance_frequency == 'monthly':
            freq_days = 21
        elif rebalance_frequency == 'quarterly':
            freq_days = 63
        else:
            freq_days = 21  # Default to monthly
        
        start_idx = lookback_period
        end_idx = len(returns_data) - 1
        
        current_weights = np.ones(len(available_assets)) / len(available_assets)  # Initial equal weights
        
        for i in range(start_idx, end_idx, freq_days):
            rebalance_date = returns_data.index[i]
            rebalance_dates.append(rebalance_date)
            
            # Get historical data for estimation
            hist_data = self.data[self.data['date'] <= rebalance_date.strftime('%Y-%m-%d')]
            
            if len(hist_data) < lookback_period:
                continue
            
            try:
                # Create temporary optimizer with historical data
                temp_optimizer = AlphaBetaOptimizer(
                    hist_data, self.market_index, self.risk_free_rate
                )
                
                # Generate signals and optimize
                temp_optimizer.generate_alpha_signals(available_assets)
                temp_optimizer.estimate_risk_models(available_assets)
                opt_result = temp_optimizer.optimize_portfolio(available_assets, method=method)
                
                if opt_result['success']:
                    current_weights = opt_result['weights']
                
            except Exception as e:
                print(f"Optimization failed at {rebalance_date}: {e}")
                # Keep previous weights
            
            # Calculate portfolio returns for next period
            next_period_end = min(i + freq_days, end_idx)
            
            for j in range(i, next_period_end):
                if j >= len(returns_data):
                    break
                
                date = returns_data.index[j]
                asset_returns = returns_data.loc[date, available_assets].values
                
                # Handle missing returns
                if np.any(np.isnan(asset_returns)):
                    portfolio_return = 0
                else:
                    portfolio_return = np.sum(current_weights * asset_returns)
                
                backtest_results.append({
                    'date': date,
                    'portfolio_return': portfolio_return,
                    'weights': current_weights.copy(),
                    'assets': available_assets.copy()
                })
        
        # Convert to DataFrame
        if backtest_results:
            backtest_df = pd.DataFrame(backtest_results)
            
            # Calculate cumulative returns
            backtest_df['cumulative_return'] = (1 + backtest_df['portfolio_return']).cumprod()
            
            # Calculate performance metrics
            returns_series = backtest_df['portfolio_return']
            
            performance_metrics = {
                'total_return': backtest_df['cumulative_return'].iloc[-1] - 1,
                'annual_return': np.mean(returns_series) * 252,
                'annual_volatility': np.std(returns_series) * np.sqrt(252),
                'sharpe_ratio': np.mean(returns_series) / np.std(returns_series) * np.sqrt(252) if np.std(returns_series) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(backtest_df['cumulative_return'])
            }
            
            print(f"Backtest completed:")
            print(f"  Total Return: {performance_metrics['total_return']:.2%}")
            print(f"  Annual Return: {performance_metrics['annual_return']:.2%}")
            print(f"  Annual Volatility: {performance_metrics['annual_volatility']:.2%}")
            print(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
            
            return backtest_df
        else:
            return pd.DataFrame()
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def generate_optimization_report(self, 
                                   optimization_result: Dict = None,
                                   save_path: str = None) -> str:
        """
        Generate comprehensive optimization report.
        
        Args:
            optimization_result: Optimization results
            save_path: Path to save report
            
        Returns:
            Report text
        """
        if optimization_result is None:
            optimization_result = self.optimization_results
        
        if not optimization_result:
            return "No optimization results available."
        
        report = []
        report.append("="*60)
        report.append("ALPHA-BETA PORTFOLIO OPTIMIZATION REPORT")
        report.append("="*60)
        
        # Portfolio Summary
        if optimization_result.get('success', False):
            report.append(f"\nOPTIMIZATION STATUS: SUCCESS")
            report.append(f"METHOD: {optimization_result.get('method', 'Unknown').upper()}")
            
            # Portfolio Weights
            report.append(f"\n{'='*60}")
            report.append("OPTIMAL PORTFOLIO WEIGHTS")
            report.append(f"{'='*60}")
            
            assets = optimization_result.get('assets', [])
            weights = optimization_result.get('weights', [])
            
            for asset, weight in zip(assets, weights):
                alpha_score = self.alpha_scores.get(asset, {}).get('alpha_score', 0)
                beta_estimate = self.beta_estimates.get(asset, {}).get('primary_beta', 1)
                
                report.append(f"{asset}: {weight:.2%}")
                report.append(f"  Alpha Score: {alpha_score:.4f}")
                report.append(f"  Beta Estimate: {beta_estimate:.4f}")
            
            # Portfolio Metrics
            report.append(f"\n{'='*60}")
            report.append("PORTFOLIO PERFORMANCE METRICS")
            report.append(f"{'='*60}")
            
            metrics = [
                'expected_return', 'volatility', 'sharpe_ratio',
                'effective_assets', 'max_weight', 'concentration'
            ]
            
            for metric in metrics:
                if metric in optimization_result:
                    value = optimization_result[metric]
                    if metric in ['expected_return', 'volatility']:
                        report.append(f"{metric.replace('_', ' ').title()}: {value:.2%}")
                    elif metric == 'sharpe_ratio':
                        report.append(f"Sharpe Ratio: {value:.4f}")
                    elif metric == 'effective_assets':
                        report.append(f"Effective Number of Assets: {value:.2f}")
                    elif metric in ['max_weight', 'concentration']:
                        report.append(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            
            # Risk Contributions
            if 'risk_contributions' in optimization_result:
                report.append(f"\nRISK CONTRIBUTIONS:")
                risk_contribs = optimization_result['risk_contributions']
                for asset, contrib in risk_contribs.items():
                    report.append(f"  {asset}: {contrib:.2%}")
        
        else:
            report.append(f"\nOPTIMIZATION STATUS: FAILED")
            report.append(f"MESSAGE: {optimization_result.get('message', 'Unknown error')}")
        
        # Alpha Signal Summary
        report.append(f"\n{'='*60}")
        report.append("ALPHA SIGNAL SUMMARY")
        report.append(f"{'='*60}")
        
        if self.alpha_scores:
            for asset, scores in self.alpha_scores.items():
                report.append(f"{asset}:")
                report.append(f"  Alpha Score: {scores['alpha_score']:.4f}")
                report.append(f"  Factor Count: {scores['factor_count']}")
                report.append(f"  Signal Strength: {scores['signal_strength']:.4f}")
        
        # Beta Estimate Summary
        report.append(f"\n{'='*60}")
        report.append("BETA ESTIMATE SUMMARY")
        report.append(f"{'='*60}")
        
        if self.beta_estimates:
            for asset, estimates in self.beta_estimates.items():
                report.append(f"{asset}:")
                report.append(f"  Primary Beta: {estimates['primary_beta']:.4f}")
                report.append(f"  Beta Consensus: {estimates['beta_consensus']:.4f}")
                report.append(f"  Beta Uncertainty: {estimates['beta_uncertainty']:.4f}")
        
        report.append(f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_text = "\n".join(report)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text


def main():
    """Example usage of AlphaBetaOptimizer."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== Alpha-Beta Portfolio Optimizer Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'SPY']
    
    # Get price data
    df_list = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        import yfinance as yf
        stock_data = yf.download(ticker, period="2y", progress=False)
        
        for date, row in stock_data.iterrows():
            df_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'tic': ticker,
                'close': row['Close']
            })
    
    df = pd.DataFrame(df_list)
    
    # Initialize optimizer
    optimizer = AlphaBetaOptimizer(df, market_index='SPY')
    
    # Test assets (exclude market index)
    test_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG']
    
    # Generate alpha signals
    print("\n1. Generating alpha signals...")
    alpha_signals = optimizer.generate_alpha_signals(test_assets, include_fundamental=False)
    
    print("Alpha signals summary:")
    print(alpha_signals.round(4).to_string(index=False))
    
    # Estimate risk models
    print("\n2. Estimating beta risk models...")
    beta_estimates = optimizer.estimate_risk_models(test_assets)
    
    print("Beta estimates summary:")
    print(beta_estimates.round(4).to_string(index=False))
    
    # Portfolio optimization - different methods
    methods = ['max_sharpe', 'min_variance', 'max_utility', 'risk_parity']
    optimization_results = {}
    
    for method in methods:
        print(f"\n3.{methods.index(method)+1}. Optimizing portfolio using {method}...")
        
        try:
            result = optimizer.optimize_portfolio(
                test_assets, 
                method=method,
                risk_aversion=2.0 if method == 'max_utility' else 1.0
            )
            optimization_results[method] = result
            
            if result['success']:
                print(f"✓ {method} optimization successful")
                print(f"  Expected Return: {result.get('expected_return', 0):.2%}")
                print(f"  Volatility: {result.get('volatility', 0):.2%}")
                print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}")
                
                print(f"  Top 3 Holdings:")
                weights = result['weights']
                assets = result['assets']
                top_indices = np.argsort(weights)[-3:][::-1]
                
                for idx in top_indices:
                    print(f"    {assets[idx]}: {weights[idx]:.2%}")
            else:
                print(f"✗ {method} optimization failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Error in {method} optimization: {e}")
    
    # Generate comprehensive report for best method
    best_method = 'max_sharpe'  # Default choice
    if best_method in optimization_results:
        print(f"\n4. Generating comprehensive report for {best_method}...")
        report_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/alpha_beta_optimization_report.txt'
        
        optimizer.optimization_results = optimization_results[best_method]
        report = optimizer.generate_optimization_report(save_path=report_path)
        
        print("Sample report excerpt:")
        print("\n".join(report.split("\n")[:20]))  # Show first 20 lines
    
    # Simple backtest
    print(f"\n5. Running simple backtest...")
    try:
        backtest_results = optimizer.backtest_strategy(
            test_assets[:4],  # Use fewer assets for faster backtest
            rebalance_frequency='quarterly',
            lookbook_period=120,
            method='max_sharpe'
        )
        
        if not backtest_results.empty:
            print("✓ Backtest completed successfully")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
    
    # Save optimization comparison
    print(f"\n6. Saving results...")
    
    # Create comparison DataFrame
    comparison_data = []
    for method, result in optimization_results.items():
        if result['success']:
            row = {
                'method': method,
                'expected_return': result.get('expected_return', 0),
                'volatility': result.get('volatility', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'effective_assets': result.get('effective_assets', 0),
                'max_weight': result.get('max_weight', 0),
                'concentration': result.get('concentration', 0)
            }
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/optimization_methods_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"Method comparison saved to: {comparison_path}")
        print("\nOptimization methods comparison:")
        print(comparison_df.round(4).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("ALPHA-BETA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Assets analyzed: {len(test_assets)}")
    print(f"Methods tested: {len(methods)}")
    print(f"Successful optimizations: {sum(1 for r in optimization_results.values() if r.get('success'))}")


if __name__ == "__main__":
    main()