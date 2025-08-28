"""
CVaR-Based Risk Models for Beta Estimation
Conditional Value at Risk (Expected Shortfall) approaches for systematic risk estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats, optimize
from scipy.stats import norm, t as student_t
from sklearn.linear_model import LinearRegression, QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import QuantileRegressor
    QUANTILE_REG_AVAILABLE = True
except ImportError:
    QUANTILE_REG_AVAILABLE = False


class CVaRRiskModels:
    """
    CVaR-based risk model implementations for beta estimation.
    
    This class provides various CVaR (Conditional Value at Risk) approaches
    for systematic risk estimation, including tail-risk beta and 
    expected shortfall-based measures.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 market_index: str = 'SPY',
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize with return data.
        
        Args:
            data: DataFrame with asset returns
            market_index: Market index symbol
            risk_free_rate: Annual risk-free rate
            confidence_levels: List of confidence levels for VaR/CVaR calculations
        """
        self.data = data.copy()
        self.market_index = market_index
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252
        self.confidence_levels = confidence_levels
        
        # Prepare data
        self.returns_data = self._prepare_returns_data()
        
    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data."""
        if 'close' in self.data.columns:
            returns_list = []
            
            for tic in self.data['tic'].unique():
                tic_data = self.data[self.data['tic'] == tic].copy()
                tic_data = tic_data.sort_values('date')
                tic_data['returns'] = tic_data['close'].pct_change()
                returns_list.append(tic_data[['date', 'tic', 'returns']])
            
            returns_df = pd.concat(returns_list, ignore_index=True)
            returns_wide = returns_df.pivot(index='date', columns='tic', values='returns')
            returns_wide.index = pd.to_datetime(returns_wide.index)
            
            return returns_wide
        else:
            return self.data.copy()
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', 'cornish_fisher'
            
        Returns:
            VaR estimate
        """
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            return np.nan
        
        if method == 'historical':
            # Historical simulation
            return np.percentile(clean_returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric (normal) VaR
            mean = clean_returns.mean()
            std = clean_returns.std()
            z_score = norm.ppf(1 - confidence_level)
            return mean + z_score * std
        
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion
            mean = clean_returns.mean()
            std = clean_returns.std()
            skewness = clean_returns.skew()
            kurtosis = clean_returns.kurtosis()
            
            z = norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher adjustment
            z_cf = (z + 
                    (z**2 - 1) * skewness / 6 + 
                    (z**3 - 3*z) * (kurtosis - 3) / 24 - 
                    (2*z**3 - 5*z) * skewness**2 / 36)
            
            return mean + z_cf * std
        
        else:
            # Default to historical
            return np.percentile(clean_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            method: 'historical', 'parametric'
            
        Returns:
            CVaR estimate
        """
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            return np.nan
        
        if method == 'historical':
            # Historical CVaR
            var_threshold = self.calculate_var(clean_returns, confidence_level, 'historical')
            tail_losses = clean_returns[clean_returns <= var_threshold]
            
            if len(tail_losses) == 0:
                return var_threshold
            
            return tail_losses.mean()
        
        elif method == 'parametric':
            # Parametric (normal) CVaR
            mean = clean_returns.mean()
            std = clean_returns.std()
            
            # For normal distribution: CVaR = μ - σ * φ(Φ^(-1)(α)) / α
            # where α = 1 - confidence_level
            alpha = 1 - confidence_level
            z_alpha = norm.ppf(alpha)
            phi_z = norm.pdf(z_alpha)
            
            return mean - std * phi_z / alpha
        
        else:
            return self.calculate_cvar(returns, confidence_level, 'historical')
    
    def estimate_cvar_beta(self, 
                          asset: str, 
                          confidence_level: float = 0.95,
                          method: str = 'regression',
                          window: int = 252) -> Dict:
        """
        Estimate CVaR-based beta.
        
        Args:
            asset: Asset symbol
            confidence_level: Confidence level for CVaR
            method: 'regression', 'ratio', 'conditional'
            window: Estimation window
            
        Returns:
            CVaR beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return {'cvar_beta': np.nan, 'linear_beta': np.nan}
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if len(asset_returns) < window:
            recent_asset = asset_returns
            recent_market = market_returns
        else:
            recent_asset = asset_returns.tail(window)
            recent_market = market_returns.tail(window)
        
        if len(recent_asset) < 50:
            return {'cvar_beta': np.nan, 'linear_beta': np.nan}
        
        # Excess returns
        asset_excess = recent_asset - self.daily_rf_rate
        market_excess = recent_market - self.daily_rf_rate
        
        # Calculate linear beta for comparison
        linear_beta = np.cov(asset_excess, market_excess)[0, 1] / np.var(market_excess)
        
        if method == 'regression':
            # CVaR beta using quantile regression on tail events
            return self._cvar_beta_regression(
                asset_excess, market_excess, confidence_level, linear_beta
            )
        
        elif method == 'ratio':
            # CVaR beta as ratio of CVaRs
            return self._cvar_beta_ratio(
                asset_excess, market_excess, confidence_level, linear_beta
            )
        
        elif method == 'conditional':
            # Conditional beta during market stress
            return self._cvar_beta_conditional(
                asset_excess, market_excess, confidence_level, linear_beta
            )
        
        else:
            return {'cvar_beta': np.nan, 'linear_beta': linear_beta}
    
    def _cvar_beta_regression(self, 
                            asset_returns: pd.Series, 
                            market_returns: pd.Series,
                            confidence_level: float,
                            linear_beta: float) -> Dict:
        """Estimate CVaR beta using quantile regression on tail events."""
        try:
            # Identify market stress periods (negative tail events)
            market_var = self.calculate_var(market_returns, confidence_level, 'historical')
            stress_mask = market_returns <= market_var
            
            if stress_mask.sum() < 10:  # Need sufficient tail observations
                return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'insufficient_data'}
            
            # Extract tail data
            tail_asset = asset_returns[stress_mask]
            tail_market = market_returns[stress_mask]
            
            # Quantile regression on tail data (if available)
            if QUANTILE_REG_AVAILABLE:
                try:
                    # Use median regression on tail data
                    qr = QuantileRegressor(quantile=0.5, alpha=0.01)
                    X = tail_market.values.reshape(-1, 1)
                    y = tail_asset.values
                    
                    qr.fit(X, y)
                    cvar_beta = qr.coef_[0]
                    alpha = qr.intercept_
                    
                except:
                    # Fallback to OLS on tail data
                    cvar_beta = np.cov(tail_asset, tail_market)[0, 1] / np.var(tail_market)
                    alpha = tail_asset.mean() - cvar_beta * tail_market.mean()
            else:
                # OLS on tail data
                cvar_beta = np.cov(tail_asset, tail_market)[0, 1] / np.var(tail_market)
                alpha = tail_asset.mean() - cvar_beta * tail_market.mean()
            
            # Additional metrics
            tail_r_squared = np.corrcoef(tail_asset, tail_market)[0, 1]**2
            
            return {
                'cvar_beta': cvar_beta,
                'linear_beta': linear_beta,
                'tail_alpha': alpha,
                'tail_r_squared': tail_r_squared,
                'tail_observations': len(tail_asset),
                'method': 'regression'
            }
            
        except Exception as e:
            print(f"Error in CVaR regression beta: {e}")
            return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'error'}
    
    def _cvar_beta_ratio(self, 
                        asset_returns: pd.Series, 
                        market_returns: pd.Series,
                        confidence_level: float,
                        linear_beta: float) -> Dict:
        """Estimate CVaR beta as ratio of CVaRs."""
        try:
            # Calculate CVaR for both asset and market
            asset_cvar = self.calculate_cvar(asset_returns, confidence_level, 'historical')
            market_cvar = self.calculate_cvar(market_returns, confidence_level, 'historical')
            
            if np.isnan(asset_cvar) or np.isnan(market_cvar) or market_cvar == 0:
                return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'nan_cvar'}
            
            # CVaR beta as ratio
            cvar_beta = asset_cvar / market_cvar
            
            return {
                'cvar_beta': cvar_beta,
                'linear_beta': linear_beta,
                'asset_cvar': asset_cvar,
                'market_cvar': market_cvar,
                'method': 'ratio'
            }
            
        except Exception as e:
            print(f"Error in CVaR ratio beta: {e}")
            return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'error'}
    
    def _cvar_beta_conditional(self, 
                             asset_returns: pd.Series, 
                             market_returns: pd.Series,
                             confidence_level: float,
                             linear_beta: float) -> Dict:
        """Estimate conditional beta during market stress periods."""
        try:
            # Define market stress as worst (1-confidence_level) quantile
            market_var = self.calculate_var(market_returns, confidence_level, 'historical')
            
            # Conditional beta during stress
            stress_mask = market_returns <= market_var
            
            if stress_mask.sum() < 10:
                return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'insufficient_stress'}
            
            stress_asset = asset_returns[stress_mask]
            stress_market = market_returns[stress_mask]
            
            # Conditional covariance and variance
            stress_beta = np.cov(stress_asset, stress_market)[0, 1] / np.var(stress_market)
            
            # Non-stress beta for comparison
            normal_mask = ~stress_mask
            if normal_mask.sum() > 10:
                normal_asset = asset_returns[normal_mask]
                normal_market = market_returns[normal_mask]
                normal_beta = np.cov(normal_asset, normal_market)[0, 1] / np.var(normal_market)
            else:
                normal_beta = linear_beta
            
            return {
                'cvar_beta': stress_beta,
                'linear_beta': linear_beta,
                'normal_beta': normal_beta,
                'stress_observations': stress_mask.sum(),
                'normal_observations': normal_mask.sum(),
                'beta_ratio': stress_beta / normal_beta if normal_beta != 0 else np.nan,
                'method': 'conditional'
            }
            
        except Exception as e:
            print(f"Error in conditional CVaR beta: {e}")
            return {'cvar_beta': linear_beta, 'linear_beta': linear_beta, 'method': 'error'}
    
    def calculate_downside_cvar_beta(self, 
                                   asset: str, 
                                   confidence_level: float = 0.95,
                                   window: int = 252) -> Dict:
        """
        Calculate downside CVaR beta (only during market downturns).
        
        Args:
            asset: Asset symbol
            confidence_level: Confidence level
            window: Estimation window
            
        Returns:
            Downside CVaR beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return {'downside_cvar_beta': np.nan}
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        # Use recent data
        if len(asset_returns) >= window:
            recent_asset = asset_returns.tail(window) - self.daily_rf_rate
            recent_market = market_returns.tail(window) - self.daily_rf_rate
        else:
            recent_asset = asset_returns - self.daily_rf_rate
            recent_market = market_returns - self.daily_rf_rate
        
        # Only consider market down days
        down_mask = recent_market < 0
        
        if down_mask.sum() < 20:
            return {'downside_cvar_beta': np.nan, 'downside_observations': down_mask.sum()}
        
        down_asset = recent_asset[down_mask]
        down_market = recent_market[down_mask]
        
        try:
            # Calculate CVaR for downside periods
            asset_downside_cvar = self.calculate_cvar(down_asset, confidence_level)
            market_downside_cvar = self.calculate_cvar(down_market, confidence_level)
            
            # Downside beta
            if market_downside_cvar != 0 and not np.isnan(market_downside_cvar):
                downside_cvar_beta = asset_downside_cvar / market_downside_cvar
            else:
                # Fallback to covariance method
                downside_cvar_beta = np.cov(down_asset, down_market)[0, 1] / np.var(down_market)
            
            # Regular downside beta for comparison
            downside_linear_beta = np.cov(down_asset, down_market)[0, 1] / np.var(down_market)
            
            return {
                'downside_cvar_beta': downside_cvar_beta,
                'downside_linear_beta': downside_linear_beta,
                'asset_downside_cvar': asset_downside_cvar,
                'market_downside_cvar': market_downside_cvar,
                'downside_observations': down_mask.sum(),
                'downside_correlation': np.corrcoef(down_asset, down_market)[0, 1]
            }
            
        except Exception as e:
            print(f"Error calculating downside CVaR beta: {e}")
            return {'downside_cvar_beta': np.nan, 'downside_observations': down_mask.sum()}
    
    def estimate_time_varying_cvar_beta(self, 
                                      asset: str, 
                                      confidence_level: float = 0.95,
                                      window: int = 120,
                                      method: str = 'rolling') -> pd.DataFrame:
        """
        Estimate time-varying CVaR beta.
        
        Args:
            asset: Asset symbol
            confidence_level: Confidence level
            window: Rolling window size
            method: 'rolling' or 'exponential'
            
        Returns:
            DataFrame with time-varying CVaR beta estimates
        """
        if asset not in self.returns_data.columns or self.market_index not in self.returns_data.columns:
            return pd.DataFrame()
        
        asset_returns = self.returns_data[asset].dropna()
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns[common_dates]
        market_returns = market_returns[common_dates]
        
        if len(asset_returns) < window * 2:
            return pd.DataFrame()
        
        results = []
        
        for i in range(window, len(asset_returns)):
            if method == 'rolling':
                window_asset = asset_returns.iloc[i-window:i]
                window_market = market_returns.iloc[i-window:i]
            else:  # exponential weighting
                # Simple exponential weighting
                decay = 0.94
                weights = np.array([decay**j for j in range(window-1, -1, -1)])
                weights = weights / weights.sum()
                
                window_asset = asset_returns.iloc[i-window:i] * weights
                window_market = market_returns.iloc[i-window:i] * weights
            
            # Calculate CVaR beta for this window
            try:
                window_result = self.estimate_cvar_beta(
                    asset, confidence_level, 'conditional', window=len(window_asset)
                )
                
                result = {
                    'date': asset_returns.index[i],
                    'cvar_beta': window_result.get('cvar_beta', np.nan),
                    'linear_beta': window_result.get('linear_beta', np.nan),
                    'method': window_result.get('method', 'unknown')
                }
                
                # Add CVaR estimates
                asset_excess = window_asset - self.daily_rf_rate
                market_excess = window_market - self.daily_rf_rate
                
                result['asset_cvar'] = self.calculate_cvar(asset_excess, confidence_level)
                result['market_cvar'] = self.calculate_cvar(market_excess, confidence_level)
                
                results.append(result)
                
            except Exception as e:
                result = {
                    'date': asset_returns.index[i],
                    'cvar_beta': np.nan,
                    'linear_beta': np.nan,
                    'asset_cvar': np.nan,
                    'market_cvar': np.nan,
                    'method': 'error'
                }
                results.append(result)
        
        return pd.DataFrame(results).set_index('date')
    
    def calculate_portfolio_cvar_beta(self, 
                                    assets: List[str], 
                                    weights: Optional[np.ndarray] = None,
                                    confidence_level: float = 0.95,
                                    method: str = 'ratio') -> Dict:
        """
        Calculate portfolio CVaR beta.
        
        Args:
            assets: List of asset symbols
            weights: Portfolio weights
            confidence_level: Confidence level
            method: CVaR beta estimation method
            
        Returns:
            Portfolio CVaR beta estimates
        """
        if weights is None:
            weights = np.ones(len(assets)) / len(assets)
        
        weights = np.array(weights)
        
        # Check available assets
        available_assets = [a for a in assets if a in self.returns_data.columns]
        
        if len(available_assets) == 0:
            return {'portfolio_cvar_beta': np.nan}
        
        # Get returns for available assets
        asset_returns = self.returns_data[available_assets].dropna()
        
        if self.market_index not in self.returns_data.columns:
            return {'portfolio_cvar_beta': np.nan}
        
        market_returns = self.returns_data[self.market_index].dropna()
        
        # Align data
        common_dates = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns.loc[common_dates]
        market_returns = market_returns[common_dates]
        
        # Calculate portfolio returns
        asset_indices = [assets.index(asset) for asset in available_assets]
        portfolio_weights = weights[asset_indices]
        portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Renormalize
        
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        
        # Create temporary asset entry for portfolio
        temp_returns = self.returns_data.copy()
        temp_returns['PORTFOLIO'] = portfolio_returns
        
        original_returns = self.returns_data
        self.returns_data = temp_returns
        
        try:
            # Calculate portfolio CVaR beta
            portfolio_result = self.estimate_cvar_beta('PORTFOLIO', confidence_level, method)
            
            # Also calculate individual asset CVaR betas for decomposition
            individual_results = {}
            for asset in available_assets:
                individual_results[asset] = self.estimate_cvar_beta(asset, confidence_level, method)
            
            # Calculate contribution
            contributions = {}
            total_cvar_beta = portfolio_result.get('cvar_beta', np.nan)
            
            for i, asset in enumerate(available_assets):
                asset_cvar_beta = individual_results[asset].get('cvar_beta', np.nan)
                if not np.isnan(asset_cvar_beta):
                    contributions[asset] = portfolio_weights[i] * asset_cvar_beta
                else:
                    contributions[asset] = np.nan
            
            return {
                'portfolio_cvar_beta': total_cvar_beta,
                'portfolio_linear_beta': portfolio_result.get('linear_beta', np.nan),
                'individual_cvar_betas': {asset: result.get('cvar_beta', np.nan) 
                                        for asset, result in individual_results.items()},
                'contributions': contributions,
                'effective_weights': dict(zip(available_assets, portfolio_weights)),
                'method': method
            }
            
        finally:
            # Restore original data
            self.returns_data = original_returns
    
    def analyze_cvar_beta_stability(self, 
                                  asset: str, 
                                  confidence_levels: List[float] = None,
                                  methods: List[str] = None,
                                  window: int = 252) -> pd.DataFrame:
        """
        Analyze CVaR beta stability across confidence levels and methods.
        
        Args:
            asset: Asset symbol
            confidence_levels: List of confidence levels to test
            methods: List of methods to test
            window: Estimation window
            
        Returns:
            DataFrame with stability analysis results
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        if methods is None:
            methods = ['regression', 'ratio', 'conditional']
        
        results = []
        
        for conf_level in confidence_levels:
            for method in methods:
                try:
                    result = self.estimate_cvar_beta(asset, conf_level, method, window)
                    
                    result_row = {
                        'asset': asset,
                        'confidence_level': conf_level,
                        'method': method,
                        'cvar_beta': result.get('cvar_beta', np.nan),
                        'linear_beta': result.get('linear_beta', np.nan),
                        'estimation_method': result.get('method', method)
                    }
                    
                    # Add method-specific metrics
                    if method == 'regression':
                        result_row['tail_observations'] = result.get('tail_observations', np.nan)
                        result_row['tail_r_squared'] = result.get('tail_r_squared', np.nan)
                    elif method == 'ratio':
                        result_row['asset_cvar'] = result.get('asset_cvar', np.nan)
                        result_row['market_cvar'] = result.get('market_cvar', np.nan)
                    elif method == 'conditional':
                        result_row['stress_observations'] = result.get('stress_observations', np.nan)
                        result_row['beta_ratio'] = result.get('beta_ratio', np.nan)
                    
                    results.append(result_row)
                    
                except Exception as e:
                    print(f"Error in stability analysis for {asset}, {conf_level}, {method}: {e}")
        
        results_df = pd.DataFrame(results)
        
        # Add stability metrics
        if len(results_df) > 1:
            # CVaR beta stability
            cvar_betas = results_df['cvar_beta'].dropna()
            if len(cvar_betas) > 1:
                results_df['cvar_beta_std'] = cvar_betas.std()
                results_df['cvar_beta_range'] = cvar_betas.max() - cvar_betas.min()
        
        return results_df


def main():
    """Example usage of CVaRRiskModels."""
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    print("=== CVaR Risk Models Example ===")
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']  # Include market index
    
    # Get price data and convert to required format
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
    
    # Initialize CVaR models
    cvar_models = CVaRRiskModels(df, market_index='SPY')
    
    print(f"\nEstimating CVaR-based risk models...")
    
    # Test asset
    asset = 'AAPL'
    
    # 1. Basic VaR and CVaR calculations
    print(f"\n1. VaR and CVaR for {asset}:")
    asset_returns = cvar_models.returns_data[asset].dropna() - cvar_models.daily_rf_rate
    
    for conf_level in [0.95, 0.99]:
        var = cvar_models.calculate_var(asset_returns, conf_level, 'historical')
        cvar = cvar_models.calculate_cvar(asset_returns, conf_level, 'historical')
        
        print(f"  {conf_level*100:.0f}% VaR: {var:.4f}")
        print(f"  {conf_level*100:.0f}% CVaR: {cvar:.4f}")
    
    # 2. CVaR Beta Estimation - Different Methods
    print(f"\n2. CVaR Beta Estimation for {asset}:")
    
    methods = ['regression', 'ratio', 'conditional']
    cvar_results = {}
    
    for method in methods:
        result = cvar_models.estimate_cvar_beta(asset, 0.95, method)
        cvar_results[method] = result
        
        print(f"\n  {method.capitalize()} Method:")
        print(f"    CVaR Beta: {result.get('cvar_beta', 'N/A'):.4f}")
        print(f"    Linear Beta: {result.get('linear_beta', 'N/A'):.4f}")
        
        if method == 'regression':
            print(f"    Tail Observations: {result.get('tail_observations', 'N/A')}")
        elif method == 'ratio':
            print(f"    Asset CVaR: {result.get('asset_cvar', 'N/A'):.6f}")
            print(f"    Market CVaR: {result.get('market_cvar', 'N/A'):.6f}")
        elif method == 'conditional':
            print(f"    Stress Observations: {result.get('stress_observations', 'N/A')}")
            print(f"    Normal Beta: {result.get('normal_beta', 'N/A'):.4f}")
            print(f"    Beta Ratio: {result.get('beta_ratio', 'N/A'):.4f}")
    
    # 3. Downside CVaR Beta
    print(f"\n3. Downside CVaR Beta for {asset}:")
    downside_result = cvar_models.calculate_downside_cvar_beta(asset, 0.95)
    
    print(f"Downside CVaR Beta: {downside_result.get('downside_cvar_beta', 'N/A'):.4f}")
    print(f"Downside Linear Beta: {downside_result.get('downside_linear_beta', 'N/A'):.4f}")
    print(f"Downside Observations: {downside_result.get('downside_observations', 'N/A')}")
    print(f"Downside Correlation: {downside_result.get('downside_correlation', 'N/A'):.4f}")
    
    # 4. Time-Varying CVaR Beta (sample)
    print(f"\n4. Time-Varying CVaR Beta Analysis:")
    tv_cvar_beta = cvar_models.estimate_time_varying_cvar_beta(asset, 0.95, window=60)
    
    if not tv_cvar_beta.empty:
        recent_cvar_beta = tv_cvar_beta['cvar_beta'].dropna()
        if len(recent_cvar_beta) > 0:
            print(f"Recent CVaR Beta: {recent_cvar_beta.iloc[-1]:.4f}")
            print(f"CVaR Beta Volatility: {recent_cvar_beta.std():.4f}")
            print(f"CVaR Beta Range: [{recent_cvar_beta.min():.4f}, {recent_cvar_beta.max():.4f}]")
    
    # 5. Portfolio CVaR Beta
    print(f"\n5. Portfolio CVaR Beta Analysis:")
    portfolio_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    portfolio_result = cvar_models.calculate_portfolio_cvar_beta(
        portfolio_assets, 
        confidence_level=0.95, 
        method='conditional'
    )
    
    print(f"Portfolio CVaR Beta: {portfolio_result.get('portfolio_cvar_beta', 'N/A'):.4f}")
    print(f"Portfolio Linear Beta: {portfolio_result.get('portfolio_linear_beta', 'N/A'):.4f}")
    
    print("Individual CVaR Beta Contributions:")
    contributions = portfolio_result.get('contributions', {})
    for asset_name, contribution in contributions.items():
        weight = portfolio_result['effective_weights'].get(asset_name, 0)
        print(f"  {asset_name}: {contribution:.4f} (weight: {weight:.2%})")
    
    # 6. Stability Analysis
    print(f"\n6. CVaR Beta Stability Analysis:")
    stability_df = cvar_models.analyze_cvar_beta_stability(asset)
    
    if not stability_df.empty:
        print("Stability across methods and confidence levels:")
        display_cols = ['confidence_level', 'method', 'cvar_beta', 'linear_beta']
        available_cols = [col for col in display_cols if col in stability_df.columns]
        print(stability_df[available_cols].round(4).to_string(index=False))
    
    # Save results
    output_path = '/Users/mengfanlong/Downloads/Portfolio_Optimization_system/cvar_risk_analysis.csv'
    
    # Create comprehensive results
    summary_results = []
    for test_asset in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        reg_result = cvar_models.estimate_cvar_beta(test_asset, 0.95, 'regression')
        ratio_result = cvar_models.estimate_cvar_beta(test_asset, 0.95, 'ratio')
        cond_result = cvar_models.estimate_cvar_beta(test_asset, 0.95, 'conditional')
        downside_result = cvar_models.calculate_downside_cvar_beta(test_asset, 0.95)
        
        summary_results.append({
            'asset': test_asset,
            'linear_beta': reg_result.get('linear_beta', np.nan),
            'cvar_beta_regression': reg_result.get('cvar_beta', np.nan),
            'cvar_beta_ratio': ratio_result.get('cvar_beta', np.nan),
            'cvar_beta_conditional': cond_result.get('cvar_beta', np.nan),
            'downside_cvar_beta': downside_result.get('downside_cvar_beta', np.nan),
            'asset_cvar_95': ratio_result.get('asset_cvar', np.nan),
            'downside_observations': downside_result.get('downside_observations', np.nan)
        })
    
    results_df = pd.DataFrame(summary_results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"CVaR risk analysis completed for {len(summary_results)} assets")


if __name__ == "__main__":
    main()